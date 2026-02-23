import concurrent.futures
import os
import subprocess
import re
import logging

from pathlib import Path
from typing import Union, Tuple, Optional
from io import StringIO

import pandas as pd
import numpy as np

from microhh_processing.enums import Cluster, Processor, Partition, MailType
from microhh_processing.constants import PrintColor
from microhh_processing import utils

logger = logging.getLogger(__name__)


class AlpsSetup:

    # define number of cores used on Alps
    PROCESSOR_VERSIONS = {
        # no gpu feature, cpu = cores per socket (64) * sockets (2)
        "eiger": {"cpu": 128},
        # no cpu frature, gpu = cores per socket (72) * sockets (4)
        "santis": {"gpu": 288},
        # not implemented
        "daint": {None},
        # not implemented
        "clariden": {None},
        # not implemented
        "todi": {None},
    }

    PARTITION_CONDITIONS = {
        "debug": {"time_limit": 30 * 60, "max_nodes": 10},
        "normal": {"time_limit": 24 * 60 * 60, "max_nodes": 480},
        "xfer": {"time_limit": 24 * 60 * 60, "max_nodes": 1},
        "prepost": {"time_limit": 30 * 60, "max_nodes": 1},
        "low": {"time_limit": 24 * 30 * 60, "max_nodes": 124},
    }

    def __init__(
        self,
        cluster: Union[Cluster, str],
        processor: Union[Processor, str],
        mpi: bool,
        account: str,
        mail_type: Union[MailType, str] = MailType.NONE,
        email: Optional[str] = None,
        run_time: str = "00:30:00",
        nodes: int = 1,
        ntasks_per_core: int = 1,
        ntasks_per_node: Optional[int] = None,
        cpus_per_task: int = 1,
        gpus_per_task: int = 1,
        partition: Union[Partition, str] = Partition.DEBUG,
        uenv: str = "",
    ):
        """A class handling everything around the workflow on cscs's supercomputers.
        Is used to create and submit job scripts.

        cluster (str) ['clariden', 'daint', 'eiger', 'santis', 'todi']:
            Cluster on which to run jobs.
        processor (str) ['cpu', 'gpu']:
            Processor version to be used to run the model.
        mpi (bool):
            Indicates if mpi should be enabled.
        account (str):
            Account on which the job should be submitted from.
        mail_type (str):
            Specify which events should trigger a mail to the user.
            Options are 'NONE', 'BEGIN', 'END', 'FAIL', 'REQUEUE', 'ALL'
        email (str):
            E-mail adress of the user.
        run_time (str, optional. Default: "00:30:00"):
            Specify the job duration. Determines the queuing time.
            Format: "HH:MM:SS".
        nodes (int, optional. Default: 1):
            Number of nodes to use for running MicroHH.
        ntasks_per_core (int, optional. Default: 1):
            Number of tasks per core. If > 1, hyperthreading is turned on.
        ntasks_per_node (int, optional):
            Number of tasks per node.
            If not specified, the default for 'cpu' is set to
            128 and 288 for 'gpu'. If 'mpi = False', it is set to 1.
        cpus_per_task (int, optional. Default: 1):
            Number of cpus per task.
            Specify the number of cpus per task.
            Defines the number of OpenMP threads per MPI rank.
            The maximum value depends on the number of tasks per node.
        gpus_per_task (int, optional. Default: 1):
            Number of gpus per task.
            Specify the number of gpus per task.
        partition (str, optional. Default: 'debug')
            ['debug', 'normal', 'prepost', 'low']:
            Queue to be used on alps.
        uenv (str, optional):
            User environment to be used for running a job
            e.g. 'prgenv-gnu'
        """
        self._cluster = cluster
        self._processor = processor
        self.mpi = mpi
        self.account = account
        self.mail_type = mail_type.value if isinstance(mail_type, MailType) else str(mail_type.upper())
        self.email = email
        self.run_time = run_time
        self.nodes = nodes
        self.ntasks_per_core = ntasks_per_core
        self.ntasks_per_node = ntasks_per_node or self._get_default_ntasks_per_node()
        self.cpus_per_task = cpus_per_task
        self.gpus_per_task = gpus_per_task
        self.partition = partition
        self.uenv = uenv

        self._check_wall_time_valid()
        self._check_node_number_valid()
        self._check_processor_valid()
        self._check_mail_setup_valid()

        # check if the user wants to remotely submit jobs via ssh
        self._set_ssh_connect()

    @property
    def cluster(self) -> str:
        cluster_val = self._cluster.value if isinstance(self._cluster, Cluster) else str(self._cluster).lower()
        if cluster_val in ["santis", "eiger"]:
            return cluster_val
        elif cluster_val in self.PROCESSOR_VERSIONS.keys():
            raise NotImplementedError(f"Cluster '{cluster_val}' is not yet implemented.")
        else:
            raise ValueError(f"{self._cluster} is invalid.")

    @property
    def processor(self) -> str:
        processor_val = self._processor.value if isinstance(self._processor, Processor) else str(self._processor).lower()
        if processor_val in self.PROCESSOR_VERSIONS[self.cluster]:
            return processor_val
        else:
            raise ValueError(
                f"{processor_val} is not available on {self.cluster}."
            )

    @property
    def partition_key(self) -> str:
        """Get partition value as string for dictionary lookups."""
        return self.partition.value if isinstance(self.partition, Partition) else str(self.partition).lower()

    def _get_default_ntasks_per_node(self) -> int:
        """Determines the default number of tasks per node
        if not specified in __init__

        Returns:
            ntasks_per_node (int):
                Number of tasks per node to be used depending on the cluster on alps.
                If 'mpi' is disabled, the tasks per node are set to 1.
        """
        if self.mpi:
            return (
                self.PROCESSOR_VERSIONS[self.cluster][self.processor]
                * self.ntasks_per_core
            )
        else:
            return 1

    def _check_wall_time_valid(self):
        """Checks if the setup is valid to be run on alps based
        on the selected queue, wall time and number of nodes.

        Raises:
            ValueError: Raise error if the wall time is
                too long for the selected queue.
            ValueError: Raise error if the number of nodes is
                too large for the selected queue.
        """
        # Check if CSCS setup is valid
        run_time_delta = self._parse_run_time(self.run_time)
        conditions = self.PARTITION_CONDITIONS.get(self.partition_key, {})
        if (
            conditions.get("time_limit")
            and run_time_delta.total_seconds() > conditions["time_limit"]
        ):
            raise ValueError("Invalid time limit for the selected partition.")
        
    def _check_node_number_valid(self):
        conditions = self.PARTITION_CONDITIONS.get(self.partition_key, {})
        if conditions.get("max_nodes") and self.nodes > conditions["max_nodes"]:
            raise ValueError("Invalid number of nodes for the selected partition.")
        
    def _check_processor_valid(self):
        if self.processor not in self.PROCESSOR_VERSIONS[self.cluster]:
            raise ValueError(
                f"{self.processor} is not available on {self.cluster}."
            )

    def _check_mail_setup_valid(self):
        mail_val = self.mail_type.value if isinstance(self.mail_type, MailType) else self.mail_type
        if mail_val != "NONE" and self.email is None:
            raise ValueError(
                "Please provide an email address to receive notifications."
            )
        
    def _set_ssh_connect(self):
        # get name of the current machine
        current_cluster = os.environ["CLUSTER_NAME"]
        if current_cluster != self.cluster:
            user = os.environ["USER"]
            self.ssh_connect = f"{user}@{self.cluster}"
            logger.info(f"SSH connection set to {self.ssh_connect}")
        else:
            self.ssh_connect = None

    def _parse_run_time(self, run_time: str) -> pd.Timedelta:
        """Parse the run_time string and return a timedelta object

        Args:
            run_time (str):
                Specify the job duration. Determines the queuing time.
                Format: "HH:MM:SS".

        Raises:
            ValueError: Raise error if the format is not "HH:MM:SS".

        Returns:
            pd.Timedelta: Timedelta of the wall clock time.
        """

        try:
            return pd.to_timedelta(run_time)
        except ValueError:
            raise ValueError("Invalid run_time format. Use HH:MM:SS")

    def process_case_on_alps(
        self,
        case,
        py_file_path: str,
        job_name: str,
        python_path: str = "python",
        additional_args: dict = {},
        overwrite: bool = False,
        start_jobs: bool = True,
    ):
        """Submit a bash script to process a case on alps.

        Args:
            case (init.Case):
                Case which should be processed.
            py_file_path (str):
                File to the python script which should be executed to process the case.
            job_name (str):
                Name of the job. Determines the name of the resulting slurm.out script
                and the job name in the queue.
            python_path (str):
                Path of the python environment to use. The default is /users/USER/bin/python3
            additional_args (dict, optional):
                Addtional arguments for the function called in the python script. Defaults to {}.
            start_jobs (bool, optional):
                Specify if the job should be starte. Defaults to True.
        """
        job_name_formatted = job_name.replace(" ", "_").replace("-", "_")

        # create command for bash script
        cmd = f"srun {python_path} {py_file_path} --case-name {case.name} --case-suffix {case.suffix}"

        if additional_args:
            additional_args_str = " ".join(
                [f"{key}={str(value)}" for key, value in additional_args.items()]
            )
            cmd = f"{cmd} --additional-args {additional_args_str}"

        # submit bash script to run post processing in parallel on alps
        job_file = Path(case.scratch_path, f"{job_name_formatted}.sbash")

        if job_file.exists() and overwrite:
            os.remove(job_file)

        if not job_file.exists() or overwrite:
            job_file = self.create_bash_script(
                commands=cmd,
                job_name=job_name_formatted,
                file_path=job_file,
                scratch_path=case.scratch_path,
            )
        else:
            logger.info(f"Reading existing bash script {job_file}")

        if start_jobs:
            # Launch the job on alps
            job_id = execute_bash_script(job_file, wait=False)

    def create_bash_script(
        self,
        commands: Union[str, list],
        job_name: str,
        file_path: Union[str, Path],
        scratch_path: Union[str, Path],
    ) -> Union[str, Path]:
        """Create bash script to run a simulation.
        For more info, see 
        - https://docs.cscs.ch/running/slurm/
        - https://slurm.schedmd.com/sbatch.html

        Args:
            start_jobs (bool, optional. Default: False):
                Specify if the job should be started.
            wait (bool, optional. Default: False):
                Specify if the next job should only be submitted
                after the previous one has stopped.
        """
        # make sure commands is a list:
        commands = utils.object_to_list(commands)

        multithread = "multithread" if self.ntasks_per_core > 1 else "nomultithread"
        constraint = "mc" if self.processor == "cpu" else "gpu"

        with open(file_path, "w") as bash_script:
            header = "#!/usr/bin/env bash\n\n"

            header += f"#SBATCH --exclusive\n"
            header += f"#SBATCH --job-name={job_name} \n"
            header += f"#SBATCH --account={self.account}\n"
            header += f"#SBATCH --mail-type={self.mail_type}\n"
            if self.email is not None:
                header += f"#SBATCH --mail-user={self.email}\n"
            header += f"#SBATCH --time={self.run_time}\n"
            header += f"#SBATCH --nodes={self.nodes}\n"
            header += f"#SBATCH --ntasks-per-core={self.ntasks_per_core}\n"
            header += f"#SBATCH --ntasks-per-node={self.ntasks_per_node}\n"
            header += f"#SBATCH --cpus-per-task={self.cpus_per_task}\n"
            if self.processor == "gpu":
                header += f"#SBATCH --gpus-per-task={self.gpus_per_task}\n"
            header += f"#SBATCH --partition={self.partition} \n"
            header += f"#SBATCH --constraint={constraint}\n"
            header += f"#SBATCH --hint={multithread}\n"
            header += f"#SBATCH --output={job_name}_%j.out\n"
            header += f"#SBATCH --chdir={str(scratch_path)}\n"
            if self.uenv:
                header += f"#SBATCH --uenv={self.uenv}\n\n"
                header += 'export LD_LIBRARY_PATH="/user-environment/env/default/lib:/user-environment/env/default/lib64:$LD_LIBRARY_PATH"\n'
            else:
                header += "\n"

            header += "export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK\n"
            header += "\n"

            # append commands to the file
            for command in commands:
                header += f"{command}\n"

            bash_script.write(header)

        return file_path


def execute_bash_script(job_file: Union[str, Path], wait: bool = True, ssh_connect: str = None) -> int:
    """Run a simulation using sbatch

    Args:
        job_file (Union[str, Path]):
            Specify the path of the bash script which should be run.
        wait (bool, optional. Default: False):
            Specify if the next job should only be submitted
            after the previous one has stopped.
        ssh_connect (str):
            If specified, the command will be executed on a remote server via ssh.
            The string should be in the format "user@hostname" or 
            "hostname" if the remote machine is configured for passwordless ssh access.

    """
    job_file = Path(job_file)

    cmd = ["ssh", ssh_connect, "sbatch"] if ssh_connect else ["sbatch"]

    if wait:
        cmd.append("--wait")
        
    cmd.append(str(job_file))

    result = subprocess.run(
        cmd, capture_output=True, text=True, check=True, cwd=job_file.parent
    )
    logger.info(f"Starting {str(job_file)}")

    job_id = _get_job_id(result=result)

    return job_id


def _get_job_id(result: subprocess.CompletedProcess[str]) -> int:

    output = result.stdout.strip()

    # Parse the job ID
    job_id = None
    if output.startswith("Submitted batch job"):
        job_id = output.split()[-1]

    return int(job_id)


def jobs_in_queue(
    omit: list = [
        "TRES_PER_NODE",
        "MIN_CPUS",
        "MIN_TMP_DISK",
        "GROUP",
        "OVER_SUBSCRIBE",
        "COMMENT",
        "MIN_MEMORY",
        "REQ_NODES",
        "COMMAND",
        "PRIORITY",
        "QOS",
        "ST",
        "RESERVATION",
        "WCKEY",
        "EXC_NODES",
        "NICE",
        "S:C:T",
        "EXEC_HOST",
        "DEPENDENCY",
        "ARRAY_JOB_ID",
        "SOCKETS_PER_NODE",
        "CORES_PER_SOCKET",
        "THREADS_PER_CORE",
        "ARRAY_TASK_ID",
        "CONTIGUOUS",
        "NODELIST(REASON)",
        "UID",
        "LICENSES",
        "CORE_SPEC",
        "SCHEDNODES",
    ],
    add_progress: bool = False,
    ssh_connect: str = None,
) -> pd.DataFrame:
    """Get information about jobs in the queue.

    Args:
        omit (list, optional):
            Specify which columns of the output should be omitted.

    Returns:
        pd.DataFrame: Information about jobs in the queue.
    """

    # get the user name
    user = os.environ["USER"]

    # call squeue returning only all variables of the jobs
    cmd = ["ssh", ssh_connect] if ssh_connect else []
    cmd.extend(["squeue", "-u", user, "-o", "%all"])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Failed to run squeue"
            f"{' via ssh to ' + ssh_connect if ssh_connect else ''}:\n"
            f"{e.stderr.strip() or e.stdout.strip()}"
        ) from e

    output = result.stdout.strip()

    # parse data
    data = output.split("\n")
    headers = data[0].split("|")
    rows = [x for x in data[1:] if x]

    df = pd.DataFrame([row.split("|") for row in rows], columns=headers)

    # add simulation progress
    if add_progress:
        df = add_progress_to_df(df)

    # remove omitted columns
    df = df.drop(columns=omit)

    return df


def cancel_all() -> None:

    user = os.environ["USER"]
    subprocess.run(["scancel -u", user], check=True)


def add_progress_to_df(df: pd.DataFrame) -> pd.DataFrame:
    """Add MicroHH simulation progress to the DataFrame containing the jobs in the queue.

    Args:
        df (pd.DataFrame):
            DataFrame containing the jobs in the queue

    Returns:
        pd.DataFrame:
            DataFrame containing the jobs in the queue with simulation progress.
    """
    import microhh_processing.initialisation as init

    progress_percentages = []

    for index, row in df.iterrows():
        job_dir = row["WORK_DIR"]
        try:
            # read case
            case = init.read_case(job_dir)
            current_timestep, max_timestep = get_simulation_progress(case)
            progress_percentage = current_timestep / max_timestep * 100
            progress_percentages.append(int(progress_percentage))
        except:
            progress_percentages.append("NA")

    df["PROGRESS (%)"] = progress_percentages

    return df


def print_simulation_progress(case):
    """Print the progress of a simulation based on the saved binary output
    files of MicroHH"""

    current_timestep, max_timestep = get_simulation_progress(case)
    progress_percentage = current_timestep / max_timestep * 100

    if progress_percentage == 0.0:
        color = PrintColor.RED
    elif progress_percentage == 100.0:
        color = PrintColor.GREEN
    else:
        color = PrintColor.YELLOW

    end_color = PrintColor.ENDC

    print(
        f"{color.value}Simulation progress of {case.name_spec}: {progress_percentage:.1f}% ({current_timestep}s out of {max_timestep}s){end_color.value}"
    )


def get_simulation_progress(case) -> Tuple[int, int]:
    """Get current simulation progress of a MicroHH simulation based on the
    binary grid files generated when running the model.

    Args:
        case (init.Case):
            Case for which the simulation progress should be returned.

    Returns:
        Tuple[int, int]:
            Current and max time step of a simulation.
    """
    current_timestep = 0

    def process_file(file):
        if file.is_file() and file.name.startswith("time."):
            try:
                return int(file.name[5:])
            except ValueError:
                return 0
        return 0

    # Using ThreadPoolExecutor to parallelize the file processing
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_file, file)
            for file in case.file_directory.iterdir()
        ]
        for future in concurrent.futures.as_completed(futures):
            timestep = future.result()
            current_timestep = max(current_timestep, timestep)

    max_timestep = int(case.ini_file["time"]["endtime"])

    return current_timestep, max_timestep


def extract_job_id(case) -> int | None:
    """
    Extracts the number represented by '*' in a pattern like 'run_microhh_*.out'.
    Returns the number as an integer, or None if not found.
    """
    files = case.file_directory.glob("run_microhh_*.out")
    for file in files:
        match = re.search(r"run_microhh_(\d+)\.out", file.name)
        if match:
            return int(match.group(1))
    return None


def get_sacct_info(jobid, fields=None, ssh_connect=None) -> pd.DataFrame:
    """
    Query Slurm accounting (sacct) for a given job ID and return a pandas DataFrame.

    Parameters
    ----------
    jobid : int or str
        The Slurm job ID to query.
    fields : list of str, optional
        List of sacct fields to include.
        Defaults to ['JobID', 'JobName', 'NNodes', 'NCPUS', 'Elapsed', 'State'].

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the sacct output.
    """
    if fields is None:
        fields = ["JobID", "JobName", "NNodes", "NCPUS", "Elapsed", "State"]

    # Build sacct command
    cmd = ["ssh", ssh_connect] if ssh_connect else []

    cmd.extend([
        "sacct",
        "-j",
        str(jobid),
        "-P",  # pipe-delimited
        f"--format={','.join(fields)}",
    ])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Error running sacct: {e.stderr.strip()}")

    # Load into pandas
    data = StringIO(result.stdout.strip())
    df = pd.read_csv(data, sep="|")

    # Clean up whitespace in column names and values
    df.columns = df.columns.str.strip()
    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)

    # remove rows with JobID.batch or JobID.extern or JobID.0
    df = df[~df["JobID"].str.contains(r"\.batch$|\.extern$|\.0$", regex=True)]

    return df

def calculate_expected_runtime(benchmark, simulation_time: pd.Timedelta) -> pd.Timedelta:
    """Calculate the expected runtime based on a benchmark and a given time to be simulated."""
    simulation_time_s = simulation_time.total_seconds()
    simulated_time_s = float(benchmark["simulated_time_s"])
    elapsed_time_s = float(benchmark["elapsed_time_s"])
    expected_runtime = (elapsed_time_s / simulated_time_s) * simulation_time_s

    return pd.to_timedelta(expected_runtime, unit="s")


__all__ = [
    "AlpsSetup",
    "execute_bash_script",
    "jobs_in_queue",
    "cancel_all",
    "add_progress_to_df",
    "print_simulation_progress",
    "get_simulation_progress",
    "extract_job_id",
    "get_sacct_info"
    ]
