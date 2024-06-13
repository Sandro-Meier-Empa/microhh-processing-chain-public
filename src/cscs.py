import concurrent.futures
import os
import subprocess
import sys

from pathlib import Path
from typing import Union, Tuple

import pandas as pd


sys.path.append(os.path.join("/users", os.environ["USER"], "microhh", "python"))
sys.path.append(
    os.path.join("/users", os.environ["USER"], "microhh", "processing_chain", "src")
)
import microhh_tools as mht
import initialisation as init


class DaintSetup:

    # define number of cores used on daint multicore (cpu) and hybrid (gpu)
    PROCESSOR_VERSIONS = {"cpu": 36, "gpu": 12}

    PARTITION_CONDITIONS = {
        "debug": {"time_limit": 30 * 60, "max_nodes": 10},
        "normal": {"time_limit": 24 * 60 * 60, "max_nodes": 480},
        "prepost": {"time_limit": 30 * 60, "max_nodes": 1},
        "low": {"time_limit": 24 * 30 * 60, "max_nodes": 124},
    }

    def __init__(
        self,
        processor: str,
        mpi: bool,
        account: str,
        mail_type: str,
        email: str,
        run_time: str = "00:30:00",
        nodes: int = 1,
        ntasks_per_core: int = 1,
        ntasks_per_node: int = None,
        cpus_per_task: int = 1,
        large_memory: bool = False,
        partition: str = "debug",
    ):
        """TODO

        processor (str) ['cpu', 'gpu']:
            Processor version to be used to run the model.
        mpi (bool):
            Indicates if mpi should be enabled.
        account (str):
            TODO
        mail_type (str):
            TODO (NONE, BEGIN, END, FAIL, REQUEUE, ALL)
        email (str):
            TODO
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
            36 and 12 for 'gpu'. If 'mpi = False', it is set to 1.
        cpus_per_task (int, optional. Default: 1):
            Number of cpus per task.
            Specify the number of cpus per task.
            Defines the number of OpenMP threads per MPI rank.
            The maximum value depends on the number of tasks per node.
        large_memory (bool, optional. Defaults False):
            Specifies if nodes with large memory should be used.
        partition (str, optional. Default: 'debug')
            ['debug', 'normal', 'prepost', 'low']:
            Queue to be used on daint.
        """
        self.processor = processor.lower()
        self.mpi = mpi
        self.account = account
        self.mail_type = mail_type
        self.email = email
        self.run_time = run_time
        self.nodes = nodes
        self.ntasks_per_core = ntasks_per_core
        self.cpus_per_task = cpus_per_task
        self.large_memory = large_memory
        self.partition = partition
        self.ntasks_per_node = ntasks_per_node or self._get_default_ntasks_per_node()

        self._check_cscs_setup_valid()

    def _get_default_ntasks_per_node(self) -> int:
        """Determines the default number of tasks per node
        if not specified in __init__

        Returns:
            ntasks_per_node (int):
                Number of tasks per node to be used.
                Set to 36 for 'cpu' and 12 for 'gpu' on daint.
                If 'mpi' is disabled, the tasks per node are set to 1.
        """
        if self.mpi:
            return (
                self.PROCESSOR_VERSIONS[self.processor] * self.ntasks_per_core
                if self.processor in self.PROCESSOR_VERSIONS
                else 1
            )
        else:
            return 1

    def _check_cscs_setup_valid(self):
        """Checks if the setup is valid to be run on daint based
        on the selected queue, wall time and number of nodes.

        Raises:
            ValueError: Raise error if the wall time is
                too long for the selected queue.
            ValueError: Raise error if the number of nodes is
                too large for the selected queue.
        """
        # Check if CSCS setup is valid
        run_time_delta = self._parse_run_time(self.run_time)
        conditions = self.PARTITION_CONDITIONS.get(self.partition, {})
        if (
            conditions.get("time_limit")
            and run_time_delta.total_seconds() > conditions["time_limit"]
        ):
            raise ValueError("Invalid time limit for the selected partition.")
        if conditions.get("max_nodes") and self.nodes > conditions["max_nodes"]:
            raise ValueError("Invalid number of nodes for the selected partition.")

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

    def create_bash_script(
        self,
        commands: Union[str, list],
        job_name: str,
        file_path: Union[str, Path],
        scratch_path: Union[str, Path],
    ) -> Union[str, Path]:
        """Create bash script to run a simulation.
        For more info, see https://slurm.schedmd.com/sbatch.html

        Args:
            start_jobs (bool, optional. Default: False):
                Specify if the job should be started.
            wait (bool, optional. Default: False):
                Specify if the next job should only be submitted
                after the previous one has stopped.
        """
        # make sure commands is a list:
        import processing as proc

        commands = proc.object_to_list(commands)

        multithread = "multithread" if self.ntasks_per_core > 1 else "nomultithread"
        constraint = "mc" if self.processor == "cpu" else "gpu"

        with open(file_path, "w") as bash_script:
            header = "#!/bin/bash -l \n\n"

            header += f"#SBATCH --job-name={job_name} \n"
            header += f"#SBATCH --account={self.account}\n"
            header += f"#SBATCH --mail-type={self.mail_type}\n"
            header += f"#SBATCH --mail-user={self.email}\n"
            header += f"#SBATCH --time={self.run_time}\n"
            header += f"#SBATCH --nodes={self.nodes}\n"
            header += f"#SBATCH --ntasks-per-core={self.ntasks_per_core}\n"
            header += f"#SBATCH --ntasks-per-node={self.ntasks_per_node}\n"
            header += f"#SBATCH --cpus-per-task={self.cpus_per_task}\n"
            if self.large_memory:
                header += "#SBATCH --mem=120GB"
            header += f"#SBATCH --partition={self.partition} \n"
            header += f"#SBATCH --constraint={constraint}\n"
            header += f"#SBATCH --hint={multithread}\n"
            header += f"#SBATCH --chdir={str(scratch_path)}\n\n"

            header += "export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK\n"
            if self.processor == "gpu":
                header += "export CRAY_CUDA_MPS=1\n\n"
            else:
                header += "\n"

            # append commands to the file
            for command in commands:
                header += f"{command}\n"

            bash_script.write(header)

        return file_path

    @staticmethod
    def execute_bash_script(job_file: Path, wait: bool = True):
        """Run a simulation using sbatch

        Args:
            job_file (Path):
                Specify the path of the bash script which should be run.
            wait (bool, optional. Default: False):
                Specify if the next job should only be submitted
                after the previous one has stopped.
        """

        print(f"Starting {job_file=}")
        os.system(f"sbatch {'--wait' if wait else ''} {job_file}")


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
    add_progress: bool = True,
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
    squeue = subprocess.Popen(
        ["squeue", "-u", user, "-o", "%all"], stdout=subprocess.PIPE
    )

    # get the output
    output = squeue.communicate()[0]

    # decode the output
    output = output.decode("utf-8")

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


def add_progress_to_df(df: pd.DataFrame) -> pd.DataFrame:
    progress_percentages = []

    for case_path in df["NAME"]:
        parts = case_path.split(".", 1)
        case_name = parts[0]
        case_suffix = parts[1] if len(parts) > 1 else ""
        # create case
        case = init.Case(
            case_name=case_name,
            suffix=case_suffix,
            scratch_path=Path("/scratch/snx3000"),
            store_path=Path(""),
            location="scratch",
        )
        current_timestep, max_timestep = get_simulation_progress(case)
        progress_percentage = current_timestep / max_timestep * 100
        progress_percentages.append(int(progress_percentage))

    df["PROGRESS (%)"] = progress_percentages

    return df


def print_simulation_progress(case: init.Case):
    """Print the progress of a simulation based on the saved binary output
    files of MicroHH"""

    current_timestep, max_timestep = get_simulation_progress(case)
    progress_percentage = current_timestep / max_timestep * 100

    print(
        f"Simulation progress of {case.case_path}: {progress_percentage:.0f}% ({current_timestep}s ouf of {max_timestep}s)"
    )


def get_simulation_progress(case: init.Case) -> Tuple[int, int]:
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
            executor.submit(process_file, file) for file in case.scratch_path.iterdir()
        ]
        for future in concurrent.futures.as_completed(futures):
            timestep = future.result()
            current_timestep = max(current_timestep, timestep)

    ini_file = mht.Read_namelist(case.ini_path)
    max_timestep = int(ini_file["time"]["endtime"])

    return current_timestep, max_timestep


def delete_empty_folders(scratch_path: Union[str, Path]):
    """
    Finds and prints empty directories on your personal scratch. If the user
    agrees, all directories found are deleted.

    Args:
        scratch_path (str):
            The path to search for empty directories. Default is your personal scratch.
    """

    personal_scratch_path = Path(scratch_path, os.environ["USER"])

    print_command = ["find", personal_scratch_path, "-type", "d", "-empty", "-print"]
    delete_command = ["find", personal_scratch_path, "-type", "d", "-empty", "-delete"]

    # list empty folders
    list_folders = subprocess.run(print_command, capture_output=True, text=True)

    count = len(list_folders.stdout.split("\n")) - 1

    if count > 0:
        print(list_folders.stdout)

        ask_delete = input("Should these folders be deleted? (y/n)").lower()
        while ask_delete not in ["yes", "y", "no", "n"]:
            ask_delete = input("Enter 'y' (delete) or 'n' (don't delete).").lower()

        if ask_delete in ["yes", "y"]:
            delete_folders = subprocess.run(
                delete_command, capture_output=True, text=True
            )
            print("deleting...")
        else:
            print("No folder has been deleted.")

    else:
        print("No empty folder found")
