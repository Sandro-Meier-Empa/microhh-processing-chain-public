import glob
import importlib
import os
import shutil
import subprocess
import sys

from pathlib import Path
from typing import Union

import netCDF4 as nc
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.join("/users", os.environ["USER"], "microhh", "python"))
sys.path.append(
    os.path.join("/users", os.environ["USER"], "microhh", "processing_chain", "src")
)
import microhh_tools as mht
import processing as proc

sys.path.append(os.path.join("/users", os.environ["USER"], "LS2D"))
import ls2d


# molar mass (g/mol)
gas_mass = {
    "air": 28.9664,  # dry air
    "co": 28.01,
    "co2": 44.01,
    "h2o": 18.01528,
    "no2": 46.0055,
    "no": 30.01,
    "o3": 48.00,
    "so2": 64.066,
    "ch4": 16.0425,
}


class Case:

    def __init__(
        self,
        case_name: str,
        suffix: str,
        scratch_path: str,
        store_path: str,
        overwrite: bool = False,
        method: str = "new",
    ):
        """Define a case for MicroHH.
        Can be used for...
            ...initialising the model (module 'Initialisation')
            ...pre-processing (module 'PreProcessing')
            ...post-processing (module 'PostProcessing')
            ...visualisation (module 'Visualisation')

        Args:
            case_name (str):
                Name of the case. Must be set up in microhh/cases
            suffix (str):
                Suffix for the specific model run.
                Is combined as {case_name}.{suffix}.
            scratch_path (str):
                Path of personal folder on 'scratch'.
            store_path (str):
                Path of personal folder on 'store'.

        Raises:
            FileNotFoundError: Raise error if the case is not set up
            in microhh/cases
        """
        self.user = os.environ["USER"]
        self.case_name = case_name
        self.suffix = suffix
        self.case_path = f"{case_name}.{suffix}" if suffix else f"{case_name}"
        self.scratch_path = os.path.join(scratch_path, self.user, self.case_path)
        self.store_path = os.path.join(store_path, self.case_path)
        self.method = method

        # check if method is correct
        if method not in ["default", "new"]:
            raise ValueError(f"{method} is not available. Use 'default' or 'new'.")

        # check if the case is set up in the directory 'microhh/cases/'
        if method == "default":
            self.check_case_setup()

        self.handle_case(overwrite)

    def check_case_setup(self):
        """Check if the case is set up.

        Raises:
            FileNotFoundError: If the case does not exist or required files are missing.
        """
        case_setup_path = os.path.join(
            "/users", self.user, "microhh", "cases", self.case_name
        )
        if not os.path.exists(case_setup_path):
            raise FileNotFoundError(f"Case '{self.case_name}' does not exist.")

        # Check for the existence of {case_name}_input.py and {case_name}.ini files
        input_py_path = os.path.join(case_setup_path, f"{self.case_name}_input.py")
        ini_file_path = os.path.join(case_setup_path, f"{self.case_name}.ini")

        if not os.path.exists(input_py_path):
            raise FileNotFoundError(
                f"{self.case_name}_input.py does not exist in the case directory."
            )

        if not os.path.exists(ini_file_path):
            raise FileNotFoundError(
                f"{self.case_name}.ini does not exist in the case directory."
            )

    def handle_case(self, overwrite: bool = False):
        """Create the case by creating a directory on 'scratch' and
        copying the '{case_name}' file to the folder.

        Args:
            method (str) ['default', 'new']:
                Specify if the {case_name}.ini file should be
                generated copied from 'microhh/cases/{case_name}/' or if it is
                newly generated using the function modify_ini
            overwrite (bool, optional. Default: False):
                Specify if an existing folder and content should be overwritten.

        Raises:
            FileExistsError: Raise error if the folder exists and
            'overwrite=False'.
        """

        # set up directory on scratch
        if not os.path.exists(self.scratch_path):
            print(f"Creating {self.scratch_path}")
            os.makedirs(self.scratch_path)
            self.ini_path = Path(self.scratch_path) / f"{self.case_name}.ini"
        else:
            if overwrite:
                shutil.rmtree(self.scratch_path)
                print(f"Overwriting {self.scratch_path}")
                os.makedirs(self.scratch_path)
                self.ini_path = Path(self.scratch_path) / f"{self.case_name}.ini"
            else:
                self.ini_path = self.get_file_location(file=f"{self.case_name}.ini")

    def get_file_location(self, file):
        file_store_path = Path(self.store_path) / file
        file_scratch_path = Path(self.scratch_path) / file

        if file_store_path.exists():
            file_path = file_store_path
            print(f"Reading {self.case_path}/{file} from store.")
        elif file_scratch_path.exists():
            file_path = file_scratch_path
            print(f"Reading {self.case_path}/{file} from scratch.")
        else:
            raise FileNotFoundError(
                f"{self.case_name}{file} does not exist on scratch or store."
            )

        return file_path


class Initialisation:

    def __init__(self, case: Case):
        """Initialise the model run by modifying the .ini file and creating
        the {case_name}_input.nc file.

        Args:
            case (Case):
                Case which should be initialised.
            method (str) ['default', 'new']:
                Specify if the {case_name}_input.nc file should be
                generated by running the {case_name}_input.py file or
                by using the function 'write_netcdf_input' in
                'LS2D/examples/microhh/microhh_ls2d_tools.py'
            mode (str) ['run', 'rerun']
        """
        self.user = case.user
        self.case_name = case.case_name
        self.suffix = case.suffix
        self.case_path = case.case_path
        self.scratch_path = case.scratch_path
        self.store_path = case.store_path
        self.ini_path = case.ini_path

    def initialise(self, method: str = "new"):
        self.method = method

        self.provide_ini()
        self.provide_initial()

    def initialise_restart(self, reuse_bash_script=True):

        # List of files to keep
        files_to_keep = [
            f"{self.case_name}.ini",
            f"{self.case_name}_input.nc",
            "microhh",
        ]

        if reuse_bash_script:
            files_to_keep.append("bash_script.sbash")
        else:
            self.create_batch_script()

        # delete existing files except {case_name}.ini, {case_name}_input.nc and executable
        for file_path in self.scratch_path.iterdir():
            if file_path.name not in files_to_keep:
                file_path.unlink()
                print(f"Deleted {file_path}")

    def initialise_rerun(
        self,
        case_to_rerun,
        reuse_bash_script: bool = True,
    ):
        # Re-run simulation based on bash_script.sbash, {case_name}.ini and {case_name}_input.nc.

        # List of files to copy
        files_to_copy = [
            f"{self.case_name}.ini",
            f"{self.case_name}_input.nc",
            "microhh",
        ]

        if reuse_bash_script:
            files_to_copy.append("bash_script.sbash")
        else:
            self.create_batch_script()

        # copy files from store to scratch
        for file_name in files_to_copy:
            for file in glob.glob(os.path.join(case_to_rerun.store_path, file_name)):
                shutil.copy(file, self.scratch_path)
                print(f"Successfully copied '{file}' to '{self.scratch_path}'.")

    def provide_ini(self):

        # default: copy .ini file from 'microhh/cases/'
        if self.method == "default":
            case_path = Path("/users", self.user, "microhh", "cases", self.case_name)
            ini_file_path = case_path / f"{self.case_name}.ini"

            if ini_file_path.exists():
                # Copy the existing file to the scratch path
                shutil.copy(str(ini_file_path), str(self.scratch_path))

        # new: fill .ini file using LS2D package
        if self.method == "new":
            # Create an empty file in the scratch path
            self.ini_path.touch()

        self.ini_file = mht.Read_namelist(self.ini_path)

    def provide_initial(self):
        # default: use {case_name}_input.py to create {case_name}_input.nc
        if self.method == "default":
            py_script_path = Path(
                "/users",
                self.user,
                "microhh",
                "cases",
                self.case_name,
                f"{self.case_name}_input.py",
            )

            if py_script_path.exists():
                subprocess.run(["python", str(py_script_path)], cwd=self.scratch_path)
            else:
                print(f"{py_script_path} not found.")

    def modify_ini(self, settings: dict, overwrite: bool = True):
        """Modify the .ini file of a case on scratch.

        Args:
            settings (dict):
                Dictionary containing the groups, variables and values
                which should be changed in the .ini file.
            overwrite (bool, optional. Default: True):
                Specify if the .ini file should be overwritten.
        """
        for group, variables in settings.items():
            for variable, value in variables.items():
                if value is None:
                    self.ini_file.remove_value(group=group, variable=variable)
                else:
                    self.ini_file.set_value(group=group, variable=variable, value=value)

        self.ini_file.save(self.ini_path, allow_overwrite=overwrite)


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
        account: str = "em05",
        mail_type: str = "ALL",
        email: str = "sandro.meier@empa.ch",
        run_time: str = "00:30:00",
        nodes: int = 1,
        ntasks_per_core: int = 1,
        ntasks_per_node: int = None,
        cpus_per_task: int = 1,
        large_memory: bool = False,
        partition: str = "debug",
    ):
        """_summary_

        processor (str) ['cpu', 'gpu']:
            Processor version to be used to run the model.
        mpi (bool):
            Indicates if mpi should be enabled.
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

        self.job_name = None
        self.scratch_path = None
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
        self, commands: Union[str, list], file_path, job_name, scratch_path
    ):
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


def interpolate_to_model_profile(
    grid: ls2d.grid,
    variable_profile: Union[list, np.array] = None,
    input_profile: Union[list, np.array] = None,
):
    """Interpolate a given vertical profile to the height levels of
    the MicroHH model.

    Args:
        grid: TODO
        variable_profile (Union[list, np.array], optional. Default: None):
            Value of the variable at these heights. If not specified,
            the variable is set to 0.
        input_profile (Union[list, np.array], optional. Default: None):
            Original height levels of the data.

    Returns:
        np.array: Values interpolated to the height levels of
        the MicroHH model.
    """
    model_profile = grid.z

    if variable_profile is not None:
        # Convert variable_profile to a NumPy array if it's a list
        if isinstance(variable_profile, list):
            variable_profile = np.array(variable_profile)

        # Convert input_profile to a NumPy array if it's a list
        if isinstance(input_profile, list):
            input_profile = np.array(input_profile)

        # Handle the case where variable_profile is a single value
        if np.isscalar(variable_profile):
            model_var_profile = np.full(len(model_profile), variable_profile)
        else:
            model_var_profile = np.interp(
                model_profile, input_profile, variable_profile
            )
    else:
        model_var_profile = np.zeros(len(model_profile))

    return model_var_profile


def read_initial_cond(case: Case) -> xr.Dataset:
    """Read the netCDF file containing the initial conditions
    of a model run ({case_name}_input.nc).

    Args:
        case (Case):
            Case for which the initial conditions should be read.
        source (str) ['scratch', 'store']:
            Source from which the initial conditions should be read from.

    Raises:
        ValueError: Raise error if 'source' is invalid.

    Returns:
        xr.Dataset: netCDF of initial conditions.
    """

    fname = case.get_file_location(file=f"{case.case_name}_input.nc")
    return _combine_nc_groups(fname)


def read_statistics(case: Case) -> xr.Dataset:
    """Read the netCDF file containing the initial conditions
    of a model run ({case_name}_input.nc).

    Args:
        case (Case):
            Case for which the initial conditions should be read.
        source (str) ['scratch', 'store']:
            Source from which the initial conditions should be read from.

    Raises:
        ValueError: Raise error if 'source' is invalid.

    Returns:
        xr.Dataset: netCDF of initial conditions.
    """

    fname = case.get_file_location(file=f"{case.case_name}.default.0000000.nc")
    return _combine_nc_groups(fname)


def _combine_nc_groups(fname):
    groups = [""] + list(nc.Dataset(fname).groups.keys())
    merged_dataset = xr.Dataset()
    for group in groups:
        ds = xr.open_dataset(fname, group=group, decode_times=False)
        merged_dataset = xr.merge([merged_dataset, ds])

    return merged_dataset


def convert_to_kmol_per_s(value: Union[int, float], unit: str, gas: str) -> float:
    """Convert a given quantity of gas to kmol/s.
    Used to specify the source strength of a modelled source in MicroHH.

    Args:
        value (Union[int, float]):
            Quantitiy of gas with unit 'unit'.
        unit (str):
            Current unit of gas quantity.
        gas (str):
            Name of the gas for which the quantity should be converted.

    Raises:
        ValueError: Raise errors for invalid mass units and gases.

    Returns:
        float: Quantity of gas in kmol/s
    """

    # Define conversion factors
    mass_factors = {
        "M": 1e6,  # mega
        "k": 1e3,  # kilo
        "": 1e0,
        "m": 1e-3,  # milli
    }

    time_factors = {
        "s": 1,
        "min": 60,
        "h": 60 * 60,
        "a": 60 * 60 * 24 * 365,
    }

    gas = gas.lower()

    # split into mass and time unit
    if "/" in unit:
        mass_unit, time_unit = unit.split("/")
    else:
        mass_unit, time_unit = unit.split()

    # remove whitespace
    mass_unit = mass_unit.strip()
    time_unit = time_unit.strip()

    # remove "-" and "1"
    time_unit = time_unit.replace("-", "")
    time_unit = time_unit.replace("1", "")

    # Check if mass unit is valid
    if mass_unit.endswith("g") or mass_unit.endswith("t"):
        if mass_unit[:-1] not in mass_factors:
            raise ValueError(f'Invalid mass unit "{mass_unit}"')
    elif mass_unit.endswith("mol"):
        if mass_unit[:-3] not in mass_factors:
            raise ValueError(f'Invalid mass unit "{mass_unit}"')
    else:
        raise ValueError(f'Invalid mass unit "{mass_unit}"')

    # Check if time unit is valid
    if time_unit not in time_factors:
        raise ValueError(f'Invalid time unit "{time_unit}"')

    # Check if gas is valid
    if gas not in gas_mass:
        raise ValueError(f'Invalid gas "{gas}".')

    if mass_unit.endswith("g"):
        value /= mass_factors[mass_unit[:-1]]
        value /= gas_mass[gas]
    elif mass_unit.endswith("t"):
        value /= 1e6 * mass_factors[mass_unit[:-1]]
        value /= gas_mass[gas]
    elif mass_unit.endswith("mol"):
        value /= mass_factors[mass_unit[:-3]]
    else:
        raise ValueError(f'Failed to parse unit "{mass_unit}".')

    # Convert time values
    value /= time_factors[time_unit]  # mol/s

    return value * 1e3  # kmol/s
