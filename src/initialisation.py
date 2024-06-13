import glob
import os
import shutil
import subprocess
import sys

from pathlib import Path
from typing import Union

import netCDF4 as nc
import xarray as xr
import numpy as np

sys.path.append(os.path.join("/users", os.environ["USER"], "microhh", "python"))
sys.path.append(os.path.join("/users", os.environ["USER"], "processing_chain", "src"))
sys.path.append(os.path.join("/users", os.environ["USER"], "LS2D"))

import microhh_tools as mht
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
        scratch_path: Union[str, Path],
        store_path: Union[str, Path],
        location: str,
        method: str = "new",
        overwrite: bool = False,
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
            location (str):
                TODO ('scratch', 'store')
            overwrite (bool, default: False):
                TODO
            method (str, default: 'new'):
                ('default', 'new')

        Raises:
            ValueError: TODO
        """
        self.user = os.environ["USER"]
        self.case_name = case_name
        self.suffix = suffix
        self.case_path = f"{case_name}.{suffix}" if suffix else f"{case_name}"
        self.scratch_path = Path(scratch_path, self.user, self.case_path)
        self.store_path = Path(store_path, self.case_path)

        self._handle_method(method)
        self._handle_file_directory(location)
        self.ini_path = Path(self.file_directory, f"{self.case_name}.ini")

        self._handle_case(overwrite)

    def _handle_method(self, method):
        # check if method is correct
        if method not in ["default", "new"]:
            raise ValueError(f"{method} is not available. Use 'default' or 'new'.")

        # check if the case is set up in the directory 'microhh/cases/'
        if method == "default":
            self._check_case_setup()

    def _handle_file_directory(self, location):
        # check if the case is set up in the directory 'microhh/cases/'
        if location == "scratch":
            self.file_directory = self.scratch_path
        elif location == "store":
            self.file_directory = self.store_path
        else:
            raise ValueError(f"{location} is not available. Use 'scratch' or 'store'.")

    def _check_case_setup(self):
        """Check if the case is set up.

        Raises:
            FileNotFoundError: If the case does not exist or required files are missing.
        """
        case_setup_path = Path("/users", self.user, "microhh", "cases", self.case_name)
        if not case_setup_path.exists():
            raise FileNotFoundError(f"Case '{self.case_name}' does not exist.")

        # Check for the existence of {case_name}_input.py and {case_name}.ini files
        input_py_path = Path(case_setup_path, f"{self.case_name}_input.py")
        ini_file_path = Path(case_setup_path, f"{self.case_name}.ini")

        if input_py_path.exists():
            raise FileNotFoundError(
                f"{self.case_name}_input.py does not exist in the case directory."
            )

        if not ini_file_path.exists():
            raise FileNotFoundError(
                f"{self.case_name}.ini does not exist in the case directory."
            )

    def _handle_case(self, overwrite: bool = False):
        """Create the case by creating a directory on 'scratch' and
        copying the '{case_name}' file to the folder.

        Args:
            method (str) ['default', 'new']:
                Specify if the {case_name}.ini file should be
                generated copied from 'microhh/cases/{case_name}/' or if it is
                newly generated using the function modify_ini
            location (s):
                TODO
            overwrite (bool, optional. Default: False):
                Specify if an existing folder and content should be overwritten.

        """

        # set up directory on scratch
        if self.file_directory == self.scratch_path:
            if not self.scratch_path.exists():
                print(f"Creating {self.scratch_path}")
                os.makedirs(self.scratch_path)
            elif overwrite:
                shutil.rmtree(self.scratch_path)
                print(f"Overwriting {self.scratch_path}")
                os.makedirs(self.scratch_path)
            else:
                print(f"Reading from {self.file_directory}")
        else:
            print(f"Reading from {self.file_directory}")


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

    def initialise_restart(self, reuse_bash_script: bool = True):

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
        case_to_rerun: Case,
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
        for filename in files_to_copy:
            for file in glob.glob(Path(case_to_rerun.store_path, filename)):
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


def interpolate_to_model_profile(
    grid: ls2d.grid,
    variable_profile: Union[list, np.ndarray] = None,
    input_profile: Union[list, np.ndarray] = None,
) -> np.ndarray:
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

    Returns:
        xr.Dataset: netCDF of initial conditions.
    """

    fname = Path(case.file_directory, f"{case.case_name}_input.nc")

    return _combine_nc_groups(fname)


def read_statistics(case: Case) -> xr.Dataset:
    """Read the netCDF file containing the initial conditions
    of a model run ({case_name}_input.nc).

    Args:
        case (Case):
            Case for which the initial conditions should be read.


    Returns:
        xr.Dataset: netCDF of initial conditions.
    """

    file_path = Path(case.file_directory, f"{case.case_name}.default.0000000.nc")

    return _combine_nc_groups(file_path)


def _combine_nc_groups(file_path: Union[str, Path]) -> xr.Dataset:
    groups = [""] + list(nc.Dataset(file_path).groups.keys())
    merged_dataset = xr.Dataset()
    for group in groups:
        ds = xr.open_dataset(file_path, group=group, decode_times=False)
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
