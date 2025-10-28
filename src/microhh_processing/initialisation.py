import os
import shutil
import subprocess
import sys

from pathlib import Path
from typing import Union, Literal

import netCDF4 as nc4
import xarray as xr
import numpy as np

import microhh_py.microhh_tools as mht
import ls2d

import microhh_processing.misc as misc
import microhh_processing.io as io


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
        name: str,
        suffix: str,
        scratch_path: Union[str, Path],
        store_path: Union[str, Path],
        microhh_executable_path: Union[str, Path],
        location: Literal["scratch", "store"],
    ):
        """Define a case for MicroHH.
        Can be used for...
            ...initialising the model (module 'Initialisation')
            ...pre-processing (module 'PreProcessing')
            ...post-processing (module 'PostProcessing')
            ...visualisation (module 'Visualisation')

        Args:
            name (str):
                Name of the case. Must be set up in microhh/cases
            suffix (str):
                Suffix for the specific model run.
                Is combined as {name}.{suffix}.
            scratch_path (str):
                Path of personal folder on 'scratch'.
            store_path (str):
                Path of personal folder on 'store'.
            location (Literal["scratch", "store"]):
                Location where the case should be handled (read and write)
                Options are 'scratch' and 'store'.
        """
        self.user = os.environ["USER"]
        self.name = name
        self.suffix = suffix
        self.name_spec = f"{name}.{suffix}" if suffix else f"{name}"
        self.scratch_path = Path(scratch_path, self.name_spec)
        self.store_path = Path(store_path, self.name_spec)
        self.microhh_executable_path = Path(microhh_executable_path)

        self._handle_file_directory(location)
        self.ini_path = Path(self.file_directory, f"{self.name}.ini")

    # add method when printing case
    def __repr__(self):
        # format ini file content as string
        # read ini file and format content
        ini_content = ""
        if self.ini_path.exists():
            with open(self.ini_path, "r") as f:
                ini_content = f.read()

        return (
            f"Case(\n"
            f"  name='{self.name}',\n"
            f"  suffix='{self.suffix}',\n"
            f"  path='{self.file_directory}'\n"
            f"  ini_file=\n"
            f"{ini_content}\n"
            f")"
        )

    def _handle_file_directory(self, location):
        # check if the case is set up in the directory 'microhh/cases/'
        if location == "scratch":
            self.file_directory = self.scratch_path
        elif location == "store":
            self.file_directory = self.store_path
        else:
            raise ValueError(f"{location} is not available. Use 'scratch' or 'store'.")

    def initialise(
        self,
        method: Literal["default", "interactive", "rerun"] = "interactive",
        overwrite: bool = False,
    ):
        """Initialise a case by creating or reading in a directory.

        Args:
            method (str, default: 'interactive'):
                Method of how the case should be set up.
                Options are...
                    ...'default': use python script to create .ini file and
                    initial conditions.
                    ...'interactive': create .ini file and initial conditions object
                    oriented.
                    ...'rerun': use .ini file and initial condition from store.
            overwrite (bool, default: False):
                Specify if existing case should be overwritten.
        """
        self.method = method

        self._handle_directory(overwrite)
        self._handle_running_method()

    def _handle_directory(self, overwrite: bool = False):
        """Create the case by creating a directory on 'scratch'.

        Args:
            overwrite (bool, optional. Default: False):
                Specify if an existing folder and content should be overwritten.
        """

        # set up directory on scratch
        if not self.file_directory.exists():
            print(f"Creating {self.file_directory}")
            os.makedirs(self.file_directory)
        elif overwrite:
            shutil.rmtree(self.file_directory)
            print(f"Overwriting {self.file_directory}")
            os.makedirs(self.file_directory)

    def _handle_running_method(self):
        """Check if the running method is correct.
        Accepted values are
            'default': Create .ini file and initial conditions by running a python script.
            'interactive': Create .ini file and initial conditions object oriented.
            'rerun': Rerun a case which is stored on store.
        """
        # check if the case is set up in the directory 'microhh/cases/'
        if self.method == "default":
            self._check_case_setup()
            self._provide_ini()
            self._create_initial()
        elif self.method == "interactive":
            self._provide_ini()
        elif self.method == "rerun":
            raise NotImplementedError(
                "Rerunning a case is not implemented yet. Please use 'default' or 'interactive'."
            )
        else:
            raise ValueError(
                f"{self.method} is not available. Use 'default' or 'interactive'."
            )

    def _check_case_setup(self):
        """Check if the case is set up.

        Raises:
            FileNotFoundError: If the case does not exist or required files are missing.
        """
        case_setup_path = Path("/users", self.user, "microhh", "cases", self.name)
        if not case_setup_path.exists():
            raise FileNotFoundError(f"Case '{self.name}' does not exist.")

        # Check for the existence of {name}_input.py and {name}.ini files
        input_py_path = Path(case_setup_path, f"{self.name}_input.py")
        ini_file_path = Path(case_setup_path, f"{self.name}.ini")

        if not input_py_path.exists():
            raise FileNotFoundError(
                f"{self.name}_input.py does not exist in the case directory."
            )

        if not ini_file_path.exists():
            raise FileNotFoundError(
                f"{self.name}.ini does not exist in the case directory."
            )

    def _provide_ini(self):
        """Provide .ini file based on the selected method when initialising a case.
        If method="default", it will be created by running a pyhton file.
        If method="interactive", an empty file is created which can be modified later.
        """

        # default: copy .ini file from 'microhh/cases/'
        if self.method == "default":
            name_spec = Path("/users", self.user, "microhh", "cases", self.name)
            ini_file_path = name_spec / f"{self.name}.ini"

            if ini_file_path.exists():
                # Copy the existing file to the scratch path
                shutil.copy(str(ini_file_path), str(self.scratch_path))

        # interactive: fill .ini file using LS2D package
        if self.method == "interactive":
            # Create an empty file in the scratch path
            self.ini_path.touch()

        self.ini_file = mht.Read_namelist(self.ini_path)

    def _create_initial(self):
        # default: use {name}_input.py to create {name}_input.nc
        if self.method == "default":
            py_script_path = Path(
                "/users",
                self.user,
                "microhh",
                "cases",
                self.name,
                f"{self.name}_input.py",
            )

            if py_script_path.exists():
                subprocess.run(["python", str(py_script_path)], cwd=self.scratch_path)
            else:
                print(f"{py_script_path} not found.")

    def overwrite_ini(self, settings: dict = {}, overwrite: bool = True):
        """Write the .ini file of a case.
        If the .ini file already exists, it will be overwritten.

        Args:
            overwrite (bool, optional. Default: True):
                Specify if the .ini file should be overwritten.
        """
        self.ini_file.groups = {}

        self.modify_ini(settings=settings, overwrite=overwrite)

    def modify_ini(self, settings: dict, overwrite: bool = True):
        """Modify the .ini file of a case.
        Values populated with None will be removed from the .ini file.
        Values which are not specified will remain unchanged.

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

    def open_mhh_in_vscode(self, filetype: Literal["ini", "out", "sbash", "sbash_out"]):
        """Open the specified file in VS Code.
        filetype can be "ini", "out", "sbash", "sbash_out"
        """
        if filetype == "ini":
            filename = f"{self.name}.ini"
        elif filetype == "out":
            filename = f"{self.name}.out"
        elif filetype == "sbash":
            filename = f"run_microhh.sbash"
        elif filetype == "sbash_out":
            # Use wildcard matching
            matches = list(Path(self.path).glob("run_microhh*.out"))
            if not matches:
                raise FileNotFoundError("No sbash_out files found.")
            filename = max(matches, key=lambda f: f.stat().st_mtime)
        else:
            raise ValueError(
                f"{filetype} is not available. Use 'ini', 'out', 'sbash' or 'sbash_out'."
            )

        file_path = self.file_directory / filename

        if file_path.exists():
            io.open_in_vscode(file_path)
        else:
            print(f"{file_path} does not exist.")

    def delete_on_scratch(self):
        """Delete the scratch directory"""
        shutil.rmtree(self.scratch_path)
        print(f"Deleting {self.name_spec} from scratch.")


def interpolate_to_model_profile(
    grid: Union[
        np.ndarray,
        ls2d.grid.Grid_equidist,
        ls2d.grid.Grid_linear_stretched,
        ls2d.grid.Grid_stretched,
        ls2d.grid.Grid_stretched_manual,
    ],
    variable_value: Union[float, int] = 0.0,
    variable_profile: Union[list, np.ndarray] = None,
    input_profile: Union[list, np.ndarray] = None,
) -> np.ndarray:
    """Interpolate a given vertical profile to the height levels of
    the MicroHH model. Can also be used to create a constant profile

    Args:
        grid: MicroHH grid created using the LS2D module.
        variable_value (Union[float, int], optional. Default: 0.0):
            Constant value of the variable.
            Used if no variable profile is provided.
        variable_profile (Union[list, np.ndarray], optional. Default: None):
            Vertical profile of the variable.
        input_profile (Union[list, np.ndarray], optional. Default: None):
            Height levels of the provided variable profile.

    Returns:
        np.array: Values interpolated to the height levels of
        the MicroHH model.
    """
    # check if variable value or variable profile is provided
    if variable_value is None and variable_profile is None:
        raise ValueError(
            "Please provide either a variable value or a variable profile."
        )

    if isinstance(grid, np.ndarray):
        model_profile = grid
    elif isinstance(
        grid,
        (
            ls2d.grid.Grid_equidist,
            ls2d.grid.Grid_linear_stretched,
            ls2d.grid.Grid_stretched,
            ls2d.grid.Grid_stretched_manual,
        ),
    ):
        model_profile = grid.z
    else:
        raise ValueError(
            "Provided grid is not valid. Please provide a np.ndarray or ls2d.grid."
        )

    # convert model profile and variable profile to numpy arrays
    if variable_profile is not None:
        variable_profile = np.array(variable_profile)
    if input_profile is not None:
        input_profile = np.array(input_profile)

    if variable_profile is None:
        model_var_profile = np.full(len(model_profile), variable_value)
    else:
        model_var_profile = np.interp(model_profile, input_profile, variable_profile)

    return model_var_profile


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


def write_netcdf_input(
    name: str,
    path: Union[str, Path],
    float_type: str,
    init_profiles: dict,
    tdep_surface: dict = None,
    tdep_ls: dict = None,
    radiation: dict = None,
    soil: dict = None,
):
    """
    Function for writing the MicroHH NetCDF input

    Args:
        name (str): Name of the simulated case
        path (Union[str, Path]): Path where the NetCDF file should be stored
        float_type (str): Float type for the NetCDF file ('f4' or 'f8')
        init_profiles (dict): Dictionary containing the initial profiles
        tdep_surface (dict, optional): Dictionary containing the time dependent surface values
        tdep_ls (dict, optional): Dictionary containing the time dependent atmospheric values
        radiation (dict, optional): Dictionary containing the radiation values
        soil (dict, optional): Dictionary containing the soil values
    """

    def add_variable(nc_group, name, dims, data, float_type):
        """
        Add variable to NetCDF file (or group), and write data
        """
        if dims is None:
            var = nc_group.createVariable(name, float_type)
            var[:] = data
        else:
            var = nc_group.createVariable(name, float_type, dims)
            var[:] = data[:]

    def is_array(data):
        """
        Check if value if array or scalar
        """
        if isinstance(data, np.ndarray) or isinstance(data, xr.DataArray):
            return True
        return False

    # Define new NetCDF file
    fname = Path(path) / f"{name}_input.nc"

    # Check if the file exists and remove it if yes
    if fname.exists():
        fname.unlink()

    # Create a new NetCDF file
    nc_file = nc4.Dataset(fname, mode="w", datamodel="NETCDF4")

    # Create height dimension, and set height coordinate
    nc_file.createDimension("z", init_profiles["z"].size)
    add_variable(nc_file, "z", ("z"), init_profiles["z"], float_type)

    # Create a group called "init" for the initial profiles.
    nc_group_init = nc_file.createGroup("init")

    # Set the initial profiles
    for name, data in init_profiles.items():
        # Switch between vector and scalar values
        dims = "z" if is_array(data) else None
        add_variable(nc_group_init, name, dims, data, float_type)

    # Create a group called "timedep" for the time dependent input
    if tdep_surface is not None or tdep_ls is not None:
        nc_group_timedep = nc_file.createGroup("timedep")

    # Write the time dependent surface values
    if tdep_surface is not None:
        nc_group_timedep.createDimension(
            "time_surface", tdep_surface["time_surface"].size
        )

        for name, data in tdep_surface.items():
            add_variable(nc_group_timedep, name, ("time_surface"), data, float_type)

    # Write the time dependent atmospheric values
    if tdep_ls is not None:
        nc_group_timedep.createDimension("time_ls", tdep_ls["time_ls"].size)

        for name, data in tdep_ls.items():
            dims = ("time_ls") if name == "time_ls" else ("time_ls", "z")
            add_variable(nc_group_timedep, name, dims, data, float_type)

    # Write the radiation values
    if radiation is not None:
        nc_group_rad = nc_file.createGroup("radiation")

        nc_group_rad.createDimension("lay", radiation["p_lay"].size)
        nc_group_rad.createDimension("lev", radiation["p_lev"].size)

        for name, data in radiation.items():
            # Switch between vector and scalar values
            if not is_array(data):
                dims = None
            else:
                dims = ("lay") if data.size == radiation["p_lay"].size else ("lev")

            add_variable(nc_group_rad, name, dims, data, float_type)

    # Write the soil values
    if soil is not None:
        nc_group_soil = nc_file.createGroup("soil")
        nc_group_soil.createDimension("z", soil["z"].size)

        for name, data in soil.items():
            add_variable(nc_group_soil, name, "z", data, float_type)

    nc_file.close()
