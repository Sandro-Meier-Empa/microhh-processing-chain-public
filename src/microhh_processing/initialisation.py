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
            scratch_path (Union[str, Path]):
                Path of personal folder on 'scratch'.
            store_path (Union[str, Path]):
                Path of personal folder on 'store'.
            microhh_executable_path (Union[str, Path]):
                Path to the MicroHH executable.
            file_directory (Union[str, Path], optional):
                Directory where the model output files are stored.
        """
        self.user = os.environ["USER"]
        self.name = name
        self.suffix = suffix
        self.name_spec = f"{name}.{suffix}" if suffix else f"{name}"

    # add method when printing case
    def __repr__(self):
        # format ini file content as string
        # read ini file and format content
        ini_content = ""
        if not hasattr(self, "ini_path"):
            pass
        elif self.ini_path.exists():
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

    def _initialise(
        self,
        scratch_path: Union[str, Path],
        microhh_executable_path: Union[str, Path],
        method: Literal["default", "interactive", "rerun"] = "interactive",
        init_from: Union[str, Path] = None,
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
        self.scratch_path = Path(scratch_path, self.name_spec)
        self.microhh_executable_path = Path(microhh_executable_path)
        self.file_directory = self.scratch_path
        self.ini_path = Path(self.scratch_path, f"{self.name}.ini")

        self._handle_method(method, init_from, overwrite)
        self._handle_scratch_dir(overwrite)
        self._handle_running_method(init_from)

    def _handle_method(
        self,
        method: Literal["default", "interactive", "rerun"],
        init_from: Union[str, Path] = None,
        overwrite: bool = False,
    ):
        self.method = method

        if self.method in ["default", "rerun"] and init_from is None:
            raise ValueError(
                "When using 'default' or 'rerun' method, please provide "
                "the path of the directory containing `.ini` and initial condition files."
            )

        if self.method == "rerun" and overwrite:
            raise ValueError(
                "When using 'rerun' method, overwriting existing files is not allowed. "
                "Please rename your case."
            )

    def _handle_scratch_dir(self, overwrite: bool = False):
        """Create the case by creating a directory on 'scratch'.

        Args:
            overwrite (bool, optional. Default: False):
                Specify if an existing folder and content should be overwritten.
        """

        # set up directory on scratch
        if not self.scratch_path.exists():
            print(f"Creating {self.scratch_path}")
            os.makedirs(self.scratch_path)
        elif overwrite:
            # ask for user input to confirm overwriting
            user_input = input(
                f"Are you sure you want to overwrite {self.scratch_path}? (y/n): "
            )
            if user_input.lower() == "y":
                shutil.rmtree(self.scratch_path)
                print(f"Overwriting {self.scratch_path}")
                os.makedirs(self.scratch_path)

    def _handle_running_method(self, init_from: Union[str, Path]):
        """Check if the running method is correct.
        Accepted values are
            'default': Create .ini file and initial conditions by running a python script.
            'interactive': Create .ini file and initial conditions object oriented.
            'rerun': Rerun a case which is on store or scratch.
        """
        if self.method == "default":
            case_path = Path(init_from)
            self._check_case_setup(case_path)
            self._copy_ini(case_path)
            self._create_initial()
        elif self.method == "interactive":
            self.ini_path.touch()
        elif self.method == "rerun":
            case_path = Path(init_from)
            self._copy_ini(case_path)
            self._copy_initial(case_path)
        else:
            raise ValueError(
                f"{self.method} is not available. Use 'default', 'interactive' or 'rerun'."
            )
        self.ini_file = mht.Read_namelist(self.ini_path)

    def _check_case_setup(self, case_path: Union[str, Path]):
        """Check if the case is set up.

        Raises:
            FileNotFoundError: If the case does not exist or required files are missing.
        """
        if not case_path.exists():
            raise FileNotFoundError(f"Case '{self.name}' does not exist.")

        self._check_required_files(case_path)

    def _check_required_files(self, case_path: Union[str, Path]):
        """Check for the existence of {name}_input.py and {name}.ini files"""
        input_py_path = Path(case_path, f"{self.name}_input.py")
        ini_file_path = Path(case_path, f"{self.name}.ini")

        for required_file in [input_py_path, ini_file_path]:
            if not required_file.exists():
                raise FileNotFoundError(
                    f"Required file '{required_file.name}' is missing"
                )

    def _copy_single_file(
        self,
        case_path: Union[str, Path],
        pattern: str,
        new_filename: str,
    ):
        """Generic helper to copy a single file matching a pattern from case_path to scratch."""
        case_path = Path(case_path)
        matched_files = list(case_path.rglob(pattern))

        if len(matched_files) != 1:
            raise FileNotFoundError(
                f"Expected one file in '{case_path}', found {len(matched_files)}."
            )

        src = matched_files[0]
        dst = self.scratch_path / new_filename
        shutil.copy(str(src), str(dst))

    def _copy_ini(self, case_path: Union[str, Path]):
        """Provide .ini file based on the default method when initialising a case."""
        self._copy_single_file(
            case_path=case_path,
            pattern="*.ini",
            new_filename=f"{self.name}.ini",
        )

    def _copy_initial(self, case_path: Union[str, Path]):
        """Provide initial conditions file based on the default method when initialising a case."""
        self._copy_single_file(
            case_path=case_path,
            pattern="*_input.nc",
            new_filename=f"{self.name}_input.nc",
        )

    def _remove_existing_files(self):
        """Remove all existing files except microhh, .ini and _input.nc files"""
        files_to_keep = [
            f"{self.name}.ini",
            f"{self.name}_input.nc",
            "microhh",
        ]
        user_input = input(
            "Are you sure you want to remove all "
            f"files in {self.scratch_path}? (y/n): "
        )
        if user_input.lower() != "y":
            return

        for item in self.scratch_path.iterdir():
            if item.name not in files_to_keep:
                if item.is_file():
                    item.unlink()

    def _create_initial(self):
        """Create the initial conditions for a case.
        Uses the python script located in 'microhh/cases/{name}/'
        to create the initial conditions.
        """
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
            matches = list(Path(self.scratch_path).glob("run_microhh*.out"))
            if not matches:
                raise FileNotFoundError("No sbash_out files found.")
            filename = max(matches, key=lambda f: f.stat().st_mtime)
        else:
            raise ValueError(
                f"{filetype} is not available. Use 'ini', 'out', 'sbash' or 'sbash_out'."
            )

        file_path = self.scratch_path / filename

        if file_path.exists():
            os.chdir(self.scratch_path)
            io.open_in_vscode(file_path)
        else:
            print(f"{file_path} does not exist.")


def initialise_case(
    name: str,
    suffix: str,
    scratch_path: Union[str, Path],
    microhh_executable_path: Union[str, Path],
    method: Literal["default", "interactive", "rerun"] = "interactive",
    init_from: Union[str, Path] = None,
    overwrite: bool = False,
) -> Case:
    """Initialise a MicroHH case.
    Args:
        name (str):
            Name of the case
        suffix (str):
            Suffix for the specific model run.
            Is combined as {name}.{suffix}.
        scratch_path (Union[str, Path]):
            Path of personal folder on 'scratch'.
        store_path (Union[str, Path]):
            Path of personal folder on 'store'.
        microhh_executable_path (Union[str, Path]):
            Path to the MicroHH executable.
    Returns:
        Case: Initialised MicroHH case.
    """
    case = Case(
        name=name,
        suffix=suffix,
    )
    case._initialise(
        scratch_path=scratch_path,
        microhh_executable_path=microhh_executable_path,
        method=method,
        init_from=init_from,
        overwrite=overwrite,
    )
    return case


def read_case(case_directory: Union[str, Path]) -> Case:
    """Read in an existing MicroHH case.

    Args:
        case_directory (Union[str, Path]):
            Directory where the case is stored.
    Returns:
        Case: Read MicroHH case.
    """
    case_directory = Path(case_directory)
    case_name = Path(case_directory).name

    name_parts = case_name.split(".")
    name = name_parts[0]
    suffix = name_parts[1] if len(name_parts) > 1 else ""

    case = Case(
        name=name,
        suffix=suffix,
    )
    case.file_directory = case_directory
    case.ini_path = Path(case.file_directory, f"{case.name}.ini")

    if not case.ini_path.exists():
        raise FileNotFoundError(
            f".ini file for case '{case.name}' does not exist in {case.file_directory}."
        )

    case.ini_file = mht.Read_namelist(case.ini_path)

    return case


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
