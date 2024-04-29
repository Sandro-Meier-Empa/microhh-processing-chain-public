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

    def __init__(self, case_name: str, suffix: str, scratch_path: str, store_path: str):
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
        self.ini_path = None
        self.ini_file = None

        # check if case is set up
        case_setup_path = os.path.join(
            "/users", self.user, "microhh", "cases", self.case_name
        )
        if not os.path.exists(case_setup_path):
            raise FileNotFoundError(f"Case '{self.case_name}' does not exist.")

    def create_case(self, overwrite: bool = False):
        """Create the case by creating a directory on 'scratch' and
        copying the '{case_name}' file to the folder.

        Args:
            overwrite (bool, optional. Default: False):
                Specify if an existing folder and content should be overwritten.

        Raises:
            FileExistsError: Raise error if the folder exists and
            'overwrite=False'.
        """
        ini_path = self._construct_ini_path(self.scratch_path)

        if not os.path.exists(self.scratch_path):
            print(f"Creating {self.scratch_path}")
        elif overwrite:
            shutil.rmtree(self.scratch_path)
            print(f"Overwriting {self.scratch_path}")
        else:
            raise FileExistsError(
                f"Case {self.scratch_path} already exists. "
                f"Choose new suffix or set 'overwrite=True'."
            )

        os.makedirs(self.scratch_path)
        shutil.copy(
            os.path.join(
                "/users",
                self.user,
                "microhh",
                "cases",
                self.case_name,
                f"{self.case_name}.ini",
            ),
            ini_path,
        )

        self.ini_path = ini_path
        self.ini_file = mht.Read_namelist(self.ini_path)

    def read_case(self, source: str):
        """Read a set up the case from 'source'.

        Args:
            source (str) ['scratch', 'store']:
                Source from where the case should be read.

        Raises:
            ValueError: Raise error if 'source' is invalid.
            FileNotFoundError: Raise error if the .ini file was
            not found on 'source'.
        """
        if source == "scratch":
            source_path = self.scratch_path
        elif source == "store":
            source_path = self.store_path
        else:
            raise ValueError(
                f"Invalid source: '{source}'. " f"Use 'scratch' or 'store'."
            )

        ini_path = self._construct_ini_path(source_path)

        if not os.path.exists(ini_path):
            raise FileNotFoundError(f"File {ini_path} could not be found.")

        self.ini_path = ini_path
        self.ini_file = mht.Read_namelist(self.ini_path)

    def _construct_ini_path(self, base_path: str) -> str:
        """Construct the full path to the .ini file."""
        return os.path.join(base_path, f"{self.case_name}.ini")


class Initialisation:

    def __init__(self, case: Case):
        """Initialise the model run by modifying the .ini file and creating
        the {case_name}_input.nc file.

        Args:
            case (Case):
                Case which should be initialised.
        """

        case.read_case("scratch")

        self.user = case.user
        self.case_name = case.case_name
        self.suffix = case.suffix
        self.case_path = case.case_path
        self.scratch_path = case.scratch_path
        self.store_path = case.store_path
        self.ini_path = case.ini_path
        self.ini_file = case.ini_file
        self.vars = {}
        self.dims = {}

    def interpolate_to_model_profile(
        self,
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

        kmax = self.ini_file["grid"]["ktot"]
        self.model_profile = grid.z

        if variable_profile is not None:
            # Convert variable_profile to a NumPy array if it's a list
            if isinstance(variable_profile, list):
                variable_profile = np.array(variable_profile)

            # Convert input_profile to a NumPy array if it's a list
            if isinstance(input_profile, list):
                input_profile = np.array(input_profile)

            # Handle the case where variable_profile is a single value
            if np.isscalar(variable_profile):
                model_var_profile = np.full(kmax, variable_profile)
            else:
                model_var_profile = np.interp(
                    self.model_profile, input_profile, variable_profile
                )
        else:
            model_var_profile = np.zeros(kmax)

        return model_var_profile

    def change_inflow_outflow(self, overwrite: bool = False):
        """Change the inflow and outflow in the group 'boundar' in the .ini
        file dynamically based on the wind direction.

        Args:
            overwrite (bool, optional. Default: False):
                Specify if the .ini file should be overwritten.

        Raises:
            ValueError: Raise error if the file would be changed but
            'overwrite=False'
        """
        save_ini = False

        # Define direction values based on mean wind components (u, v)
        direction_mapping = {
            (0, 0): {
                "west": "outflow",
                "north": "outflow",
                "east": "outflow",
                "south": "outflow",
            },
            (1, 0): {
                "west": "inflow",
                "north": "outflow",
                "east": "outflow",
                "south": "outflow",
            },
            (1, 1): {
                "west": "inflow",
                "north": "outflow",
                "east": "outflow",
                "south": "inflow",
            },
            (0, 1): {
                "west": "outflow",
                "north": "outflow",
                "east": "outflow",
                "south": "inflow",
            },
            (-1, 1): {
                "west": "outflow",
                "north": "outflow",
                "east": "inflow",
                "south": "inflow",
            },
            (-1, 0): {
                "west": "outflow",
                "north": "outflow",
                "east": "inflow",
                "south": "outflow",
            },
            (-1, -1): {
                "west": "outflow",
                "north": "inflow",
                "east": "inflow",
                "south": "outflow",
            },
            (0, -1): {
                "west": "outflow",
                "north": "inflow",
                "east": "outflow",
                "south": "outflow",
            },
            (1, -1): {
                "west": "inflow",
                "north": "inflow",
                "east": "outflow",
                "south": "outflow",
            },
        }

        wind_means = {}
        for wind in ["u", "v"]:
            if wind in self.ini_file["force"]["lslist"].split(","):
                wind_means[wind] = np.nanmean(self.vars[f"{wind}_ls"]["data"])
            else:
                wind_means[wind] = np.nanmean(self.vars[wind]["data"])

        u_mean = wind_means["u"]
        v_mean = wind_means["v"]

        # Determine direction values based on mean wind components
        direction_values = direction_mapping.get(
            (np.sign(u_mean), np.sign(v_mean)), None
        )

        if direction_values is None:
            raise ValueError("Invalid wind direction.")

        # Set direction values in the INI file
        for direction, value in direction_values.items():
            if self.ini_file["boundary"][f"flow_direction[{direction}]"] != value:
                print(direction, value)
                self.ini_file.set_value(
                    group="boundary",
                    variable=f"flow_direction[{direction}]",
                    value=value,
                )
                save_ini = True

        # Save changes to the INI file
        if save_ini:
            self.ini_file.save(self.ini_path, allow_overwrite=overwrite)
            print(f"Modifying {self.ini_path}")

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


def read_initial_cond(case: Case, source: str) -> xr.Dataset:
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

    if source == "scratch":
        source_path = case.scratch_path
    elif source == "store":
        source_path = case.store_path
    else:
        raise ValueError(f"Invalid source: '{source}'. " f"Use 'scratch' or 'store'.")

    fname = os.path.join(source_path, f"{case.case_name}_input.nc")
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
