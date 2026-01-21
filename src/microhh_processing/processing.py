import glob
import os
import dask
import pyproj
import re
import scipy
import shutil
import subprocess
import sys
import warnings
import zipfile
import logging

from pathlib import Path
from typing import Literal, Union, Tuple

import pandas as pd
import xarray as xr
import numpy as np

import ucat

import microhh_py.microhh_tools as mht
import microhh_py.convert_3d_to_nc as convert_3d_to_nc
import microhh_py.cross_to_nc as cross_to_nc

import microhh_processing.initialisation as init
import microhh_processing.cscs as cscs
import microhh_processing.wind as wind_proc
import microhh_processing.misc as misc
import microhh_processing.io as io

logger = logging.getLogger(__name__)

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


class PreProcessing:
    def __init__(
        self,
        case: init.Case,
        alps_setup: cscs.AlpsSetup,
    ):
        """Class used for pre-processing and running MicroHH cases

        Args:
            case (init.Case):
                Case to pre-process and run.
            alps_setup (cscs.AlpsSetup):
                Object containing the setup variables used for creating a bash script to run the MicroHH simulations.

        """
        self.case = case
        self.name = case.name
        self.suffix = case.suffix
        self.name_spec = case.name_spec
        self.scratch_path = case.scratch_path
        self.microhh_executable_path = case.microhh_executable_path
        self.ini_path = case.ini_path
        self.ini_file = mht.Read_namelist(self.ini_path)
        self.processor = alps_setup.processor
        self.alps_setup = alps_setup

        self._check_microhh_build_exists()

    def _check_microhh_build_exists(self):
        """Check if the selected model build exists.

        Raises:
            ValueError: Raise error if the model build does not exist.
        """
        # Check if build exists
        if self.alps_setup.processor == "gpu" and self.alps_setup.mpi:
            raise ValueError("MPI is not enabled for GPU accelerated version.")

    def prepare_simulation(
        self,
        overwrite: bool = False,
        start_jobs: bool = False,
        wait: bool = False,
    ):
        """Prepare MicroHH for running a simulation and executes the simulation.
        - Copy the initial conditions {name}_input.nc
        - Copy the specified build of MicroHH to scratch
        - Create bash script and execute it to run the model

        Args:
            start_jobs (bool, optional. Default: False):
                Specify if the job should be started.
            wait (bool, optional. Default: False):
                Specify if the next job should only be submitted
                after the previous one has stopped.

        Raises:
            FileNotFoundError: Raise error if oen of the files cannot be found on scatch.
        """

        # copy microHH to directory
        self.copy_executable()

        # check if all files are present
        files_required = [
            f"{self.name}_input.nc",
            f"{self.name}.ini",
            "microhh",
        ]

        for filename in files_required:
            file_path = Path(self.scratch_path, filename)
            if not file_path.exists():
                raise FileNotFoundError(
                    f"Required file '{filename}' not found in '{self.scratch_path}'."
                )

        # Create bash script if it does not exist
        self.job_file = self.create_run_script(overwrite)

        # check if the .ini setup is valid
        self._check_ini_setup_valid()

        if start_jobs:
            # Launch the job on alps
            self.start_simulation(wait)

    def start_simulation(self, wait: bool = False):
        self.job_id = cscs.execute_bash_script(self.job_file, wait)

    def cancel_simulation(self):
        # check if self has attribute "job_id"
        if not hasattr(self, "job_id"):
            logger.info("Case has not been submitted yet.")

        if self.job_id is not None:
            subprocess.run(["scancel", str(self.job_id)])
            logger.info(f"Cancelling simulation {self.job_id}")

    def create_run_script(self, overwrite: bool = False) -> Path:
        job_file = Path(self.scratch_path, "run_microhh.sbash")
        if not job_file.exists() or overwrite:
            job_file = self.alps_setup.create_bash_script(
                commands=[
                    f"srun {self.scratch_path}/microhh init {self.name}",
                    f"srun {self.scratch_path}/microhh run {self.name}",
                ],
                job_name="run_microhh",
                file_path=job_file,
                scratch_path=self.scratch_path,
            )
        else:
            logger.info(f"Reading existing bash script {job_file}")

        return job_file

    def _check_ini_setup_valid(self):
        """
        Check if the setup specified in the .ini file is valid based on the daint setup and the parameters specified in the .ini file.
        """
        self._check_ini_content()
        self._adjust_mpi_settings()
        self._check_grid_match()
        self._check_nodes_match()
        self._check_grid_order()
        self._check_thermo()
        self._check_forcing()
        self._check_source_position()
        self._check_source_height()
        self._check_source_sigma()
        self._check_source_list()
        self._check_buffer_height()
        self._check_surface_roughness_height()

    def _check_ini_content(self):
        """check if the .ini file is not empty"""
        if len(self.ini_file.groups) == 0:
            raise ValueError(f"{self.name}.ini is empty.")

    def _adjust_mpi_settings(self):
        """Adjust MPI settings if MPI is disabled."""
        if (
            not self.alps_setup.mpi
            and not self.ini_file["master"]["npx"] == 1
            and not self.ini_file["master"]["npy"] == 1
        ):
            for dim in ["x", "y"]:
                self.ini_file.set_value(group="master", variable=f"np{dim}", value=1)
            self.ini_file.save(self.ini_path, allow_overwrite=True)
            logger.warning("Overwriting 'npx' and 'npy' because MPI is disabled.")

    def _check_grid_match(self):
        """Check if the grid size matches the number of processes."""
        itot, jtot, ktot = (
            self.ini_file["grid"]["itot"],
            self.ini_file["grid"]["jtot"],
            self.ini_file["grid"]["ktot"],
        )
        npx, npy = self.ini_file["master"]["npx"], self.ini_file["master"]["npy"]

        if itot % npx != 0:
            raise ValueError(f"itot ({itot}) is not divisible by npx ({npx})")
        if itot % npy != 0:
            raise ValueError(f"itot ({itot}) is not divisible by npy ({npy})")
        if jtot % npx != 0:
            raise ValueError(f"jtot ({jtot}) is not divisible by npx ({npx})")
        if jtot % npy != 0:
            raise ValueError(f"jtot ({jtot}) is not divisible by npy ({npy})")
        if ktot % npx != 0:
            raise ValueError(f"ktot ({ktot}) is not divisible by npx ({npx})")

    def _check_nodes_match(self):
        """Check if the number of nodes matches the expected value."""
        expected_nodes = self.ini_file["master"]["npx"] * self.ini_file["master"]["npy"]
        if expected_nodes != self.alps_setup.ntasks_per_node:
            raise ValueError(
                f"Task per node ({self.alps_setup.ntasks_per_node}) does "
                f"not match the expected value."
            )

    def _check_grid_order(self):
        """Check that the grid order matches the orders of the advection schemes."""
        swspatial = self.ini_file["grid"].get("swspatialorder", None)
        swpres = swadvec = None

        if "pres" in self.ini_file.groups:
            swpres = self.ini_file.get("pres", {}).get("swpres", None)

        if "advec" in self.ini_file.groups:
            swadvec = str(self.ini_file["advec"].get("swadvec", None))

        if swspatial is None:
            return

        if swpres is not None and int(swpres) != int(swspatial):
            raise ValueError(
                f"Grid order 'swspatialorder' ({swspatial}) does not match pressure advection order 'swpres' ({swpres})."
            )

        if swadvec is not None and int(swadvec[0]) != int(swspatial):
            raise ValueError(
                f"Grid order 'swspatialorder' ({swspatial}) does not match scalar advection order 'swadvec' ({swadvec})."
            )

    def _check_thermo(self):
        """Check if the thermo settings are valid."""
        if "thermo" not in self.ini_file.groups:
            return

        swthermo = self.ini_file["thermo"].get("swthermo", None)
        swbasestate = self.ini_file["thermo"].get("swbasestate", None)
        thref0 = self.ini_file["thermo"].get("thref0", None)

        if swbasestate is None:
            return

        if swbasestate == "boussinesq":
            if swthermo == "dry":
                if thref0 is None:
                    raise ValueError(
                        "'thref0' must be specified when using "
                        "'swbasestate'='boussinesq' and 'swthermo'='dry'."
                    )

        elif swbasestate == "anelastic":
            swspatial = self.ini_file["grid"].get("swspatialorder", None)
            if swspatial == 4:
                raise ValueError(
                    "'swbasestate'='anelastic' is not supported for 'swspatialorder'=4"
                )
        else:
            raise ValueError(
                f"Invalid value for 'swbasestate': {swbasestate}. "
                "Valid options are 'boussinesq' or 'anelastic'."
            )

    def _check_forcing(self):
        """Check if the specified forcing is valid."""
        if "force" in self.ini_file.groups:
            force = self.ini_file["force"].get("swlspres", None)

            if force == "geo":
                fc = self.ini_file["force"].get("fc", None)
                if fc is None or fc == 0:
                    raise ValueError(
                        "Coriolis parameter 'fc' must be specified and non-zero "
                        "when using 'swlspres = geo' forcing."
                    )
            elif force == "dpdx":
                dpdx = self.ini_file["force"].get("dpdx", None)
                if dpdx is None:
                    raise ValueError(
                        "'dpdx' must be specified when using 'swlspres = dpdx' forcing."
                    )
            elif force == "uflux":
                uflux = self.ini_file["force"].get("uflux", None)
                if uflux is None:
                    raise ValueError(
                        "'uflux' must be specified when using "
                        "'swlspres = uflux' forcing."
                    )
            elif force == 0:
                pass
            else:
                raise ValueError(
                    f"Invalid value for 'swlspres': {force}. "
                    "Valid options are 'geo', 'dpdx', or 'uflux'."
                )

    def _check_source_position(self):
        """Check if the source position conflicts with the domain size."""
        if "source" in self.ini_file.groups:
            for coord in ["x", "y", "z"]:
                source_0 = np.atleast_1d(self.ini_file["source"][f"source_{coord}0"])
                sigma = np.atleast_1d(self.ini_file["source"][f"sigma_{coord}"])

                if len(source_0) != len(sigma):
                    raise ValueError(
                        f"source_{coord}0 and sigma_{coord} must have the same length."
                    )

                grid_size = float(self.ini_file["grid"][f"{coord}size"])

                for s0, sig in zip(source_0, sigma):
                    source_ext = s0 + 2 * sig
                    if source_ext > grid_size:
                        raise ValueError(
                            f"{coord}-positioning conflicts with the domain size."
                        )

    def _check_source_height(self):
        """Check if source height is greater than vertical resolution"""
        if "source" in self.ini_file.groups:
            source_heights = np.atleast_1d(self.ini_file["source"]["source_z0"])

            vert_grid_extent = float(self.ini_file["grid"]["zsize"])
            vert_grid_n = float(self.ini_file["grid"]["ktot"])
            vert_res = vert_grid_extent / vert_grid_n

            for source_height in source_heights:
                if source_height < vert_res:
                    logger.warning(
                        f"Source height of {source_height}m "
                        f"is smaller than vertical resolution of {vert_res}m"
                    )
                    # raise ValueError(
                    #     f"Source height of {source_height}m "
                    #     f"is smaller than vertical resolution of {vert_res}m"
                    # )

    def _check_source_sigma(self):
        """Check if the source std is greater than zero."""
        if "source" in self.ini_file.groups:
            for coord in ["x", "y", "z"]:
                sigmas = np.atleast_1d(self.ini_file["source"][f"sigma_{coord}"])

                for sigma in sigmas:
                    if sigma == 0:
                        raise ValueError(f"'sigma_{coord}' cannot be 0.")

    def _check_source_list(self):
        """Check if all required variables have the same length."""
        if "source" in self.ini_file.groups:
            sourcelist = np.atleast_1d(self.ini_file["source"]["sourcelist"])
            source_x0 = np.atleast_1d(self.ini_file["source"]["source_x0"])
            source_y0 = np.atleast_1d(self.ini_file["source"]["source_y0"])
            source_z0 = np.atleast_1d(self.ini_file["source"]["source_z0"])
            sigma_x = np.atleast_1d(self.ini_file["source"]["sigma_x"])
            sigma_y = np.atleast_1d(self.ini_file["source"]["sigma_y"])
            sigma_z = np.atleast_1d(self.ini_file["source"]["sigma_z"])
            strength = np.atleast_1d(self.ini_file["source"]["strength"])
            swvmr = np.atleast_1d(self.ini_file["source"]["swvmr"])

            if not (
                len(sourcelist)
                == len(source_x0)
                == len(source_y0)
                == len(source_z0)
                == len(sigma_x)
                == len(sigma_y)
                == len(sigma_z)
                == len(strength)
                == len(swvmr)
            ):
                raise ValueError("All source variables must have the same length.")

    def _check_buffer_height(self):
        """Check if the buffer height lies within the domain."""
        if "buffer" in self.ini_file.groups:
            zstart = self.ini_file["buffer"].get("zstart", 0)
            zsize = self.ini_file["grid"]["zsize"]
            if zstart > zsize:
                raise ValueError(
                    f"Buffer height={zstart} lies outside of the domain "
                    f" with zsize={zsize}."
                )

    def _check_surface_roughness_height(self):
        """Check if the surface roughness is smaller than the lowest vertical grid level."""
        # read in {case}_input.nc file:
        initial_cond = io.open_initial_cond(self.case)
        grid_values = initial_cond.z.values
        non_neg_grid_values = [value for value in grid_values if value >= 0]
        lowest_grid_level = min(non_neg_grid_values)
        warning_threshold = 0.9 * lowest_grid_level

        if "boundary" in self.ini_file.groups:
            for roughness_type in ["z0m", "z0h"]:
                if roughness_type in self.ini_file["boundary"]:
                    surface_roughness = self.ini_file["boundary"][roughness_type]
                    if surface_roughness >= lowest_grid_level:
                        logger.warning(
                            f"Surface roughness ({roughness_type} = {surface_roughness} m) "
                            f"is larger than the lowest level of the grid (={lowest_grid_level} m). "
                            "This may cause the model to crash."
                        )
                    elif surface_roughness >= warning_threshold:
                        logger.warning(
                            f"Surface roughness ({roughness_type} = {surface_roughness} m) "
                            f"similar to the lowest grid level (={lowest_grid_level} m). "
                            "This may cause instability in the model."
                        )

    def copy_executable(self):
        """Copy MicroHH executable to scratch"""
        microhh_executable_path = self.microhh_executable_path / "microhh"

        shutil.copy(microhh_executable_path, self.scratch_path)


class PostProcessing:
    def __init__(self, case: init.Case):
        """Class used for post-processing MicroHH cases and handling the output.

        Args:
            case (init.Case):
                Case to post-process.
        """
        self.case = case
        self.name = case.name
        self.suffix = case.suffix
        self.name_spec = case.name_spec
        self.ini_path = case.ini_path
        self.ini_file = case.ini_file
        self.file_directory = case.file_directory

    def run_postprocessing(
        self,
        py_file_path: Union[str, Path],
        gas: Union[str, list] = None,
        prognostic_vars=["u", "v", "th"],
        precision: Literal["single", "double"] = "double",
        start_time=None,
        end_time=None,
        method: Literal[
            "mean", "median", "height", "concentration_weighted"
        ] = "height",
        height: Union[int, float] = None,
        profile_gas: str = None,
        overwrite: bool = False,
        save: bool = True,
        chunks: Union[str, dict] = None,
    ):
        """Run all post-processing steps to...
            ... create netCDF files from grid output files
            ... convert gas mixing ratios to concentrations
            ... vertically integrate the concentrations
            ... vertically integrate wind data

        Args:
            gas (Union[str, list]):
                Gas or list of gases to process to concentrations and vertically integrate.
                If not given it will process all gases specified in the .ini file.
            prognostic_vars (list, optional):
                Modelled variables which should be processed.
                E.g. ["u", "v", "w", "qt", "th"]
            precision (str, optional):
                Precision used for the created .nc file. Defaults to "double".
            start_time (int, optional. Default: None):
                Start time in s of the data to process.
            end_time (int, optional. Default: None):
                End time in s of the data to process.
            to_unit (str, optional):
                Units of the trace gas columns. Defaults to "kg m-2".
            default_p_and_t (bool, optional):
                Indicate if the standard values for pressure and temperature should be
                used to convert mixing ratios to concentrations or if the
                modelled pressure and temperature should be used. Defaults to False.
            method (str, optional):
                Method to vertically integrate wind speeds. Available options are
                "mean", "median", "height", "concentration_weighted". Defaults to "height".
            height (Union[int, float], optional):
                Height at which the wind speed should be used if method = 'height'. Defaults to None.
            profile_gas (str, optional):
                Gas used to weight the wind speed if method = 'concentration_weighted'.
            overwrite (bool, optional):
                Specify if the resulting dataset should be overwritten if it exists. Defaults to False.
            save (bool, optional):
                Specify if the resulting dataset should be saved. Defaults to True.
            chunks (Union[str, dict], optional):
                Chunks used when opening the netCDF files with xarray. Defaults to None.
        """
        if gases is None:
            gases = self.ini_file["fields"]["slist"]
        else:
            gases = np.atleast_1d(gas).tolist()

        vars = prognostic_vars + gases

        logger.info(f"########## START PROCESSING OF {self.name_spec} ##########")

        # skip post-processing if simulation failed at start
        current_time_step, max_time_step = cscs.get_simulation_progress(self.case)
        if current_time_step == 0:
            logger.error("ERROR: Simulation failed. Skipped post-processing.")
            return

        self.create_3d_output(
            vars=vars,
            py_file_path=py_file_path,
            overwrite=overwrite,
            precision=precision,
            start_time=start_time,
            end_time=end_time,
        )

        if profile_gas is None and method == "concentration_weighted":
            profile_gas = gases[0]
            logger.warning(
                f"No profile_gas specified. Using '{profile_gas}' as profile_gas."
            )

        wind_processing = wind_proc.WindProcessing(self.case)
        wind_2D = wind_processing.create_2D_wind(
            method=method,
            gas=profile_gas,
            height=height,
            save=save,
            overwrite=overwrite,
            chunks=chunks,
        )
        logger.info(f"########## FINISH PROCESSING OF {self.name_spec} ##########")

    def create_3d_output(
        self,
        vars: Union[str, list],
        overwrite: bool = False,
        precision: Literal["single", "double"] = "single",
        start_time=None,
        end_time=None,
        py_file_path: Union[str, Path] = None,
    ):
        """Create 3D netCDF files for the variables specified

        Args:
            vars (Union[str, list]):
                Modelled variables which should be processed.
                E.g. ["u", "v", "w", "ch4", "qt", "th"]
            py_file_path (Union[str, Path]):
                Path to the directory withing /microhh containing the 3d_to_nc.py file.
            overwrite (bool, option. Default: False):
                Specify if existing files on 'scratch' and
                'store' should be overwritten.
            precision (str, optional. Default: "double"):
                Precision used for the created .nc file.
            start_time (int, optional. Default: None):
                Start time in s of the data to process.
            end_time (int, optional. Default: None):
                End time in s of the data to process.
        """
        vars = np.atleast_1d(vars).tolist()
        vars_to_process = vars.copy()

        if start_time is None:
            start_time = self.ini_file["time"]["starttime"]
        if end_time is None:
            end_time = self.ini_file["time"]["endtime"]
        sample_time = self.ini_file["time"]["savetime"]

        for var in vars:
            nc_file_path = Path(self.file_directory, f"{var}.nc")

            if nc_file_path.exists():
                if overwrite:
                    os.remove(nc_file_path)
                    logger.info(f"Overwriting {nc_file_path}")
                else:
                    vars_to_process.remove(var)

        if py_file_path is not None:
            # run as python script
            if vars_to_process:
                post_command = (
                    f"{sys.executable} 3d_to_nc.py -d {self.file_directory} "
                    f"-f {self.name}.ini -v {' '.join(vars_to_process)} "
                    f"-p {precision} -t0 {start_time} -t1 {end_time} "
                    f"-tstep {sample_time}"
                )
                subprocess.run(
                    post_command,
                    shell=True,
                    cwd=py_file_path,
                )
        else:
            # run as module
            if vars_to_process:
                convert_3d_to_nc.run(
                    directory=self.file_directory,
                    filename=f"{self.name}.ini",
                    vars=vars_to_process,
                    precision=precision,
                    starttime=start_time,
                    endtime=end_time,
                    sampletime=sample_time,
                )

    def create_cross_output(
        self,
        vars: Union[str, list],
        modes: list = None,
        indexes: list = None,
        overwrite: bool = False,
        precision: Literal["single", "double"] = "single",
        start_time=None,
        end_time=None,
        sample_time=None,
        nocompression: bool = False,
        nprocs: int = None,
        py_file_path: Union[str, Path] = None,
    ):
        """Create cross-section netCDF files for the variables specified.

        Args:
            vars (Union[str, list]):
                Modelled variables which should be processed.
                E.g. ["u", "v", "w", "ch4", "qt", "th"]
            modes (list, optional):
                Cross-section modes to process. Available options are "xy", "xz", "yz".
                If not provided, modes are determined from the .ini file.
            indexes (list, optional):
                Indices of the cross-sections to process. Defaults to None.
            overwrite (bool, optional. Default: False):
                Specify if existing files should be overwritten.
            precision (str, optional. Default: "single"):
                Precision used for the created .nc file.
            start_time (int, optional. Default: None):
                Start time in s of the data to process.
            end_time (int, optional. Default: None):
                End time in s of the data to process.
            sample_time (int, optional. Default: None):
                Sample time in s of the cross-section output.
            nocompression (bool, optional. Default: False):
                Disable compression in the netCDF file.
            nprocs (int, optional. Default: None):
                Number of worker processes for parallel processing.
            py_file_path (Union[str, Path], optional):
                Path to the directory containing the cross_to_nc.py file.
                If None, runs as module.
        """
        vars = np.atleast_1d(vars).tolist()
        vars_to_process = vars.copy()

        if start_time is None:
            start_time = self.ini_file["time"]["starttime"]
        if end_time is None:
            end_time = self.ini_file["time"]["endtime"]
        if sample_time is None:
            sample_time = self.ini_file["cross"]["sampletime"]

        # Determine modes to check for existing files
        cross_modes = ["xy", "xz", "yz"]
        if modes is None:
            modes_to_check = list(self.ini_file["cross"].keys() & set(cross_modes))
        else:
            modes_to_check = modes

        for var in vars:
            var_exists = False
            for mode in modes_to_check:
                nc_file_path = Path(self.file_directory, f"{var}.{mode}.nc")
                if nc_file_path.exists():
                    if overwrite:
                        os.remove(nc_file_path)
                        logger.info(f"Overwriting {nc_file_path}")
                    else:
                        var_exists = True
                        break
            if var_exists and var in vars_to_process:
                vars_to_process.remove(var)

        if py_file_path is not None:
            # run as python script
            if vars_to_process:
                post_command = (
                    f"{sys.executable} cross_to_nc.py -d {self.file_directory} "
                    f"-f {self.name}.ini -v {' '.join(vars_to_process)} "
                    f"-p {precision} -t0 {start_time} -t1 {end_time} "
                    f"-tstep {sample_time}"
                )
                if modes is not None:
                    post_command += f" -m {' '.join(modes)}"
                if indexes is not None:
                    post_command += f" -x {' '.join(map(str, indexes))}"
                if nocompression:
                    post_command += " -c"
                if nprocs is not None:
                    post_command += f" -n {nprocs}"

                subprocess.run(
                    post_command,
                    shell=True,
                    cwd=py_file_path,
                )
        else:
            # run as module
            if vars_to_process:
                cross_to_nc.run(
                    directory=self.file_directory,
                    filename=f"{self.name}.ini",
                    vars=vars_to_process,
                    modes=modes,
                    indexes=indexes,
                    precision=precision,
                    starttime=start_time,
                    endtime=end_time,
                    sampletime=sample_time,
                    nocompression=nocompression,
                    nprocs=nprocs,
                )

    def move_to_store(
        self,
        store_path: Union[str, Path],
        overwrite: bool = False,
        move_var_files: bool = False,
        vars: Union[str, list] = None,
        move_processed_files: bool = True,
    ):
        """Move files used for running MicroHH to store for reproducibility.

        Args:
            overwrite (bool, optional):
                Specify if existing files on 'store' should be overwritten.
            move_var_files (bool, optional):
                Specify if the .nc files of the variable files should be moved as well.
                Defaults to False.
            vars (Union[str, list], optional):
                Names of the variables to move if 'move_var_files' is True. Defaults to None.
            move_processed_files (bool, optional):
                Specify if the processed files (e.g. {gas}_2D.nc) should be moved. Defaults to True.
        """
        store_path = Path(store_path) / self.name_spec

        # Handle target path
        if not store_path.exists():
            store_path.mkdir(parents=True, exist_ok=True)
        elif overwrite:
            shutil.rmtree(store_path)
            logger.warning(f"Overwriting {store_path}.")
            store_path.mkdir(parents=True)

        # Copy files for reproducibility
        file_patterns = [
            "*" + self.name + ".ini",
            "*" + self.name + "_input.nc",
            "*" + self.name + ".out",
            "*" + self.name + ".default.*.nc",
            "run_microhh_*.out",
            "run_microhh.sbash",
            "microhh",
        ]
        for pattern in file_patterns:
            self.move_pattern_to_store(
                pattern, store_path=store_path, overwrite=overwrite
            )

        if move_var_files:
            vars = np.atleast_1d(vars).tolist()
            for var in vars:
                pattern = f"{var}.nc"
                self.move_pattern_to_store(
                    pattern, store_path=store_path, overwrite=overwrite
                )

        if move_processed_files:
            for pattern in ["*_2D.nc"]:
                self.move_pattern_to_store(
                    pattern, store_path=store_path, overwrite=overwrite
                )

    def move_pattern_to_store(
        self, pattern: str, store_path: Union[str, Path], overwrite: bool = False
    ):
        """Move files whith a certain pattern to store.

        Args:
            pattern (str):
                Pattern of files which should be moved.
            overwrite (bool):
                Specify if existing files should be overwritten.
        """

        for file in self.file_directory.glob(pattern):
            dest_file = Path(store_path, file.name)
            if not dest_file.exists() or overwrite:
                shutil.copy(file, store_path)
                logger.info(f"Successfully copied '{file}' to '{store_path}'.")
            else:
                logger.info(f"'{file}' already exists in '{store_path}'. Skipping.")

    def zip_grid_files(self, remove_files: bool = False):
        """Zip all binary grid files together.

        Args:
            remove_files (bool, optional):
                Specify if the grid files should be removed after zipping them together.
                Defaults to False.
        """
        zip_filename = Path(self.file_directory, "binary_grid_files.zip")

        if zip_filename.exists():
            logger.info(f"{zip_filename} already exists. Skipping...")
            return

        pattern = re.compile(r"^.*\.\d{7}$")
        matching_files = [
            file for file in self.file_directory.iterdir() if pattern.match(file.name)
        ]

        with zipfile.ZipFile(zip_filename, "w") as zip_file:
            for file in matching_files:
                zip_file.write(file, file.name)

        logger.info(f"{len(matching_files)} files were zipped to '{zip_filename}'.")

        if remove_files:
            for file in matching_files:
                file.unlink()

    def unzip_grid_files(self, remove_zip_file: bool = False):
        """Unzip the binay grid files to scratch"""
        zip_filename = Path(self.file_directory, "binary_grid_files.zip")

        if not zip_filename.exists():
            logger.warning(f"{zip_filename} does not exist. Skipping...")
            return

        with zipfile.ZipFile(zip_filename, "r") as zip_ref:
            zip_ref.extractall(self.file_directory)

        logger.info(
            f"Files from '{zip_filename}' successfully extracted to '{self.file_directory}'"
        )

        if remove_zip_file:
            zip_filename.unlink()

    def open_3d_prognostic_variable(
        self, variable: str, chunks: Union[str, dict] = None
    ) -> xr.Dataset:
        """Open 3D output netCDF file of 'variable' from model output.

        Args:
            variable (str):
                Variable name of dataset to open.

        Returns:
            xr.Dataset: 3D netCDF file from model output.
        """
        file_path = Path(self.file_directory, f"{variable.lower()}.nc")

        if not file_path.exists():
            self.create_3d_output(vars=variable.lower())

        data = io.open_microhh_nc(case=self.case, variable=variable, chunks=chunks)
        data = self._nc_assign_names_and_attributes(data)
        data = self._capitalise_variable_names(data)

        return data

    def _nc_assign_names_and_attributes(self, data: xr.Dataset) -> xr.Dataset:
        """Assign long_name and unit attributes to variables in a .nc file.

        Args:
            data (xr.Dataset):
                MicroHH dataset to which the attributes and units should be assigned.
        """
        # Define the dictionary with variable attributes using a dictionary comprehension
        swvmr = np.all(self.ini_file["source"]["swvmr"])
        if swvmr:
            unit = "mol mol-1"
            prefix = "volume"
        else:
            unit = "kg kg-1"
            prefix = "mass"

        variables = {
            var: (f"{var} {prefix} mixing ratio", unit) for var in gas_mass.keys()
        }
        variables.update(
            {
                "th": ("potential temperature", "K"),
                "thl": ("liquid water potential temperature", "K"),
                "qt": ("total water mixing ratio", "kg kg-1"),
                "u": ("zonal wind component", "m s-1"),
                "v": ("meridional wind component", "m s-1"),
                "w": ("vertical wind component", "m s-1"),
                "p": ("air pressure", "Pa"),
            }
        )

        # Assign attributes to variables
        for var in data.variables:
            if var.lower() in variables:
                long_name, unit = variables[var.lower()]
                data[var].attrs.update({"long_name": long_name, "units": unit})

        return data

    def _capitalise_variable_names(self, dataset: xr.Dataset) -> xr.Dataset:
        """Capitalize variable names in an xr.Dataset if they match
        keys in the gas_mass dictionary.

        Args:
            dataset (xr.Dataset): The xarray dataset to process.

        Returns:
            xr.Dataset: The dataset with variable names capitalized if they match
            keys in the gas_mass dictionary.
        """
        for var_name in list(dataset.variables):
            new_var_name = self._capitalise_gas_key(var_name)
            if new_var_name != var_name:
                dataset = dataset.rename({var_name: new_var_name})
        return dataset

    def _capitalise_gas_key(self, variable_name: str) -> str:
        """Capitalise gases to make them consistent with 'ddeq'"""
        for key in gas_mass.keys():
            if key in variable_name:
                return key.upper()
        return variable_name

    def handle_prognostic_variables(
        self,
        vars: Union[str, list],
        filename: str,
        overwrite: bool = False,
        save: bool = True,
        chunks: dict = None,
        interpolate: bool = True,
    ) -> xr.Dataset:
        """Combine .nc files of different variables to one file.

        Args:
            vars (Union[str, list]):
                Modelled variables which should be processed.
                E.g. ["u", "v", "w", "ch4", "qt", "th"]
            filename (str):
                Name of the file without path or file extension.
            overwrite (bool. Default: False):
                Indicate if an existing file should be overwritten.
            save (bool. Default: True):
                Indicate if the created file should be saved.
            chunks (dict):
                Chunks used when processing the netCDF files with xarray.

        Returns:
            xr.Dataset: Dataset with combined variables.
        """
        file_path = Path(self.file_directory, f"{filename}.nc")

        # Check if the output file already exists
        if file_path.exists() and not overwrite:
            # If the file exists, open and return it
            return io.open_microhh_nc(case=self.case, variable=filename, chunks=chunks)

        # combine the prognostic variables into one dataset
        combined_dataset = self.combine_prognostic_variables(
            vars=vars, chunks=chunks, interpolate=interpolate
        )

        # Save the combined dataset to a new NetCDF file
        if save:
            logger.info(f"Saving {file_path}")
            combined_dataset.to_netcdf(file_path)

        return combined_dataset

    def combine_prognostic_variables(
        self,
        vars: Union[str, list],
        data: xr.Dataset = None,
        chunks: Union[str, dict] = None,
        interpolate: bool = True,
    ) -> xr.Dataset:
        """Combine multiple prognostic variables of the MicroHH output to one
        dataset. Specify either the variables to combine or give a dataset to
        which the new variables should be added.

        Args:
            vars (Union[str, list]):
                Variable(s) to add together or to assign to the given dataset.
            data (xr.Dataset):
                If given, the variable(s) are added to this dataset
            chunks (Union[str, dict], optional):
                Chunks used when processing the netCDF files with xarray.

        Returns:
            xr.Dataset:
                Dataset containing the combined variables.
        """
        vars = np.atleast_1d(vars).tolist()
        datasets = [] if data is None else [data]

        for var in vars:
            dataset = self.open_3d_prognostic_variable(var, chunks=chunks)

            if interpolate:
                # homogenise dimensions
                dataset = self.interpolate_to_grid_midpoint(dataset)
                # dataset = dataset.persist()
            datasets.append(dataset)

        # Merge all datasets into one
        logger.info("Combining datasets...")
        combined_dataset = xr.merge(datasets, combine_attrs="no_conflicts")

        return combined_dataset

    def interpolate_to_grid_midpoint(self, data: xr.Dataset) -> xr.Dataset:
        """Interpolate variables at grid edges 'h' to the middle of the grid.

        Args:
            data (xr.Dataset): Dataset with the variables to interpolate.

        Returns:
            xr.Dataset: Dataset with the interpolated variables
        """
        dim = self._get_dims_at_grid_boundaries(data)

        if dim is None:
            return data

        new_dim_name = dim[:-1]
        data = data.rename({dim: new_dim_name})

        logger.info(f"Interpolating {dim} to {new_dim_name}")

        if dim == "zh":
            # Use initial conditions since vertical grid can be stretched
            midpoints = io.open_initial_cond(self.case).z
        else:
            spacing = data[new_dim_name].diff(new_dim_name)
            half = spacing.median(skipna=True) / 2
            midpoints = data[new_dim_name] + half

        data_interpolated = data.interp(
            {new_dim_name: midpoints}, kwargs={"fill_value": "extrapolate"}
        )

        return data_interpolated

    def _get_dims_at_grid_boundaries(self, data: xr.Dataset) -> str:
        """Get the name of the dimension that is sampled at the boundaries of the
        MicroHH grid

        Args:
            data (xr.Dataset):
                Dataset which contains a dimension which is sampled at the boundaries
                of the grid.

        Returns:
            str: Name of dimension which is sampled at the grid boundaries.
        """
        # iterate over the dimensions of the Dataset
        for dim in list(data.dims):
            # check if the dimension name ends with 'h'
            if dim.endswith("h"):
                return dim

        return None

    def add_gaussian_noise(
        self, data: xr.Dataset, var: str, std: float, noise_units: str
    ) -> xr.Dataset:
        """
        Add Gaussian noise to a specified variable in the dataset.

        Args:
            data (xr.Dataset):
                The dataset containing the variable to which noise will be added.
            var (str):
                The name of the variable to which Gaussian noise will be added.
            std (float):
                The standard deviation of the Gaussian noise distribution.
            unit (str):
                Unit of the noise added.

        Returns:
            xr.Dataset: The dataset with the specified variable modified to include Gaussian noise.
        """
        var = var.upper()

        # Create a deep copy of the dataset to avoid modifying the original in-place
        data_copy = data.copy(deep=True)

        # convert noise to same unit as data
        data_units = data[var].attrs.get("units", None)

        if data_units is None:
            raise ValueError(
                f"Units for variable '{var}' are not defined in the dataset."
            )

        std_conv = ucat.convert_columns(
            std, from_unit=noise_units, to_unit=data_units, molar_mass=var
        )

        # assume no bias (center = 0) but noise (= std_conv)
        noise = np.random.normal(0, std_conv, data_copy[var].size)
        noise = noise.reshape(data_copy[var].shape)

        data_copy[var] = data[var] + noise
        data_copy[var].attrs.update(data[var].attrs)
        data_copy[var].attrs["noise_level"] = std_conv

        # add std_conv for variable based on artificial noise
        data_copy[f"{var}_std"] = (data[var].dims, np.full_like(data[var], std_conv))
        data_copy[f"{var}_std"].attrs["units"] = data_units

        return data_copy

    def degrade_resolution(
        self, data: xr.Dataset, zonal_res: int, meridional_res: int
    ) -> xr.Dataset:
        """Degrade resolution of 3D or 2D MicroHH output by aggregating pixels.

        Args:
            data (xr.Dataset): 3D or 2D data
            zonal_res (int):
                Number of pixels in zonal direction which should be aggregated.
            meridional_res (int):
                Number of pixels in meridional direction which should be aggregated.

        Returns:
            xr.Dataset: Data with coarser resolution.
        """
        spatial_dims = {
            ("x", "y"): ("x", "y"),
            ("lon", "lat"): ("lon", "lat"),
            ("nrows", "nobs"): ("nrows", "nobs"),
            ("longitude", "latitude"): ("longitude", "latitude"),
        }

        for dims, (zonal_dim, meridional_dim) in spatial_dims.items():
            if all(dim in data.dims for dim in dims):
                degraded_data = data.coarsen(
                    {zonal_dim: zonal_res, meridional_dim: meridional_res},
                    boundary="trim",
                ).mean()
                return degraded_data

        raise ValueError(f"Unknown spatial dimensions: {data.dims}")

    def create_sources_dataset(
        self,
        lat: Union[list, np.ndarray],
        lon: Union[list, np.ndarray],
        source_names: Union[list, np.ndarray] = None,
        source_labels: Union[list, np.ndarray] = None,
        source_diameter: Union[list, np.ndarray] = None,
    ) -> xr.Dataset:
        """Create a sources dataset used for the plume detection and in 'ddeq'.

        Args:
            lat (Union[list, np.ndarray]):
                List of latitudes for the locations of a source.
            lon (Union[list, np.ndarray]):
                List of longitudes for the locations of a source.
            source_names (Union[list, np.ndarray], optional. Default: None):
                List of names of the sources. Should not contain any special
                characters. If not specified, the sources will be numbered
                starting with source1 etc.
            source_labels (Union[list, np.ndarray], optional. Default: None):
                List of names of the sources used for plotting.
                If not specified, the sources will be numbered
                starting with source1 etc.
            source_diameter (Union[list, np.ndarray], optional. Default: None):
                Diameter (in pixels) around the soruce which are considered
                to assign a plume to a source. If not specified, 5 * pixel_size
                is used.

        Returns:
            xr.Dataset: Dataset with source information.
        """

        source_list = self.ini_file["source"]["sourcelist"]
        nr_sources = len(source_list)

        source_names = (
            source_names or source_labels or [f"source{i+1}" for i in range(nr_sources)]
        )
        source_labels = source_labels or source_names

        if source_diameter is None:
            pixel_size = np.max(
                [
                    self.ini_file["grid"]["xsize"] / self.ini_file["grid"]["itot"],
                    self.ini_file["grid"]["ysize"] / self.ini_file["grid"]["jtot"],
                ]
            )
            source_diameter = np.full(nr_sources, 5 * pixel_size)  # in m

        data = {
            "lon": (["source"], np.array(lon, dtype=float)),
            "lat": (["source"], np.array(lat, dtype=float)),
            "source_x0": (
                ["source"],
                np.array((self.ini_file["source"]["source_x0"]), dtype=float),
            ),
            "source_y0": (
                ["source"],
                np.array(self.ini_file["source"]["source_y0"], dtype=float),
            ),
            "source_z0": (
                ["source"],
                np.array(self.ini_file["source"]["source_z0"], dtype=float),
            ),
            "label": (
                ["source"],
                np.array(source_labels, dtype=object),
            ),
            "diameter": (
                ["source"],
                np.array(source_diameter, dtype=float),
            ),
            "strength": (["source"], self.model_emissions_to_kg_per_h()),
        }

        coords = {"source": np.array(source_names, dtype=object)}

        sources = xr.Dataset(data, coords=coords)
        sources.lon.attrs["units"] = "longitude of source"
        sources.lat.attrs["units"] = "latitude of source"
        sources.diameter.attrs["units"] = "m"
        sources.strength.attrs["units"] = "kg h-1"

        for dim in ["x", "y", "z"]:
            sources[f"source_{dim}0"].attrs["units"] = "m"
            sources[f"source_{dim}0"].attrs[
                "description"
            ] = f"{dim} position in MicroHH grid"

        return sources

    def convert_microhh_grid_to_coords(
        self,
        data: xr.Dataset,
        source: xr.DataArray = None,
        target_crs: str = "epsg:4326",
        microhh_crs: str = "epsg:32632",
    ) -> xr.Dataset:
        """Convert the modelled MicroHH grid into a lat/lon grid.
        If lat and lon are specified in the .ini file, these values are
        used as domain centres.
        If 'source' and 'sources' are provided, the lat and lon of the source
        is used to convert the grid such that the source is located at the
        latitude and longitude specified in the 'sources' dataset.

        Args:
            data (xr.Dataset):
                Dataset for which the modelled grid should be transformed to
                a lat/lon grid.
            source (str, optional):
                Source simulated in the dataset 'data'.
            sources (xr.Dataset, optional):
                Dataset which contains information about the modelled source.
            target_crs (str, optional. Default: 'epsg:4326'):
                Coordinate reference system to which the modelled data should
                be transformed to.
            microhh_crs (str, optional. Default: 'epsg:32632'):
                Metric coordinate reference system which matches the desired
                lat/lon location of the source.
                Default 'epsg:32632' is suited for sources in Europe.

        Returns:
            xr.Dataset: Dataset with transformed coordinates.
        """

        # if lat/lon is defined in .ini file: use these values to convert grid
        if "lat" in self.ini_file["grid"] and source is None and sources is None:
            lat = self.ini_file["grid"]["lat"]
            lon = self.ini_file["grid"]["lon"]
            grid_x = self.ini_file["grid"]["xsize"] / 2
            grid_y = self.ini_file["grid"]["ysize"] / 2
        elif source is not None:
            lat = source.lat.values
            lon = source.lon.values
            grid_x = source.source_x0.values
            grid_y = source.source_y0.values
        else:
            raise ValueError(
                f"Provide 'lat' and 'lon' in the {self.name} "
                f"file or 'source' and 'sources' as arguments"
            )

        # convert coordinates to metric crs
        transformer = pyproj.Transformer.from_crs(target_crs, microhh_crs)
        x_o, y_o = transformer.transform(lat, lon)

        # create 2D grid with x and y values
        xx, yy = np.meshgrid(data["x"], data["y"])

        # shift grid of x and y values so that the source or domain center
        # is at the location given by lat and lon
        shifted_grid_x = xx - grid_x + x_o
        shifted_grid_y = yy - grid_y + y_o

        # transform shifted grid to lat / lon
        lat_grid, lon_grid = transformer.transform(
            shifted_grid_x, shifted_grid_y, direction="INVERSE"
        )

        time_attrs = {
            "long_name": "time [UTC]",
        }
        lon_attrs = {
            "units": "degrees_east",
            "standard_name": "longitude",
            "long_name": "longitude",
            "epsg": target_crs.split(":")[-1],
        }
        lat_attrs = {
            "units": "degrees_north",
            "standard_name": "latitude",
            "long_name": "latitude",
            "epsg": target_crs.split(":")[-1],
        }

        data = data.rename({"y": "nrows", "x": "nobs"})

        # define the coordinates
        coords = {
            "time": (["time"], data.time.data, time_attrs),
            "lon": (["nrows", "nobs"], lon_grid, lon_attrs),
            "lat": (["nrows", "nobs"], lat_grid, lat_attrs),
        }

        # extract only the variables that are not a dimension
        vars_to_include = {
            var: data[var] for var in data.variables if var not in data.dims
        }

        # create the new dataset
        data = xr.Dataset(vars_to_include, coords=coords).drop_vars(["nrows", "nobs"])

        return data

    def convert_microhh_timestep_to_date(
        self,
        data: xr.Dataset,
        start_date: Union[str, pd.Timestamp] = None,
        format="%Y-%m-%d %H:%M:%S",
    ) -> xr.Dataset:
        """Convert time into a timestamp to make the
        dataset digestible by `ddeq`.
        Use 'datetime_utc' if available in the .ini file.

        Args:
            data (xr.Dataset):
                The dataset to convert.
            start_date (Union[str, Timestamp], optional):
                The start date as a string or a pandas Timestamp object.

        Returns:
            xr.Dataset:
                The dataset with the time converted to a timestamp.
        """
        # use 'datetime_utc' if available in the .ini file
        if "datetime_utc" in self.ini_file["time"] and start_date is None:
            start_date = self.ini_file["time"]["datetime_utc"]
        elif "datetime_utc" in self.ini_file["time"] and start_date is not None:
            start_date = start_date
            logger.info(
                f"Using {start_date} to overwrite {self.ini_file['time']['datetime_utc']} "
                f"in .ini file"
            )
        else:
            start_date = start_date

        # Check if start_date is a string and convert it to a Timestamp if necessary
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date, format=format)

        time_attrs = data.time.attrs

        # check if the time coordinate is alreay given as datetime:
        if data["time"].dtype == "<M8[ns]":
            # overwrite datetime
            time_diff = pd.Timestamp(data["time"][0].values) - start_date
            data["time"] = data["time"] - time_diff

        elif data["time"].dtype in ["int64", "float64"]:
            # convert time to timedelta
            data["time"] = pd.to_timedelta(data["time"], unit="s")
            # add Timedelta to the start date
            data["time"] = start_date + data["time"]

        else:
            raise ValueError("Data type of 'time' was not recognized.")

        data["time"].attrs = time_attrs

        return data

    def model_emissions_to_kg_per_h(self) -> np.ndarray:
        """
        Convert the prescribed emissions from the .ini file to kg/h

        Returns:
            np.array: Emitted quantities in kg/h
        """
        sourcelist = np.array(self.ini_file["source"]["sourcelist"])
        swvmr = np.array(self.ini_file["source"]["swvmr"])
        strength = np.array(self.ini_file["source"]["strength"])

        # check if all sources emit the same gas
        unique_sourcelist = np.unique(sourcelist)
        if len(unique_sourcelist) > 1:
            raise ValueError(
                "Emissions can only be converted to kg/h if all sources "
                "emit the same gas."
            )

        if all(swvmr):  # it true, units are kmol s-1
            from_unit = "kmol s-1"
        else:  # units are kg s-1
            from_unit = "kg s-1"

        strength = ucat.convert_mass_per_time_unit(
            strength,
            from_unit=from_unit,
            to_unit="kg h-1",
            gas_mass=unique_sourcelist[0].lower(),
        )

        return strength

    def convert_to_concentration(
        self,
        data: xr.Dataset,
        gas: str = "ch4",
        to_unit: str = "kg m-3",
        chunks: Union[str, dict] = None,
    ) -> xr.Dataset:
        """Convert the gas given in data to concentrations using modelled pressure
        and potential temperature.

        Args:
            data (xr.Dataset):
                Dataset containing the variable which should be converted to concentrations.
            gas (str, optional):
                Gas for which the columns should be computed. Defaults to "ch4".
            to_unit (str, optional):
                Concentration units to which the gas should be converted.
                Defaults to "kg m-3".
        """
        required_prognostic_vars = [gas, "th"]

        # check if all required prognostic variables are in the dataset
        for var in required_prognostic_vars:
            if var not in data.data_vars:
                logger.info(f"Adding '{var}' to the dataset.")
                data = self.combine_prognostic_variables(
                    vars=var, data=data, chunks=chunks
                )

        # check if pressure is in the dataset
        if "phydro" not in data.data_vars:
            logger.info(f"Adding 'phydro' to the dataset.")
            data = self.add_statistics_variable(data, "phydro")

        # convert potential temperature to absolute temperature
        if "ta" not in data.data_vars:
            logger.info(f"Calculating absolute temperature.")
            data = self.add_absolute_temperature(data)

        from_unit = data[gas].attrs.get("units", None)

        logger.info(
            f"Converting {gas} mixing ratios [{from_unit}] to concentrations [{to_unit}].",
            "This might take a while.",
        )

        data[gas] = ucat.convert_points(
            data[gas],
            from_unit=from_unit,
            to_unit=to_unit,
            molar_mass=gas,
            p=data.phydro,
            T=data.ta,
        )

        return data

    def add_statistics_variable(self, data: xr.Dataset, variable: str) -> xr.Dataset:
        # read in statistics file
        stats = io.open_statistics(case=self.case)

        # check if variable is in statistics file
        if variable not in stats.data_vars:
            raise KeyError(f"No variable named '{variable}'.")

        # check if target file has the necessary dimensions:
        for dim in stats.sizes:
            if dim not in data.dims or dim not in data.coords:
                data = data.assign_coords({dim: stats[dim]})

        if "time" in stats[variable].dims and "z" in stats[variable].dims:
            variable_values = np.tile(
                stats[variable].values[:, :, np.newaxis, np.newaxis],
                (1, 1, data.y.size, data.x.size),
            )
            data[variable] = (
                ("time", "z", "y", "x"),
                variable_values,
                stats[variable].attrs,
            )
        if "time" in stats[variable].dims and "zh" in stats[variable].dims:
            variable_values = np.tile(
                stats[variable].values[:, :, np.newaxis, np.newaxis],
                (1, 1, data.y.size, data.x.size),
            )
            data[variable] = (
                ("time", "zh", "y", "x"),
                variable_values,
                stats[variable].attrs,
            )

        return data

    def add_absolute_temperature(self, data: xr.Dataset) -> xr.Dataset:
        """FIXME

        Args:
            data (xr.Dataset): _description_

        Returns:
            xr.Dataset: _description_
        """
        required_vars = ["phydro", "phydroh"]

        for var in required_vars:
            if var not in data.data_vars:
                logger.info(f"Adding '{var}' to the dataset.")
                data = self.add_statistics_variable(data, variable=var)

        # Calculate absolute temperature
        data["ta"] = PostProcessing.calculate_absolute_temperature(
            theta=data.th, p=data.phydro, p0=data.phydroh.sel(zh=0)
        )

        # Add attributes to 'ta' variable
        data["ta"].attrs["long_name"] = "Absolute temperature"
        data["ta"].attrs["units"] = "K"

        return data

    @staticmethod
    def calculate_absolute_temperature(theta, p, p0):
        """
        Calculate absolute temperature from potential temperature and pressure.

        FIXME
        """
        # Constants
        R = scipy.constants.R  # Specific gas constant for dry air in J/(kg·K)
        cp = 1004  # Specific heat capacity at constant pressure for dry air in J/(kg·K)

        # Calculate absolute temperature
        absolute_temperature = theta * (p / p0) ** (R / cp)

        return absolute_temperature

    def calculate_3D_tke(
        self, data: xr.Dataset, chunks: Union[str, dict] = None
    ) -> xr.Dataset:
        """Calculate the TKE based on the 3D prognostic variables of MicroHH

        Args:
            data (xr.Dataset):
                Dataset containing the required variables. If it does not contain
                all variables, they are added.

        Returns:
            xr.Dataset:
                Dataset containing TKE as a variable.
        """
        wind_components = ["u", "v", "w"]

        for var in wind_components:
            if var not in data.data_vars:
                data = self.combine_prognostic_variables(
                    vars=var, data=data, chunks=chunks
                )

        # compute temporal deviation for each wind component
        for wind_component in wind_components:
            data[f"{wind_component}_std"] = data[wind_component] - data[
                wind_component
            ].mean(dim="time")
            data[f"{wind_component}_std"].attrs["units"] = data[wind_component].attrs[
                "units"
            ]
            data[f"{wind_component}_std"].attrs[
                "description"
            ] = f"deviation of {wind_component} wind speed from mean"

        # compute TKE
        data["TKE"] = 0.5 * (
            data["u_std"] ** 2 + data["v_std"] ** 2 + data["w_std"] ** 2
        ).mean(dim="time")
        data[f"{wind_component}_std"].attrs["units"] = "m2 s-2"
        data[f"{wind_component}_std"].attrs["long_name"] = "turbulent kinetic energy"

        return data

    def calculate_mass_budget(
        self, read_method: Literal["binary", "netcdf"] = "binary"
    ) -> dict:
        """Calculate the mass budget of the modelled scalars.

        Args:
            read_method (Literal['binary', 'netcdf'], optional. Default: 'binary'):
                Method to read the modelled scalar fields. Choose 'binary' to read
                the binary output files or 'netcdf' to read the netCDF output files.
        """

        stats = io.open_statistics(case=self.case)
        rhoref = stats.rhoref.values
        zh = stats.zh.values
        dz = zh[1:] - zh[:-1]

        itot = self.case.ini_file["grid"]["itot"]
        jtot = self.case.ini_file["grid"]["jtot"]
        ktot = self.case.ini_file["grid"]["ktot"]

        xsize = self.case.ini_file["grid"]["xsize"]
        ysize = self.case.ini_file["grid"]["ysize"]

        dx = xsize / itot
        dy = ysize / jtot

        scalars = np.atleast_1d(self.case.ini_file["source"]["sourcelist"])
        strength = np.array(self.case.ini_file["source"]["strength"])

        time_range = list(
            range(
                0,
                int(self.case.ini_file["time"]["endtime"] + 1),
                int(self.case.ini_file["time"]["savetime"]),
            )
        )

        def read_mass_from_binary(scalar, time_range):
            mass_array = []
            dtype = get_dtype()
            for time in time_range:
                fld = np.fromfile(
                    f"{self.case.file_directory}/{scalar}.{time:07d}", dtype
                )
                fld = fld.reshape((ktot, jtot, itot))
                mass = np.sum(rhoref[:, None, None] * fld * dx * dy * dz[:, None, None])
                mass_array.append(float(mass))
            return mass_array

        def read_mass_from_netcdf(scalar, time_range):
            ds = io.open_microhh_nc(self.case, variable=scalar)
            # Assume variable name matches scalar, dimensions (time, z, y, x)
            arr = ds[scalar].values
            mass_array = []
            for i, time in enumerate(time_range):
                fld = arr[i]
                mass = np.sum(rhoref[:, None, None] * fld * dx * dy * dz[:, None, None])
                mass_array.append(float(mass))
            ds.close()
            return mass_array

        def get_dtype():
            grid = mht.Read_grid(itot, jtot, ktot)

            if grid.prec == "d":
                return np.float64
            else:
                return np.float32

        mass_dict = {"time": time_range}
        for i, scalar in enumerate(scalars):
            if read_method == "binary":
                mass_dict[scalar] = read_mass_from_binary(scalar, time_range)
            elif read_method == "netcdf":
                mass_dict[scalar] = read_mass_from_netcdf(scalar, time_range)
            else:
                raise ValueError("Unknown read_method: choose 'binary' or 'netcdf'")

        return mass_dict

    def case_get_sacct_info(self) -> pd.DataFrame:
        """Get SLURM sacct information for the current case.

        Returns:
            pd.DataFrame: DataFrame with sacct information.
        """
        if self.case.job_id is None:
            raise ValueError("No job_id found in the case.")

        try:
            df = cscs.get_sacct_info(self.case.job_id)
            return df
        except Exception as e:
            raise RuntimeError(f"Error retrieving sacct information: {e}")

    def calculate_centerline_position(
        self,
        data: xr.Dataset,
        variable: str,
        dim: str,
        ydim_name: str = "y",
        zdim_name: str = "z",
    ):
        """Calculate the centerline position of the plume in the dataset.
        According to Raznjevic et al. (2022)"""

        if dim not in [ydim_name, zdim_name]:
            raise ValueError(f"Dimension '{dim}' not recognized.")

        dim_data = data[dim]

        num = (data[variable] * dim_data).integrate(coord=[zdim_name, ydim_name])
        den = data[variable].integrate(coord=[zdim_name, ydim_name])

        return num / den


############# other functions #############
def object_to_list(var) -> list:
    if var is None:
        raise ValueError("Input should not be 'None'")
    if isinstance(var, str):
        # parse comma separated string
        var = [s.strip() for s in var.split(",")]

        # convert to numeric if possible
        # Try converting to numbers
        converted_items = []
        for item in var:
            try:
                # Try integer first
                converted_items.append(int(item))
            except ValueError:
                try:
                    # Try float if int fails
                    converted_items.append(float(item))
                except ValueError:
                    # Keep as string if both fail
                    converted_items.append(item)

        return converted_items

    if isinstance(var, xr.DataArray):
        var = var.values.tolist()
    elif isinstance(var, np.ndarray):
        var = var.tolist()
    elif not isinstance(var, list):
        var = [var]
    return var

from typing import Iterable, Union
from decimal import Decimal, InvalidOperation
import numpy as np


def to_cs_list(lst: Union[list, np.ndarray], n: int = 1) -> str:
    """
    From Python iterable to comma separated string.
    Handles float precision artifacts (e.g. 209.20000000000002 -> 209.2)
    while preserving small numbers (e.g. 0.00000123).
    """

    # Helper to clean individual items
    def clean_fmt(x):
        if isinstance(x, (float, np.floating)):
            # :.15g keeps up to 15 significant digits.
            return f"{x:.15g}"
        return str(x)

    # 1. Convert all items to clean strings
    clean_items = [clean_fmt(x) for x in lst]

    # 2. Repeat the list n times (as per your original logic)
    # Using list multiplication is faster than a loop for string joining
    repeated_items = clean_items * n

    # 3. Join once (More efficient than string concatenation += in a loop)
    return ",".join(repeated_items)


def get_ini_dict_from_file(file_path: Union[str, Path]) -> dict:
    """Read .ini file and return as dictionary.

    Args:
        file_path (Union[str, Path]):
            Path to the .ini file.

    Returns:
        dict: Dictionary with the .ini file contents.
    """
    ini_file = mht.Read_namelist(file_path)
    return ini_file.groups
