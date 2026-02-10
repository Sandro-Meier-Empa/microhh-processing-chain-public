import glob
import json
import os
import dask
import platform
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
        self._check_micrhh_build_architecture()

    def _check_microhh_build_exists(self):
        """Check if the selected model build exists.

        Raises:
            ValueError: Raise error if the model build does not exist.
        """
        # Check if build exists
        if self.alps_setup.processor == "gpu" and self.alps_setup.mpi:
            raise ValueError("MPI is not enabled for GPU accelerated version.")
        
    def _check_micrhh_build_architecture(self):
        """
        Check if the architecture of the selected MicroHH build matches the system architecture.
        """
        if not os.path.exists(self.microhh_executable_path):
            return False, f"Error: File {self.microhh_executable_path} not found."

        # Get the current system architecture (e.g., 'x86_64' or 'aarch64')
        current_arch = platform.machine().lower()
        

        result = subprocess.check_output(
            ['file', self.microhh_executable_path],
            stderr=subprocess.STDOUT).decode()
        result_lower = result.lower()

        if "aarch64" in result_lower:
            exec_arch = "aarch64"
        elif "x86-64" in result_lower:
            exec_arch = "x86_64"
        else:
            exec_arch = "unknown"

        if current_arch != exec_arch:
            logger.error(f"Mismatch: System is {current_arch}, but binary is {exec_arch}.")

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
        self.job_file = self.create_run_script(overwrite, mode="run")

        # check if the .ini setup is valid
        self._check_ini_setup_valid()

        if start_jobs:
            # Launch the job on alps
            self.start_simulation(wait)

    def prepare_restart_simulation(
        self,
        restart_case: init.Case,
        restart_time: int,
        overwrite: bool = False,
        start_jobs: bool = False,
        wait: bool = False,
    ):

        # copy microHH to directory
        self.copy_executable()

        # copy microHH restart files
        restart_str = self.get_restart_str(restart_time)
        self.copy_restart_files(origin=restart_case.file_directory, timestr=restart_str)

        # Create bash script if it does not exist
        self.job_file = self.create_run_script(overwrite, mode="rerun")

        # Check if .ini file has been updated for restart
        self._check_restart_ini_updated(restart_time)

        # check if the .ini setup is valid
        self._check_ini_setup_valid()

        if start_jobs:
            # Launch the job on alps
            self.start_simulation(wait)

    def _check_restart_ini_updated(self, restart_time: int):
        """Check if the .ini file has been updated for a restart simulation.

        Args:
            restart_time (int):
                Time in seconds from which the simulation is restarted.
        """
        starttime = self.ini_file["time"]["starttime"]

        if starttime != restart_time:
            raise ValueError(
                f"starttime ({starttime}s) in .ini file does not match "
                f"the specified restart_time ({restart_time}s)."
            )
        endtime = self.ini_file["time"]["endtime"]
        if endtime <= restart_time:
            raise ValueError(
                f"endtime ({endtime}s) must be larger than restart_time ({restart_time}s)."
            )

    def start_simulation(self, wait: bool = False):
        """Start the MicroHH simulation by executing the bash script on CSCS Alps.
        Check if another job is in the queue"""
        running_jobs = cscs.jobs_in_queue()
        jobs_in_debug_queue = False

        # check if any other job is running in the debug queue
        if not running_jobs.empty:
            jobs_in_debug_queue = running_jobs[running_jobs["PARTITION"] == "debug"]

        if self.alps_setup.partition == "debug" and jobs_in_debug_queue:
            logger.warning(
                "There are other jobs in the queue. "
                "No new job will be started in the debug partition."
            )

            self.case.job_id = None
            return

        self.case.job_id = cscs.execute_bash_script(self.job_file, wait)

    def cancel_simulation(self):
        # check if self has attribute "job_id"
        if not hasattr(self, "job_id"):
            logger.info("Case has not been submitted yet.")

        if self.case.job_id is not None:
            subprocess.run(["scancel", str(self.case.job_id)])
            logger.info(f"Cancelling simulation {self.case.job_id}")

    def create_run_script(
        self, overwrite: bool = False, mode: Literal["run", "rerun"] = "run"
    ) -> Path:
        job_file = Path(self.scratch_path, "run_microhh.sbash")

        commands = self._get_microhh_run_commands(mode)

        if not job_file.exists() or overwrite:
            job_file = self.alps_setup.create_bash_script(
                commands=commands,
                job_name="run_microhh",
                file_path=job_file,
                scratch_path=self.scratch_path,
            )
        else:
            logger.info(f"Reading existing bash script {job_file}")

        return job_file

    def _get_microhh_run_commands(
        self, mode: Literal["run", "rerun"] = "run"
    ) -> list[str]:
        if mode == "run":
            commands = [
                f"srun {self.scratch_path}/microhh init {self.name}",
                f"srun {self.scratch_path}/microhh run {self.name}",
            ]
        elif mode == "rerun":
            commands = [
                f"srun {self.scratch_path}/microhh run {self.name}",
            ]
        else:
            raise ValueError("mode must be either 'run' or 'rerun'")

        return commands

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
        self._check_sample_times()

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
        swspatial = self.ini_file.get("grid", "swspatialorder", None)
        swpres = self.ini_file.get("pres", "swpres", None)
        swadvec = str(self.ini_file.get("advec", "swadvec", None))

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

        swthermo = self.ini_file.get("thermo", "swthermo", None)
        swbasestate = self.ini_file.get("thermo", "swbasestate", None)
        thref0 = self.ini_file.get("thermo", "thref0", None)

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
            swspatial = self.ini_file.get("grid", "swspatialorder", None)
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
        force = self.ini_file.get("force", "swlspres", None)
        fc = self.ini_file.get("force", "fc", None)
        dpdx = self.ini_file.get("force", "dpdx", None)

        if force is None:
            return
        elif force == 0:
            return
        if force == "geo":
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
        else:
            raise ValueError(
                f"Invalid value for 'swlspres': {force}. "
                "Valid options are 'geo', 'dpdx', or 'uflux'."
            )

    def _check_source_position(self):
        """Check if the source position conflicts with the domain size."""
        if "source" not in self.ini_file.groups:
            return

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
        if "source" not in self.ini_file.groups:
            return
        
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

    def _check_source_sigma(self):
        """Check if the source std is valid.
        It must be >0 and between grid size/4 and grid size/2 since the source is
        simulated within 4*sigma."""
        coords = ["x", "y", "z"]
        cells = ["itot", "jtot", "ktot"]

        if "source" not in self.ini_file.groups:
            return
    
        for coord, cell in zip(coords, cells):
            sigmas = np.atleast_1d(self.ini_file["source"][f"sigma_{coord}"])
            res = float(self.ini_file["grid"][f"{coord}size"]) / float(
                self.ini_file["grid"][cell]
            )

            for sigma in sigmas:
                # check if sigma is zero
                if sigma <= 0:
                    raise ValueError(f"'sigma_{coord}' cannot be <0.")

                # check if sigma is >= grid size/4 to grid size/2
                if not (res / 4 <= sigma <= res / 2):
                    logger.warning(
                        f"'sigma_{coord}' must be between {res / 4} and {res / 2}. "
                        f"With your current setup, it could be that the source is not simulated."
                    )

    def _check_source_list(self):
        """Check if all required variables have the same length."""
        if "source" not in self.ini_file.groups:
            return

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
        if "buffer" not in self.ini_file.groups:
            return
    
        zstart = self.ini_file.get("buffer", "zstart", 0)
        zsize = self.ini_file.get("grid", "zsize")
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

        if "boundary" not in self.ini_file.groups:
            return
        
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

    def _check_sample_times(case) -> None:
        sample_time_3d = case.ini_file.get("time", "savetime", None)
        sample_time_2d = case.ini_file.get("cross", "sampletime", None)
        sample_time_stats = case.ini_file.get("stats", "sampletime", None)

        # Remove None values
        sample_times = [sample_time_3d, sample_time_2d, sample_time_stats]
        sample_times = [s for s in sample_times if s is not None]

        # Logic: If more than one unique value exists among the found keys, warn the user
        if len(set(sample_times)) > 1:
            logger.warning(
                "Sample times are not the same. It is recommended to have the same "
                f"sample time for 3d output '{sample_time_3d:.0f}'s, 2d output '{sample_time_2d:.0f}'s, and "
                f"statistics '{sample_time_stats:.0f}'s for postprocessing convenience."
            )

    def copy_executable(self):
        """Copy MicroHH executable to scratch"""
        microhh_executable_path = self.microhh_executable_path / "microhh"

        shutil.copy(microhh_executable_path, self.scratch_path)

    def copy_restart_files(self, origin: Union[str, Path], timestr: str):
        """Copy MicroHH restart files to scratch. The .ini and input.nc files
        are already copied when creating the case using case.initialise_case()."""

        origin = Path(origin)

        file_patterns = [
            "grid.0000000",
            "fftwplan.0000000",
            "thermo_basestate.0000000",
            "rhoref.0000000",
            f"*.{timestr}",
        ]

        restart_files = []
        for pattern in file_patterns:
            restart_files.extend(origin.glob(pattern))

        for file in restart_files:
            shutil.copy(file, self.scratch_path)


    def get_restart_str(self, endtime: int) -> str:
        """Get endtime string used for restart files."""
        iotimeprec = self.ini_file.get("time", "iotimeprec", 0)
        endtime_io = int(round(endtime * 10 ** (-iotimeprec)))

        return "{0:07d}".format(endtime_io)
