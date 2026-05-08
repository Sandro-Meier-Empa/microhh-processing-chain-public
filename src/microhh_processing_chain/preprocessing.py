import fnmatch
import os
import platform
import shutil
import subprocess
import logging

from pathlib import Path
from typing import Union, Optional, List

import numpy as np
import xarray as xr

import microhh_processing_chain.initialisation as init
import microhh_processing_chain.cscs as cscs
from microhh_processing_chain import io, utils
from microhh_processing_chain.enums import RunMode

logger = logging.getLogger(__name__)


class PreProcessing:
    def __init__(
        self,
        case: init.Case,
        alps_setup: cscs.AlpsSetup,
        microhh_executable_path: Union[str, Path],
        jobreport_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Class used for pre-processing and running MicroHH cases

        Args:
            case (init.Case):
                Case to pre-process and run.
            alps_setup (cscs.AlpsSetup):
                Object containing the setup variables used for creating a bash script to run the MicroHH simulations.
            microhh_executable_path (Union[str, Path]):
                Path to the MicroHH executable or the directory containing the MicroHH executable.
            jobreport_path (Optional[Union[str, Path]]):
                Path to the GPU job report executable. Only supported on GPU.
        """
        self.case = case
        self.alps_setup = alps_setup
        self._microhh_executable_path = microhh_executable_path
        self.jobreport_path = jobreport_path

        self.microhh_executable_path = self._get_microhh_executable_path()
        self._check_microhh_build_architecture()
        self._check_jobpreport_gpu()

    def _get_microhh_executable_path(self) -> Path:
        microhh_executable_path = Path(self._microhh_executable_path)

        if microhh_executable_path.is_file():
            return microhh_executable_path

        mhh_files = list(microhh_executable_path.glob("microhh"))

        if len(mhh_files) != 1:
            raise ValueError(
                f"Expected exactly one 'microhh' executable in {microhh_executable_path}, but found {len(mhh_files)}."
            )

        return mhh_files[0]

    def _check_microhh_build_architecture(self) -> None:
        """Check if the architecture of the selected MicroHH build matches the system architecture."""
        if not os.path.exists(self.microhh_executable_path):
            logger.warning(f"File {self.microhh_executable_path} not found.")

        arch = self._get_architecture()

        if arch is None:
            logger.warning(
                "Unable to determine system architecture. Skipping architecture check."
            )
            return

        result = subprocess.check_output(
            ["file", self.microhh_executable_path], stderr=subprocess.STDOUT
        ).decode()
        result_lower = result.lower()

        if "aarch64" in result_lower:
            exec_arch = "aarch64"
        elif "x86-64" in result_lower:
            exec_arch = "x86_64"
        else:
            exec_arch = "unknown"

        if arch != exec_arch:
            logger.error(f"Mismatch: System is {arch}, but binary is {exec_arch}.")

    def _get_architecture(self) -> Optional[str]:
        """Get the architecture of the system.
        If ssh_connect is provided, get the architecture of the remote system.
        Returns:
            str: Architecture of the system (e.g., 'x86_64', 'aarch64').
        """
        if self.alps_setup.ssh_connect:
            try:
                subprocess.check_output(
                    ["ssh", self.alps_setup.ssh_connect, "uname -m"],
                    stderr=subprocess.STDOUT,
                    text=True,  # Python 3.7+
                ).strip()
            except subprocess.CalledProcessError as e:
                logger.warning(
                    f"Failed to connect to {self.alps_setup.ssh_connect}. "
                    f"SSH error:\n{e.output.strip()}"
                )
                return None
        else:
            return str(platform.machine())

    def _check_jobpreport_gpu(self) -> None:
        """Make sure that jobreport is only provided when using GPU partition and raise an error if not."""
        if self.alps_setup.processor != "gpu" and self.jobreport_path is not None:
            raise ValueError(
                "GPU jobreport is only supported when using GPU partition."
            )

    def prepare_simulation(
        self,
        overwrite: bool = False,
        start_jobs: bool = False,
        wait: bool = False,
    ) -> None:
        """Prepare the case for running the MicroHH simulation by
        - checking if the .ini setup is valid,
        - copying the MicroHH executable to the scratch directory,
        - checking if all required files are present,
        - creating a bash script to run the simulation

        """
        # check if the .ini setup is valid
        self._check_ini_setup_valid()

        # copy microHH to directory
        self._copy_executable()

        # Check if required files are present
        self._check_required_files()

        # Create bash script if it does not exist
        self.job_file = self.create_run_script(overwrite, mode=RunMode.RUN)

        if start_jobs:
            # Launch the job on alps
            self.start_simulation(wait)

    def _check_required_files(self) -> None:
        """Check if all required files are present in the scratch directory."""
        files_required = [
            f"{self.case.name}_input.nc",
            f"{self.case.name}.ini",
            "microhh",
        ]

        for filename in files_required:
            file_path = Path(self.case.scratch_path, filename)
            if not file_path.exists():
                raise FileNotFoundError(
                    f"Required file '{filename}' not found in '{self.case.scratch_path}'."
                )

    def prepare_restart_simulation(
        self,
        restart_case: init.Case,
        restart_time: int,
        overwrite: bool = False,
        start_jobs: bool = False,
        wait: bool = False,
    ) -> None:

        # Check if .ini file has been updated for restart
        self._check_restart_ini_updated(restart_time)

        # check if the .ini setup is valid
        self._check_ini_setup_valid()

        # copy microHH to directory
        self._copy_executable()

        # copy microHH restart files
        restart_str = self.get_restart_str(restart_time)
        self.copy_restart_files(origin=restart_case.file_directory, timestr=restart_str)

        # Create bash script if it does not exist
        self.job_file = self.create_run_script(overwrite, mode=RunMode.RERUN)

        if start_jobs:
            # Launch the job on alps
            self.start_simulation(wait)

    def _check_restart_ini_updated(self, restart_time: int) -> None:
        """Check if the .ini file has been updated for a restart simulation.

        Args:
            restart_time (int):
                Time in seconds from which the simulation is restarted.
        """
        start_time = self.case.ini_file["time"]["starttime"]

        if start_time != restart_time:
            raise ValueError(
                f"start time ({start_time}s) in .ini file does not match "
                f"the specified restart_time ({restart_time}s)."
            )
        end_time = self.case.ini_file["time"]["endtime"]
        if end_time <= restart_time:
            raise ValueError(
                f"end time ({end_time}s) must be larger than restart_time ({restart_time}s)."
            )

    def start_simulation(self, wait: bool = False) -> None:
        """Start the MicroHH simulation by executing the bash script on CSCS Alps.

        Args:
            wait (bool, optional. Default: False):
                Specify if the next job should only be submitted
                after the previous one has stopped.
        """
        running_jobs = cscs.jobs_in_queue()
        jobs_in_debug_queue = False

        # check if any other job is running in the debug queue
        if not running_jobs.empty:
            jobs_in_debug_queue = not running_jobs[
                running_jobs["PARTITION"] == "debug"
            ].empty

        if self.alps_setup.partition == "debug" and jobs_in_debug_queue:
            logger.warning(
                "There are other jobs in the queue. "
                "No new job will be started in the debug partition."
            )

            self.case.job_id = None
            return

        self.case.job_id = cscs.execute_bash_script(
            self.job_file, wait, ssh_connect=self.alps_setup.ssh_connect
        )

    def cancel_simulation(self) -> None:
        # check if self has attribute "job_id"
        if not hasattr(self, "job_id"):
            logger.info("Case has not been submitted yet.")

        if self.case.job_id is not None:
            subprocess.run(["scancel", str(self.case.job_id)])
            logger.info(f"Cancelling simulation {self.case.job_id}")

    def create_run_script(
        self, overwrite: bool = False, mode: Union[RunMode, str] = RunMode.RUN
    ) -> Path:
        job_file = Path(self.case.scratch_path, "run_microhh.sbash")

        commands = _get_microhh_run_commands(
            case_list=[self.case],
            mode=mode,
            jobreport_path=(
                str(self.jobreport_path) if self.jobreport_path is not None else None
            ),
        )

        if not job_file.exists() or overwrite:
            job_file = self.alps_setup.create_bash_script(
                commands=commands,
                job_name="run_microhh",
                file_path=job_file,
                scratch_path=self.case.scratch_path,
            )
        else:
            logger.info(f"Reading existing bash script {job_file}")

        return Path(job_file)

    def _check_ini_setup_valid(self) -> None:
        """Check if the .ini file setup is valid.

        Validates based on the Alps setup and the parameters specified in the .ini file.
        """
        self._check_ini_content()
        self._adjust_mpi_settings()
        self._check_grid_match()
        self._check_nodes_match()
        self._check_grid_order()
        self._check_dump()
        self._check_thermo()
        self._check_forcing()
        self._check_boundary_logic()
        self._check_forcing_logic()
        self._check_thermo_logic()
        self._check_source_position()
        self._check_source_height()
        self._check_source_sigma()
        self._check_source_list()
        self._check_tdep_source_location()
        self._check_tdep_source_strength()
        self._check_buffer_height()
        self._check_surface_roughness_height()
        self._check_sample_times()

    def _check_ini_content(self) -> None:
        """check if the .ini file is not empty"""
        if len(self.case.ini_file.groups) == 0:
            raise ValueError(f"{self.case.name}.ini is empty.")

    def _adjust_mpi_settings(self) -> None:
        """Adjust MPI settings if MPI is disabled."""
        if (
            self.alps_setup.processor == "gpu"
            and not self.case.ini_file["master"]["npx"] == 1
            and not self.case.ini_file["master"]["npy"] == 1
        ):
            for dim in ["x", "y"]:
                self.case.ini_file.set_value(
                    group="master", variable=f"np{dim}", value=1
                )
            self.case.ini_file.save(self.case.ini_path, allow_overwrite=True)
            logger.info("Overwriting 'npx' and 'npy' because MPI is disabled.")

    def _check_grid_match(self) -> None:
        """Check if the grid size matches the number of processes."""
        itot, jtot, ktot = (
            self.case.ini_file["grid"]["itot"],
            self.case.ini_file["grid"]["jtot"],
            self.case.ini_file["grid"]["ktot"],
        )
        npx, npy = (
            self.case.ini_file["master"]["npx"],
            self.case.ini_file["master"]["npy"],
        )

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

    def _check_nodes_match(self) -> None:
        """Check if the number of nodes matches the expected value."""
        expected_cores = (
            self.case.ini_file["master"]["npx"] * self.case.ini_file["master"]["npy"]
        )
        if self.alps_setup.ntasks_per_node is None:
            raise ValueError("Alps setup requires 'ntasks_per_node' to validate nodes.")

        requested_cores = self.alps_setup.ntasks_per_node * self.alps_setup.nodes
        if expected_cores != requested_cores:
            raise ValueError(
                f"Requested cores ({requested_cores}) does not match the "
                f"expected cores ({expected_cores})."
            )

    def _check_grid_order(self) -> None:
        """Check that the grid order matches the orders of the advection schemes."""
        swspatial = self.case.ini_file.get("grid", "swspatialorder", None)
        swpres = self.case.ini_file.get("pres", "swpres", None)
        swadvec = str(self.case.ini_file.get("advec", "swadvec", None))

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

    def _check_thermo(self) -> None:
        """Check if the thermo settings are valid."""

        swthermo = self.case.ini_file.get("thermo", "swthermo", None)
        swbasestate = self.case.ini_file.get("thermo", "swbasestate", None)
        thref0 = self.case.ini_file.get("thermo", "thref0", None)

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
            swspatial = self.case.ini_file.get("grid", "swspatialorder", None)
            if swspatial == 4:
                raise ValueError(
                    "'swbasestate'='anelastic' is not supported for 'swspatialorder'=4"
                )
        else:
            raise ValueError(
                f"Invalid value for 'swbasestate': {swbasestate}. "
                "Valid options are 'boussinesq' or 'anelastic'."
            )

    def _check_forcing(self) -> None:
        """Check if the specified forcing is valid."""
        force = self.case.ini_file.get("force", "swlspres", None)
        fc = self.case.ini_file.get("force", "fc", None)
        dpdx = self.case.ini_file.get("force", "dpdx", None)

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
            dpdx = self.case.ini_file["force"].get("dpdx", None)
            if dpdx is None:
                raise ValueError(
                    "'dpdx' must be specified when using 'swlspres = dpdx' forcing."
                )
        elif force == "uflux":
            uflux = self.case.ini_file["force"].get("uflux", None)
            if uflux is None:
                raise ValueError(
                    "'uflux' must be specified when using 'swlspres = uflux' forcing."
                )
        else:
            raise ValueError(
                f"Invalid value for 'swlspres': {force}. "
                "Valid options are 'geo', 'dpdx', or 'uflux'."
            )

    def _check_alignment(
        self,
        group: str,
        key: str,
        default: Union[str, bool],
        var_pattern: str,
        nc_group: str = "init",
        required_by_list: str = "",
    ) -> bool:
        """
        Generic check: If switch is True, variables must exist in nc_group.
        """
        switch_val = self.case.ini_file.get(group, key, default)

        is_active = (
            bool(switch_val)
            and switch_val != "0"
            and switch_val != 0
            and str(switch_val).lower() != "false"
        )

        available_vars = self._get_vars_in_group(nc_group)
        matching_vars = [v for v in available_vars if fnmatch.fnmatch(v, var_pattern)]
        has_vars = len(matching_vars) > 0

        if is_active and not has_vars:
            raise ValueError(
                f"Switch [{group}] {key} is active, but no variables matching "
                f"'{var_pattern}' found in NetCDF group '{nc_group}'."
            )
        if not is_active and has_vars:
            logger.warning(
                f"Switch [{group}] {key} is not active, but variables matching "
                f"'{var_pattern}' were found in NetCDF group '{nc_group}'."
            )
        if not is_active:
            return False

        # Check if items in a list (like timedeplist) actually exist as variables in initial conditions
        if required_by_list:
            config_list = self.case.ini_file.get(group, required_by_list, [])
            config_list = utils.object_to_list(config_list)
            if len(config_list) == 0:
                raise ValueError(
                    f"Switch [{group}] {key} is active, but the required list "
                    f"[{group}] {required_by_list} is empty."
                )
            for item in config_list:
                item_pattern = var_pattern.replace("*", item).replace("?", item)
                if not any(fnmatch.fnmatch(v, item_pattern) for v in available_vars):
                    raise ValueError(
                        f"Variable '{item}' in [{group}] {required_by_list} "
                        f"not found as '{item_pattern}' in group '{nc_group}'."
                    )

        return is_active

    def _get_vars_in_group(self, group_name: str):
        """Helper to get variable names from a specific NetCDF group."""
        try:
            initial_cond = xr.open_dataset(
                Path(self.case.file_directory, f"{self.case.name}_input.nc"),
                group=group_name,
            )
            return [str(v) for v in initial_cond.variables]
        except Exception:
            return []

    def _check_cross(self) -> None:
        swcross = self.case.ini_file.get("cross", "swcross", False)
        crosslist = self.case.ini_file.get("cross", "crosslist", []) or []
        if swcross and len(crosslist) == 0:
            raise ValueError("swcross is True, but crosslist is empty.")

    def _check_dump(self):
        swdump = self.case.ini_file.get("dump", "swdump", False)
        swdumplist = self.case.ini_file.get("dump", "dumplist", []) or []
        if swdump and len(swdumplist) == 0:
            raise ValueError("swdump is True, but dumplist is empty.")

    def _check_forcing_logic(self):
        # 1. Large Scale Forcing
        swls = self.case.ini_file.get("force", "swls", False)
        swtdep_ls = self.case.ini_file.get("force", "swtimedep_ls", False)

        if swtdep_ls and not swls:
            raise ValueError(
                "swtimedep_ls is True, but swls is False. swls must be enabled for LS forcing."
            )

        if swls:
            if swtdep_ls:
                self._check_alignment(
                    "force", "swtimedep_ls", False, "*_ls", "timedep", "timedeplist_ls"
                )
            else:
                self._check_alignment("force", "swls", False, "*_ls", "init", "lslist")

        # 2. Geostrophic Forcing
        swlspres = self.case.ini_file.get("force", "swlspres", False)
        swtdep_geo = self.case.ini_file.get("force", "swtimedep_geo", False)

        if swtdep_geo and swlspres != "geo":
            raise ValueError("swtimedep_geo is True, but swlspres is not 'geo'.")

        if swlspres == "geo":
            nc_grp = "timedep" if swtdep_geo else "init"
            self._check_alignment("force", "swlspres", False, "u_geo", nc_grp)
            self._check_alignment("force", "swlspres", False, "v_geo", nc_grp)

        # 3. Subsidence (w_ls)
        swwls = self.case.ini_file.get("force", "swwls", False)
        swtdep_wls = self.case.ini_file.get("force", "swtimedep_wls", False)

        if swtdep_wls and not swwls:
            raise ValueError("swtimedep_wls is True, but swwls is disabled.")

        if swwls:
            nc_grp = "timedep" if swtdep_wls else "init"
            self._check_alignment("force", "swwls", False, "w_ls", nc_grp)

        # 4. Nudging
        swnudge = self.case.ini_file.get("force", "swnudge", False)
        swtdep_nudge = self.case.ini_file.get("force", "swtimedep_nudge", False)

        if swtdep_nudge and not swnudge:
            raise ValueError("swtimedep_nudge is True, but swnudge is False.")

        # nudgefac is always required in 'init' group if nudging is on
        self._check_alignment("force", "swnudge", False, "nudgefac", "init")
        if swtdep_nudge:
            self._check_alignment(
                "force",
                "swtimedep_nudge",
                False,
                "*_nudge",
                "timedep",
                "timedeplist_nudge",
            )
        else:
            self._check_alignment(
                "force", "swnudge", False, "*_nudge", "init", "nudgelist"
            )

    def _check_boundary_logic(self):
        # Requires time_surface and *_sbot in timedep group
        self._check_alignment("boundary", "swtimedep", False, "time_surface", "timedep")
        self._check_alignment(
            "boundary", "swtimedep", False, "*", "timedep", "timedeplist"
        )

    def _check_thermo_logic(self):
        self._check_alignment("thermo", "swtimedep_pbot", False, "p_sbot", "timedep")

    def _check_grid_logic(self):
        """Checks spatial order consistency."""
        spatial = self.case.ini_file.get("grid", "swspatialorder", None)
        if spatial is None:
            return

        # Check Pressure and Advection match Spatial order
        for grp, key in [("pres", "swpres"), ("advec", "swadvec")]:
            val = str(self.case.ini_file.get(grp, key, spatial))
            # Extract first digit for advection schemes like '2i5'
            if int(val[0]) != int(spatial):
                raise ValueError(
                    f"[{grp}] {key} ({val}) must match [grid] swspatialorder ({spatial})"
                )

    def _check_source_position(self) -> None:
        """Check if the source position conflicts with the domain size."""
        if "source" not in self.case.ini_file.groups:
            return

        for coord in ["x", "y", "z"]:
            source_0 = np.atleast_1d(self.case.ini_file["source"][f"source_{coord}0"])
            sigma = np.atleast_1d(self.case.ini_file["source"][f"sigma_{coord}"])

            if len(source_0) != len(sigma):
                raise ValueError(
                    f"source_{coord}0 and sigma_{coord} must have the same length."
                )

            grid_size = float(self.case.ini_file["grid"][f"{coord}size"])

            for s0, sig in zip(source_0, sigma):
                source_ext = s0 + 2 * sig
                if source_ext > grid_size:
                    raise ValueError(
                        f"{coord}-positioning conflicts with the domain size."
                    )

    def _check_source_height(self) -> None:
        """Check if source height is greater than vertical resolution"""
        if "source" not in self.case.ini_file.groups:
            return

        source_heights = np.atleast_1d(self.case.ini_file["source"]["source_z0"])
        lowest_grid_level = self._get_lowest_grid_level()

        for source_height in source_heights:
            if source_height < lowest_grid_level:
                logger.info(
                    f"Source height of {source_height}m "
                    f"is smaller than the lowest grid level of {lowest_grid_level}m"
                )

    def _check_source_sigma(self) -> None:
        """Check if the source std is valid.
        It must be >0 and between grid size/4 and grid size/2 since the source is
        simulated within 4*sigma."""

        if "source" not in self.case.ini_file.groups:
            return

        # check if sigma is zero
        for coord in ["x", "y", "z"]:
            sigmas = np.atleast_1d(self.case.ini_file["source"][f"sigma_{coord}"])
            if np.any(sigmas <= 0):
                raise ValueError(f"'sigma_{coord}' cannot be <=0.")

    def _check_source_list(self) -> None:
        """Check if all required variables have the same length."""
        if "source" not in self.case.ini_file.groups:
            return

        sourcelist = np.atleast_1d(self.case.ini_file["source"]["sourcelist"])
        source_x0 = np.atleast_1d(self.case.ini_file["source"]["source_x0"])
        source_y0 = np.atleast_1d(self.case.ini_file["source"]["source_y0"])
        source_z0 = np.atleast_1d(self.case.ini_file["source"]["source_z0"])
        sigma_x = np.atleast_1d(self.case.ini_file["source"]["sigma_x"])
        sigma_y = np.atleast_1d(self.case.ini_file["source"]["sigma_y"])
        sigma_z = np.atleast_1d(self.case.ini_file["source"]["sigma_z"])
        strength = np.atleast_1d(self.case.ini_file["source"]["strength"])
        swvmr = np.atleast_1d(self.case.ini_file["source"]["swvmr"])

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

    def _check_tdep_source_location(self) -> None:
        """Check if switches for dtep source, tdep source list and initial conditions align"""
        self._check_alignment(
            group="source",
            key="swtimedep_location",
            default=False,
            var_pattern="source_?0_*",
            nc_group="timedep",
        )

    def _check_tdep_source_strength(self) -> None:
        """Check if switches for dtep source, tdep source list and initial conditions align"""
        self._check_alignment(
            group="source",
            key="swtimedep_strength",
            default=False,
            var_pattern="source_strength_*",
            nc_group="timedep",
        )

    def _check_buffer_height(self) -> None:
        """Check if the buffer height lies within the domain."""
        if "buffer" not in self.case.ini_file.groups:
            return

        zstart = self.case.ini_file.get("buffer", "zstart", 0)
        zsize = self.case.ini_file.get("grid", "zsize")
        if zsize is None:
            return
        if zstart > zsize:
            raise ValueError(
                f"Buffer height={zstart} lies outside of the domain "
                f" with zsize={zsize}."
            )

    def _check_surface_roughness_height(self) -> None:
        """Check if the surface roughness is smaller than the lowest vertical grid level."""
        # read in {case}_input.nc file:
        lowest_grid_level = self._get_lowest_grid_level()
        warning_threshold = 0.9 * lowest_grid_level

        if "boundary" not in self.case.ini_file.groups:
            return

        for roughness_type in ["z0m", "z0h"]:
            if roughness_type in self.case.ini_file["boundary"]:
                surface_roughness = self.case.ini_file["boundary"][roughness_type]
                if surface_roughness >= lowest_grid_level:
                    logger.warning(
                        f"Surface roughness ({roughness_type} = {surface_roughness} m) "
                        f"is larger than the lowest level of the grid (={lowest_grid_level} m). "
                        "This will result in NaNs in the output or cause the model to crash."
                    )
                elif surface_roughness >= warning_threshold:
                    logger.warning(
                        f"Surface roughness ({roughness_type} = {surface_roughness} m) "
                        f"similar to the lowest grid level (={lowest_grid_level} m). "
                        "This will result in NaNs in the output or cause the model to crash."
                    )

    def _check_sample_times(self) -> None:
        sample_time_2d = self.case.ini_file.get("cross", "sampletime", None)
        sample_time_3d = self.case.ini_file.get("dump", "sampletime", None)
        sample_time_stats = self.case.ini_file.get("stats", "sampletime", None)

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

    def _get_lowest_grid_level(self) -> float:
        """Get the lowest atmospheric grid level from the {case}_input.nc file."""
        initial_cond = io.open_initial_cond(self.case)
        grid_values = initial_cond.z.values
        non_neg_grid_values = [value for value in grid_values if value >= 0]
        lowest_grid_level = min(non_neg_grid_values)
        return lowest_grid_level

    def _copy_executable(self) -> None:
        """Copy MicroHH executable to scratch"""

        shutil.copy(self.microhh_executable_path, self.case.scratch_path)

    def copy_restart_files(self, origin: Union[str, Path], timestr: str) -> None:
        """Copy MicroHH restart files to scratch. The .ini and input.nc files
        are already copied when creating the case using self.case.initialise_case()."""

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
            shutil.copy(file, self.case.scratch_path)

    def get_restart_str(self, endtime: int) -> str:
        """Get endtime string used for restart files."""
        iotimeprec = self.case.ini_file.get("time", "iotimeprec", 0) or 0
        endtime_io = int(round(endtime * 10 ** (-iotimeprec)))

        return "{0:07d}".format(endtime_io)


def _get_microhh_run_commands(
    case_list: List[init.Case],
    mode: Union[RunMode, str] = RunMode.RUN,
    cpus_per_task_per_case: int = 72,
    gpus_per_task_per_case: int = 1,
    jobreport_path: Optional[str] = None,
) -> list[str]:
    """Generate the shell commands to run MicroHH.

    Args:
        case_list: List of cases to run.
        mode: Run mode ('run' for init+run, 'rerun' for run only).
        cpus_per_task_per_case: CPUs per srun step (multi-GPU only).
        gpus_per_task_per_case: GPUs per srun step (multi-GPU only).
        jobreport_path: Optional wrapper prefix for the final phase
            (multi-GPU only).

    Returns:
        List of shell command strings.
    """
    case_list = list(case_list)
    mode_val = mode.value if isinstance(mode, RunMode) else str(mode).lower()

    if mode_val == RunMode.RUN.value:
        phases = ["init", "run"]
    elif mode_val == RunMode.RERUN.value:
        phases = ["run"]
    else:
        raise ValueError("mode must be either 'run' or 'rerun'")

    if len(case_list) == 1:
        return _get_single_run_command(
            phases, case=case_list[0], jobreport_path=jobreport_path
        )

    elif len(case_list) > 1 and len(case_list) <= 4:
        return _get_multi_run_commands(
            phases,
            case_list=case_list,
            cpus_per_task_per_case=cpus_per_task_per_case,
            gpus_per_task_per_case=gpus_per_task_per_case,
            jobreport_path=jobreport_path,
        )
    else:
        raise ValueError(
            f"Expected between 1 and 4 cases, but got {len(case_list)}. For a "
            f"multi-GPU run, please provide 4 cases to use compute resources efficiently."
        )


def _get_single_run_command(
    phases: list[str], case: init.Case, jobreport_path: Optional[str] = None
) -> list[str]:
    """Generate srun commands for a single-case run, e.g. on a CPU node.

    Args:
        phases: List of MicroHH phases to execute (e.g. ["init", "run"]).
        case: The case to run.
        jobreport_path: Optional jobreport wrapper for the run phase.

    Returns:
        List of shell command strings.
    """
    commands = []
    last_phase_idx = len(phases) - 1
    for phase_idx, phase in enumerate(phases):
        if jobreport_path and phase_idx == last_phase_idx:
            commands.append(
                f"srun {jobreport_path} -- {case.scratch_path}/microhh {phase} {case.name}"
            )
        else:
            commands.append(f"srun {case.scratch_path}/microhh {phase} {case.name}")
    return commands


def _get_multi_run_commands(
    phases: list[str],
    case_list: List[init.Case],
    cpus_per_task_per_case: int = 72,
    gpus_per_task_per_case: int = 1,
    jobreport_path: Optional[str] = None,
) -> list[str]:
    """Generate srun commands that launch up to 4 cases in parallel on a GPU node.

    Args:
        phases: List of MicroHH phases to execute (e.g. ["init", "run"]).
        case_list: Cases to run in parallel, one per GPU.
        cpus_per_task_per_case: CPUs per srun step.
        gpus_per_task_per_case: GPUs per srun step.
        jobreport_path: Optional wrapper prefix for the final phase.

    Returns:
        List of shell command strings.
    """
    if len(case_list) > 4:
        raise ValueError(
            f"Expected at most 4 cases for multi-GPU run, but got {len(case_list)}."
        )
    case_name = _get_common_case_name(case_list)
    n_cases = len(case_list)
    executable = "./microhh"

    commands = []

    # Bash array with case directories
    commands.append("# Define the cases to run")
    commands.append("DIRS=(")
    for case in case_list:
        commands.append(f'    "{case.scratch_path}"')
    commands.append(")")
    commands.append("")

    last_phase_idx = len(phases) - 1
    for phase_idx, phase in enumerate(phases):
        commands.append(f"# Launch '{phase}' phase")
        commands.append(f"for i in {{0..{n_cases - 1}}}; do")

        srun_prefix = (
            f"    srun --exclusive"
            f" --ntasks-per-node=1"
            f" --cpus-per-task={cpus_per_task_per_case}"
            f" --gpus-per-task={gpus_per_task_per_case}"
            f" --chdir=${{DIRS[$i]}}"
            f" --output=${{DIRS[$i]}}/{phase}_microhh_%j.out"
        )

        if jobreport_path and phase_idx == last_phase_idx:
            srun_cmd = f"{srun_prefix} \\\n         {jobreport_path} -- {executable} {phase} {case_name} &"
        else:
            srun_cmd = f"{srun_prefix} \\\n         {executable} {phase} {case_name} &"

        commands.append(srun_cmd)
        commands.append("done")
        commands.append("")
        commands.append(f"# Wait for all '{phase}' phases to complete")
        commands.append("wait")
        commands.append("")

    return commands


def create_run_4x_script(
    cases: List[init.Case],
    alps_setup: cscs.AlpsSetup,
    scratch_path: str,
    overwrite: bool = False,
    jobreport_path: Optional[str] = None,
) -> Path:
    """Run 4 prepared simulations on GPU."""
    ncases = len(cases)

    if ncases != 4:
        logger.warning(f"Expected 4 cases, but got {len(cases)}.")

    master_job_file = Path(scratch_path, f"run_microhh_{ncases}x.sbash")

    commands = _get_microhh_run_commands(
        case_list=cases, mode=RunMode.RUN, jobreport_path=jobreport_path
    )

    if not master_job_file.exists() or overwrite:
        master_job_file = alps_setup.create_bash_script(
            commands=commands,
            job_name=f"run_microhh_{ncases}x",
            file_path=master_job_file,
            scratch_path=scratch_path,
        )
    else:
        logger.info(f"Reading existing bash script {master_job_file}")

    return Path(master_job_file)


def _get_common_case_name(cases: list[init.Case]) -> str:
    # Check if all cases have the same base name as this is also how the
    # .ini file and initial conditions are called.
    case_names = np.unique([case.name for case in cases])

    if len(case_names) > 1:
        raise ValueError(
            "Cases must have the same case name for the multi-gpu setup to work."
        )

    return str(case_names[0])


__all__ = [
    "PreProcessing",
    "create_run_4x_script",
]
