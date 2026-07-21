"""Pre-processing and job-preparation helpers for MicroHH cases."""

import fnmatch
import logging
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Union

import cscs
import ls2d
import numpy as np
import xarray as xr

import microhh_processing_chain.initialisation as init
from microhh_processing_chain import io, utils
from microhh_processing_chain.enums import RunMode

logger = logging.getLogger(__name__)


class PreProcessing:
    """Class used for pre-processing and running MicroHH cases.

    Args:
        case:
            Case to pre-process and run.
        alps_setup:
            Object containing the setup variables used for creating a bash
            script to run the MicroHH simulations.
        microhh_executable_path:
            Path to the MicroHH executable or the directory containing the
            MicroHH executable.
        jobreport_path:
            Path to the GPU job report executable. Only supported on GPU.
    """

    def __init__(
        self,
        case: init.Case,
        alps_setup: cscs.AlpsSetup,
        microhh_executable_path: Union[str, Path],
        jobreport_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self.case = case
        self.alps_setup = alps_setup
        self._microhh_executable_path = microhh_executable_path
        self.jobreport_path = jobreport_path

        self.microhh_executable_path = self._get_microhh_executable_path()
        self._check_microhh_build_architecture()
        self._check_jobpreport_gpu()

    def _get_microhh_executable_path(self) -> Path:
        """Resolve the MicroHH executable path from a file or containing directory."""
        microhh_executable_path = Path(self._microhh_executable_path)

        if microhh_executable_path.is_file():
            return microhh_executable_path

        mhh_files = list(microhh_executable_path.glob("microhh"))

        if len(mhh_files) != 1:
            raise ValueError(
                "Expected exactly one 'microhh' executable in "
                f"{microhh_executable_path}, but found {len(mhh_files)}."
            )

        return mhh_files[0]

    def _check_microhh_build_architecture(self) -> None:
        """Check if MicroHH executable matches the system architecture."""
        if not Path(self.microhh_executable_path).exists():
            logger.warning("File %s not found.", self.microhh_executable_path)

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
            logger.error("Mismatch: System is %s, but binary is %s.", arch, exec_arch)

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
                    "Failed to connect to %s. SSH error:\n%s",
                    self.alps_setup.ssh_connect,
                    e.output.strip(),
                )
                return None
        else:
            return str(platform.machine())

    def _check_jobpreport_gpu(self) -> None:
        """Check if GPU jobreport is only provided when using GPU partition."""
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
        """Prepare the case for running the MicroHH simulation.

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
        self.slurm_job = self.create_run_script(overwrite, mode=RunMode.RUN)

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
                    f"Required file '{filename}' not found in "
                    f"'{self.case.scratch_path}'."
                )

    def prepare_restart_simulation(
        self,
        restart_case: init.Case,
        restart_time: int,
        overwrite: bool = False,
        start_jobs: bool = False,
        wait: bool = False,
    ) -> None:
        """Prepare the case for running a restart MicroHH simulation.

        Args:
            restart_case: The case from which to restart.
                Must have the same name as the current case.
            restart_time: Time in seconds from which the simulation is restarted.
                Must match the start time in the .ini file.
            overwrite: Whether to overwrite existing files.
            start_jobs: Whether to start the simulation after
                preparation.
            wait: Whether to wait for the job to finish before
                returning.
        """
        # Check if .ini file has been updated for restart
        self._check_restart_ini_updated(restart_time)

        # check if the .ini setup is valid
        self._check_ini_setup_valid()

        # copy microHH to directory
        self._copy_executable()

        # copy microHH restart files
        restart_str = self.get_restart_str(restart_time)
        self.copy_restart_files(origin=restart_case.directory, timestr=restart_str)

        # Create bash script if it does not exist
        self.slurm_job = self.create_run_script(overwrite, mode=RunMode.RERUN)

        if start_jobs:
            # Launch the job on alps
            self.start_simulation(wait)

    def _check_restart_ini_updated(self, restart_time: int) -> None:
        """Check if the .ini file has been updated for a restart simulation.

        Args:
            restart_time:
                Time in seconds from which the simulation is restarted.
        """
        start_time = self.case.config["time"]["starttime"]

        if start_time != restart_time:
            raise ValueError(
                f"start time ({start_time}s) in .ini file does not match "
                f"the specified restart_time ({restart_time}s)."
            )
        end_time = self.case.config["time"]["endtime"]
        if end_time <= restart_time:
            raise ValueError(
                f"end time ({end_time}s) must be larger than "
                f"restart_time ({restart_time}s)."
            )

    def start_simulation(self, wait: bool = False) -> None:
        """Start the MicroHH simulation by executing the bash script on CSCS Alps.

        Args:
            wait:
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

        self.case.job_id = self.slurm_job.start(wait=wait)

    def cancel_simulation(self) -> None:
        """Cancel the running MicroHH simulation."""
        # check if self has attribute "job_id"
        if not hasattr(self, "job_id"):
            logger.info("Case has not been submitted yet.")

        if self.case.job_id is not None:
            subprocess.run(["scancel", str(self.case.job_id)])
            logger.info("Cancelling simulation %s", self.case.job_id)

    def create_run_script(
        self, overwrite: bool = False, mode: Union[RunMode, str] = RunMode.RUN
    ) -> cscs.SlurmJob:
        """Create a bash script to run the MicroHH simulation."""
        job_file = Path(self.case.scratch_path, "run_microhh.sbatch")

        commands = _get_microhh_run_commands(
            case_list=[self.case],
            mode=mode,
            jobreport_path=(
                str(self.jobreport_path) if self.jobreport_path is not None else None
            ),
        )

        if not job_file.exists() or overwrite:
            slurm_job = self.alps_setup.create_sbatch_script(
                commands=commands,
                job_name="run_microhh",
                file_path=job_file,
                scratch_path=self.case.scratch_path,
            )
        else:
            logger.info("Reading existing sbatch script %s", job_file)
            slurm_job = cscs.SlurmJob(job_file, ssh_connect=self.alps_setup.ssh_connect)

        return slurm_job

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
        self._check_initial_conditions_times()

    def _check_ini_content(self) -> None:
        """Check if the .ini file is not empty."""
        if len(self.case.config.groups) == 0:
            raise ValueError(f"{self.case.name}.ini is empty.")

    def _adjust_mpi_settings(self) -> None:
        """Adjust MPI settings if MPI is disabled."""
        if (
            self.alps_setup.processor == "gpu"
            and self.case.config["master"]["npx"] != 1
            and self.case.config["master"]["npy"] != 1
        ):
            for dim in ["x", "y"]:
                self.case.config.set_value(group="master", variable=f"np{dim}", value=1)
            self.case.config.save(self.case.ini_file_path, allow_overwrite=True)
            logger.info("Overwriting 'npx' and 'npy' because MPI is disabled.")

    def _check_grid_match(self) -> None:
        """Check if the grid size matches the number of processes."""
        itot, jtot, ktot = (
            self.case.config["grid"]["itot"],
            self.case.config["grid"]["jtot"],
            self.case.config["grid"]["ktot"],
        )
        npx, npy = (
            self.case.config["master"]["npx"],
            self.case.config["master"]["npy"],
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
            self.case.config["master"]["npx"] * self.case.config["master"]["npy"]
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
        swspatial = self.case.config.get("grid", "swspatialorder", None)
        swpres = self.case.config.get("pres", "swpres", None)
        swadvec = str(self.case.config.get("advec", "swadvec", None))

        if swspatial is None:
            return

        if swpres is not None and int(swpres) != int(swspatial):
            raise ValueError(
                "Grid order 'swspatialorder' "
                f"({swspatial}) does not match pressure advection order "
                f"'swpres' ({swpres})."
            )

        if swadvec is not None and int(swadvec[0]) != int(swspatial):
            raise ValueError(
                f"Grid order 'swspatialorder' ({swspatial}) does not match "
                f"scalar advection order 'swadvec' ({swadvec})."
            )

    def _check_thermo(self) -> None:
        """Check if the thermo settings are valid."""
        swthermo = self.case.config.get("thermo", "swthermo", None)
        swbasestate = self.case.config.get("thermo", "swbasestate", None)
        thref0 = self.case.config.get("thermo", "thref0", None)

        if swbasestate is None:
            return

        if swbasestate == "boussinesq":
            if swthermo == "dry" and thref0 is None:
                raise ValueError(
                    "'thref0' must be specified when using "
                    "'swbasestate'='boussinesq' and 'swthermo'='dry'."
                )

        elif swbasestate == "anelastic":
            swspatial = self.case.config.get("grid", "swspatialorder", None)
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
        force = self.case.config.get("force", "swlspres", None)
        fc = self.case.config.get("force", "fc", None)
        dpdx = self.case.config.get("force", "dpdx", None)

        if force is None or force == 0:
            return
        if force == "geo":
            if fc is None or fc == 0:
                raise ValueError(
                    "Coriolis parameter 'fc' must be specified and non-zero "
                    "when using 'swlspres = geo' forcing."
                )
        elif force == "dpdx":
            dpdx = self.case.config["force"].get("dpdx", None)
            if dpdx is None:
                raise ValueError(
                    "'dpdx' must be specified when using 'swlspres = dpdx' forcing."
                )
        elif force == "uflux":
            uflux = self.case.config["force"].get("uflux", None)
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
        """Check that active switches have matching variables in a NetCDF group."""
        switch_val = self.case.config.get(group, key, default)

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
                "Switch [%s] %s is not active, but variables matching "
                "'%s' were found in NetCDF group '%s'.",
                group,
                key,
                var_pattern,
                nc_group,
            )
        if not is_active:
            return False

        # Check if items in a list (like timedeplist) exist as variables in ic
        if required_by_list:
            config_list = self.case.config.get(group, required_by_list, [])
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

    def _get_vars_in_group(self, group_name: str) -> list[str]:
        """Get variable names from a specific NetCDF group."""
        try:
            initial_cond = xr.open_dataset(
                Path(self.case.directory, f"{self.case.name}_input.nc"),
                group=group_name,
            )
            return [str(v) for v in initial_cond.variables]
        except Exception:
            return []

    def _check_cross(self) -> None:
        """Check that enabled cross-section output has a configured variable list."""
        swcross = self.case.config.get("cross", "swcross", False)
        crosslist = self.case.config.get("cross", "crosslist", []) or []
        if swcross and len(crosslist) == 0:
            raise ValueError("swcross is True, but crosslist is empty.")

    def _check_dump(self) -> None:
        """Check that enabled dump output has a configured variable list."""
        swdump = self.case.config.get("dump", "swdump", False)
        swdumplist = self.case.config.get("dump", "dumplist", []) or []
        if swdump and len(swdumplist) == 0:
            raise ValueError("swdump is True, but dumplist is empty.")

    def _check_forcing_logic(self) -> None:
        """Check consistency between forcing switches and initial-condition groups."""
        # 1. Large Scale Forcing
        swls = self.case.config.get("force", "swls", False)
        swtdep_ls = self.case.config.get("force", "swtimedep_ls", False)

        if swtdep_ls and not swls:
            raise ValueError(
                "swtimedep_ls is True, but swls is False. "
                "swls must be enabled for LS forcing."
            )

        if swls:
            if swtdep_ls:
                self._check_alignment(
                    "force", "swtimedep_ls", False, "*_ls", "timedep", "timedeplist_ls"
                )
            else:
                self._check_alignment("force", "swls", False, "*_ls", "init", "lslist")

        # 2. Geostrophic Forcing
        swlspres = self.case.config.get("force", "swlspres", False)
        swtdep_geo = self.case.config.get("force", "swtimedep_geo", False)

        if swtdep_geo and swlspres != "geo":
            raise ValueError("swtimedep_geo is True, but swlspres is not 'geo'.")

        if swlspres == "geo":
            nc_grp = "timedep" if swtdep_geo else "init"
            self._check_alignment("force", "swlspres", False, "u_geo", nc_grp)
            self._check_alignment("force", "swlspres", False, "v_geo", nc_grp)

        # 3. Subsidence (w_ls)
        swwls = self.case.config.get("force", "swwls", False)
        swtdep_wls = self.case.config.get("force", "swtimedep_wls", False)

        if swtdep_wls and not swwls:
            raise ValueError("swtimedep_wls is True, but swwls is disabled.")

        if swwls:
            nc_grp = "timedep" if swtdep_wls else "init"
            self._check_alignment("force", "swwls", False, "w_ls", nc_grp)

        # 4. Nudging
        swnudge = self.case.config.get("force", "swnudge", False)
        swtdep_nudge = self.case.config.get("force", "swtimedep_nudge", False)

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

    def _check_boundary_logic(self) -> None:
        """Check consistency between boundary switches and timedep input variables."""
        # Requires time_surface and *_sbot in timedep group
        self._check_alignment("boundary", "swtimedep", False, "time_surface", "timedep")
        self._check_alignment(
            "boundary", "swtimedep", False, "*", "timedep", "timedeplist"
        )

    def _check_thermo_logic(self) -> None:
        """Check consistency between thermo switches and timedep input variables."""
        self._check_alignment("thermo", "swtimedep_pbot", False, "p_sbot", "timedep")

    def _check_grid_logic(self) -> None:
        """Check spatial-order consistency."""
        spatial = self.case.config.get("grid", "swspatialorder", None)
        if spatial is None:
            return

        # Check Pressure and Advection match Spatial order
        for grp, key in [("pres", "swpres"), ("advec", "swadvec")]:
            val = str(self.case.config.get(grp, key, spatial))
            # Extract first digit for advection schemes like '2i5'
            if int(val[0]) != int(spatial):
                raise ValueError(
                    f"[{grp}] {key} ({val}) must match [grid] "
                    f"swspatialorder ({spatial})."
                )

    def _check_source_position(self) -> None:
        """Check if the source position conflicts with the domain size."""
        if "source" not in self.case.config.groups:
            return

        for coord in ["x", "y", "z"]:
            source_0 = np.atleast_1d(self.case.config["source"][f"source_{coord}0"])
            sigma = np.atleast_1d(self.case.config["source"][f"sigma_{coord}"])

            if len(source_0) != len(sigma):
                raise ValueError(
                    f"source_{coord}0 and sigma_{coord} must have the same length."
                )

            grid_size = float(self.case.config["grid"][f"{coord}size"])

            for s0, sig in zip(source_0, sigma):
                source_ext = s0 + 2 * sig
                if source_ext > grid_size:
                    raise ValueError(
                        f"{coord}-positioning conflicts with the domain size."
                    )

    def _check_source_height(self) -> None:
        """Check if source height is greater than vertical resolution."""
        if "source" not in self.case.config.groups:
            return

        source_heights = np.atleast_1d(self.case.config["source"]["source_z0"])
        lowest_grid_level = self._get_lowest_grid_level()

        for source_height in source_heights:
            if source_height < lowest_grid_level:
                logger.info(
                    "Source height of %sm is smaller than the lowest grid level of %sm",
                    source_height,
                    lowest_grid_level,
                )

    def _check_source_sigma(self) -> None:
        """Check if the source std is valid.

        It must be >0 and between grid size/4 and grid size/2 since the source is
        simulated within 4*sigma.
        """
        if "source" not in self.case.config.groups:
            return

        # check if sigma is zero
        for coord in ["x", "y", "z"]:
            sigmas = np.atleast_1d(self.case.config["source"][f"sigma_{coord}"])
            if np.any(sigmas <= 0):
                raise ValueError(f"'sigma_{coord}' cannot be <=0.")

    def _check_source_list(self) -> None:
        """Check if all required variables have the same length."""
        if "source" not in self.case.config.groups:
            return

        sourcelist = np.atleast_1d(self.case.config["source"]["sourcelist"])
        source_x0 = np.atleast_1d(self.case.config["source"]["source_x0"])
        source_y0 = np.atleast_1d(self.case.config["source"]["source_y0"])
        source_z0 = np.atleast_1d(self.case.config["source"]["source_z0"])
        sigma_x = np.atleast_1d(self.case.config["source"]["sigma_x"])
        sigma_y = np.atleast_1d(self.case.config["source"]["sigma_y"])
        sigma_z = np.atleast_1d(self.case.config["source"]["sigma_z"])
        strength = np.atleast_1d(self.case.config["source"]["strength"])
        swvmr = np.atleast_1d(self.case.config["source"]["swvmr"])

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
        """Check if switches for source location and initial conditions align."""
        self._check_alignment(
            group="source",
            key="swtimedep_location",
            default=False,
            var_pattern="source_?0_*",
            nc_group="timedep",
        )

    def _check_tdep_source_strength(self) -> None:
        """Check if switches for source strength and initial conditions align."""
        self._check_alignment(
            group="source",
            key="swtimedep_strength",
            default=False,
            var_pattern="source_strength_*",
            nc_group="timedep",
        )

    def _check_buffer_height(self) -> None:
        """Check if the buffer height lies within the domain."""
        if "buffer" not in self.case.config.groups:
            return

        zstart = self.case.config.get("buffer", "zstart", 0)
        zsize = self.case.config.get("grid", "zsize")
        if zsize is None:
            return
        if zstart > zsize:
            raise ValueError(
                f"Buffer height={zstart} lies outside of the domain "
                f" with zsize={zsize}."
            )

    def _check_surface_roughness_height(self) -> None:
        """Check if surface roughness is smaller than the lowest vertical grid level."""
        # read in {case}_input.nc file:
        lowest_grid_level = self._get_lowest_grid_level()
        warning_threshold = 0.9 * lowest_grid_level

        if "boundary" not in self.case.config.groups:
            return

        for roughness_type in ["z0m", "z0h"]:
            if roughness_type in self.case.config["boundary"]:
                surface_roughness = self.case.config["boundary"][roughness_type]
                if surface_roughness >= lowest_grid_level:
                    logger.warning(
                        "Surface roughness (%s = %s m) is larger than the lowest "
                        "level of the grid (=%s m). This will result in NaNs in "
                        "the output or cause the model to crash.",
                        roughness_type,
                        surface_roughness,
                        lowest_grid_level,
                    )
                elif surface_roughness >= warning_threshold:
                    logger.warning(
                        "Surface roughness (%s = %s m) similar to the lowest grid "
                        "level (=%s m). This will result in NaNs in the output or "
                        "cause the model to crash.",
                        roughness_type,
                        surface_roughness,
                        lowest_grid_level,
                    )

    def _check_sample_times(self) -> None:
        """Warn when 2D, 3D, and statistics outputs use different sample times."""
        sample_time_2d = self.case.config.get("cross", "sampletime", None)
        sample_time_3d = self.case.config.get("dump", "sampletime", None)
        sample_time_stats = self.case.config.get("stats", "sampletime", None)

        # Remove None values
        sample_times = [sample_time_3d, sample_time_2d, sample_time_stats]
        sample_times = [s for s in sample_times if s is not None]

        # If more than one unique value exists among the found keys, warn the user
        if len(set(sample_times)) > 1:
            logger.warning(
                "Sample times are not the same. It is recommended to have the same "
                "sample time for 3d output '%.0f's, 2d output '%.0f's, and "
                "statistics '%.0f's for postprocessing convenience.",
                sample_time_3d,
                sample_time_2d,
                sample_time_stats,
            )

    def _check_initial_conditions_times(self) -> None:
        """Check if the times in boundary conditions match config.

        Check if the times specified in the boundary conditions match the time
        specified in the .ini file.
        """
        ic = io.open_initial_cond(self.case)

        time_coords = [name for name in ic.coords if "time" in str(name)]

        for time_coord in time_coords:
            ic_start_time = int(ic[time_coord].values[0])
            ic_end_time = int(ic[time_coord].values[-1])

            ini_start_time = int(self.case.config["time"]["starttime"])
            ini_end_time = int(self.case.config["time"]["endtime"])

            # If times do not match exactly, log a general warning.
            if ini_start_time != ic_start_time or ini_end_time != ic_end_time:
                logger.warning(
                    "Times in .ini file do not match exactly the times in initial "
                    "conditions variable '%s': ini start=%ss, ic start=%ss; "
                    "ini end=%ss, ic end=%ss.",
                    time_coord,
                    ini_start_time,
                    ic_start_time,
                    ini_end_time,
                    ic_end_time,
                )

            # Raise value errors for problematic mismatches
            if ini_start_time < ic_start_time:
                raise ValueError(
                    f"Start time in .ini file ({ini_start_time}s) is earlier "
                    f"than the start time in initial conditions variable "
                    f"'{time_coord}' ({ic_start_time}s)."
                )
            if ini_end_time > ic_end_time:
                raise ValueError(
                    f"End time in .ini file ({ini_end_time}s) is later "
                    f"than the end time in initial conditions variable "
                    f"'{time_coord}' ({ic_end_time}s)."
                )

    def _get_lowest_grid_level(self) -> float:
        """Get the lowest atmospheric grid level from the {case}_input.nc file."""
        initial_cond = io.open_initial_cond(self.case)
        grid_values = initial_cond.z.values
        non_neg_grid_values = [value for value in grid_values if value >= 0]
        lowest_grid_level = min(non_neg_grid_values)
        return lowest_grid_level

    def _copy_executable(self) -> None:
        """Copy MicroHH executable to scratch."""
        shutil.copy(self.microhh_executable_path, self.case.scratch_path)

    def copy_restart_files(self, origin: Union[str, Path], timestr: str) -> None:
        """Copy MicroHH restart files to scratch.

        The .ini and input.nc files are already copied when creating the case using
        self.case.initialise_case().
        """
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
            # check if file exists in target directory
            target_file = Path(self.case.scratch_path, file.name)
            if target_file.exists():
                logger.info(
                    "Restart file %s already exists in scratch directory. "
                    "Skipping copy.",
                    target_file,
                )
                continue
            else:
                shutil.copy(file, self.case.scratch_path)

    def get_restart_str(self, endtime: int) -> str:
        """Get endtime string used for restart files."""
        iotimeprec = self.case.config.get("time", "iotimeprec", 0) or 0
        endtime_io = round(endtime * 10 ** (-iotimeprec))

        return f"{endtime_io:07d}"


def _get_microhh_run_commands(
    case_list: list[init.Case],
    mode: Union[RunMode, str] = RunMode.RUN,
    cpus_per_task_per_case: int = 72,
    gpus_per_task_per_case: int = 1,
    jobreport_path: Optional[str] = None,
) -> list[str]:
    """Generate the shell commands to run MicroHH.

    Args:
        case_list: list of cases to run.
        mode: Run mode, determines which phases of MicroHH to execute.
            "run" (default) executes both "init" and "run" phases, while
            "rerun" and "restart" execute only the "run" phase.
        cpus_per_task_per_case: CPUs per srun step (multi-GPU only).
        gpus_per_task_per_case: GPUs per srun step (multi-GPU only).
        jobreport_path: Optional wrapper prefix for the final phase
            (multi-GPU only).

    Returns:
        list of shell command strings.
    """
    case_list = list(case_list)
    mode_val = mode.value if isinstance(mode, RunMode) else str(mode).lower()

    if mode_val == RunMode.RUN.value:
        phases = ["init", "run"]
    elif mode_val in (RunMode.RERUN.value, RunMode.RESTART.value):
        phases = ["run"]
    else:
        raise ValueError("mode must be either 'run', 'rerun', or 'restart'")

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
            "multi-GPU run, please provide 4 cases to use compute resources "
            "efficiently."
        )


def _get_single_run_command(
    phases: list[str], case: init.Case, jobreport_path: Optional[str] = None
) -> list[str]:
    """Generate srun commands for a single-case run, e.g. on a CPU node.

    Args:
        phases: list of MicroHH phases to execute (e.g. ["init", "run"]).
        case: The case to run.
        jobreport_path: Optional jobreport wrapper for the run phase.

    Returns:
        list of shell command strings.
    """
    commands = []
    last_phase_idx = len(phases) - 1
    for phase_idx, phase in enumerate(phases):
        if jobreport_path and phase_idx == last_phase_idx:
            cmd = (
                f"srun {jobreport_path} -- "
                f"{case.scratch_path}/microhh {phase} {case.name}"
            )
        else:
            cmd = f"srun {case.scratch_path}/microhh {phase} {case.name}"
        commands.append(cmd)
    return commands


def _get_multi_run_commands(
    phases: list[str],
    case_list: list[init.Case],
    cpus_per_task_per_case: int = 72,
    gpus_per_task_per_case: int = 1,
    jobreport_path: Optional[str] = None,
) -> list[str]:
    """Generate srun commands that launch up to 4 cases in parallel on a GPU node.

    Args:
        phases: list of MicroHH phases to execute (e.g. ["init", "run"]).
        case_list: Cases to run in parallel, one per GPU.
        cpus_per_task_per_case: CPUs per srun step.
        gpus_per_task_per_case: GPUs per srun step.
        jobreport_path: Optional wrapper prefix for the final phase.

    Returns:
        list of shell command strings.
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
            srun_cmd = (
                f"{srun_prefix} \\\n         {jobreport_path} -- "
                f"{executable} {phase} {case_name} &"
            )
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
    cases: list[init.Case],
    alps_setup: cscs.AlpsSetup,
    scratch_path: str,
    mode: Union[RunMode, str] = RunMode.RUN,
    overwrite: bool = False,
    jobreport_path: Optional[str] = None,
) -> cscs.SlurmJob:
    """Create a SLURM script that runs prepared simulations on a GPU node."""
    ncases = len(cases)
    run_mode = mode.value if isinstance(mode, RunMode) else str(mode).lower()

    if ncases != 4:
        logger.warning("Expected 4 cases, but got %d.", len(cases))

    master_job_file = Path(scratch_path, f"{run_mode}_microhh_{ncases}x.sbatch")

    commands = _get_microhh_run_commands(
        case_list=cases, mode=mode, jobreport_path=jobreport_path
    )

    if not master_job_file.exists() or overwrite:
        slurm_job = alps_setup.create_sbatch_script(
            commands=commands,
            job_name=f"{run_mode}_microhh_{ncases}x",
            file_path=master_job_file,
            scratch_path=scratch_path,
        )
    else:
        logger.info("Reading existing sbatch script %s", master_job_file)
        slurm_job = cscs.SlurmJob(master_job_file, ssh_connect=alps_setup.ssh_connect)

    return slurm_job


def _get_common_case_name(cases: list[init.Case]) -> str:
    """Return the common base case name required for multi-GPU runs."""
    # Check if all cases have the same base name as this is also how the
    # .ini file and initial conditions are called.
    case_names = np.unique([case.name for case in cases])

    if len(case_names) > 1:
        raise ValueError(
            "Cases must have the same case name for the multi-gpu setup to work."
        )

    return str(case_names[0])


def interpolate_to_model_profile(
    grid: Union[
        np.ndarray,
        ls2d.grid.Grid_equidist,
        ls2d.grid.Grid_linear_stretched,
        ls2d.grid.Grid_stretched,
        ls2d.grid.Grid_stretched_manual,
    ],
    variable_value: Union[float, int] = 0.0,
    variable_profile: Optional[Union[list, np.ndarray]] = None,
    input_profile: Optional[Union[list, np.ndarray]] = None,
) -> np.ndarray:
    """Interpolate a vertical profile to the MicroHH model height levels.

    Can also be used to create a constant profile by specifying only
    *variable_value*.

    Args:
        grid: MicroHH grid object (from LS2D) or a height-level array.
        variable_value: Constant value used when *variable_profile* is
            not provided.
        variable_profile: Vertical profile of the variable to
            interpolate.
        input_profile: Height levels corresponding to
            *variable_profile*.

    Returns:
        Values interpolated (or filled) to the model height levels.

    Raises:
        ValueError: If neither *variable_value* nor *variable_profile*
            is provided, or if *grid* is not a recognised type.
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
        raise TypeError(
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
        if input_profile is None:
            raise ValueError(
                "Please provide 'input_profile' when 'variable_profile' is given."
            )
        model_var_profile = np.interp(model_profile, input_profile, variable_profile)

    return model_var_profile


__all__ = [
    "PreProcessing",
    "create_run_4x_script",
    "interpolate_to_model_profile",
]
