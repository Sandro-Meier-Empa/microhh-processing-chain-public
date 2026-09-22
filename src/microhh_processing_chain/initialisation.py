"""Case lifecycle helpers for MicroHH simulations."""

import json
import logging
import re
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Optional, Union

import cscs
import microhh_py.microhh_tools as mht
import pandas as pd

from microhh_processing_chain import utils
from microhh_processing_chain.enums import FileType, InitMethod
from microhh_processing_chain.spatial import Domain

logger = logging.getLogger(__name__)


class Case:
    """Define a case for MicroHH.

    Can be used for initialising the model, pre-processing,
    post-processing, and visualisation.

    Args:
        name: Name of the case. Must be set up in ``microhh/cases``.
        suffix: Suffix for the specific model run.
            Combined as ``{name}.{suffix}``.
    """

    def __init__(
        self,
        name: str,
        suffix: str,
    ) -> None:
        self.name = name
        self.suffix = suffix
        self.full_name = f"{name}.{suffix}" if suffix else f"{name}"

        # define paths that will be set during initinitialisation
        self._directory: Optional[Path] = None
        self._ini_file_path: Optional[Path] = None
        self._scratch_path: Optional[Path] = None

        # define attributes that will be set during initialisation
        self._config: Optional[mht.Read_namelist] = None
        self.job_id: Optional[int] = None

        return None

    @property
    def directory(self) -> Path:
        """Get the directory of the case."""
        if self._directory is None:
            raise RuntimeError("File directory has not been set yet.")
        return self._directory

    @directory.setter
    def directory(self, value: Union[str, Path]) -> None:
        """Set the directory of the case."""
        self._directory = Path(value)

    @property
    def ini_file_path(self) -> Path:
        """Get the path to the INI file of the case."""
        if self._ini_file_path is None:
            raise RuntimeError("INI file path has not been set yet.")
        return self._ini_file_path

    @ini_file_path.setter
    def ini_file_path(self, value: Union[str, Path]) -> None:
        """Set the path to the INI file of the case."""
        self._ini_file_path = Path(value)

    @property
    def scratch_path(self) -> Path:
        """Get the scratch path of the case."""
        if self._scratch_path is None:
            raise RuntimeError("Scratch path has not been set yet.")
        return self._scratch_path

    @scratch_path.setter
    def scratch_path(self, value: Union[str, Path]) -> None:
        """Set the scratch path of the case."""
        self._scratch_path = Path(value)

    @property
    def out_file_path(self) -> Path:
        """Get the path to the output file of the case."""
        return Path(self.directory, f"{self.name}.out")

    @property
    def slurm_job(self) -> cscs.SlurmJob:
        """Get the Slurm job associated with the case."""
        job_files = list(Path(self.directory).glob("run_microhh*.sbatch"))
        if not job_files:
            raise FileNotFoundError(
                f"No sbatch files found in {self.directory}. "
                "Please ensure the case has been initialised and submitted."
            )

        if len(job_files) > 1:
            logger.warning(
                "Multiple sbatch files found in %s. Using the last one: %s",
                self.directory,
                job_files[-1],
            )
        return cscs.SlurmJob(
            path=job_files[-1],  # Use the last sbatch file if multiple exist
            job_id=self.job_id,
        )

    @property
    def initial_conditions_file_path(self) -> Path:
        """Get the path to the initial conditions file of the case."""
        return Path(self.directory, f"{self.name}_input.nc")

    @property
    def statistics_file_path(self) -> Path:
        """Path to the statistics file of the case."""
        start_time = int(self.config["time"]["starttime"])
        return Path(self.directory, f"{self.name}.default.{start_time:07d}.nc")

    @property
    def config(self) -> mht.Read_namelist:
        """Get the configuration of the case from the INI file."""
        if self._config is None:
            raise RuntimeError("Config has not been loaded yet.")
        return self._config

    @config.setter
    def config(self, value: mht.Read_namelist) -> None:
        """Set the loaded INI configuration."""
        self._config = value

    @property
    def run_state(self) -> str:
        """Get run state of the case based on SLURM sacct information."""
        if self.job_id is None:
            return "NOT SUBMITTED"

        sacct_info = cscs.get_sacct_info(self.job_id)

        # filter sacct info for the specific job name
        try:
            sacct_info = sacct_info[sacct_info["JobName"] == "run_microhh"]
            state_series = pd.Series(sacct_info["State"])
            return str(state_series.iloc[0])
        except (KeyError, IndexError):
            return "UNKNOWN"

    # add method when printing case
    def __repr__(self) -> str:
        """Return a string representation of the case.

        Includes the name, suffix, directory, and INI file content.
        """
        ini_content = ""
        if self._ini_file_path is not None and self._ini_file_path.exists():
            with self._ini_file_path.open("r") as f:
                ini_content = f.read()

        case_str = (
            f"Case(\n"
            f"  name='{self.name}',\n"
            f"  suffix='{self.suffix}',\n"
            f"  path='{self.directory}'\n"
        )

        if self.job_id is not None:
            case_str += f"  job_id={self.job_id},\n"

        if self.run_state is not None:
            case_str += f"  run_state='{self.run_state}',\n"

        if ini_content:
            case_str += f"  ini_file=\n{ini_content}\n)"

        return case_str

    def _initialise(
        self,
        scratch_path: Union[str, Path],
        method: Union[InitMethod, str] = InitMethod.INTERACTIVE,
        init_from: Optional[Union[str, Path]] = None,
        overwrite: bool = False,
    ) -> None:
        """Initialise a case by creating or reading in a directory.

        Args:
            scratch_path: Path of target folder on *scratch*.
            method: Method of how the case should be set up.
                Accepted values are:

                - ``'default'``: Create ``.ini`` file and initial conditions
                  by running a Python script.
                - ``'interactive'``: Create ``.ini`` file and initial
                  conditions interactively.
                - ``'rerun'``: Rerun an existing case from *store* or
                  *scratch*.
            init_from: Path to the case used as a template.
                Required when *method* is ``'default'`` or ``'rerun'``.
            overwrite: Whether to overwrite an existing case directory.
        """
        self.scratch_path = Path(scratch_path, self.full_name)
        self.directory = self.scratch_path
        self.ini_file_path = Path(self.scratch_path, f"{self.name}.ini")

        self._handle_method(method, init_from)
        self._handle_scratch_dir(overwrite)
        self._handle_running_method(init_from=init_from)

        self.job_id = self.job_id if self.job_id is not None else self._get_job_id()

    def _handle_method(
        self,
        method: Union[InitMethod, str],
        init_from: Optional[Union[str, Path]] = None,
    ) -> None:
        """Validate and normalise the initialisation method.

        Args:
            method: Initialisation method (``'default'``,
                ``'interactive'``, or ``'rerun'``).
            init_from: Path to the source case directory. Required for
                ``'default'`` and ``'rerun'`` methods.

        Raises:
            ValueError: If *method* is ``'default'`` or ``'rerun'`` and
                *init_from* is not provided.
        """
        # Normalize method to string for comparison
        method_val = (
            method.value if isinstance(method, InitMethod) else str(method).lower()
        )
        self.method = method_val

        if (
            self.method in [InitMethod.DEFAULT.value, InitMethod.RERUN.value]
            and init_from is None
        ):
            msg = (
                "When using 'default' or 'rerun' method, please provide "
                "the path of the directory containing `.ini` and initial "
                "condition files."
            )
            raise ValueError(msg)

    def _handle_scratch_dir(self, overwrite: bool = False) -> None:
        """Create the case directory on *scratch*.

        If the directory already exists and *overwrite* is ``True``, the
        user is prompted for confirmation before the existing directory
        is removed and re-created.

        Args:
            overwrite: Whether to overwrite an existing directory.
        """
        # set up directory on scratch
        if not self.scratch_path.exists():
            logger.info("Creating %s", self.scratch_path)
            self.scratch_path.mkdir(parents=True)
        elif overwrite:
            with utils.confirm_action(
                f"Are you sure you want to overwrite {self.scratch_path}?"
            ) as confirmed:
                if confirmed:
                    shutil.rmtree(self.scratch_path)
                    logger.info("Overwriting %s", self.scratch_path)
                    self.scratch_path.mkdir(parents=True)

    def _handle_running_method(
        self, init_from: Optional[Union[str, Path]] = None
    ) -> None:
        """Execute the initialisation logic for the selected method.

        Depending on ``self.method``, this will:

        - ``'default'``: Validate the source case, create initial
          conditions, and copy ``.ini`` and input files.
        - ``'interactive'``: Create an empty ``.ini`` file.
        - ``'rerun'``: Copy ``.ini`` and input files from an existing case.

        Args:
            init_from: Path to the source case directory.

        Raises:
            ValueError: If ``self.method`` is not a recognised value.
        """
        if self.method == InitMethod.DEFAULT.value:
            if init_from is None:
                raise ValueError(
                    "Please provide 'init_from' for default initialisation."
                )
            case_path = Path(init_from)
            self._check_case_setup(case_path)
            self._create_initial(case_path)
            self._copy_ini(case_path)
            self._copy_initial(case_path)
        elif self.method == InitMethod.INTERACTIVE.value:
            self.ini_file_path.touch()
        elif self.method == InitMethod.RERUN.value:
            if init_from is None:
                raise ValueError("Please provide 'init_from' for rerun initialisation.")
            case_path = Path(init_from)
            self._copy_ini(case_path)
            self._copy_initial(case_path)
        else:
            raise ValueError(
                f"{self.method} is not available. "
                "Use 'default', 'interactive' or 'rerun'."
            )
        self.config = mht.Read_namelist(self.ini_file_path)

    def _get_job_id(self) -> Optional[int]:
        """Look for the job ID in the ``run_microhh*.out`` file.

        Returns:
            The SLURM job ID extracted from the output filename,
            or ``None`` if no output file is found.
        """
        out_files = list(Path(self.directory).glob("run_microhh*.out"))

        if not out_files:
            return None

        job_id: Optional[int] = None
        for file in out_files:
            file_name = file.name
            match = re.search(r"\d{6}", file_name)
            if match is not None:
                job_id = int(match.group())

        return job_id

    def _check_case_setup(self, case_path: Union[str, Path]) -> None:
        """Verify that the case directory exists and contains required files.

        Args:
            case_path: Path to the case directory to validate.

        Raises:
            FileNotFoundError: If the case directory does not exist or
                required files are missing.
        """
        case_path = Path(case_path)

        if not case_path.exists():
            raise FileNotFoundError(f"Case '{self.name}' does not exist.")

        self._check_required_files(case_path)

    def _check_required_files(self, case_path: Union[str, Path]) -> None:
        """Check for the existence of ``{name}_input.py`` and ``{name}*.ini`` files.

        Args:
            case_path: Path to the case directory.

        Raises:
            FileNotFoundError: If ``{name}_input.py`` or a matching
                ``.ini`` file is missing.
        """
        case_path = Path(case_path)

        input_py_path = case_path / f"{self.name}_input.py"
        ini_files = list(case_path.glob(f"{self.name}*.ini"))

        if not input_py_path.exists():
            raise FileNotFoundError(f"Required file '{input_py_path.name}' is missing")

        if not ini_files:
            raise FileNotFoundError(f"No .ini file found matching '{self.name}*.ini'")

    def _copy_single_file(
        self,
        case_path: Union[str, Path],
        pattern: str,
        new_filename: str,
        exclude_pattern: Optional[str] = None,
    ) -> None:
        """Copy a single file matching *pattern* from *case_path* to scratch.

        Args:
            case_path: Source directory to search for the file.
            pattern: Glob pattern used to locate the file.
            new_filename: Name for the copied file in the scratch directory.
            exclude_pattern: Substring to exclude from matched filenames.

        Raises:
            FileNotFoundError: If zero or more than one file matches
                the pattern.
        """
        case_path = Path(case_path)
        matched_files = list(case_path.rglob(pattern))

        if exclude_pattern:
            matched_files = [p for p in matched_files if exclude_pattern not in p.name]

        if len(matched_files) != 1:
            raise FileNotFoundError(
                f"Expected one file with pattern '{pattern}' in '{case_path}', "
                f"found {len(matched_files)}."
            )

        src = matched_files[0]
        dst = self.scratch_path / new_filename
        shutil.copy(str(src), str(dst))

    def _copy_ini(self, case_path: Union[str, Path]) -> None:
        """Copy the ``.ini`` file from *case_path* to the scratch directory.

        Args:
            case_path: Source directory containing the ``.ini`` file.
        """
        self._copy_single_file(
            case_path=case_path,
            pattern="*.ini",
            new_filename=f"{self.name}.ini",
            exclude_pattern="_base.ini",
        )

    def _copy_initial(self, case_path: Union[str, Path]) -> None:
        """Copy the initial conditions file from `case_path` to scratch.

        Args:
            case_path: Source directory containing the input file.
        """
        self._copy_single_file(
            case_path=case_path,
            pattern="*_input.nc",
            new_filename=f"{self.name}_input.nc",
        )

    def _remove_existing_files(self) -> None:
        """Remove all files in the scratch directory except core case files.

        The ``.ini``, ``_input.nc``, and ``microhh`` executable files are
        preserved. The user is prompted for confirmation before deletion.
        """
        files_to_keep = [
            f"{self.name}.ini",
            f"{self.name}_input.nc",
            "microhh",
        ]
        with utils.confirm_action(
            f"Are you sure you want to remove all files in {self.scratch_path}?"
        ) as confirmed:
            if not confirmed:
                return

            for item in self.scratch_path.iterdir():
                if item.name not in files_to_keep and item.is_file():
                    item.unlink()

    def _create_initial(self, case_path: Union[str, Path]) -> None:
        """Create the initial conditions by running the case's input script.

        Executes ``{name}_input.py`` located in *case_path* to generate
        the initial condition files.

        Args:
            case_path: Directory containing the ``{name}_input.py`` script.
        """
        py_script_path = Path(case_path, f"{self.name}_input.py")

        if py_script_path.exists():
            subprocess.run(["python", str(py_script_path)], cwd=case_path)
        else:
            logger.info("%s not found.", py_script_path)

    def overwrite_ini(
        self, settings: Optional[dict] = None, overwrite: bool = True
    ) -> None:
        """Overwrite the ``.ini`` file of the case with the given settings.

        Clears all existing groups before applying *settings*.

        Args:
            settings: Nested dictionary of ``{group: {variable: value}}``
                pairs to write.
            overwrite: Whether to allow overwriting the existing file.
        """
        if settings is None:
            settings = {}

        self.config.groups = {}

        self.modify_ini(settings=settings, overwrite=overwrite)

    def modify_ini(self, settings: dict, overwrite: bool = True) -> None:
        """Modify the ``.ini`` file of the case in-place.

        Values set to ``None`` are removed. Variables not present in
        *settings* remain unchanged.

        Args:
            settings: Nested dictionary of ``{group: {variable: value}}``
                pairs to update. Use ``None`` as a value to remove a
                variable.
            overwrite: Whether to allow overwriting the existing file.
        """
        for group, variables in settings.items():
            for variable, value in variables.items():
                if value is None:
                    self.config.remove_value(group=group, variable=variable)
                else:
                    self.config.set_value(group=group, variable=variable, value=value)

        self.config.save(self.ini_file_path, allow_overwrite=overwrite)
        self.config = mht.Read_namelist(self.ini_file_path)

    def open_mhh_in_vscode(self, filetype: Union[FileType, str]) -> None:
        """Open the specified file in VS Code.

        Args:
            filetype: File type to open. Can be a :class:`FileType` enum
                member or one of ``'ini'``, ``'out'``, ``'sbatch'``,
                ``'sbatch_out'``.

        Raises:
            FileNotFoundError: If no matching file is found (for
                ``'sbatch_out'``).
            ValueError: If *filetype* is not a recognised value.
        """
        # Normalize filetype to string
        filetype_val = (
            filetype.value if isinstance(filetype, FileType) else str(filetype).lower()
        )

        if filetype_val == FileType.INI.value:
            filename = f"{self.name}.ini"
        elif filetype_val == FileType.OUT.value:
            filename = f"{self.name}.out"
        elif filetype_val == FileType.SBATCH.value:
            filename = "run_microhh.sbatch"
        elif filetype_val == FileType.SBATCH_OUT.value:
            # Use wildcard matching
            matches = list(Path(self.directory).glob("run_microhh*.out"))
            if not matches:
                raise FileNotFoundError("No sbatch_out files found.")
            filename = max(matches, key=lambda f: f.stat().st_mtime)
        else:
            raise ValueError(
                f"{filetype} is not available. "
                "Use 'ini', 'out', 'sbatch' or 'sbatch_out'."
            )

        file_path = self.directory / filename

        if file_path.exists():
            utils.open_in_vscode(file_path)
        else:
            logger.info("%s does not exist.", file_path)

    def calculate_benchmark(
        self, save: bool = False, ssh_connect: Optional[str] = None
    ) -> dict:
        """Calculate benchmark information from the ``.out`` file of the case.

        Args:
            save: If ``True``, persist the benchmark as a JSON file in
                the case directory.
            ssh_connect: SSH connection string for retrieving SLURM
                accounting data from a remote cluster.

        Returns:
            Dictionary containing grid resolution, simulated/elapsed
            time, and resource usage metrics.
        """
        benchmark = {}

        simulated_time_s = self._get_simulated_time()
        simulated_time = pd.to_timedelta(simulated_time_s, unit="s")

        # get setup from ini file
        itot = self.config["grid"]["itot"]
        jtot = self.config["grid"]["jtot"]
        ktot = self.config["grid"]["ktot"]

        xsize = self.config["grid"]["xsize"]
        ysize = self.config["grid"]["ysize"]
        zsize = self.config["grid"]["zsize"]

        # get saccing info
        if self.job_id is None:
            raise ValueError("No SLURM job ID found for this case.")

        info = cscs.get_sacct_info(self.job_id, ssh_connect=ssh_connect)

        # get first row of info
        info = info.iloc[0]
        nnodes = info["NNodes"]
        ncpus = info["NCPUS"]
        elapsed = pd.to_timedelta(info["Elapsed"])

        benchmark["itot"] = itot
        benchmark["jtot"] = jtot
        benchmark["ktot"] = ktot
        benchmark["dx"] = xsize / itot
        benchmark["dy"] = ysize / jtot
        benchmark["dz"] = zsize / ktot

        benchmark["simulated_time"] = str(simulated_time)
        benchmark["simulated_time_s"] = simulated_time_s
        benchmark["elapsed_time"] = str(elapsed)
        benchmark["elapsed_time_s"] = elapsed.total_seconds()
        benchmark["nnodes"] = int(nnodes)
        benchmark["ncpus"] = int(ncpus)

        if save:
            # Save benchmark information as a .json file
            json_file_path = self.directory / f"{self.name}_benchmark.json"
            with json_file_path.open("w") as f:
                json.dump(benchmark, f, indent=4)

        return benchmark

    def load_benchmark(self) -> dict:
        """Load benchmark information from a JSON file in the case directory.

        Returns:
            Dictionary containing the benchmark information.

        Raises:
            FileNotFoundError: If the benchmark JSON file does not exist.
        """
        json_file_path = self.directory / f"{self.name}_benchmark.json"

        if not json_file_path.exists():
            raise FileNotFoundError(
                f"Benchmark file '{json_file_path}' does not exist."
            )

        with json_file_path.open("r") as f:
            benchmark = json.load(f)

        # add some newly calculated fields
        if "simulated_time" not in benchmark:
            benchmark["simulated_time"] = pd.to_timedelta(
                benchmark["simulated_time_s"], unit="s"
            )

        if "elapsed_time_s" not in benchmark:
            benchmark["elapsed_time_s"] = pd.to_timedelta(
                benchmark["elapsed_time"]
            ).total_seconds()

        return benchmark

    def _get_simulated_time(self) -> float:
        """Get the total simulated time from the ``.out`` file of the case.

        Returns:
            Total simulated time in seconds.

        Raises:
            FileNotFoundError: If the ``.out`` file does not exist.
        """
        out_file_path = self.directory / f"{self.name}.out"

        if not out_file_path.exists():
            raise FileNotFoundError(f"{out_file_path} does not exist.")

        df = pd.read_csv(out_file_path, sep=r"\s+")
        last_time_value = df["TIME"].values[-1]

        return float(last_time_value)

    def rename(
        self, new_name: Optional[str] = None, new_suffix: Optional[str] = None
    ) -> None:
        """Rename this MicroHH case by renaming its directory and ``.ini`` file.

        Args:
            new_name: New name for the case. Keeps the existing name
                if not provided.
            new_suffix: New suffix for the case. Keeps the existing
                suffix if not provided.

        Raises:
            ValueError: If neither *new_name* nor *new_suffix* is provided.
            FileExistsError: If the target directory already exists.
        """
        if new_name is None and new_suffix is None:
            raise ValueError("Please provide a new name or a new suffix for the case.")

        new_name = new_name if new_name is not None else self.name
        new_suffix = new_suffix if new_suffix is not None else self.suffix
        new_name_spec = f"{new_name}.{new_suffix}" if new_suffix else f"{new_name}"
        new_file_directory = self.directory.parent / new_name_spec

        if new_file_directory.exists():
            raise FileExistsError(f"Directory '{new_file_directory}' already exists.")

        # Rename the directory
        shutil.move(str(self.directory), str(new_file_directory))

        # Update the case attributes
        self.name = new_name
        self.suffix = new_suffix
        self.full_name = new_name_spec
        self._directory = new_file_directory
        self._ini_file_path = Path(new_file_directory, f"{new_name}.ini")

    def _config_list(self, group: str, variable: str) -> list[str]:
        """Return an INI value as a list of non-empty strings."""
        value = self.config.get(group, variable, None)
        values = value if isinstance(value, list) else [value]
        return [str(item).strip() for item in values if item not in (None, "")]

    def _config_bool(self, group: str, variable: str, default: bool) -> bool:
        """Return a boolean INI value while handling string representations."""
        value = self.config.get(group, variable, default)
        if isinstance(value, str):
            return value.lower() in {"1", "true", "yes", "on"}
        return bool(value)

    def _prognostic_scalar_names(self) -> set[str]:
        """Return the scalar fields saved by MicroHH for the active setup."""
        scalars = set(self._config_list("fields", "slist"))

        swthermo = str(self.config.get("thermo", "swthermo", "0")).lower()
        thermo_scalars = {
            "dry": {"th"},
            "moist": {"thl", "qt"},
            "buoy": {"b"},
        }
        scalars.update(thermo_scalars.get(swthermo, set()))

        swmicro = str(self.config.get("micro", "swmicro", "0")).lower()
        microphysics_scalars = {
            "2mom_warm": {"qr", "nr"},
            "nsw6": {"qr", "qs", "qg"},
        }
        scalars.update(microphysics_scalars.get(swmicro, set()))

        swdiff_default = self.config.get("grid", "swspatialorder", "2")
        swdiff = str(self.config.get("diff", "swdiff", swdiff_default)).lower()
        if swdiff == "tke2":
            scalars.add("sgstke")

        return scalars

    def _microhh_output_stems(self) -> set[str]:
        """Return time-suffixed output stems for the active MicroHH setup."""
        scalars = self._prognostic_scalar_names()
        stems = {
            "fftwplan",
            "grid",
            "rhoref",
            "thermo_basestate",
            "time",
            "u",
            "v",
            "w",
            *scalars,
        }

        swthermo = str(self.config.get("thermo", "swthermo", "0")).lower()
        if swthermo == "moist":
            stems.update({"thl_bot", "qt_bot"})

        swboundary = str(self.config.get("boundary", "swboundary", "default")).lower()
        if swboundary == "surface":
            stems.update({"dudz_mo", "dvdz_mo"})
            if swthermo != "0":
                stems.add("dbdz_mo")
            if not self._config_bool("boundary", "swconstantz0", True):
                stems.add("obuk")

            default_sbcbot = self.config.get("boundary", "sbcbot", None)
            for scalar in scalars:
                sbcbot = self.config.get(
                    "boundary", f"sbcbot[{scalar}]", default_sbcbot
                )
                if str(sbcbot).lower() == "flux":
                    stems.add(f"{scalar}_gradbot")
        elif swboundary == "surface_bulk":
            stems.update({"dudz_mo", "dvdz_mo", "dbdz_mo"})
        elif swboundary == "surface_lsm":
            stems.update(
                {
                    "dudz_mo",
                    "dvdz_mo",
                    "dbdz_mo",
                    "wl_skin",
                    "thl_bot_veg",
                    "thl_bot_soil",
                    "thl_bot_wet",
                    "qt_bot_veg",
                    "qt_bot_soil",
                    "qt_bot_wet",
                }
            )
            if not self._config_bool("boundary", "swconstantz0", True):
                stems.update({"obuk_veg", "obuk_soil", "obuk_wet"})

        if self._config_bool("dump", "swdump", False):
            stems.update(self._config_list("dump", "dumplist"))

        return stems

    def _is_microhh_output(self, path: Path) -> bool:
        """Return whether *path* is an output created directly by MicroHH."""
        if not path.is_file() and not path.is_symlink():
            return False

        name = path.name
        if name == f"{self.name}.out":
            return True

        case_netcdf_pattern = rf"{re.escape(self.name)}\..+\.\d{{7,}}\.nc"
        if re.fullmatch(case_netcdf_pattern, name):
            return True

        cross_names = (
            self._config_list("cross", "crosslist")
            if self._config_bool("cross", "swcross", False)
            else []
        )
        for cross_name in cross_names:
            cross_pattern = rf"{re.escape(cross_name)}\.(?:xy|xz|yz)\..+\.\d{{7,}}"
            if re.fullmatch(cross_pattern, name):
                return True

        timed_output_match = re.fullmatch(r"(.+)\.(\d{7,})", name)
        if timed_output_match is None:
            return False

        stem, time_str = timed_output_match.groups()
        if stem in self._microhh_output_stems():
            return True

        if stem in {"t_soil", "theta_soil"}:
            swboundary = str(
                self.config.get("boundary", "swboundary", "default")
            ).lower()
            if swboundary != "surface_lsm":
                return False
            return (
                self._config_bool("land_surface", "swhomogeneous", True)
                or int(time_str) > 0
            )

        return False

    def _get_clean_paths(
        self, additional_patterns: Optional[list[str]] = None
    ) -> list[Path]:
        """Collect MicroHH outputs and explicitly requested additional paths."""
        paths = {
            path for path in self.directory.iterdir() if self._is_microhh_output(path)
        }

        if additional_patterns is not None and not isinstance(
            additional_patterns, list
        ):
            raise TypeError("Additional cleanup patterns must be provided as a list.")

        for pattern in additional_patterns or []:
            if not isinstance(pattern, str):
                raise TypeError("Additional cleanup patterns must be strings.")

            pattern_path = Path(pattern)
            if (
                not pattern
                or pattern_path.is_absolute()
                or len(pattern_path.parts) != 1
                or "**" in pattern
                or pattern in {".", ".."}
            ):
                raise ValueError(
                    "Additional cleanup patterns must match names directly inside "
                    "the case directory."
                )

            paths.update(
                path
                for path in self.directory.glob(pattern)
                if path.parent == self.directory
            )

        return sorted(paths, key=lambda path: path.name)

    def clean(self, additional_patterns: Optional[list[str]] = None) -> None:
        """Remove outputs created by MicroHH from this case.

        Args:
            additional_patterns: Additional filenames or glob patterns to remove.
                Patterns are matched directly inside the case directory and may
                explicitly select files or directories that are otherwise preserved.

        The paths selected for deletion are printed before the user is prompted
        for confirmation. Nothing is deleted unless the user enters ``y``.
        """
        paths_to_delete = self._get_clean_paths(additional_patterns)
        if not paths_to_delete:
            print(f"No files to delete in {self.directory}.")
            return

        print(f"The following paths will be deleted from {self.directory}:")
        for path in paths_to_delete:
            suffix = "/" if path.is_dir() and not path.is_symlink() else ""
            print(f"  {path.name}{suffix}")

        with utils.confirm_action("Continue with cleanup?") as confirmed:
            if not confirmed:
                logger.info("Cleanup cancelled for %s.", self.directory)
                return

            for path in paths_to_delete:
                if path.is_symlink() or path.is_file():
                    path.unlink(missing_ok=True)
                elif path.is_dir():
                    shutil.rmtree(path)
            logger.info("Cleaned case at %s.", self.directory)

    def delete(self) -> None:
        """Delete this MicroHH case by removing its directory.

        The user is prompted for confirmation before deletion.

        Raises:
            FileNotFoundError: If the case directory does not exist.
        """
        if not self.directory.exists():
            raise FileNotFoundError(
                f"Case directory '{self.directory}' does not exist."
            )

        with utils.confirm_action(
            f"Are you sure you want to delete the case at {self.directory}?"
        ) as confirmed:
            if confirmed:
                shutil.rmtree(self.directory)
                logger.info("Deleted case at %s.", self.directory)

    def summarise_files(self) -> None:
        """Summarise output files in the case directory."""
        files = [f.name for f in self.directory.iterdir() if f.is_file()]

        # Files written exclusively by `microhh init`. Other restart files written
        # during init can gain later timestamps and remain complete file series.
        initialisation_names = {
            "fftwplan.0000000",
            "grid.0000000",
            "rhoref.0000000",
        }

        # Regex patterns for timestamped files
        # Pattern 1: name.XXXXXXX  (e.g. ch4.0000000, u.0000020)
        # Pattern 2: name.XXX.XXXXXXX (e.g. ch4_path.xy.000.0000000)
        # Special: name.nc (e.g. u.nc, v.nc)
        ts_pattern = re.compile(r"^(.+?)\.(\d{7})$")
        ts2_pattern = re.compile(
            r"^(.+\.\d{3})\.(\d{7})$"
        )  # for ch4_path.xy.NNN.TTTTTTT
        nc_pattern = re.compile(r"^(.+?)\.nc$")

        initialisation_files = []
        timestamped = defaultdict(list)  # base_name -> [timestamps]
        nc_files = []
        other_files = []

        for fname in sorted(files):
            if fname in initialisation_names:
                initialisation_files.append(fname)
                continue

            # Try the two-level timestamp pattern first (more specific)
            m2 = ts2_pattern.match(fname)
            m1 = ts_pattern.match(fname)
            mn = nc_pattern.match(fname)

            if m2:
                timestamped[m2.group(1)].append(int(m2.group(2)))
            elif m1:
                timestamped[m1.group(1)].append(int(m1.group(2)))
            elif mn:
                nc_files.append(fname)
            else:
                other_files.append(fname)

        # --- Print summary ---
        print(f"Case directory: {self.directory}\n")

        if initialisation_files:
            print("=" * 60)
            print("INITIALISATION FILES")
            print("=" * 60)
            for f in initialisation_files:
                print(f"  {f}")
            print()

        print("=" * 60)
        print("TIMESTAMPED FILE SERIES")
        print("=" * 60)
        for base in sorted(timestamped):
            ts = sorted(timestamped[base])
            n = len(ts)
            t0, t1 = ts[0], ts[-1]
            # Detect uniform step
            steps = {ts[i + 1] - ts[i] for i in range(len(ts) - 1)} if n > 1 else {0}
            if len(steps) == 1:
                dt = steps.pop()
                step_str = f"step={dt}"
            else:
                step_str = "irregular steps"
            print(f"  {base:<40s}  {n:>4d} files  [{t0:07d} - {t1:07d}]  {step_str}")

        if nc_files:
            print()
            print("=" * 60)
            print("NETCDF FILES")
            print("=" * 60)
            for f in sorted(nc_files):
                print(f"  {f}")

        log_files = [f for f in other_files if f.endswith((".out", ".sbatch"))]
        other_files = [f for f in other_files if not f.endswith((".out", ".sbatch"))]

        if log_files:
            print()
            print("=" * 60)
            print("LOGS & JOB SCRIPTS")
            print("=" * 60)
            for f in sorted(log_files):
                print(f"  {f}")

        if other_files:
            print()
            print("=" * 60)
            print("OTHER FILES")
            print("=" * 60)
            for f in sorted(other_files):
                print(f"  {f}")

    def copy_files_to_case(self, case: "Case", files: list) -> None:
        """Copy given files from this case to another case.

        Args:
            case: Target case to which the files will be copied.
            files: List of filenames to copy to the target case.
        """
        for file in files:
            src = self.directory / file
            dst = case.directory / file
            if src.exists():
                shutil.copy(str(src), str(dst))
                logger.info("Copied %s to %s", file, case.directory)
            else:
                logger.warning("File %s does not exist.", src)

    def domain(self) -> Domain:
        """Creates a Domain from the Case ini file.

        Returns:
            instance of spacial.Domain
        """
        domain = Domain(
            xsize=self.config.grid.xsize,
            ysize=self.config.grid.ysize,
            itot=self.config.grid.itot,
            jtot=self.config.grid.jtot,
            lon=self.config.grid.lon,
            lat=self.config.grid.lat,
            anchor="center",
        )
        return domain


def compare_ini(case_path1: Union[str, Path], case_path2: Union[str, Path]) -> None:
    """Compare two ``.ini`` files in a Git-diff style.

    Group and variable ordering is ignored. Differences are printed with
    ANSI colours: red for deletions/modifications in *case_path1*, green
    for additions/modifications in *case_path2*.

    Args:
        case_path1: Path to the first case directory (or file).
        case_path2: Path to the second case directory (or file).
    """
    # ANSI Colors
    red = "\033[91m"
    green = "\033[92m"
    reset = "\033[0m"

    case_path1 = Path(case_path1)
    case_path2 = Path(case_path2)

    # get parent folder if a file path is provided
    if case_path1.is_file():
        case_path1 = case_path1.parent
    if case_path2.is_file():
        case_path2 = case_path2.parent

    case1 = read_case(case_path1)
    case2 = read_case(case_path2)

    # 1. Identify all unique groups involved
    self_groups = set(case1.config.groups)
    other_groups = set(case2.config.groups)
    all_groups = self_groups | other_groups

    for group in sorted(all_groups):
        # Case A: Group completely removed in case2
        if group in self_groups and group not in other_groups:
            print(f"{red}[Group: {group}] removed{reset}")
            # Show all variables in this group as deleted
            for var in case1.config.groups[group]:
                val = case1.config[group][var]
                print(f"{red}- {var} = {val}{reset}")
            continue

        # Case B: Group newly added in case2
        if group not in self_groups and group in other_groups:
            print(f"{green}[Group: {group}] added{reset}")
            # Show all variables in this group as added
            for var in case2.config.groups[group]:
                val = case2.config[group][var]
                print(f"{green}+ {var} = {val}{reset}")
            continue

        # Case C: Group exists in both (Check for variable differences)
        self_vars = set(case1.config.groups[group])
        other_vars = set(case2.config.groups[group])
        all_vars = self_vars | other_vars

        # Buffer changes so we only print the group header if there is a difference
        changes = []

        for var in sorted(all_vars):
            # Variable removed
            if var in self_vars and var not in other_vars:
                val = case1.config[group][var]
                changes.append(f"{red}- {var} = {val}{reset}")

            # Variable added
            elif var not in self_vars and var in other_vars:
                val = case2.config[group][var]
                changes.append(f"{green}+ {var} = {val}{reset}")

            # Variable modified
            else:
                val_self = case1.config[group][var]
                val_other = case2.config[group][var]

                # Convert to string for comparison to avoid type mismatches
                if str(val_self) != str(val_other):
                    changes.append(f"{red}- {var} = {val_self}{reset}")
                    changes.append(f"{green}+ {var} = {val_other}{reset}")

        # Only print if changes were found in this group
        if changes:
            print(f"[{group}]")
            for change in changes:
                print(change)


def initialise_case(
    name: str,
    suffix: str,
    scratch_path: Union[str, Path],
    method: Union[InitMethod, str] = InitMethod.INTERACTIVE,
    init_from: Optional[Union[str, Path]] = None,
    overwrite: bool = False,
) -> Case:
    """Initialise a MicroHH case.

    Args:
        name: Name of the case.
            For the 'default' method, the case must be set up in ``microhh/cases``
            with an ``{name}_input.py`` file and a matching ``{name}*.ini`` file.
        suffix: Suffix for the specific model run.
            Combined as ``{name}.{suffix}``.
        scratch_path: Path of personal folder on *scratch*.
        method: Method of how the case should be set up.
        init_from: Path to the case used as a template for the new case.
            Required when *method* is ``'default'`` or ``'rerun'``.
        overwrite: Whether to overwrite an existing case directory.

    Returns:
        The initialised MicroHH case.
    """
    case = Case(
        name=name,
        suffix=suffix,
    )
    case._initialise(
        scratch_path=scratch_path,
        method=method,
        init_from=init_from,
        overwrite=overwrite,
    )
    return case


def read_case(
    case_directory: Union[str, Path], scratch_path: Optional[Union[str, Path]] = None
) -> Case:
    """Read an existing MicroHH case from disk.

    Args:
        case_directory: Directory where the case is stored.
        scratch_path: Optional path to the scratch directory.

    Returns:
        The loaded MicroHH case.

    Raises:
        FileNotFoundError: If the ``.ini`` file does not exist in
            *case_directory*.
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
    case.directory = case_directory
    case.ini_file_path = Path(case.directory, f"{case.name}.ini")
    case.scratch_path = (
        Path(scratch_path, case.full_name) if scratch_path else case.directory
    )
    case.job_id = case.job_id if case.job_id is not None else case._get_job_id()

    if not case.ini_file_path.exists():
        raise FileNotFoundError(
            f".ini file for case '{case.name}' does not exist in {case.directory}."
        )

    case.config = mht.Read_namelist(case.ini_file_path)

    return case


__all__ = [
    "Case",
    "compare_ini",
    "initialise_case",
    "read_case",
]
