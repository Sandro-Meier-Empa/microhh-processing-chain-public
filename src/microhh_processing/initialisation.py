import json
import os
import re
import shutil
import subprocess
import logging

from pathlib import Path
from typing import Union, Optional

import pandas as pd
import numpy as np

import microhh_py.microhh_tools as mht
import ls2d

from microhh_processing.enums import InitMethod, FileType
from microhh_processing import cscs, utils

logger = logging.getLogger(__name__)


class Case:
    def __init__(
        self,
        name: str,
        suffix: str,
    ) -> None:
        """Define a case for MicroHH.

        Can be used for initialising the model, pre-processing,
        post-processing, and visualisation.

        Args:
            name: Name of the case. Must be set up in ``microhh/cases``.
            suffix: Suffix for the specific model run.
                Combined as ``{name}.{suffix}``.
        """
        self.name = name
        self.suffix = suffix
        self.name_spec = f"{name}.{suffix}" if suffix else f"{name}"
        self.job_id = None

    @property
    def run_state(self) -> str:
        """Get run state of the case based on SLURM sacct information."""
        if self.job_id is None:
            return "NOT SUBMITTED"

        sacct_info = cscs.get_sacct_info(self.job_id)

        # filter sacct info for the specific job name
        try:
            sacct_info = sacct_info[sacct_info["JobName"] == "run_microhh"]
            return sacct_info["State"].values[0]
        except (KeyError, IndexError):
            return "UNKNOWN"

    # add method when printing case
    def __repr__(self) -> str:
        # format ini file content as string
        # read ini file and format content
        ini_content = ""
        if not hasattr(self, "ini_path"):
            pass
        elif self.ini_path.exists():
            with open(self.ini_path, "r") as f:
                ini_content = f.read()

        case_str = (
            f"Case(\n"
            f"  name='{self.name}',\n"
            f"  suffix='{self.suffix}',\n"
            f"  path='{self.file_directory}'\n"
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
        self.scratch_path = Path(scratch_path, self.name_spec)
        self.file_directory = self.scratch_path
        self.ini_path = Path(self.scratch_path, f"{self.name}.ini")

        self._handle_method(method, init_from)
        self._handle_scratch_dir(overwrite)
        self._handle_running_method(init_from)

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
            raise ValueError(
                "When using 'default' or 'rerun' method, please provide "
                "the path of the directory containing `.ini` and initial condition files."
            )

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
            logger.info(f"Creating {self.scratch_path}")
            os.makedirs(self.scratch_path)
        elif overwrite:
            with utils.confirm_action(
                f"Are you sure you want to overwrite {self.scratch_path}?"
            ) as confirmed:
                if confirmed:
                    shutil.rmtree(self.scratch_path)
                    logger.info(f"Overwriting {self.scratch_path}")
                    os.makedirs(self.scratch_path)

    def _handle_running_method(self, init_from: Union[str, Path]) -> None:
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
            case_path = Path(init_from)
            self._check_case_setup(case_path)
            self._create_initial(case_path)
            self._copy_ini(case_path)
            self._copy_initial(case_path)
        elif self.method == InitMethod.INTERACTIVE.value:
            self.ini_path.touch()
        elif self.method == InitMethod.RERUN.value:
            case_path = Path(init_from)
            self._copy_ini(case_path)
            self._copy_initial(case_path)
        else:
            raise ValueError(
                f"{self.method} is not available. Use 'default', 'interactive' or 'rerun'."
            )
        self.ini_file = mht.Read_namelist(self.ini_path)

    def _get_job_id(self) -> Optional[int]:
        """Look for the job ID in the ``run_microhh*.out`` file.

        Returns:
            The SLURM job ID extracted from the output filename,
            or ``None`` if no output file is found.
        """
        out_files = list(Path(self.file_directory).glob("run_microhh*.out"))

        if not out_files:
            return None

        for file in out_files:
            file_name = file.name
            job_id = int(re.search(r"\d+", file_name).group())

        return job_id

    def _check_case_setup(self, case_path: Union[str, Path]) -> None:
        """Verify that the case directory exists and contains required files.

        Args:
            case_path: Path to the case directory to validate.

        Raises:
            FileNotFoundError: If the case directory does not exist or
                required files are missing.
        """
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
                f"Expected one file with pattern '{pattern}' in '{case_path}', found {len(matched_files)}."
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
        """Copy the initial conditions (``*_input.nc``) file from *case_path* to scratch.

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
                if item.name not in files_to_keep:
                    if item.is_file():
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
            logger.info(f"{py_script_path} not found.")

    def overwrite_ini(self, settings: dict = {}, overwrite: bool = True) -> None:
        """Overwrite the ``.ini`` file of the case with the given settings.

        Clears all existing groups before applying *settings*.

        Args:
            settings: Nested dictionary of ``{group: {variable: value}}``
                pairs to write.
            overwrite: Whether to allow overwriting the existing file.
        """
        self.ini_file.groups = {}

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
                    self.ini_file.remove_value(group=group, variable=variable)
                else:
                    self.ini_file.set_value(group=group, variable=variable, value=value)

        self.ini_file.save(self.ini_path, allow_overwrite=overwrite)
        self.ini_file = mht.Read_namelist(self.ini_path)

    def open_mhh_in_vscode(self, filetype: Union[FileType, str]) -> None:
        """Open the specified file in VS Code.

        Args:
            filetype: File type to open. Can be a :class:`FileType` enum
                member or one of ``'ini'``, ``'out'``, ``'sbash'``,
                ``'sbash_out'``.

        Raises:
            FileNotFoundError: If no matching file is found (for
                ``'sbash_out'``).
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
        elif filetype_val == FileType.SBASH.value:
            filename = "run_microhh.sbash"
        elif filetype_val == FileType.SBASH_OUT.value:
            # Use wildcard matching
            matches = list(Path(self.file_directory).glob("run_microhh*.out"))
            if not matches:
                raise FileNotFoundError("No sbash_out files found.")
            filename = max(matches, key=lambda f: f.stat().st_mtime)
        else:
            raise ValueError(
                f"{filetype} is not available. Use 'ini', 'out', 'sbash' or 'sbash_out'."
            )

        file_path = self.file_directory / filename

        if file_path.exists():
            utils.open_in_vscode(file_path)
        else:
            logger.info(f"{file_path} does not exist.")

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
        itot = self.ini_file["grid"]["itot"]
        jtot = self.ini_file["grid"]["jtot"]
        ktot = self.ini_file["grid"]["ktot"]

        xsize = self.ini_file["grid"]["xsize"]
        ysize = self.ini_file["grid"]["ysize"]
        zsize = self.ini_file["grid"]["zsize"]

        # get saccing info
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
            json_file_path = self.file_directory / f"{self.name}_benchmark.json"
            with open(json_file_path, "w") as f:
                json.dump(benchmark, f, indent=4)

        return benchmark

    def load_benchmark(self) -> dict:
        """Load benchmark information from a JSON file in the case directory.

        Returns:
            Dictionary containing the benchmark information.

        Raises:
            FileNotFoundError: If the benchmark JSON file does not exist.
        """
        json_file_path = self.file_directory / f"{self.name}_benchmark.json"

        if not json_file_path.exists():
            raise FileNotFoundError(
                f"Benchmark file '{json_file_path}' does not exist."
            )

        with open(json_file_path, "r") as f:
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
        out_file_path = self.file_directory / f"{self.name}.out"

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
        new_file_directory = self.file_directory.parent / new_name_spec

        if new_file_directory.exists():
            raise FileExistsError(f"Directory '{new_file_directory}' already exists.")

        # Rename the directory
        shutil.move(str(self.file_directory), str(new_file_directory))

        # Update the case attributes
        self.name = new_name
        self.suffix = new_suffix
        self.name_spec = new_name_spec
        self.file_directory = new_file_directory
        self.ini_path = Path(new_file_directory, f"{new_name}.ini")

    def clean(self) -> None:
        """Clean this MicroHH case by removing all files and directories.

        The following files are preserved:

        - ``{name}.ini``
        - ``{name}_input.nc``
        - ``microhh`` executable

        The user is prompted for confirmation before deletion.
        """
        files_to_keep = [
            f"{self.name}.ini",
            f"{self.name}_input.nc",
            "microhh",
        ]
        with utils.confirm_action(
            f"Are you sure you want to clean the case at {self.file_directory}?"
        ) as confirmed:
            if confirmed:
                for item in self.file_directory.iterdir():
                    if item.name not in files_to_keep:
                        if item.is_file():
                            item.unlink()
                        elif item.is_dir():
                            shutil.rmtree(item)
                logger.info(f"Cleaned case at {self.file_directory}.")

    def delete(self) -> None:
        """Delete this MicroHH case by removing its directory.

        The user is prompted for confirmation before deletion.

        Raises:
            FileNotFoundError: If the case directory does not exist.
        """
        if not self.file_directory.exists():
            raise FileNotFoundError(
                f"Case directory '{self.file_directory}' does not exist."
            )

        with utils.confirm_action(
            f"Are you sure you want to delete the case at {self.file_directory}?"
        ) as confirmed:
            if confirmed:
                shutil.rmtree(self.file_directory)
                logger.info(f"Deleted case at {self.file_directory}.")


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
    RED = "\033[91m"
    GREEN = "\033[92m"
    RESET = "\033[0m"

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
    self_groups = set(case1.ini_file.groups)
    other_groups = set(case2.ini_file.groups)
    all_groups = self_groups | other_groups

    for group in sorted(all_groups):
        # Case A: Group completely removed in case2
        if group in self_groups and group not in other_groups:
            print(f"{RED}[Group: {group}] removed{RESET}")
            # Show all variables in this group as deleted
            for var in case1.ini_file.groups[group]:
                val = case1.ini_file[group][var]
                print(f"{RED}- {var} = {val}{RESET}")
            continue

        # Case B: Group newly added in case2
        if group not in self_groups and group in other_groups:
            print(f"{GREEN}[Group: {group}] added{RESET}")
            # Show all variables in this group as added
            for var in case2.ini_file.groups[group]:
                val = case2.ini_file[group][var]
                print(f"{GREEN}+ {var} = {val}{RESET}")
            continue

        # Case C: Group exists in both (Check for variable differences)
        self_vars = set(case1.ini_file.groups[group])
        other_vars = set(case2.ini_file.groups[group])
        all_vars = self_vars | other_vars

        # Buffer changes so we only print the group header if there is an actual difference
        changes = []

        for var in sorted(all_vars):
            # Variable removed
            if var in self_vars and var not in other_vars:
                val = case1.ini_file[group][var]
                changes.append(f"{RED}- {var} = {val}{RESET}")

            # Variable added
            elif var not in self_vars and var in other_vars:
                val = case2.ini_file[group][var]
                changes.append(f"{GREEN}+ {var} = {val}{RESET}")

            # Variable modified
            else:
                val_self = case1.ini_file[group][var]
                val_other = case2.ini_file[group][var]

                # Convert to string for comparison to avoid type mismatches
                if str(val_self) != str(val_other):
                    changes.append(f"{RED}- {var} = {val_self}{RESET}")
                    changes.append(f"{GREEN}+ {var} = {val_other}{RESET}")

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
    case.file_directory = case_directory
    case.ini_path = Path(case.file_directory, f"{case.name}.ini")
    case.scratch_path = Path(scratch_path, case.name_spec) if scratch_path else None
    case.job_id = case.job_id if case.job_id is not None else case._get_job_id()

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


__all__ = [
    "Case",
    "compare_ini",
    "initialise_case",
    "read_case",
    "interpolate_to_model_profile",
]
