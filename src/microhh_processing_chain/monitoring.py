"""MicroHH simulation monitoring helpers."""

import concurrent.futures
import re
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from microhh_processing_chain import io
from microhh_processing_chain.constants import PrintColor

if TYPE_CHECKING:
    from microhh_processing_chain.initialisation import Case


def add_progress_to_df(df: pd.DataFrame) -> pd.DataFrame:
    """Add MicroHH simulation progress to a DataFrame containing queued jobs.

    A job's ``WORK_DIR`` is either a case directory (single-case job) or the
    folder holding the cases of a multi-GPU job (see
    :func:`~microhh_processing_chain.preprocessing.create_run_4x_script`). The
    cases of a multi-GPU job are the subdirectories that contain the job's run
    log ``run_microhh_<JOBID>.out``.

    Args:
        df: DataFrame containing jobs in the queue, with the columns ``JOBID``
            and ``WORK_DIR``.

    Returns:
        Jobs in the queue with two added columns: ``PROGRESS [%]``, the
        progress of the slowest case of each job (``NaN`` for jobs without
        readable MicroHH cases), and ``CASE_PROGRESS [%]``, the progress of
        each case of a multi-case job.
    """
    progress = []
    case_progress = []

    for _, row in df.iterrows():
        # ``squeue -o %all`` has two JOBID columns (equal except for array jobs),
        # so the row lookup can return a Series instead of a scalar
        job_id = row["JOBID"]
        if isinstance(job_id, pd.Series):
            job_id = job_id.iloc[0]
        try:
            cases = _job_cases(Path(str(row["WORK_DIR"])), str(job_id))
            percentages = [_case_progress_percentage(case) for case in cases]
        except Exception:
            percentages = []

        progress.append(round(min(percentages), 1) if percentages else np.nan)
        case_progress.append(
            " / ".join(f"{p:.1f}" for p in percentages) if len(percentages) > 1 else ""
        )

    df["PROGRESS [%]"] = progress
    df["CASE_PROGRESS [%]"] = case_progress

    return df


def _job_cases(job_dir: Path, job_id: str) -> list["Case"]:
    """Return the MicroHH cases run by a Slurm job.

    Args:
        job_dir: Working directory of the job.
        job_id: Slurm job ID.

    Returns:
        The case in *job_dir* for a single-case job, otherwise the cases in the
        subdirectories of *job_dir* that contain ``run_microhh_<job_id>.out``,
        sorted by directory name. Empty if there are neither.
    """
    import microhh_processing_chain.initialisation as init

    try:
        return [init.read_case(job_dir)]
    except FileNotFoundError:
        pass

    case_dirs = sorted(
        log.parent for log in job_dir.glob(f"*/run_microhh_{job_id}.out")
    )
    return [init.read_case(case_dir) for case_dir in case_dirs]


def _case_progress_percentage(case: "Case") -> float:
    """Return the progress of a case in percent of its end time.

    Uses the model time in the MicroHH ``.out`` file, which is updated every
    ``outputiter`` steps, and falls back to the latest restart file while the
    ``.out`` file has no time step yet.
    """
    try:
        current, end = get_simulation_progress_from_out(case)
    except (IndexError, KeyError, ValueError):
        current, end = 0.0, 0
    if current == 0:
        current, end = get_simulation_progress_from_files(case)
    return float(current) / end * 100


def print_simulation_progress(case: "Case") -> None:
    """Print the progress of a MicroHH simulation."""
    current_timestep, max_timestep = get_simulation_progress_from_out(case)
    progress_percentage = current_timestep / max_timestep * 100

    if progress_percentage == 0.0:
        color = PrintColor.RED
    elif progress_percentage == 100.0:
        color = PrintColor.GREEN
    else:
        color = PrintColor.YELLOW

    end_color = PrintColor.ENDC

    print(
        f"{color.value}Simulation progress of {case.full_name}: "
        f"{progress_percentage:.1f}% ({current_timestep}s out of {max_timestep}s)"
        f"{end_color.value}"
    )


def get_simulation_progress_from_files(case: "Case") -> tuple[int, int]:
    """Get current simulation progress from binary MicroHH grid files."""
    current_timestep = 0

    def process_file(file: Path) -> int:
        if file.is_file() and file.name.startswith("time."):
            try:
                return int(file.name[5:])
            except ValueError:
                return 0
        return 0

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_file, file) for file in case.directory.iterdir()
        ]
        for future in concurrent.futures.as_completed(futures):
            timestep = future.result()
            current_timestep = max(current_timestep, timestep)

    max_timestep = int(case.config["time"]["endtime"])

    return current_timestep, max_timestep


def get_simulation_progress_from_out(case: "Case") -> tuple[float, int]:
    """Get current simulation progress from the MicroHH ``.out`` file."""
    try:
        mhh_out = io.open_microhh_out(case)
        current_timestep = mhh_out["TIME"].iloc[-1]
    except FileNotFoundError:
        current_timestep = 0.0

    max_timestep = int(case.config["time"]["endtime"])

    return current_timestep, max_timestep


def extract_job_id(case: "Case") -> int | None:
    """Extract the Slurm job ID from a MicroHH run output file name."""
    files = case.directory.glob("run_microhh_*.out")
    for file in files:
        match = re.search(r"run_microhh_(\d+)\.out", file.name)
        if match:
            return int(match.group(1))
    return None


__all__ = [
    "add_progress_to_df",
    "extract_job_id",
    "get_simulation_progress_from_files",
    "get_simulation_progress_from_out",
    "print_simulation_progress",
]
