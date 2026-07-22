"""MicroHH simulation monitoring helpers."""

import concurrent.futures
import re
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from microhh_processing_chain import io
from microhh_processing_chain.constants import PrintColor

if TYPE_CHECKING:
    from microhh_processing_chain.initialisation import Case


def add_progress_to_df(df: pd.DataFrame) -> pd.DataFrame:
    """Add MicroHH simulation progress to a DataFrame containing queued jobs.

    Args:
        df:
            DataFrame containing jobs in the queue.

    Returns:
        Jobs in the queue with simulation progress.
    """
    import microhh_processing_chain.initialisation as init

    progress_percentages = []

    for _, row in df.iterrows():
        job_dir = str(row["WORK_DIR"])
        try:
            case = init.read_case(job_dir)
            current_timestep, max_timestep = get_simulation_progress_from_files(case)
            progress_percentage = current_timestep / max_timestep * 100
            progress_percentages.append(int(progress_percentage))
        except Exception:
            progress_percentages.append("NA")

    df["PROGRESS (%)"] = progress_percentages

    return df


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
