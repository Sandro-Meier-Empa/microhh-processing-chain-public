import subprocess

import xarray as xr
import numpy as np

from pathlib import Path
from typing import Union, Literal, Optional


def open_in_vscode(file_path: Union[str, Path]):
    """Open the bash script in VS Code.

    Args:
        file_path (Path):
            Specify the path of the bash script which should be opened.
    """
    subprocess.run(["code", str(file_path)])


def open_microhh_nc(case, variable: str, chunks: Union[str, dict] = None) -> xr.Dataset:
    """Open a .nc file related to MicroHH output.

    Args:
        file_path (Union[str, Path]):
            Path of the file including file name.
    """
    file_path = Path(case.file_directory, f"{variable.lower()}.nc")

    if not file_path.exists():
        # List all .nc files in the directory
        nc_files = [f.name for f in Path(case.file_directory).glob("*.nc")]
        nc_list = ", ".join(nc_files) if nc_files else "No .nc files found."

        raise FileNotFoundError(
            f"Variable '{variable}' not found in {case.file_directory}. "
            f"Available .nc files: {nc_list}"
        )

    try:
        data = xr.open_dataset(file_path, chunks=chunks)
    except ValueError:
        data = xr.open_dataset(file_path, chunks=chunks, decode_times=False)

    return data
