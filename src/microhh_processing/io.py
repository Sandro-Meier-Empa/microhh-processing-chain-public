import subprocess

import xarray as xr
import numpy as np
import netCDF4 as nc4

from pathlib import Path
from typing import Union, Literal, Optional

import microhhpy


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
        case (Case):
            The Case object containing the file directory.
        variable (str):
            The variable name to open (e.g., 'u', 'v', 'w', 'T', etc.).
        chunks (Union[str, dict], optional):
            Chunking strategy for dask. Defaults to None.
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


def open_initial_cond(case) -> xr.Dataset:
    """Read the netCDF file containing the initial conditions
    of a model run (<name>_input.nc).

    Args:
        case (Case):
            Case for which the initial conditions should be read.

    Returns:
        xr.Dataset: netCDF of initial conditions.
    """

    fname = Path(case.file_directory, f"{case.name}_input.nc")

    return _combine_nc_groups(fname)


def _combine_nc_groups(file_path: Union[str, Path]) -> xr.Dataset:
    groups = [""] + list(nc4.Dataset(file_path).groups.keys())
    merged_dataset = xr.Dataset()
    for group in groups:
        ds = xr.open_dataset(file_path, group=group, decode_times=False)
        merged_dataset = xr.merge([merged_dataset, ds], compat="override")

    return merged_dataset


def open_statistics(case) -> xr.Dataset:
    """Read the netCDF file containing the statistics of a model run
    (<name>_statistics.nc).

    Returns:
        xr.Dataset: netCDF of initial conditions.
    """
    file_path = Path(case.file_directory, f"{case.name}.default.0000000.nc")

    return _combine_nc_groups(file_path)


def open_microhh_out(case) -> list:
    path = case.file_directory / f"{case.name}.out"
    out = microhhpy.io.read_case_out(path)

    return out[0]
