import xarray as xr
import netCDF4 as nc4

from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union
from functools import wraps

import microhhpy  # pyright: ignore[reportMissingImports]

if TYPE_CHECKING:
    from microhh_processing.initialisation import Case


def open_microhh_nc(
    case: "Case",
    variable: str,
    chunks: Optional[Union[str, dict[str, int]]] = None,
) -> xr.Dataset:
    """Open a .nc file related to MicroHH output.

    Args:
        case (Case):
            The Case object containing the file directory.
        variable (str):
            The variable name to open (e.g., 'u', 'v', 'w', 'T', etc.).
        chunks (Union[str, dict], optional):
            Chunking strategy for dask. Defaults to None.
    """
    file_path = Path(case.file_directory, f"{variable}.nc")

    if not file_path.exists():
        # List all .nc files in the directory
        nc_files = [f.name for f in Path(case.file_directory).glob("*.nc")]
        nc_list = ", ".join(nc_files) if nc_files else "No .nc files found."

        raise FileNotFoundError(
            f"Variable '{variable}' not found in {case.file_directory}. "
            f"Available .nc files: {nc_list}"
        )

    try:
        data = xr.open_dataset(file_path, engine="netcdf4", chunks=chunks)
    except ValueError:
        data = xr.open_dataset(
            file_path, engine="netcdf4", chunks=chunks, decode_times=False
        )

    return data


def open_initial_cond(case: "Case") -> xr.Dataset:
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


def open_statistics(case: "Case") -> xr.Dataset:
    """Read the netCDF file containing the statistics of a model run
    (<name>_statistics.nc).

    Returns:
        xr.Dataset: netCDF of initial conditions.
    """
    file_path = Path(case.file_directory, f"{case.name}.default.0000000.nc")

    return _combine_nc_groups(file_path)


@wraps(microhhpy.io.read_case_out)
def open_microhh_out(case: "Case") -> list[Any]:
    path = case.file_directory / f"{case.name}.out"
    out = microhhpy.io.read_case_out(path)

    return out[0]


@wraps(microhhpy.io.save_case_input)
def save_case_input(
    case_name: str,
    output_dir: Union[str, Path],
    init_profiles: Optional[dict[str, Any]],
    tdep_surface: Optional[dict[str, Any]] = None,
    tdep_ls: Optional[dict[str, Any]] = None,
    tdep_source: Optional[dict[str, Any]] = None,
    tdep_chem: Optional[dict[str, Any]] = None,
    tdep_aerosol: Optional[dict[str, Any]] = None,
    tdep_radiation: Optional[dict[str, Any]] = None,
    radiation: Optional[dict[str, Any]] = None,
    soil: Optional[dict[str, Any]] = None,
    source: Optional[dict[str, Any]] = None,
    trajectories: Optional[dict[str, Any]] = None,
) -> None:

    microhhpy.io.save_case_input(
        case_name=case_name,
        init_profiles=init_profiles,
        tdep_surface=tdep_surface,
        tdep_ls=tdep_ls,
        tdep_source=tdep_source,
        tdep_chem=tdep_chem,
        tdep_aerosol=tdep_aerosol,
        tdep_radiation=tdep_radiation,
        radiation=radiation,
        soil=soil,
        source=source,
        trajectories=trajectories,
        output_dir=output_dir,
    )


__all__ = [
    "open_microhh_nc",
    "open_initial_cond",
    "open_statistics",
    "open_microhh_out",
    "save_case_input",
]
