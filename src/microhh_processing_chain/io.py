from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

import netCDF4 as nc4
import pandas as pd
import xarray as xr

import microhhpy  # pyright: ignore[reportMissingImports]

if TYPE_CHECKING:
    from microhh_processing_chain.initialisation import Case


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
    file_path = Path(case.directory, f"{variable}.nc")

    if not file_path.exists():
        # List all .nc files in the directory
        nc_files = [f.name for f in Path(case.directory).glob("*.nc")]
        nc_list = ", ".join(nc_files) if nc_files else "No .nc files found."

        raise FileNotFoundError(
            f"Variable '{variable}' not found in {case.directory}. "
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
    of a model run ({case.name}_input.nc).

    Args:
        case (Case):
            Case for which the initial conditions should be read.

    Returns:
        xr.Dataset: netCDF of initial conditions.
    """

    return _combine_nc_groups(case.initial_conditions_file_path)


def _combine_nc_groups(file_path: Union[str, Path]) -> xr.Dataset:
    groups = [""] + list(nc4.Dataset(file_path).groups.keys())
    merged_dataset = xr.Dataset()
    for group in groups:
        ds = xr.open_dataset(file_path, group=group, decode_times=False)
        merged_dataset = xr.merge([merged_dataset, ds], compat="override", join="outer")

    return merged_dataset


def open_statistics(case: "Case") -> xr.Dataset:
    """Read the netCDF file containing the statistics of a model run
    (<name>_statistics.nc).

    Returns:
        xr.Dataset: netCDF of initial conditions.
    """
    return _combine_nc_groups(case.statistics_file_path)


def open_microhh_out(case: "Case") -> pd.DataFrame:
    """
    Parse a MicroHH case.out file into a pandas DataFrame.

    Adds:
    - CPUT: cumulative CPU time
    - SDPD: simulated days per day

    Args:
        case (Case): The Case object containing the file directory.

    Returns:
        pd.DataFrame: DataFrame containing the parsed case.out data.
    """
    df = pd.read_csv(case.out_file_path, sep=r"\s+")

    df["CPUT"] = df["CPUDT"].cumsum()

    dtime = df["TIME"].diff()
    df["SDPD"] = dtime / df["CPUDT"]

    df.loc[0, "SDPD"] = 0.0

    return df


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
    overwrite: Optional[bool] = False,
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
        overwrite=overwrite,
    )


def open_jobreport(path: Union[str, Path]) -> pd.DataFrame:
    """Open a jobreport file as a pandas DataFrame.

    Args:
        path (Union[str, Path]): Path where the jobreport file is located.

    Returns:
        pd.DataFrame: DataFrame containing the jobreport data.
    """
    path = Path(path)

    if path.is_file() and path.suffix == ".csv":
        csv_file = path
    else:
        # get the csv file in the directory
        csv_files = list(path.glob("jobreport*/*/*.csv"))

        if not csv_files:
            raise FileNotFoundError(f"No jobreport .csv file found in {path}")

        if len(csv_files) > 1:
            raise FileExistsError(f"Multiple .csv files found in {path}.")

        csv_file = csv_files[0]

    df = pd.read_csv(csv_file)
    df["startTime"] = pd.to_datetime(df["startTime"], unit="us")
    df["endTime"] = pd.to_datetime(df["endTime"], unit="us")
    df["runTime"] = df["endTime"] - df["startTime"]

    return df


__all__ = [
    "open_microhh_nc",
    "open_initial_cond",
    "open_statistics",
    "open_microhh_out",
    "save_case_input",
    "open_jobreport",
]
