"""Input and output helpers for MicroHH case files."""

from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union, cast

import netCDF4
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
        case: Case containing the output directory.
        variable: Variable name to open.
        chunks: Chunking strategy passed to xarray.

    Returns:
        Opened NetCDF dataset.

    Raises:
        FileNotFoundError: If the requested NetCDF file does not exist.
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
    """Read the netCDF file containing the initial conditions.

    Args:
        case: Case for which the initial conditions should be read.

    Returns:
        Initial-condition dataset with all NetCDF groups merged.
    """
    return _combine_nc_groups(case.initial_conditions_file_path)


def _combine_nc_groups(file_path: Union[str, Path]) -> xr.Dataset:
    """Combine all groups in a NetCDF file into one dataset."""
    groups = ["", *list(netCDF4.Dataset(file_path).groups.keys())]
    merged_dataset = xr.Dataset()
    for group in groups:
        ds = xr.open_dataset(file_path, group=group, decode_times=False)
        merged_dataset = xr.merge([merged_dataset, ds], compat="override", join="outer")

    return merged_dataset


def open_statistics(case: "Case") -> xr.Dataset:
    """Read the NetCDF file containing model-run statistics.

    Returns:
        Statistics dataset with all NetCDF groups merged.
    """
    return _combine_nc_groups(case.statistics_file_path)


def open_microhh_out(case: "Case") -> pd.DataFrame:
    """Parse a MicroHH case.out file into a pandas DataFrame.

    Adds:
    - CPUT: cumulative CPU time
    - SDPD: simulated days per day

    Args:
        case: Case containing the output file.

    Returns:
        Parsed output table with derived runtime columns.
    """
    df = pd.read_csv(case.out_file_path, sep=r"\s+")

    # for restarted simulations, the column names are written again in the
    # middle of the file
    bad_mask = pd.isna(pd.to_numeric(df["TIME"], errors="coerce")) & df["TIME"].notna()
    df_clean = df[~bad_mask]

    # ensure all numeric columns are numeric again
    df_clean = cast(pd.DataFrame, df_clean.apply(pd.to_numeric))

    df_clean["CPUT"] = df_clean["CPUDT"].cumsum()

    dtime = df_clean["TIME"].diff()
    df_clean["SDPD"] = dtime / df_clean["CPUDT"]

    df_clean.loc[0, "SDPD"] = 0.0

    return df_clean


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
    """Save a MicroHH ``case_input.nc`` file via microhhpy."""
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


def open_jobreport(path: Union[str, Path]) -> pd.DataFrame:
    """Open a jobreport file as a pandas DataFrame.

    Args:
        path: CSV file or directory containing a single jobreport CSV file.

    Returns:
        Jobreport table with parsed start, end, and runtime columns.

    Raises:
        FileNotFoundError: If no jobreport CSV file is found.
        FileExistsError: If multiple jobreport CSV files are found.
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
    "open_initial_cond",
    "open_jobreport",
    "open_microhh_nc",
    "open_microhh_out",
    "open_statistics",
    "save_case_input",
]
