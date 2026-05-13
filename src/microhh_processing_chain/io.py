import xarray as xr
import netCDF4 as nc4
import numpy as np
import pandas as pd

from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Sequence, Union
from functools import wraps

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


def compare_netcdf_values(
    file_paths: Sequence[Union[str, Path]],
    variables: Optional[Sequence[str]] = None,
    reference: Union[int, str, Path] = 0,
    atol: float = 0.0,
    rtol: float = 0.0,
    join: str = "exact",
) -> pd.DataFrame:
    """Compare variable values between multiple netCDF files.

    Each file is compared against one reference file. By default the first file
    is used as reference and all common numeric data variables are compared.

    Args:
        file_paths: netCDF files to compare.
        variables: Variable names to compare. If omitted, all numeric variables
            present in every file are compared.
        reference: Reference file, either as index in ``file_paths`` or as path.
        atol: Absolute tolerance used to decide whether values deviate.
        rtol: Relative tolerance used to decide whether values deviate.
        join: Coordinate alignment mode passed to ``xarray.align``. Use
            ``"exact"`` to require identical coordinates.

    Returns:
        DataFrame with one row per compared file and variable. Important
        columns are ``deviates``, ``max_abs_deviation``,
        ``mean_abs_deviation``, and ``n_deviating_values``.
    """
    paths = [Path(path) for path in file_paths]
    if len(paths) < 2:
        raise ValueError("At least two netCDF files are required for comparison.")

    missing_paths = [path for path in paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(f"Missing netCDF files: {missing_paths}")

    if isinstance(reference, int):
        reference_path = paths[reference]
    else:
        reference_path = Path(reference)
        if not reference_path.exists():
            raise FileNotFoundError(f"Missing reference netCDF file: {reference_path}")
        if reference_path not in paths:
            paths = [reference_path, *paths]

    datasets = [_open_dataset_no_time_decode_fallback(path) for path in paths]
    try:
        ref_ds = datasets[paths.index(reference_path)]
        compare_vars = _variables_to_compare(datasets, variables)
        rows = []

        for path, ds in zip(paths, datasets):
            if path == reference_path:
                continue

            for variable in compare_vars:
                if variable not in ref_ds or variable not in ds:
                    rows.append(
                        {
                            "reference_file": str(reference_path),
                            "compared_file": str(path),
                            "variable": variable,
                            "deviates": True,
                            "status": "missing_variable",
                            "max_abs_deviation": np.nan,
                            "mean_abs_deviation": np.nan,
                            "n_deviating_values": np.nan,
                            "n_compared_values": np.nan,
                        }
                    )
                    continue

                ref_var, other_var = xr.align(ref_ds[variable], ds[variable], join=join)
                diff = other_var - ref_var
                abs_diff = abs(diff)
                tolerance = atol + rtol * abs(ref_var)
                finite_mask = np.isfinite(ref_var) & np.isfinite(other_var)
                deviating_mask = (abs_diff > tolerance) & finite_mask

                max_abs_deviation = _to_float(abs_diff.where(finite_mask).max())
                mean_abs_deviation = _to_float(abs_diff.where(finite_mask).mean())
                n_deviating_values = int(deviating_mask.sum().compute().item())
                n_compared_values = int(finite_mask.sum().compute().item())

                rows.append(
                    {
                        "reference_file": str(reference_path),
                        "compared_file": str(path),
                        "variable": variable,
                        "deviates": n_deviating_values > 0,
                        "status": "ok",
                        "max_abs_deviation": max_abs_deviation,
                        "mean_abs_deviation": mean_abs_deviation,
                        "n_deviating_values": n_deviating_values,
                        "n_compared_values": n_compared_values,
                    }
                )

        return pd.DataFrame(rows).sort_values(
            ["compared_file", "variable"], ignore_index=True
        )
    finally:
        for ds in datasets:
            ds.close()


def _open_dataset_no_time_decode_fallback(file_path: Path) -> xr.Dataset:
    try:
        return xr.open_dataset(file_path, engine="netcdf4")
    except ValueError:
        return xr.open_dataset(file_path, engine="netcdf4", decode_times=False)


def _variables_to_compare(
    datasets: Sequence[xr.Dataset], variables: Optional[Sequence[str]]
) -> list[str]:
    if variables is not None:
        if isinstance(variables, str):
            return [variables]
        return list(variables)

    common_variables = set(datasets[0].data_vars)
    for ds in datasets[1:]:
        common_variables &= set(ds.data_vars)

    return sorted(
        var
        for var in common_variables
        if all(np.issubdtype(ds[var].dtype, np.number) for ds in datasets)
    )


def _to_float(data: xr.DataArray) -> float:
    value = data.compute().item()
    if value is None:
        return np.nan
    return float(value)


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
    "compare_netcdf_values",
    "open_microhh_out",
    "save_case_input",
]
