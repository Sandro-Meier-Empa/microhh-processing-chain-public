"""Export MicroHH plume columns to the SPECTRA ``PlumeColumn`` contract.

One exported file holds one case, one tracer and one source as a time series
of vertical columns on the native MicroHH grid. The attribute names follow
``spectra.contracts.plume_column`` (contract ``spectra.PlumeColumn`` 1.0)
without importing ``spectra``; the dataset is validated on the SPECTRA side.
"""

import datetime as dt
import json
import logging
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, Optional, Union, cast

import numpy as np
import pandas as pd
import ucat
import xarray as xr

import microhh_processing_chain
import microhh_processing_chain.initialisation as init
import microhh_processing_chain.io as io
import microhh_processing_chain.sources as sources
from microhh_processing_chain.constants import GAS_MOLAR_MASS
from microhh_processing_chain.plume import compute_total_slant_column
from microhh_processing_chain.postprocessing import (
    PostProcessing,
    convert_microhh_timestep_to_date,
)

logger = logging.getLogger(__name__)

PLUME_COLUMN_CONTRACT = "spectra.PlumeColumn"
PLUME_COLUMN_VERSION = "1.0"
COLUMN_UNITS = "kg m-2"
EMISSION_RATE_UNITS = "kg s-1"
WIND_UNITS = "m s-1"
LES_MODEL = "MicroHH"
PRODUCER = "microhh_processing_chain.export"

ColumnSource = Literal["path_xy", "3d"]
TimeSelection = Union[slice, Sequence[float], np.ndarray]

# Snapshots requested as model seconds must match an output time this closely.
_TIME_TOLERANCE_S = 0.5
_SPACING_RTOL = 1e-6
_LOG_HEAD_BYTES = 65536
_GIT_HASH_PATTERN = re.compile(r"Microhh git-hash:\s*(\S+)", re.IGNORECASE)


def export_plume_column(
    case: init.Case,
    tracer: str = "ch4",
    *,
    gas: Optional[str] = None,
    source: ColumnSource = "path_xy",
    source_index: int = 0,
    times: Optional[TimeSelection] = None,
    start_time: Optional[Union[str, pd.Timestamp]] = None,
    wind_height: Optional[float] = None,
    include_wind: bool = True,
    altitudes_m: Optional[Sequence[float]] = None,
    output_path: Optional[Union[str, Path, Literal[False]]] = None,
    overwrite: bool = False,
    chunks: Optional[Union[str, dict[str, int]]] = None,
) -> xr.Dataset:
    """Export the vertical columns of one tracer as a ``PlumeColumn`` dataset.

    The default ``source="path_xy"`` reads the online MicroHH cross-section
    ``{tracer}_path.xy.nc`` (``sum(rho_ref * q * dz)`` in kg m-2) unchanged.
    ``source="3d"`` integrates the 3-D tracer field instead: the mixing ratio
    is converted with :meth:`PostProcessing.convert_to_concentration`, the
    column with :func:`~microhh_processing_chain.plume.compute_total_slant_column`
    at zero zenith angle, and partial columns below ``altitudes_m`` by a
    trapezoidal integral truncated at each altitude. The 3-D concentration
    itself is not stored.

    Args:
        case: Case whose output directory holds the MicroHH files.
        tracer: Name of the MicroHH scalar, e.g. ``"ch4"`` or ``"ch4_valve_1m"``.
        gas: Gas name recorded in the output, e.g. ``"CH4"``. Inferred from
            the tracer root when it is a key of ``GAS_MOLAR_MASS``.
        source: ``"path_xy"`` for the online column cross-section or ``"3d"``
            for the vertically integrated 3-D tracer field.
        source_index: Index of the source in the ``[source]`` arrays of the
            ``.ini`` file that emits ``tracer``.
        times: Snapshots to export as model seconds: a slice selects a range,
            a sequence selects individual output times.
        start_time: UTC time of model second zero. Defaults to
            ``[time] datetime_utc`` of the ``.ini`` file; required when the
            case does not define it.
        wind_height: Height in metres at which the horizontal-mean wind is
            taken from the statistics file. Defaults to the source height.
        include_wind: Whether to add ``wind_u``/``wind_v`` from the statistics
            file.
        altitudes_m: Strictly increasing altitudes for ``column_below_altitude``;
            only used with ``source="3d"``.
        output_path: NetCDF file to write. Defaults to
            ``{case.directory}/{case_id}__{tracer}.plumecolumn.nc``; ``False``
            skips writing.
        overwrite: Whether an existing ``output_path`` may be replaced.
        chunks: Chunking passed to xarray when opening the MicroHH files.

    Returns:
        Dataset following the ``spectra.PlumeColumn`` contract.

    Raises:
        ValueError: If the source emits a volume mixing ratio, the gas cannot
            be inferred, the start time is unknown, the tracer is not emitted
            by exactly the selected source, or requested snapshots are missing.
        FileExistsError: If ``output_path`` exists and ``overwrite`` is false.
    """
    case_id = case.full_name
    gas_name = _resolve_gas(tracer, gas)
    source_settings = _source_settings(case, tracer, source_index)
    start_timestamp = _resolve_start_time(case, start_time)

    if source == "path_xy":
        dataset, source_files = _columns_from_path_xy(
            case, tracer, gas_name, times, chunks
        )
    elif source == "3d":
        dataset, source_files = _columns_from_3d(
            case, tracer, gas_name, times, altitudes_m, chunks
        )
    else:
        raise ValueError(f"source must be 'path_xy' or '3d', got {source!r}.")

    model_seconds = np.asarray(dataset["time"].values, dtype=float)
    dataset = dataset.assign_coords(
        simulation_time=(
            "time",
            model_seconds,
            {"units": "s", "long_name": "Model time since the LES start"},
        )
    )
    dataset = convert_microhh_timestep_to_date(dataset, start_timestamp)

    timedep = bool(case.config["source"]["swtimedep_strength"])
    rates = _emission_rates(case, model_seconds, source_index, timedep)
    dataset["emission_rate"] = (
        "time",
        rates,
        {"units": EMISSION_RATE_UNITS, "long_name": f"{gas_name} emission rate"},
    )
    if timedep:
        source_files.append(case.initial_conditions_file_path.name)

    if include_wind:
        height = (
            float(source_settings["source_z"])
            if wind_height is None
            else float(wind_height)
        )
        wind_u, wind_v = _mean_wind(case, model_seconds, height)
        dataset["wind_u"] = ("time", wind_u.values, wind_u.attrs)
        dataset["wind_v"] = ("time", wind_v.values, wind_v.attrs)
        source_files.append(case.statistics_file_path.name)

    dx, dy = _grid_spacing(case, dataset)
    dataset.attrs = _plume_column_attrs(
        case=case,
        case_id=case_id,
        tracer=tracer,
        gas=gas_name,
        source_index=source_index,
        source_settings=source_settings,
        dx=dx,
        dy=dy,
        start_timestamp=start_timestamp,
        timedep=timedep,
        column_source=source,
        source_files=[case.ini_file_path.name, *source_files],
    )

    if output_path is not False:
        path = (
            case.directory / f"{case_id}__{tracer}.plumecolumn.nc"
            if output_path is None
            else Path(output_path)
        )
        _write_netcdf(dataset, path, overwrite=overwrite)

    return dataset


def _resolve_gas(tracer: str, gas: Optional[str]) -> str:
    """Return the gas name, inferring it from the tracer root when possible."""
    if gas is not None:
        return gas
    for candidate in (tracer.lower(), tracer.lower().split("_")[0]):
        if candidate in GAS_MOLAR_MASS:
            return candidate.upper()
    raise ValueError(
        f"Cannot infer the gas from tracer {tracer!r}; pass gas=... explicitly. "
        f"Known gases: {sorted(GAS_MOLAR_MASS)}."
    )


def _source_settings(
    case: init.Case, tracer: str, source_index: int
) -> dict[str, float]:
    """Return position and extent of the selected source from the ``.ini`` file."""
    config = case.config["source"]
    sourcelist = [str(name) for name in np.atleast_1d(config["sourcelist"])]
    matching = [i for i, name in enumerate(sourcelist) if name == tracer]
    if source_index not in matching:
        raise ValueError(
            f"Source {source_index} does not emit tracer {tracer!r}; sources "
            f"emitting it: {matching} (sourcelist={sourcelist})."
        )
    if len(matching) > 1:
        raise ValueError(
            f"Tracer {tracer!r} is emitted by sources {matching}; the column "
            "mixes their plumes, but a PlumeColumn file holds one source."
        )
    if bool(np.atleast_1d(config["swvmr"])[source_index]):
        raise ValueError(
            f"Source {source_index} emits a volume mixing ratio (swvmr=1), so "
            f"'{tracer}_path' is not a mass column in {COLUMN_UNITS}."
        )

    def value(key: str) -> float:
        return float(np.atleast_1d(config[key])[source_index])

    sigma_x = value("sigma_x")
    sigma_y = value("sigma_y")
    return {
        "source_x": value("source_x0"),
        "source_y": value("source_y0"),
        "source_z": value("source_z0"),
        "source_sigma_x": sigma_x,
        "source_sigma_y": sigma_y,
        "source_sigma_z": value("sigma_z"),
        # 4-sigma rule, as in PostProcessing.create_sources_dataset.
        "source_diameter": 4.0 * max(sigma_x, sigma_y),
    }


def _resolve_start_time(
    case: init.Case, start_time: Optional[Union[str, pd.Timestamp]]
) -> pd.Timestamp:
    """Return model second zero as a timezone-naive UTC timestamp."""
    if start_time is None:
        start_time = case.config.get("time", "datetime_utc")
    if start_time is None:
        raise ValueError(
            f"Case {case.full_name!r} has no '[time] datetime_utc' in its .ini "
            "file; pass start_time=... (UTC time of model second zero)."
        )
    timestamp = pd.Timestamp(start_time)
    if pd.isna(timestamp):
        raise ValueError(f"Could not parse start_time {start_time!r}.")
    timestamp = cast(pd.Timestamp, timestamp)
    if timestamp.tz is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp


def _select_times(dataset: xr.Dataset, times: Optional[TimeSelection]) -> xr.Dataset:
    """Select snapshots by model seconds, exactly for sequences."""
    if times is None:
        selected = dataset
    elif isinstance(times, slice):
        selected = dataset.sel(time=times)
    else:
        requested = np.atleast_1d(np.asarray(times, dtype=float))
        if np.unique(requested).size != requested.size:
            raise ValueError("times must not contain duplicates.")
        selected = dataset.sel(time=requested, method="nearest")
        found = np.asarray(selected["time"].values, dtype=float)
        missing = np.abs(found - requested) > _TIME_TOLERANCE_S
        if missing.any():
            raise ValueError(
                f"Snapshots {requested[missing].tolist()} s are not in the output; "
                f"available times: {np.asarray(dataset['time'].values).tolist()}."
            )
    if selected.sizes["time"] == 0:
        raise ValueError(f"No snapshots selected with times={times!r}.")
    return selected


def _columns_from_path_xy(
    case: init.Case,
    tracer: str,
    gas: str,
    times: Optional[TimeSelection],
    chunks: Optional[Union[str, dict[str, int]]],
) -> tuple[xr.Dataset, list[str]]:
    """Read the online column cross-section ``{tracer}_path.xy.nc``."""
    variable = f"{tracer}_path"
    raw = io.open_microhh_nc(case, f"{variable}.xy", chunks=chunks)
    if variable not in raw:
        raise ValueError(
            f"'{variable}.xy.nc' does not contain {variable!r}; found "
            f"{list(raw.data_vars)}."
        )
    raw = _select_times(raw, times)
    dataset = raw[[variable]].rename({variable: "column"})
    dataset = _float64_axes(dataset)
    column = dataset["column"].transpose("time", "y", "x")
    if column.dtype != np.float32:
        column = column.astype("float32")
    column.attrs = _column_attrs(
        gas,
        "MicroHH online cross-section sum(rho_ref * q * dz) over the full depth",
    )
    dataset["column"] = column
    return dataset, [f"{variable}.xy.nc"]


def _columns_from_3d(
    case: init.Case,
    tracer: str,
    gas: str,
    times: Optional[TimeSelection],
    altitudes_m: Optional[Sequence[float]],
    chunks: Optional[Union[str, dict[str, int]]],
) -> tuple[xr.Dataset, list[str]]:
    """Integrate the 3-D tracer field vertically."""
    post = PostProcessing(case)
    gas_variable = gas.lower()
    data = post.open_3d_prognostic_variable(tracer, chunks=chunks)
    data = _select_times(data, times)
    if tracer != gas_variable:
        # ucat looks the molar mass up by the variable name.
        data = data.rename({tracer: gas_variable})
    data[gas_variable].attrs.setdefault("units", "kg kg-1")
    mixing_ratio = data[gas_variable].copy()

    data = post.convert_to_concentration(data, gas=gas_variable, to_unit="kg m-3")
    concentration = data[gas_variable]
    # Air density consistent with the ucat conversion above: 1 kg/kg -> rho_air.
    # ucat accepts xarray pressure and temperature fields although its
    # annotation only names scalars.
    air_density = cast(
        xr.DataArray,
        ucat.convert_points(
            xr.ones_like(mixing_ratio),
            from_unit="kg kg-1",
            to_unit="kg m-3",
            molar_mass=gas_variable,
            p=data["phydro"],  # pyright: ignore[reportArgumentType]
            T=data["ta"],  # pyright: ignore[reportArgumentType]
        ),
    )
    columns = compute_total_slant_column(
        mixing_ratio, air_density, sza=0.0, saa=0.0, dim_names=("x", "y", "z")
    )
    column = columns["microhh_vertical_column"].transpose("time", "y", "x")
    column = column.astype("float32")
    column.attrs = _column_attrs(
        gas, "Vertical integral of the 3-D tracer mass density (zero zenith angle)"
    )

    dataset = xr.Dataset({"column": column})
    dataset = dataset.assign_coords(
        time=data["time"].values, y=data["y"].values, x=data["x"].values
    )
    dataset = _float64_axes(dataset)
    if altitudes_m is not None:
        altitudes = np.asarray(altitudes_m, dtype=float)
        if (
            altitudes.ndim != 1
            or altitudes.size == 0
            or np.any(np.diff(altitudes) <= 0)
        ):
            raise ValueError("altitudes_m must be a strictly increasing sequence.")
        partial = _columns_below_altitudes(concentration, altitudes)
        partial = partial.transpose("time", "altitude", "y", "x").astype("float32")
        partial.attrs = {
            "units": COLUMN_UNITS,
            "long_name": f"{gas} column below altitude",
            "description": (
                "Trapezoidal integral of the mass density from the lowest "
                "level to the altitude (linear interpolation at the top)."
            ),
        }
        dataset["column_below_altitude"] = partial.assign_coords(
            altitude=("altitude", altitudes, {"units": "m", "long_name": "Altitude"})
        )
    return dataset, [f"{tracer}.nc", case.statistics_file_path.name]


def _columns_below_altitudes(
    concentration: xr.DataArray, altitudes: np.ndarray
) -> xr.DataArray:
    """Integrate a mass density from the lowest level to each altitude."""
    z = np.asarray(concentration["z"].values, dtype=float)
    partials: list[xr.DataArray] = []
    for altitude in altitudes:
        if altitude <= z[0]:
            partial = xr.zeros_like(concentration.isel(z=0, drop=True))
        elif altitude >= z[-1]:
            partial = concentration.integrate("z")
        else:
            below = concentration.sel(z=slice(None, altitude))
            top = concentration.interp(z=altitude).expand_dims(z=[altitude])
            partial = xr.concat([below, top], dim="z").integrate("z")
        partials.append(partial)
    return xr.concat(partials, dim="altitude")


def _float64_axes(dataset: xr.Dataset) -> xr.Dataset:
    """Store the model-second and metre axes as float64."""
    axes = {}
    for name in ("time", "y", "x"):
        attrs = {"units": "m", "long_name": f"{name} of the cell centre"}
        values = np.asarray(dataset[name].values, dtype=float)
        axes[name] = (name, values, {} if name == "time" else attrs)
    return dataset.assign_coords(axes)


def _column_attrs(gas: str, description: str) -> dict[str, str]:
    return {
        "units": COLUMN_UNITS,
        "long_name": f"{gas} vertical column",
        "column_convention": "vertical",
        "description": description,
    }


def _emission_rates(
    case: init.Case, model_seconds: np.ndarray, source_index: int, timedep: bool
) -> np.ndarray:
    """Return the emission rate of the source in kg s-1 at each snapshot."""
    if timedep:
        rates = [
            float(
                sources.get_source_strength(case, to_unit="kg/s", time_s=float(t))[
                    source_index
                ]
            )
            for t in model_seconds
        ]
        return np.asarray(rates, dtype=float)
    rate = float(sources.get_source_strength(case, to_unit="kg/s")[source_index])
    return np.full(model_seconds.shape, rate, dtype=float)


def _mean_wind(
    case: init.Case, model_seconds: np.ndarray, height: float
) -> tuple[xr.DataArray, xr.DataArray]:
    """Return the horizontal-mean wind at ``height`` for each snapshot.

    The ``/default/u`` and ``/default/v`` profiles of the statistics file are
    interpolated linearly in height and the nearest statistics sample within
    one sampling interval is used for each snapshot.
    """
    if not case.statistics_file_path.exists():
        raise FileNotFoundError(
            f"Statistics file {case.statistics_file_path} is required for the "
            "mean wind; pass include_wind=False to skip it."
        )
    profiles = io.open_statistics(case)["/default"].dataset
    z = np.asarray(profiles["z"].values, dtype=float)
    if not z.min() <= height <= z.max():
        raise ValueError(
            f"wind_height={height} m is outside the statistics levels "
            f"[{z.min()}, {z.max()}] m."
        )
    stats_time = np.asarray(profiles["time"].values, dtype=float)
    interval = float(np.median(np.diff(stats_time))) if stats_time.size > 1 else 0.0

    components = []
    for name in ("u", "v"):
        profile = profiles[name].assign_coords(time=stats_time)
        at_height = profile.interp(z=height).drop_vars("z")
        at_times = at_height.reindex(
            time=model_seconds, method="nearest", tolerance=interval
        )
        if bool(at_times.isnull().any()):
            missing = model_seconds[np.asarray(at_times.isnull().values)]
            raise ValueError(
                f"No statistics sample within {interval} s of the snapshots at "
                f"{missing.tolist()} s."
            )
        at_times.attrs = {
            "units": WIND_UNITS,
            "long_name": f"Horizontal-mean {name} wind component at {height:g} m",
            "height": float(height),
            "height_units": "m",
            "method": (
                "linear interpolation of /default/"
                f"{name} in z; nearest statistics sample in time"
            ),
        }
        components.append(at_times)
    return components[0], components[1]


def _grid_spacing(case: init.Case, dataset: xr.Dataset) -> tuple[float, float]:
    """Return ``(dx, dy)`` from the ``.ini`` grid, checked against the axes."""
    grid = case.config["grid"]
    dx = float(grid["xsize"]) / int(grid["itot"])
    dy = float(grid["ysize"]) / int(grid["jtot"])
    for name, step in (("x", dx), ("y", dy)):
        values = np.asarray(dataset[name].values, dtype=float)
        if values.size > 1 and not np.allclose(
            np.diff(values), step, rtol=_SPACING_RTOL
        ):
            raise ValueError(
                f"The '{name}' axis spacing disagrees with the .ini grid ({step} m)."
            )
    return dx, dy


def _read_les_model_version(case: init.Case) -> str:
    """Return the MicroHH git hash printed in the run log, or ``"unknown"``."""
    logs = [*sorted(case.directory.glob("run_microhh*.out")), case.out_file_path]
    for log in logs:
        try:
            with log.open(errors="replace") as handle:
                head = handle.read(_LOG_HEAD_BYTES)
        except OSError:
            continue
        match = _GIT_HASH_PATTERN.search(head)
        if match:
            return match.group(1)
    return "unknown"


def _plume_column_attrs(
    *,
    case: init.Case,
    case_id: str,
    tracer: str,
    gas: str,
    source_index: int,
    source_settings: dict[str, float],
    dx: float,
    dy: float,
    start_timestamp: pd.Timestamp,
    timedep: bool,
    column_source: str,
    source_files: list[str],
) -> dict[str, Union[str, int, float]]:
    """Assemble the global attributes of the ``spectra.PlumeColumn`` contract."""
    created = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    attrs: dict[str, Union[str, int, float]] = {
        "contract": PLUME_COLUMN_CONTRACT,
        "contract_version": PLUME_COLUMN_VERSION,
        "gas": gas,
        "les_model": LES_MODEL,
        "les_model_version": _read_les_model_version(case),
        "case_id": case_id,
        "tracer": tracer,
        "source_id": f"{case_id}:{tracer}",
        "source_index": int(source_index),
        **source_settings,
        "emission_rate_units": EMISSION_RATE_UNITS,
        # NetCDF attributes cannot hold booleans; 1/0 are used instead.
        "emission_rate_is_constant": int(not timedep),
        "passive_tracer": 1,
        "dx": dx,
        "dy": dy,
        "reference_time": start_timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "column_convention": "vertical",
        "column_source": column_source,
        "grid_frame": "model",
        "producer": PRODUCER,
        "producer_version": microhh_processing_chain.__version__,
        "created_utc": created,
        "source_directory": str(case.directory),
        "source_files": json.dumps(source_files),
    }
    return attrs


def _write_netcdf(dataset: xr.Dataset, path: Path, *, overwrite: bool) -> None:
    """Write the dataset with compressed float32 columns."""
    if path.exists():
        if not overwrite:
            raise FileExistsError(
                f"{path} already exists. Use overwrite=True to replace it."
            )
        path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)
    ny, nx = dataset.sizes["y"], dataset.sizes["x"]
    encoding: dict[str, dict] = {
        "column": {
            "dtype": "float32",
            "zlib": True,
            "complevel": 4,
            "chunksizes": (1, ny, nx),
        },
        "time": {
            "units": "seconds since 1970-01-01 00:00:00",
            "calendar": "proleptic_gregorian",
            "dtype": "float64",
        },
    }
    if "column_below_altitude" in dataset:
        encoding["column_below_altitude"] = {
            "dtype": "float32",
            "zlib": True,
            "complevel": 4,
            "chunksizes": (1, 1, ny, nx),
        }
    logger.info("Writing plume column to %s", path)
    dataset.to_netcdf(path, encoding=encoding)


__all__ = [
    "COLUMN_UNITS",
    "EMISSION_RATE_UNITS",
    "PLUME_COLUMN_CONTRACT",
    "PLUME_COLUMN_VERSION",
    "export_plume_column",
]
