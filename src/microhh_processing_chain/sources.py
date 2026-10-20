"""Source-strength and plume-rise utilities for MicroHH cases."""

import logging
from typing import Optional, Union, cast

import numpy as np
import pandas as pd
import ucat
import xarray as xr

import microhh_processing_chain.initialisation as init
import microhh_processing_chain.io as io
from microhh_processing_chain.enums import Stability

logger = logging.getLogger(__name__)


def get_source_strength(
    case: init.Case, to_unit: str, time_s: Optional[float] = None
) -> np.ndarray:
    """Get source strengths converted to desired units.

    Works both for constant and time-dependent source strengths.

    Args:
        case: Case object used to read source settings and initial conditions.
        to_unit: Desired unit for source strength.
        time_s: Time in seconds to select source strength data.
            When omitted, the maximum over all times is used.

    Returns:
        Source strengths in the desired unit.
    """
    swtimedep_strength = case.config["source"]["swtimedep_strength"]

    if swtimedep_strength:
        source_strength_arr = _get_dynamic_source_strength(case, time_s=time_s)
    else:
        source_strength_arr = np.array(case.config["source"]["strength"])

    from_unit_arr, gas_mass_arr = _get_emission_units(case)

    # convert to 1d arrays
    source_strength_arr = np.atleast_1d(source_strength_arr)
    from_unit_arr = np.atleast_1d(from_unit_arr)
    gas_mass_arr = np.atleast_1d(gas_mass_arr)

    source_strength_conv = []
    for source_strength, from_unit, gas_mass in zip(
        source_strength_arr, from_unit_arr, gas_mass_arr
    ):
        s = ucat.convert_mass_per_time_unit(
            source_strength, from_unit=from_unit, to_unit=to_unit, gas_mass=gas_mass
        )
        source_strength_conv.append(s)

    return np.array(source_strength_conv)


def _get_dynamic_source_strength(
    case: init.Case, time_s: Optional[float] = None
) -> np.ndarray:
    """Get time-dependent source strengths from initial conditions.

    Args:
        case: Case object used to read initial conditions.
        time_s: Time in seconds to select source strength data.
            When omitted, the maximum over all times is used.

    Returns:
        Source strengths from the initial-condition file.

    Raises:
        ValueError: If the initial-condition file has no ``/timedep`` group or no
            ``source_strength_*`` variables in it.
    """
    tree = io.open_initial_cond(case)
    if "/timedep" not in tree.groups:
        raise ValueError(
            f"'{case.initial_conditions_file_path}' has no '/timedep' group, which "
            "is required for time-dependent source strengths. Either rebuild the "
            "initial conditions with the source strengths or set "
            "'[source] swtimedep_strength = False'."
        )

    initial_cond = tree["/timedep"].dataset

    # get all data variables with "source_strength_*"
    strength_vars = [
        str(var)
        for var in initial_cond.data_vars
        if str(var).startswith("source_strength_")
    ]

    if len(strength_vars) == 0:
        raise ValueError(
            "No time-dependent source strength data found in initial conditions."
        )

    if time_s is None:
        source_strengths = np.array(
            [initial_cond[var].max(skipna=True).item() for var in strength_vars]
        )
    else:
        source_strengths = np.array(
            [
                initial_cond[var]
                .sel(time_source=time_s, method="nearest")
                .max(skipna=True)
                .item()
                for var in strength_vars
            ]
        )

    return source_strengths


def _get_emission_units(case: init.Case) -> tuple[np.ndarray, np.ndarray]:
    """Return source-strength units and gas masses from source settings."""
    swvmr_list = case.config["source"]["swvmr"]
    sourcelist_list = case.config["source"]["sourcelist"]

    from_unit = np.where(swvmr_list, "kmol/s", "kg/s")
    gas_mass = np.where(swvmr_list, sourcelist_list, "")

    return from_unit, gas_mass


def seconds_from_midnight(
    time: Union[pd.Timestamp, pd.DatetimeIndex],
) -> Union[int, np.ndarray]:
    """Calculate seconds from midnight for a given time.

    Args:
        time: Timestamp or DatetimeIndex for which to calculate seconds from midnight.

    Returns:
        Number of seconds from midnight to the given time (int for single Timestamp,
        array for DatetimeIndex).
    """
    if isinstance(time, pd.Timestamp):
        midnight = pd.to_datetime(time.strftime("%Y-%m-%d 00:00:00"))
        if time.tz is not None:
            midnight = midnight.tz_localize(time.tz)
        delta = time - midnight
        return int(delta.total_seconds())
    else:
        # Handle DatetimeIndex
        return np.array([seconds_from_midnight(ts) for ts in time], dtype=int)


def create_strength_profiles(
    start_times: list[str],
    end_times: list[str],
    strengths: list[float],
    reference_start: pd.Timestamp,
    reference_end: pd.Timestamp,
    from_unit: str,
    to_unit: str,
    gas_mass: Union[float, str],
    time_format: str = "%H:%M:%S",
    spacing: str = "1min",
) -> xr.Dataset:
    """Create strength profiles with 1-second ramp up/down for each time interval.

    Args:
        start_times: Start times for each interval.
        end_times: End times for each interval.
        strengths: Source strength values for each interval.
        reference_start: Start time of the output time series.
        reference_end: End time of the output time series.
        from_unit: Unit of the input strengths.
        to_unit: Desired unit for the output strengths.
        gas_mass: Molecular mass or gas name for unit conversion.
        time_format: Format of the input start and end times.
        spacing: Time spacing for the output profile.

    Returns:
        Dataset with source-by-time strength profiles.

    Raises:
        ValueError: If the reference period or any interval is inverted.
    """
    strength_values = np.array(strengths)
    strengths_converted = np.atleast_1d(
        ucat.convert_mass_per_time_unit(
            strength_values,
            from_unit,
            to_unit,
            gas_mass=gas_mass,
        )
    )

    reference_start = _parse_reference(reference_start, "reference_start")
    reference_end = _parse_reference(reference_end, "reference_end")

    if reference_start > reference_end:
        raise ValueError("reference_end must be after reference_start")

    time_range = pd.date_range(start=reference_start, end=reference_end, freq=spacing)

    results = []

    for start_time, end_time, strength in zip(
        start_times, end_times, strengths_converted
    ):
        # Parse start/end times (and compare as times, not strings)
        start_t = pd.to_datetime(start_time, format=time_format).time()
        end_t = pd.to_datetime(end_time, format=time_format).time()

        if start_t > end_t:
            raise ValueError("end_time must be after start_time")

        time_of_day = np.array([ts.time() for ts in time_range])
        mask = (time_of_day >= start_t) & (time_of_day <= end_t)

        results.append(np.where(mask, strength, 0.0))

    strength_data = np.stack(results, axis=0)

    time_sec = (time_range - reference_start).total_seconds().astype(int)

    ds = xr.Dataset(
        {
            "strength": (["source", "time"], strength_data, {"unit": to_unit}),
            "time_sec": (["time"], time_sec, {"unit": "s"}),
        },
        coords={
            "time": time_range.tz_localize(None),
            "source": np.arange(len(start_times)),
        },
    )

    return ds


def _parse_reference(ref: Union[pd.Timestamp, str], name: str) -> pd.Timestamp:
    """Parse a reference timestamp and normalise it to UTC."""
    # Allow "HH:MM:SS" strings by anchoring to a fixed UTC date.
    if isinstance(ref, str):
        parsed = pd.to_datetime(f"1970-01-01 {ref}", utc=True)
        if pd.isna(parsed):
            raise TypeError(f"{name} must be a valid time string 'HH:MM:SS'")
        return cast(pd.Timestamp, parsed)

    if isinstance(ref, pd.Timestamp):
        if pd.isna(ref):
            raise TypeError(f"{name} must not be NaT")
        return ref.tz_localize("UTC") if ref.tz is None else ref.tz_convert("UTC")

    raise TypeError(f"{name} must be a pd.Timestamp or a string 'HH:MM:SS'")


# Note: Stability enum is imported from enums.py


def briggs_momentum_rise(
    x: Union[float, np.ndarray],
    u: Union[float, np.ndarray],
    d: float,
    vs: float,
    stability: Stability = Stability.NEUTRAL,
) -> Union[float, np.ndarray]:
    """Briggs (1969) momentum-dominated plume rise Δh(x) for neutral conditions.

    Taken from Seinfeld and Pandis (2016), p. 868

    Table form used (neutral):
        near-field: Δh = 1.44 * (d*vs)^(2/3) * x^(1/3) / u^(2/3)
        final:      Δh = 3 * d * vs / u
    with the condition vs/u >= 4.

    Args:
        x: Downwind distance (m).
        u: Wind speed at stack height (m/s).
        d: Stack diameter (m).
        vs: Stack exit velocity (m/s).
        stability: Atmospheric stability regime.
            Only "neutral" is supported in this implementation.

    Returns:
        Plume rise above the source in meters.

    Raises:
        ValueError: If inputs are non-physical or stability is not neutral.
    """
    _validate_input(x, u, d, vs)

    if not isinstance(stability, Stability):
        stability = Stability(stability)

    if stability != Stability.NEUTRAL:
        raise ValueError(
            "This momentum implementation uses the Briggs neutral (1969) form."
        )

    if np.any(vs / u < 4):
        logger.warning(
            "Briggs momentum rise is only valid for vs/u >= 4. "
            "Invalid value will be masked as Nan."
        )
        valid_mask = np.where(vs / u >= 4, True, False)
    else:
        valid_mask = np.ones_like(vs, dtype=bool)

    if x == 0:
        return 0.0

    # Compute near-field and final rise
    dh_near = _generic_plume_rise_form(
        e=1.44 * (d * vs) ** (2.0 / 3.0), x=x, b=1.0 / 3.0, u=u, a=2.0 / 3.0
    )
    dh_final = _generic_plume_rise_form(e=3.0 * d * vs, x=x, b=0, u=u, a=1.0)

    # return the element wise minimum of near-field and final rise
    dh = np.minimum(dh_near, dh_final)

    return np.where(valid_mask, dh, np.nan)


def asme_momentum_rise(
    x: Union[float, np.ndarray],
    u: Union[float, np.ndarray],
    d: Union[float, np.ndarray],
    vs: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """ASME (1973) momentum-dominated plume rise Δh(x) for neutral conditions.

    Taken from Seinfeld and Pandis (2016), p. 868

    Conditions for momentum-dominated rise:
        - vs >= 10 m/s
        - vs > u
        - ΔT < 50 K

    Args:
        x: Downwind distance (m).
        u: Wind speed at stack height (m/s).
        d: Stack diameter (m).
        vs: Stack exit velocity (m/s).

    Returns:
        Plume rise above the source in meters.

    Raises:
        ValueError: If inputs are non-physical.
    """
    _validate_input(x, u, d, vs)

    if x == 0:
        return 0.0

    if np.any(vs < 10) or np.any(vs <= u):
        logger.warning(
            "ASME momentum rise is only valid for vs >= 10 m/s and vs > u. "
            "Invalid value will be masked as Nan."
        )
        valid_mask = np.where((vs >= 10) & (vs > u), True, False)
    else:
        valid_mask = np.ones_like(vs, dtype=bool)

    # Compute final rise
    dh_final = _generic_plume_rise_form(e=(d * vs) ** 1.4, x=x, b=0, u=u, a=1.4)

    return np.where(valid_mask, dh_final, np.nan)


def _generic_plume_rise_form(
    e: Union[float, np.ndarray],
    x: Union[float, np.ndarray],
    b: float,
    u: Union[float, np.ndarray],
    a: float,
) -> Union[float, np.ndarray]:
    """Generic form for plume rise equations: Δh = e * (x^b) * (u^-a).

    Args:
        e: Empirical constant.
        x: Downwind distance (m).
        b: Exponent for downwind distance.
        u: Wind speed at stack height (m/s).
        a: Exponent for wind speed.

    Returns:
        Plume rise above the source in meters.
    """
    return e * (x**b) * (u**-a)


def _validate_input(
    x: Union[float, np.ndarray],
    u: Union[float, np.ndarray],
    d: Union[float, np.ndarray],
    vs: Union[float, np.ndarray],
) -> None:
    """Validate plume-rise inputs for positive physical values."""
    if np.array(x).any() < 0:
        raise ValueError("x must be >= 0 (m).")
    if np.array(u).any() <= 0:
        raise ValueError("u must be > 0 (m/s).")
    if np.array(d).any() <= 0:
        raise ValueError("d must be > 0 (m).")
    if np.array(vs).any() <= 0:
        raise ValueError("vs must be > 0 (m/s).")


__all__ = [
    "Stability",
    "asme_momentum_rise",
    "briggs_momentum_rise",
    "create_strength_profiles",
    "get_source_strength",
    "seconds_from_midnight",
]
