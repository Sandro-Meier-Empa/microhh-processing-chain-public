from typing import List, Union

import logging

import numpy as np
import pandas as pd


import ucat
import xarray as xr
import microhh_processing.initialisation as init
import microhh_processing.io as io
from microhh_processing.enums import Stability

logger = logging.getLogger(__name__)


def get_source_strength(
    case: init.Case, to_unit: str, time_s: int = None
) -> np.ndarray:
    """Get source strengths converted to desired units.

    Works both for constant and time-dependent source strengths.

    Args:
        case: Case object needed to get initial conditions
        to_unit: Desired unit for source strength (e.g., "kg/s", "kmol/s")
        time_s: Time in seconds to select source strength data. If None, use max over all times.
    Returns:
        np.ndarray: Array of source strengths in desired units
    """
    swtimedep_strength = case.ini_file["source"]["swtimedep_strength"]

    if swtimedep_strength:
        source_strength_arr = _get_dynamic_source_strength(case, time_s=time_s)
    else:
        source_strength_arr = np.array(case.ini_file["source"]["strength"])

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


def _get_dynamic_source_strength(case: init.Case, time_s: int = None) -> np.ndarray:
    """Get time-dependent source strengths from initial conditions.

    Args:
        case: Case object needed to get initial conditions
        time_s: Time in seconds to select source strength data. If None, use max over all times.

    Returns:
        np.ndarray: Array of source strengths
    """
    # Load time-dependent strength data
    initial_cond = io.open_initial_cond(case)

    # get all data variables with "source_strength_*"
    strength_vars = [
        var for var in initial_cond.data_vars if var.startswith("source_strength_")
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


def _get_emission_units(case: init.Case) -> np.ndarray:
    swvmr_list = case.ini_file["source"]["swvmr"]
    sourcelist_list = case.ini_file["source"]["sourcelist"]

    from_unit = np.where(swvmr_list, "kmol/s", "kg/s")
    gas_mass = np.where(swvmr_list, sourcelist_list, None)

    return from_unit, gas_mass


def seconds_from_midnight(
    time: Union[pd.Timestamp, pd.DatetimeIndex],
) -> Union[int, pd.Series]:
    """Calculate seconds from midnight for a given time.

    Args:
        time: Timestamp or DatetimeIndex for which to calculate seconds from midnight.

    Returns:
        Number of seconds from midnight to the given time (int for single Timestamp,
        pd.Series for DatetimeIndex).
    """
    if isinstance(time, pd.Timestamp):
        midnight = pd.to_datetime(time.strftime("%Y-%m-%d 00:00:00"))
        if time.tz is not None:
            midnight = midnight.tz_localize(time.tz)
        delta = time - midnight
        return int(delta.total_seconds())
    else:
        # Handle DatetimeIndex
        midnight = time.normalize()
        delta = time - midnight
        return delta.total_seconds().astype(int)


def create_strength_profiles(
    start_times: List[str],
    end_times: List[str],
    strengths: List[float],
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
        start_times: List of start times in format "HH:MM:SS"
        end_times: List of end times in format "HH:MM:SS"
        strengths: List of strength values for each interval
        reference_start: Reference start time of the output time series
        reference_end: Reference end time of the output time series
        from_unit: Unit of the input strengths (e.g., "kg/s", "kmol/s")
        to_unit: Desired unit for the output strengths (e.g., "kg/s", "kmol/s")
        gas_mass: Molecular mass of the gas (float) or gas name (str) for ucat conversion
        spacing: Time spacing for the output time profile

    Returns:
        xr.Dataset with strength profiles, dimensions (time, source)
    """

    strengths = np.array(strengths)
    strengths_converted = ucat.convert_mass_per_time_unit(
        strengths,
        from_unit,
        to_unit,
        gas_mass=gas_mass,
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

        time_of_day = time_range.time
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
    # Allow "HH:MM:SS" strings by anchoring to a fixed UTC date.
    if isinstance(ref, str):
        t = pd.to_datetime(ref, format="%H:%M:%S").time()
        return pd.Timestamp.combine(pd.Timestamp("1970-01-01").date(), t).tz_localize(
            "UTC"
        )

    if isinstance(ref, pd.Timestamp):
        return ref.tz_convert("UTC")

    raise TypeError(f"{name} must be a pd.Timestamp or a string 'HH:MM:SS'")


# Note: Stability enum is imported from enums.py


def briggs_momentum_rise(
    x: Union[float, np.ndarray],
    u: Union[float, np.ndarray],
    d: Union[float, np.ndarray],
    Vs: Union[float, np.ndarray],
    stability: Stability = Stability.NEUTRAL,
) -> Union[float, np.ndarray]:
    """Briggs (1969) momentum-dominated plume rise Δh(x) for neutral conditions.

    Taken from Seinfeld and Pandis (2016), p. 868

    Table form used (neutral):
        near-field: Δh = 1.44 * (d*Vs)^(2/3) * x^(1/3) / u^(2/3)
        final:      Δh = 3 * d * Vs / u
    with the condition Vs/u >= 4.

    Args:
        x: Downwind distance (m).
        u: Wind speed at stack height (m/s).
        d: Stack diameter (m).
        Vs: Stack exit velocity (m/s).
        stability: Atmospheric stability regime.
            Only "neutral" is supported in this implementation.

    Returns:
        float: Plume rise above source Δh (m).
    """
    _validate_input(x, u, d, Vs)

    if not isinstance(stability, Stability):
        stability = Stability(stability)

    if stability != Stability.NEUTRAL:
        raise ValueError(
            "This momentum implementation uses the Briggs neutral (1969) form."
        )

    if np.any(Vs / u < 4):
        logger.warning(
            "Briggs momentum rise is only valid for Vs/u >= 4. Invalid value will be masked as Nan."
        )
        valid_mask = np.where(Vs / u >= 4, True, False)
    else:
        valid_mask = np.ones_like(Vs, dtype=bool)

    if x == 0:
        return 0.0

    # Compute near-field and final rise
    dh_near = _generic_plume_rise_form(
        E=1.44 * (d * Vs) ** (2.0 / 3.0), x=x, b=1.0 / 3.0, u=u, a=2.0 / 3.0
    )
    dh_final = _generic_plume_rise_form(E=3.0 * d * Vs, x=x, b=0, u=u, a=1.0)

    # return the element wise minimum of near-field and final rise
    dh = np.minimum(dh_near, dh_final)

    return np.where(valid_mask, dh, np.nan)


def asme_momentum_rise(
    x: Union[float, np.ndarray],
    u: Union[float, np.ndarray],
    d: Union[float, np.ndarray],
    Vs: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """ASME (1973) momentum-dominated plume rise Δh(x) for neutral conditions.

    Taken from Seinfeld and Pandis (2016), p. 868

    Conditions for momentum-dominated rise:
        - Vs >= 10 m/s
        - Vs > u
        - ΔT < 50 K

    Args:
        x: Downwind distance (m).
        u: Wind speed at stack height (m/s).
        d: Stack diameter (m).
        Vs: Stack exit velocity (m/s).

    Returns:
        float: Plume rise above source Δh (m).
    """

    _validate_input(x, u, d, Vs)

    if x == 0:
        return 0.0

    if np.any(Vs < 10) or np.any(Vs <= u):
        logger.warning(
            "ASME momentum rise is only valid for Vs >= 10 m/s and Vs > u. Invalid value will be masked as Nan."
        )
        valid_mask = np.where((Vs >= 10) & (Vs > u), True, False)
    else:
        valid_mask = np.ones_like(Vs, dtype=bool)

    # Compute final rise
    dh_final = _generic_plume_rise_form(E=(d * Vs) ** 1.4, x=x, b=0, u=u, a=1.4)

    return np.where(valid_mask, dh_final, np.nan)


def _generic_plume_rise_form(
    E: float,
    x: Union[float, np.ndarray],
    b: float,
    u: Union[float, np.ndarray],
    a: float,
) -> Union[float, np.ndarray]:
    """Generic form for plume rise equations: Δh = E * (x^b) * (u^-a).

    Args:
        E: Empirical constant.
        x: Downwind distance (m).
        b: Exponent for downwind distance.
        u: Wind speed at stack height (m/s).
        a: Exponent for wind speed.

    Returns:
        float or np.ndarray: Plume rise above source Δh (m).
    """
    return E * (x**b) * (u**-a)


def _validate_input(
    x: Union[float, np.ndarray],
    u: Union[float, np.ndarray],
    d: Union[float, np.ndarray],
    Vs: Union[float, np.ndarray],
) -> None:
    if np.array(x).any() < 0:
        raise ValueError("x must be >= 0 (m).")
    if np.array(u).any() <= 0:
        raise ValueError("u must be > 0 (m/s).")
    if np.array(d).any() <= 0:
        raise ValueError("d must be > 0 (m).")
    if np.array(Vs).any() <= 0:
        raise ValueError("Vs must be > 0 (m/s).")


__all__ = [
    "get_source_strength",
    "seconds_from_midnight",
    "create_strength_profiles",
    "Stability",
    "briggs_momentum_rise",
    "asme_momentum_rise",
]
