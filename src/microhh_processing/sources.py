from typing import List, Literal, Union
from collections.abc import Iterable

import logging

import numpy as np
import pandas as pd


import ucat
import xarray as xr
import microhh_processing.initialisation as init
import microhh_processing.io as io

logger = logging.getLogger(__name__)


def get_source_strength(
    case: init.Case, to_unit: str, time_s: int = None
) -> np.ndarray:
    """
    Get source strengths converted to desired units.
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
    """
    Get time-dependent source strengths from initial conditions.

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
):
    """
    Create strength profiles with 1-second ramp up/down for each time interval.

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

    for start_time, end_time, strength in zip(start_times, end_times, strengths_converted):
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
        return pd.Timestamp.combine(pd.Timestamp("1970-01-01").date(), t).tz_localize("UTC")

    if isinstance(ref, pd.Timestamp):
        return ref.tz_convert("UTC")

    raise TypeError(f"{name} must be a pd.Timestamp or a string 'HH:MM:SS'")