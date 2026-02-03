from typing import List, Literal, Union
from collections.abc import Iterable

import logging

import numpy as np
import pandas as pd


import ucat
import microhh_processing.initialisation as init
import microhh_processing.io as io

logger = logging.getLogger(__name__)


def get_source_strength(case: init.Case, to_unit: str) -> np.ndarray:
    swtimedep_strength = case.ini_file["source"]["swtimedep_strength"]

    if swtimedep_strength:
        source_strength_arr = _get_dynamic_source_strength(case)
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


def _get_dynamic_source_strength(case: init.Case) -> np.ndarray:
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

    return np.array(
        [initial_cond[var].max(skipna=True).item() for var in strength_vars]
    )


def _get_emission_units(case: init.Case) -> np.ndarray:
    swvmr_list = case.ini_file["source"]["swvmr"]
    sourcelist_list = case.ini_file["source"]["sourcelist"]

    from_unit = np.where(swvmr_list, "kmol/s", "kg/s")
    gas_mass = np.where(swvmr_list, sourcelist_list, None)

    return from_unit, gas_mass


def seconds_from_reference(start_time: str, reference_time: str) -> int:
    """
    Calculate seconds between start_time and reference_time.
    
    Args:
        start_time: Time string in format "HH:MM:SS"
        reference_time: Reference time string in format "HH:MM:SS"
    
    Returns:
        Number of seconds from reference_time to start_time
    """
    ref = pd.to_datetime(reference_time, format="%H:%M:%S")
    start = pd.to_datetime(start_time, format="%H:%M:%S")

    # Handle case where start_time is on next day
    if start < ref:
        start += pd.Timedelta(days=1)

    delta = start - ref
    return int(delta.total_seconds())

def create_time_profile(start_time: str, end_time: str, spacing: str = "1min") -> np.ndarray:
    base_date = "2024-01-01"
    midnight = pd.to_datetime(f"{base_date} 00:00:00")
    start = pd.to_datetime(f"{base_date} {start_time}")
    end = pd.to_datetime(f"{base_date} {end_time}")

    if end < start:
        raise ValueError(f"End time ({end_time}) must be after start time ({start_time})")

    try:
        time_range = pd.date_range(start=start, end=end, freq=spacing)
    except ValueError as e:
        raise ValueError(
            f"Invalid spacing format '{spacing}'. Use pandas formats like '1s', '5min', '1h'."
        ) from e

    seconds_from_midnight = (time_range - midnight).total_seconds()
    return seconds_from_midnight.to_numpy()


def create_strength_profiles(
    start_times: List[str],
    end_times: List[str],
    strengths: List[float],
    reference_start: str,
    reference_end: str,
    from_unit: str,
    to_unit: str,
    gas_mass: Union[float, str],
    spacing: str = "1min",
):
    """
    Create strength profiles with 1-second ramp up/down for each time interval.

    Args:
        start_times: List of start times in format "HH:MM:SS"
        end_times: List of end times in format "HH:MM:SS"
        strengths: List of strength values for each interval
        reference_start: Reference start time of the output time series in format "HH:MM:SS"
        reference_end: Reference end time of the output time series in format "HH:MM:SS"
        from_unit: Unit of the input strengths (e.g., "kg/s", "kmol/s")
        to_unit: Desired unit for the output strengths (e.g., "kg/s", "kmol/s")
        gas_mass: Molecular mass of the gas (float) or gas name (str) for ucat conversion
        spacing: Time spacing for the output time profile

    Returns:
        List of strength profiles, one for each interval
    """
    strengths = np.array(strengths)
    strengths_converted = ucat.convert_mass_per_time_unit(
        strengths,
        from_unit,
        to_unit,
        gas_mass=gas_mass
    )

    # Calculate total duration in seconds
    ref_start = pd.to_datetime(reference_start, format="%H:%M:%S")
    ref_end = pd.to_datetime(reference_end, format="%H:%M:%S")

    if ref_end < ref_start:
        raise ValueError("reference_end must be after reference_start")

    time_profile = create_time_profile(reference_start, reference_end, spacing=spacing)

    # Create result lists
    results = []

    for start_time, end_time, strength in zip(
        start_times, end_times, strengths_converted
    ):
        if end_time < start_time:
            raise ValueError("end_time must be after start_time")

        start_sec = seconds_from_reference(start_time, "00:00:00")
        end_sec = seconds_from_reference(end_time, "00:00:00")

        result = np.where((time_profile >= start_sec) & (time_profile <= end_sec), strength, 0)
        results.append(result)

    return time_profile, results
