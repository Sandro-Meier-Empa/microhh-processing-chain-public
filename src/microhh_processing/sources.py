from typing import List, Literal, Union
from collections.abc import Iterable

import numpy as np
import pandas as pd
import ucat

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

def _round_to_quarter_hour(time_str: str) -> str:
    """
    Round a timestamp to the nearest quarter hour.
    
    Args:
        time_str: Time string in format "HH:MM:SS"
    
    Returns:
        Rounded time string in format "HH:MM:SS"
    """
    time_obj = pd.to_datetime(time_str, format="%H:%M:%S")

    # Get minutes and seconds
    total_seconds = time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second

    # Round to nearest 15 minutes (900 seconds)
    rounded_seconds = round(total_seconds / 900) * 900

    # Convert back to time
    hours = rounded_seconds // 3600
    minutes = (rounded_seconds % 3600) // 60
    seconds = 0

    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def round_to_quarter_hour(time_input):
    """
    Apply _round_to_quarter_hour to a single string or an iterable of strings.
    """
    if isinstance(time_input, str):
        return _round_to_quarter_hour(time_input)

    if isinstance(time_input, Iterable):
        return [_round_to_quarter_hour(t) for t in time_input]

    raise TypeError("Input must be a time string or an iterable of time strings")


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
