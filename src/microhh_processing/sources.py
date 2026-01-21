from datetime import datetime, timedelta
from typing import List, Literal, Union

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
    ref = datetime.strptime(reference_time, "%H:%M:%S")
    start = datetime.strptime(start_time, "%H:%M:%S")
    
    # Handle case where start_time is on next day
    if start < ref:
        start += timedelta(days=1)
    
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
    time_obj = datetime.strptime(time_str, "%H:%M:%S")

    # Get minutes and seconds
    total_seconds = time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second

    # Round to nearest 15 minutes (900 seconds)
    rounded_seconds = round(total_seconds / 900) * 900

    # Convert back to time
    hours = rounded_seconds // 3600
    minutes = (rounded_seconds % 3600) // 60
    seconds = 0

    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

from collections.abc import Iterable


def round_to_quarter_hour(time_input):
    """
    Apply _round_to_quarter_hour to a single string or an iterable of strings.
    """
    if isinstance(time_input, str):
        return _round_to_quarter_hour(time_input)

    if isinstance(time_input, Iterable):
        return [_round_to_quarter_hour(t) for t in time_input]

    raise TypeError("Input must be a time string or an iterable of time strings")


# def create_time_profile(start_time: str, end_time: str, spacing: Literal["s", "min", "h"] = "min") -> np.ndarray:
#     """
#     Create a time profile array with specified spacing between start_time and end_time.
#     """
#     start = datetime.strptime(start_time, "%H:%M:%S")
#     end = datetime.strptime(end_time, "%H:%M:%S")

#     if end < start:
#         raise ValueError("end_time must be after start_time")

#     total_seconds = int((end - start).total_seconds()) + 1

#     if spacing == "s":
#         return np.arange(total_seconds, step=1)
#     elif spacing == "min":
#         return np.arange(0, total_seconds, step=60)
#     elif spacing == "h":
#         return np.arange(0, total_seconds, step=3600)
#     else:
#         raise ValueError("Invalid spacing value")


def create_time_profile(start_time: str, end_time: str, spacing: str = "1min") -> np.ndarray:
    base_date = "2024-01-01"
    midnight = pd.to_datetime(f"{base_date} 00:00:00")
    start = pd.to_datetime(f"{base_date} {start_time}")
    end = pd.to_datetime(f"{base_date} {end_time}")

    if end < start:
        raise ValueError(f"End time ({end_time}) must be after start time ({start_time})")

    try:
        time_range = pd.date_range(start=midnight, end=end, freq=spacing)
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
    spacing: Literal["s", "min", "h"] = "min"
):
    """
    Create strength profiles with 1-second ramp up/down for each time interval.
    
    Args:
        start_times: List of start times in format "HH:MM:SS"
        end_times: List of end times in format "HH:MM:SS"
        strengths: List of strength values for each interval
        reference_start: Reference start time in format "HH:MM:SS"
        reference_end: Reference end time in format "HH:MM:SS"
    
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
    ref_start = datetime.strptime(reference_start, "%H:%M:%S")
    ref_end = datetime.strptime(reference_end, "%H:%M:%S")

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

        start_sec = seconds_from_reference(start_time, reference_start)
        end_sec = seconds_from_reference(end_time, reference_start)

        result = np.where((time_profile >= start_sec) & (time_profile <= end_sec), strength, 0)
        results.append(result)

    return time_profile, results
