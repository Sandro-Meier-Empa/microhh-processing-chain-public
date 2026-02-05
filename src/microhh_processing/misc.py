import subprocess

import xarray as xr
import numpy as np
import netCDF4 as nc4

from pathlib import Path
from typing import Union, Literal, Optional, Tuple


def object_to_list(var) -> list:
    if var is None:
        raise ValueError("Input should not be 'None'")
    if isinstance(var, str):
        # parse comma separated string
        var = [s.strip() for s in var.split(",")]

        # convert to numeric if possible
        # Try converting to numbers
        converted_items = []
        for item in var:
            try:
                # Try integer first
                converted_items.append(int(item))
            except ValueError:
                try:
                    # Try float if int fails
                    converted_items.append(float(item))
                except ValueError:
                    # Keep as string if both fail
                    converted_items.append(item)

        return converted_items

    if isinstance(var, xr.DataArray):
        var = var.values.tolist()
    elif isinstance(var, np.ndarray):
        var = var.tolist()
    elif not isinstance(var, list):
        var = [var]
    return var


def to_cs_list(lst: Union[list, np.ndarray], n: int = 1, ndecimals: int = None) -> str:
    """
    From Python iterable to comma separated string.
    Handles float precision artifacts (e.g. 209.20000000000002 -> 209.2)
    while preserving small numbers (e.g. 0.00000123).

    Args:
        lst: List or array of items to convert to comma-separated string.
        n: Number of times to repeat the list (default is 1).
    """

    # Helper to clean individual items
    def clean_fmt(x):
        if isinstance(x, (float, np.floating)):
            # :.15g keeps up to 15 significant digits.
            if ndecimals is not None:
                return f"{x:.{ndecimals}f}"
            return f"{x:.15g}"
        return str(x)

    # 1. Convert all items to clean strings
    clean_items = [clean_fmt(x) for x in lst]

    # 2. Repeat the list n times (as per your original logic)
    # Using list multiplication is faster than a loop for string joining
    repeated_items = clean_items * n

    # 3. Join once (More efficient than string concatenation += in a loop)
    return ",".join(repeated_items)


def calculate_coriolis_force(lat: float) -> float:
    """Calculate the Coriolis force at a given latitude."""
    fc = 2 * 7.2921159e-5 * np.sin(np.radians(lat))
    return fc


def calculate_richardson_number(
    stats: xr.Dataset,
    threshold: Optional[float] = 1e-10,
) -> xr.DataArray:
    """Calculate the Richardson flux number."""
    denom = stats.u_w * stats.u_grad + stats.v_w * stats.v_grad  # m^2 s^-3
    Rf = stats.b_w / denom
    attrs = {
        "long_name": "Richardson number",
        "units": "1",
        "description": "Ratio of buoyant production to shear production of turbulence kinetic energy",
    }

    da = xr.DataArray(Rf, dims=Rf.dims, coords=Rf.coords, attrs=attrs)
    stats["Ri"] = da.where(np.abs(denom) > threshold)
    return stats


def get_2D_tke_threshold_time(
    stats: xr.Dataset,
    tke_threshold: bool = False,
    tke_der_threshold: bool = False,
    threshold_value=0,
) -> Tuple[int, Union[int, np.datetime64]]:

    if not tke_threshold and not tke_der_threshold:
        raise ValueError("Either 'tke_threshold' or 'tke_der_treshold' must be true.")
    if tke_threshold and tke_der_threshold:
        raise ValueError(
            "Only one of 'tke_threshold' or 'tke_der_treshold' should be true."
        )

    # Calculate the median of tke over the 'z' dimension
    tke_median = stats.tke.median(dim="z")

    if tke_threshold:
        threshold_time = tke_median.where(
            tke_median > threshold_value * tke_median.max()
        ).dropna(dim="time")
    elif tke_der_threshold:
        # Calculate the derivative of tke_median with respect to time
        tke_derivative = tke_median.differentiate(coord="time")

        # Calculate the relative derivative
        tke_median_overall = stats.tke.median(dim=["z", "time"])
        relative_derivative = tke_derivative / tke_median_overall

        # Find the time where the threshold is exceeded for the first time
        threshold_time = relative_derivative.where(
            relative_derivative > threshold_value
        ).dropna(dim="time")
    else:
        raise ValueError

    if threshold_time.size > 0:
        time_value = threshold_time.time[0].item()
        time_ind = stats["time"].where(stats["time"] == time_value).argmax().item()
        return time_ind, time_value
    else:
        raise ValueError(f"No data for TKE threshold = {threshold_value}.")
