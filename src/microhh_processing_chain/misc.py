"""Miscellaneous utility functions."""

from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap
from netCDF4 import Dataset


def make_white_inferno(
    n: int = 256, white_fraction: float = 0.25, inferno_start: float = 0.10
) -> LinearSegmentedColormap:
    """Create an inferno-like colormap with a smooth transition from white.

    Parameters
    ----------
    n : int
        Number of colors in the colormap.
    white_fraction : float
        Fraction of the colormap used for the white-to-inferno transition.
        Larger values make the transition softer.
    inferno_start : float
        Where to start sampling inferno. Higher values skip the very dark colors.
    """
    inferno = plt.get_cmap("inferno_r")

    n_white = int(n * white_fraction)
    n_inferno = n - n_white

    # First inferno color used after the white transition
    target = inferno(inferno_start)

    # Smooth interpolation from white to the first inferno color
    white = np.array([1, 1, 1, 1])
    ramp = np.linspace(0, 1, n_white)[:, None]
    transition = white * (1 - ramp) + np.array(target) * ramp

    # Then continue through inferno
    inferno_colors = inferno(np.linspace(inferno_start, 1, n_inferno))

    colors = np.vstack([transition, inferno_colors])

    return LinearSegmentedColormap.from_list("white_inferno_soft", colors)


def make_white_plasma(
    n: int = 256, white_fraction: float = 0.25, plasma_start: float = 0.10
) -> LinearSegmentedColormap:
    """Create an inferno-like colormap with a smooth transition from white.

    Parameters
    ----------
    n : int
        Number of colors in the colormap.
    white_fraction : float
        Fraction of the colormap used for the white-to-inferno transition.
        Larger values make the transition softer.
    inferno_start : float
        Where to start sampling inferno. Higher values skip the very dark colors.
    """
    inferno = plt.get_cmap("plasma_r")

    n_white = int(n * white_fraction)
    n_inferno = n - n_white

    # First inferno color used after the white transition
    target = inferno(plasma_start)

    # Smooth interpolation from white to the first inferno color
    white = np.array([1, 1, 1, 1])
    ramp = np.linspace(0, 1, n_white)[:, None]
    transition = white * (1 - ramp) + np.array(target) * ramp

    # Then continue through inferno
    inferno_colors = inferno(np.linspace(plasma_start, 1, n_inferno))

    colors = np.vstack([transition, inferno_colors])

    return LinearSegmentedColormap.from_list("white_plasma_soft", colors)


def potential_temperature(
    temperature: Union[xr.DataArray, np.ndarray, float],
    pressure: Union[xr.DataArray, np.ndarray, float],
    qv: Optional[Union[xr.DataArray, np.ndarray, float]] = None,
) -> xr.DataArray:
    """Calculate potential temperature from temperature and pressure.

    Parameters
    ----------
    temperature : xarray.DataArray
        Temperature in K.
    pressure : xarray.DataArray
        Pressure in Pa.

    Returns:
    -------
    xarray.DataArray
        Potential temperature in K.
    """
    # reference pressure in Pa
    p0 = 100000.0
    # specific gas constant for dry air in J/(kg*K)
    r_d = 287.05
    # specific heat capacity at constant pressure for dry air in J/(kg*K)
    c_p = 1004.0

    theta = temperature * (p0 / pressure) ** (r_d / c_p)
    theta.attrs = {"units": "K", "long_name": "potential_temperature"}
    if qv is None:
        return theta
    else:
        # virtual potential temperature
        theta_v = theta * (1 + 0.61 * qv)
        theta_v.attrs = {"units": "K", "long_name": "virtual_potential_temperature"}
        return theta_v


def nearest_idx(array: xr.DataArray, value: float) -> int:
    """Find the index of the nearest value in an xarray DataArray."""
    if array.ndim == 1:
        idx = np.abs(array - value).argmin().item()
        return idx
    else:
        flat_idx = np.abs(array - value).argmin()
        idx = np.unravel_index(flat_idx, array.shape)
        return idx


def nearest_lat_lon_idx(
    lat_array: Union[xr.DataArray, np.ndarray],
    lon_array: Union[xr.DataArray, np.ndarray],
    lat: float,
    lon: float,
) -> tuple[int, int]:
    """Find the 2D grid index nearest to a latitude/longitude coordinate."""
    lat_values = np.asarray(lat_array)
    lon_values = np.asarray(lon_array)

    lat_diff = lat_values - lat
    lon_diff = (lon_values - lon + 180.0) % 360.0 - 180.0
    lon_scale = np.cos(np.deg2rad(lat))

    flat_idx = np.argmin(lat_diff * lat_diff + (lon_diff * lon_scale) ** 2)
    return tuple(int(i) for i in np.unravel_index(flat_idx, lat_values.shape))


def compare_all_group_variables(
    nc_path_1: Union[str, Path],
    nc_path_2: Union[str, Path],
) -> None:
    with Dataset(nc_path_1, "r") as ds1, Dataset(nc_path_2, "r") as ds2:
        groups1 = set(ds1.groups.keys())
        groups2 = set(ds2.groups.keys())

        only_groups_1 = sorted(groups1 - groups2)
        only_groups_2 = sorted(groups2 - groups1)
        common_groups = sorted(groups1 & groups2)

        print("=" * 80)
        print("GROUP COMPARISON")
        print("=" * 80)

        print(f"\nGroups only in {nc_path_1}:")
        print("\n".join(f"  {g}" for g in only_groups_1) or "  None")

        print(f"\nGroups only in {nc_path_2}:")
        print("\n".join(f"  {g}" for g in only_groups_2) or "  None")

        print("\nGroups in both files:")
        print("\n".join(f"  {g}" for g in common_groups) or "  None")

        for group in common_groups:
            vars1 = set(ds1.groups[group].variables.keys())
            vars2 = set(ds2.groups[group].variables.keys())

            only_vars_1 = sorted(vars1 - vars2)
            only_vars_2 = sorted(vars2 - vars1)
            common_vars = sorted(vars1 & vars2)

            print("\n" + "=" * 80)
            print(f"VARIABLE COMPARISON FOR GROUP /{group}")
            print("=" * 80)

            print(f"\nOnly in {nc_path_1}:")
            print("\n".join(f"  {v}" for v in only_vars_1) or "  None")

            print(f"\nOnly in {nc_path_2}:")
            print("\n".join(f"  {v}" for v in only_vars_2) or "  None")

            print("\n" + "In both:")
            print("\n".join(f"  {v}" for v in common_vars) or "  None")
import datetime
from typing import Union

import numpy as np
import pandas as pd
from astropy import units
from astropy.coordinates import AltAz, EarthLocation, SkyCoord, get_sun
from astropy.time import Time


def calculate_sun_position(
    lat: float,
    lon: float,
    time_utc: Union[datetime.datetime, np.datetime64, pd.Timestamp],
    height: float = 0,
) -> SkyCoord:
    """Calculate the Solar Zenith Angle (SZA) for a location at a specific time.

    Parameters:
    lat (float): Latitude of the location.
    lon (float): Longitude of the location.
    time_utc (Union[datetime.datetime, np.datetime64, pd.Timestamp]): UTC time.
    height (float, optional): Height of the location in meters. Default is 0.

    Returns:
    SkyCoord: The position of the sun in AltAz coordinates.
    """
    # Create an EarthLocation object for the given latitude and longitude
    location = EarthLocation(
        lat=lat * units.deg, lon=lon * units.deg, height=height * units.m
    )

    # Convert the UTC time to an Astropy Time object
    time = Time(time_utc, scale="utc")

    # Get the position of the sun at the given time and location
    altaz_frame = AltAz(obstime=time, location=location)
    sun_altaz = get_sun(time).transform_to(altaz_frame)

    return sun_altaz


__all__ = ["calculate_sun_position"]
