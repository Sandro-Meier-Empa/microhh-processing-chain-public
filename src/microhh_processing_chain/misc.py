"""Miscellaneous utility functions."""

from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap


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
    qv: Union[xr.DataArray, np.ndarray, float] = None,
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
