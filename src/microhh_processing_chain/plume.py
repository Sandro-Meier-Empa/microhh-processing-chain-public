"""Plume geometry and spectral diagnostics."""

import numpy as np
import xarray as xr
from scipy import fftpack


def calculate_plume_centerline(
    grid_y: np.ndarray, concentration_2d: np.ndarray
) -> np.ndarray:
    """Calculate the crosswind plume centerline for each x-slice.

    Equation (2) in Raznevic et al., 2022, adapted for 2D.

    Args:
        grid_y: Crosswind coordinates aligned with the concentration field.
        concentration_2d: Concentration field indexed by x and y.

    Returns:
        Centerline y-position for each x-slice.
    """
    # Numerator: sum(c * y), Denominator: sum(c)
    numerator = np.sum(concentration_2d * grid_y, axis=1)
    denominator = np.sum(concentration_2d, axis=1)

    # Avoid division by zero for slices with no plume
    return np.divide(
        numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0
    )


def calculate_absolute_dispersion(
    grid_y: np.ndarray,
    concentration_2d: np.ndarray,
    y_mean_centerline: np.ndarray,
) -> np.ndarray:
    """Calculate absolute plume dispersion for each x-slice.

    Equation (4) in Raznevic et al., 2022, adapted for 2D.

    Args:
        grid_y: Crosswind coordinates aligned with the concentration field.
        concentration_2d: Concentration field indexed by x and y.
        y_mean_centerline: Centerline y-position for each x-slice.

    Returns:
        Absolute dispersion values.
    """
    dy_squared = (grid_y - y_mean_centerline[:, np.newaxis]) ** 2

    numerator = np.sum(concentration_2d * dy_squared, axis=1)
    denominator = np.sum(concentration_2d, axis=1)

    return np.divide(
        numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0
    )


def calculate_meandering_dispersion(y_mean_centerline: np.ndarray) -> float:
    """Calculate meandering plume dispersion.

    Equation (5) in Raznevic et al., 2022.

    Args:
        y_mean_centerline: Centerline y-position for each x-slice.

    Returns:
        Variance of the plume centerline.
    """
    return float(np.var(y_mean_centerline))


def calculate_plume_psd(
    data: xr.Dataset, variable_name: str = "CH4_mass"
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate the radially averaged plume power spectral density.

    Args:
        data: Dataset containing the plume mask, pixel area, and analysed variable.
        variable_name: Variable to analyse within the plume mask.

    Returns:
        Frequency axis and corresponding radially averaged PSD values.
    """
    plume_data = data[variable_name].where(data.plume_area == 1, drop=True).values

    plume_data -= np.nanmean(plume_data)
    plume_data = np.nan_to_num(plume_data)

    # Apply Tapering to prevent spectral leakage
    tapered_data = apply_2d_taper(plume_data)

    f = fftpack.fft2(tapered_data)
    f_shifted = fftpack.fftshift(f)
    # normlise to satisfy Parseval's Theorem
    psd_2d = (np.abs(f_shifted) ** 2) / (tapered_data.size)

    center = (psd_2d.shape[1] // 2, psd_2d.shape[0] // 2)
    psd_1d = radial_profile(psd_2d, center)

    # 3. Create the frequency axis for the 1D plot
    dx = np.nanmean(np.sqrt(data["pixel_area"].values))
    df = 1.0 / (psd_2d.shape[0] * dx)  # Assuming square-ish pixels
    freq_1d = np.arange(len(psd_1d)) * df

    return freq_1d, psd_1d


def apply_2d_taper(data: np.ndarray) -> np.ndarray:
    """Apply a 2D Hanning window before spectral analysis.

    Args:
        data: Two-dimensional plume concentration field.

    Returns:
        Tapered field.
    """
    ny, nx = data.shape
    # Create 1D windows for each axis
    win_y = np.hanning(ny)
    win_x = np.hanning(nx)

    # Combine into a 2D window (outer product)
    window_2d = np.outer(win_y, win_x)

    # Apply to the data
    return data * window_2d


def radial_profile(data: np.ndarray, center: tuple) -> np.ndarray:
    """Calculate average power at each radial distance from a center point.

    Args:
        data: Two-dimensional power field.
        center: x/y coordinates of the radial origin.

    Returns:
        Mean value for each integer radial distance.
    """
    y, x = np.indices(data.shape)
    r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    r = r.astype(int)

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr  # normalize by the number of points in each bin

    return radialprofile


__all__ = [
    "calculate_absolute_dispersion",
    "calculate_meandering_dispersion",
    "calculate_plume_centerline",
]
