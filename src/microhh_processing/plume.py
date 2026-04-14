import numpy as np
from scipy import fftpack
import xarray as xr


def calculate_plume_centerline(
    grid_y: np.ndarray, concentration_2d: np.ndarray
) -> np.ndarray:
    """Calculates the crosswind centerline position y_m for a given x-slice.

    Equation (2) in Raznevic et al., 2022, adapted for 2D.
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
    """Calculates absolute plume dispersion (sigma_ya^2).

    Equation (4) in Raznevic et al., 2022, adapted for 2D.
    """
    dy_squared = (grid_y - y_mean_centerline[:, np.newaxis]) ** 2

    numerator = np.sum(concentration_2d * dy_squared, axis=1)
    denominator = np.sum(concentration_2d, axis=1)

    return np.divide(
        numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0
    )


def calculate_meandering_dispersion(y_mean_centerline: np.ndarray) -> float:
    """Calculates meandering plume dispersion (sigma_ym^2).

    Equation (5) in Raznevic et al., 2022.
    """
    return np.var(y_mean_centerline)


def calculate_plume_psd(
    data: xr.Dataset, variable_name: str = "CH4_mass"
) -> tuple[np.ndarray, np.ndarray]:
    """Calculates the 1D radially averaged power spectral density (PSD) of the plume.

    Args:
        data (xr.Dataset): Input dataset containing the plume data and pixel area.
        variable_name (str): Name of the variable in the dataset to analyze.

    Returns:
        tuple[np.ndarray, np.ndarray]: Tuple containing the frequency axis and the
        corresponding 1D PSD values.
    """
    plume_data = data[variable_name].where(data.plume_area == 1, drop=True).values

    plume_data -= np.nanmean(plume_data)
    plume_data = np.nan_to_num(plume_data)

    # Apply Tapering to prevent spectral leakage
    tapered_data = apply_2d_taper(plume_data)

    F = fftpack.fft2(tapered_data)
    F_shifted = fftpack.fftshift(F)
    # normlise to satisfy Parseval's Theorem
    psd_2d = (np.abs(F_shifted) ** 2) / (tapered_data.size)

    center = (psd_2d.shape[1] // 2, psd_2d.shape[0] // 2)
    psd_1d = radial_profile(psd_2d, center)

    # 3. Create the frequency axis for the 1D plot
    dx = np.nanmean(np.sqrt(data["pixel_area"].values))
    df = 1.0 / (psd_2d.shape[0] * dx)  # Assuming square-ish pixels
    freq_1d = np.arange(len(psd_1d)) * df

    return freq_1d, psd_1d


def apply_2d_taper(data: np.ndarray) -> np.ndarray:
    """
    Applies a 2D Hanning window to the data to reduce spectral leakage in the FFT.

    Args:
        data (np.ndarray): 2D array of plume concentration values.

    Returns:
        np.ndarray: Tapered 2D array.
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
    """Calculates the average power at each radial distance from a center point.

    Args:
        data (np.ndarray): 2D array of power values (e.g., from the PSD).
        center (tuple): (x, y) coordinates of the center point in the data array.

    Returns:
        np.ndarray: 1D array of average power values as a function of radial distance.
    """
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    r = r.astype(int)

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr  # normalize by the number of points in each bin

    return radialprofile


__all__ = [
    "calculate_plume_centerline",
    "calculate_absolute_dispersion",
    "calculate_meandering_dispersion",
]
