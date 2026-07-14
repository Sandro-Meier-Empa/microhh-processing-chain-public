"""Plume geometry and spectral diagnostics."""

from typing import Optional, Union

import numpy as np
import xarray as xr


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
) -> xr.Dataset:
    """Calculate the plume two-dimensional spatial PSD density.

    The returned PSD density is normalized so that its integral over the
    two-dimensional spatial-frequency plane approximately preserves the
    variance of the mean-removed plume field:

        psd_2d.sum() * dkx * dky ~= np.nanvar(field)

    Args:
        data: Dataset containing the plume mask, pixel area, and analysed variable.
        variable_name: Variable to analyse within the plume mask.

    Returns:
        Dataset containing the normalized two-dimensional PSD density and
        spatial-frequency coordinates.

    Raises:
        ValueError: If the masked input field is not two-dimensional or if the
            inferred grid/window normalization is invalid.
    """
    data_2d = data.squeeze(dim="source", drop=True) if "source" in data.dims else data
    plume_field = data_2d[variable_name].where(data_2d.plume_area == 1, drop=True)

    if plume_field.ndim != 2:
        raise ValueError(
            "calculate_plume_psd expects a two-dimensional field after plume "
            f"masking; got dimensions {plume_field.dims}."
        )

    plume_data = plume_field.values.astype(float, copy=True)
    plume_data -= np.nanmean(plume_data)
    plume_data = np.nan_to_num(plume_data, nan=0.0)

    dx = float(np.sqrt(np.nanmean(data_2d["pixel_area"].values)))
    dy = dx
    if not np.isfinite(dx) or dx <= 0:
        raise ValueError(
            "Could not infer a positive finite grid spacing from pixel_area."
        )

    tapered_data, window_2d = apply_2d_taper(plume_data, return_window=True)
    window_power = float(np.mean(window_2d**2))
    if not np.isfinite(window_power) or window_power <= 0:
        raise ValueError("The Hanning window power correction is not positive.")

    ny, nx = tapered_data.shape
    field_fft = np.fft.fft2(tapered_data)
    psd_2d = (
        np.abs(np.fft.fftshift(field_fft)) ** 2 * dx * dy / (nx * ny * window_power)
    )

    kx = np.fft.fftshift(np.fft.fftfreq(nx, d=dx))
    ky = np.fft.fftshift(np.fft.fftfreq(ny, d=dy))

    input_units = data[variable_name].attrs.get("units")
    psd_attrs = {
        "long_name": f"Two-dimensional spatial PSD density of {variable_name}",
        "normalization": "sum(psd_2d) * dkx * dky approximates field variance",
    }
    if input_units:
        psd_attrs["units"] = f"({input_units})^2 m^2"

    return xr.Dataset(
        data_vars={
            "psd_2d": (
                ("ky", "kx"),
                psd_2d,
                psd_attrs,
            )
        },
        coords={
            "kx": (
                "kx",
                kx,
                {"units": "m-1", "long_name": "zonal spatial frequency"},
            ),
            "ky": (
                "ky",
                ky,
                {"units": "m-1", "long_name": "meridional spatial frequency"},
            ),
        },
        attrs={
            "dx": dx,
            "dy": dy,
            "window": "hanning_2d",
            "window_power_correction": window_power,
            "normalization": (
                "|fft2(field * window)|^2 * dx * dy / (nx * ny * mean(window^2))"
            ),
        },
    )


def calculate_radial_plume_psd(
    psd: xr.Dataset,
    psd_name: str = "psd_2d",
    nbins: Optional[int] = None,
    bin_edges: Optional[np.ndarray] = None,
) -> xr.Dataset:
    """Radially average a two-dimensional plume PSD density.

    Args:
        psd: Dataset returned by :func:`calculate_plume_psd`.
        psd_name: Name of the two-dimensional PSD variable to average.
        nbins: Number of radial wavenumber bins. Ignored when ``bin_edges`` is
            provided.
        bin_edges: Radial wavenumber bin edges in ``m-1``.

    Returns:
        Dataset containing the annular mean PSD density and a variance-preserving
        radial spectrum.

    Raises:
        ValueError: If the PSD dataset does not have compatible ``kx`` and
            ``ky`` coordinates or invalid radial bins.
    """
    if psd_name not in psd:
        raise ValueError(f"PSD variable '{psd_name}' is not present in the dataset.")

    psd_2d = psd[psd_name]
    if psd_2d.dims != ("ky", "kx"):
        raise ValueError(
            f"Expected '{psd_name}' dimensions ('ky', 'kx'); got {psd_2d.dims}."
        )

    kx = psd["kx"].values
    ky = psd["ky"].values
    if kx.size != psd_2d.sizes["kx"] or ky.size != psd_2d.sizes["ky"]:
        raise ValueError("kx/ky coordinate lengths must match the PSD shape.")

    dkx = _uniform_spacing(kx, "kx")
    dky = _uniform_spacing(ky, "ky")

    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing="xy")
    k_radius = np.sqrt(kx_grid**2 + ky_grid**2)
    values = psd_2d.values

    if bin_edges is None:
        n_bins = max(1, min(kx.size, ky.size) // 2) if nbins is None else nbins
        if n_bins < 1:
            raise ValueError("nbins must be at least 1.")
        k_max = float(np.nanmax(k_radius))
        radial_edges = np.linspace(0.0, np.nextafter(k_max, np.inf), n_bins + 1)
    else:
        radial_edges = np.asarray(bin_edges, dtype=float)
        if radial_edges.ndim != 1 or radial_edges.size < 2:
            raise ValueError("bin_edges must be a one-dimensional array of edges.")

    bin_widths = np.diff(radial_edges)
    if np.any(~np.isfinite(radial_edges)) or np.any(bin_widths <= 0):
        raise ValueError("Radial bin edges must be finite and strictly increasing.")

    n_bins = radial_edges.size - 1
    radial_psd = np.full(n_bins, np.nan)
    annular_mean_psd = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (k_radius >= radial_edges[i]) & (k_radius <= radial_edges[i + 1])
        else:
            mask = (k_radius >= radial_edges[i]) & (k_radius < radial_edges[i + 1])

        valid = mask & np.isfinite(values)
        counts[i] = int(np.count_nonzero(valid))
        if counts[i] == 0:
            continue

        annulus_sum = float(np.nansum(values[valid]))
        annular_mean_psd[i] = annulus_sum / counts[i]
        radial_psd[i] = annulus_sum * dkx * dky / bin_widths[i]

    wavenumber = 0.5 * (radial_edges[:-1] + radial_edges[1:])
    psd_units = psd_2d.attrs.get("units")
    spectrum_attrs = {
        "long_name": "Variance-preserving radial plume PSD",
        "normalization": (
            "sum(radial_spectrum * bin_width) approximates psd_2d integral"
        ),
    }
    radial_psd_attrs = {
        "long_name": "Annular mean two-dimensional plume PSD density",
    }
    if psd_units:
        radial_psd_attrs["units"] = psd_units
        spectrum_attrs["units"] = f"({psd_units}) m-1"

    return xr.Dataset(
        data_vars={
            "radial_psd": ("wavenumber", radial_psd, spectrum_attrs),
            "annular_mean_psd": ("wavenumber", annular_mean_psd, radial_psd_attrs),
            "mode_count": ("wavenumber", counts),
            "bin_width": (
                "wavenumber",
                bin_widths,
                {"units": "m-1", "long_name": "Radial bin width"},
            ),
        },
        coords={
            "wavenumber": (
                "wavenumber",
                wavenumber,
                {"units": "m-1", "long_name": "Radial spatial frequency"},
            )
        },
        attrs={
            "source_psd": psd_name,
            "dkx": dkx,
            "dky": dky,
            "normalization": (
                "radial_spectrum is the annulus-integrated PSD divided by "
                "radial bin width"
            ),
        },
    )


def _uniform_spacing(values: np.ndarray, name: str) -> float:
    """Return the uniform coordinate spacing for a one-dimensional coordinate."""
    if values.ndim != 1 or values.size < 2:
        raise ValueError(f"{name} must be a one-dimensional coordinate.")

    spacing = np.diff(values)
    mean_spacing = float(np.mean(spacing))
    if not np.allclose(spacing, mean_spacing):
        raise ValueError(f"{name} coordinate must be uniformly spaced.")

    return abs(mean_spacing)


def apply_2d_taper(
    data: np.ndarray, return_window: bool = False
) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Apply a 2D Hanning window before spectral analysis.

    Args:
        data: Two-dimensional plume concentration field.
        return_window: Return the applied window together with the tapered data.

    Returns:
        Tapered field, and optionally the applied two-dimensional window.
    """
    ny, nx = data.shape
    # Create 1D windows for each axis
    win_y = np.hanning(ny)
    win_x = np.hanning(nx)

    # Combine into a 2D window (outer product)
    window_2d = np.outer(win_y, win_x)

    # Apply to the data
    tapered_data = data * window_2d
    if return_window:
        return tapered_data, window_2d
    return tapered_data


def compute_total_slant_column(
    gas_data: xr.DataArray,
    density_data: xr.DataArray,
    sza: float,
    saa: float,
    vza: float = 0.0,
    vaa: float = 0.0,
) -> xr.Dataset:
    """Compute total 2D Slant Column Density (SCD) for CH4.

    This computes the SCD by integrating CH4 mass density along two slant
    paths (solar and viewing) accounting for horizontal displacement caused
    by zenith and azimuth angles. The implementation is vectorized and
    supports Dask-backed xarray DataArrays.

    Args:
        gas_data (xr.DataArray): 3D gas mixing ratio (e.g. CH4 in kg/kg).
        density_data (xr.DataArray): 3D air density (kg/m^3) matching gas_data.
        sza (float): Solar zenith angle in degrees.
        saa (float): Solar azimuth angle in degrees (clockwise from North).
        vza (float, optional): Viewing zenith angle in degrees. Defaults to 0.0.
        vaa (float, optional): Viewing azimuth angle in degrees. Defaults to 0.0.

    Returns:
        xr.Dataset: Total Slant Column Density (kg/m^2) with horizontal
            dimensions (y, x) and any additional dimensions (e.g. time).
    """
    if gas_data.chunks is not None:
        safe_chunks = {dim: -1 for dim in ["x", "y", "z"] if dim in gas_data.dims}
        # Keep existing chunking for other dimensions (like 'time')
        for dim, chunks in gas_data.chunkssizes.items():
            if dim not in safe_chunks:
                safe_chunks[dim] = chunks[0]
        gas_data = gas_data.chunk(safe_chunks)

    ch4_dens = gas_data * density_data

    def _compute_single_path(
        gas_dens: xr.DataArray, zenith: float, azimuth: float
    ) -> xr.DataArray:
        """Helper to calculate SCD for a single ray path line."""
        if zenith == 0.0:
            return gas_dens.integrate("z")

        zenith_rad = np.radians(zenith)
        azimuth_rad = np.radians(azimuth)

        # Calculate horizontal displacement per unit height
        dx_factor = np.tan(zenith_rad) * np.sin(azimuth_rad)
        dy_factor = np.tan(zenith_rad) * np.cos(azimuth_rad)

        # Vectorized coordinate creation:
        # (gas_dens.z * dx_factor) creates a ('z',) array.
        # Adding it to gas_dens.x ('x',) implicitly broadcasts to a
        # 2D ('z', 'x') coordinate array.
        x_target = gas_dens.x + (gas_dens.z * dx_factor)
        y_target = gas_dens.y + (gas_dens.z * dy_factor)

        da_shifted = gas_dens.interp(
            x=x_target, y=y_target, method="linear", kwargs={"fill_value": np.nan}
        )

        path_scd = da_shifted.integrate("z") / np.cos(zenith_rad)
        return path_scd

    scd_incoming = _compute_single_path(ch4_dens, sza, saa)
    scd_outgoing = _compute_single_path(ch4_dens, vza, vaa)

    # when we compute the modelled vcd, we integrate vertically one way
    # when we compute the modelled scd, we integrate two-way
    # => the scd is expected to be larger than the vcd by a factor
    # of (1/cos(SZA) + 1/cos(VZA)) for a homogeneous atmosphere.
    # => have to correct for that if we want to compare the two.
    geom_amf = 1.0 / np.cos(np.radians(sza)) + 1.0 / np.cos(np.radians(vza))
    scd_incoming /= geom_amf
    scd_outgoing /= geom_amf
    total_scd = scd_incoming + scd_outgoing

    # save in xr dataset
    scd_incoming.name = "scd_incoming"
    scd_incoming.attrs = {
        "units": "kg m-2",
        "description": (f"Incoming slant column density (SZA={sza}, SAA={saa})"),
    }
    scd_outgoing.name = "scd_outgoing"
    scd_outgoing.attrs = {
        "units": "kg m-2",
        "description": (f"Outgoing slant column density (VZA={vza}, VAA={vaa})"),
    }
    total_scd.attrs = {
        "units": "kg m-2",
        "description": ("Total slant column density "),
    }

    data_out = xr.Dataset(
        {
            "scd_incoming": scd_incoming,
            "scd_outgoing": scd_outgoing,
            "total_scd": total_scd,
        },
        coords={dim: total_scd[dim] for dim in total_scd.dims},
        attrs={
            "sza": sza,
            "saa": saa,
            "vza": vza,
            "vaa": vaa,
            "geom_amf": geom_amf,
        },
    )
    return data_out


__all__ = [
    "apply_2d_taper",
    "calculate_absolute_dispersion",
    "calculate_meandering_dispersion",
    "calculate_plume_centerline",
    "calculate_plume_psd",
    "calculate_radial_plume_psd",
    "compute_total_slant_column",
]
