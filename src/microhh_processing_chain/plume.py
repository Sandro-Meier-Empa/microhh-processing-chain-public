"""Plume geometry and spectral diagnostics."""

import logging
from collections.abc import Sequence
from typing import Optional, Union

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

AngleInput = Union[float, Sequence[float], np.ndarray]


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
    sza: AngleInput,
    saa: AngleInput,
    vza: AngleInput = 0.0,
    vaa: AngleInput = 0.0,
    dim_names: tuple[str, str, str] = ("x", "y", "z"),
) -> xr.Dataset:
    """Compute physical and retrieval-equivalent methane columns.

    The true MicroHH column is the direct vertical integral of methane mass
    density. Raw solar and viewing path integrals are physical slant columns,
    and their sum is the physical two-way slant column. The AVIRIS-4-like
    matched-filter contributions and retrieved column divide those raw paths
    by the assumed two-way geometric air-mass factor, so they are
    vertical-column equivalents rather than SCDs. The corrected shadow and
    plume branches instead divide each raw path by its own air-mass factor.
    Their sum is a double-plume diagnostic, not a physical SCD or a single
    mass-conserving VCD.

    Horizontal ray displacement is calculated from the supplied zenith and
    azimuth angles. The calculation remains vectorized for scalar or
    time-dependent angles and supports Dask-backed xarray data.

    Args:
        gas_data: Gas mass mixing ratio, such as CH4 in kg/kg.
        density_data: Air density in kg/m3 aligned with ``gas_data``.
        sza: Solar zenith angle in degrees, either scalar or a one-dimensional
            time-varying list or array.
        saa: Solar azimuth angle in degrees (clockwise from North), either
            scalar or a one-dimensional time-varying list or array.
        vza: Viewing zenith angle in degrees, either scalar or a
            one-dimensional time-varying list or array.
        vaa: Viewing azimuth angle in degrees, either scalar or a
            one-dimensional time-varying list or array.
        dim_names: Dimension names for the horizontal x, horizontal y, and
            vertical axes, respectively.

    Returns:
        Physical columns, matched-filter-equivalent retrieval products,
        corrected branch diagnostics, geometric and correction factors, and
        the input angles. Time-dependent outputs retain the input time
        coordinate.

    Raises:
        AssertionError: If ``dim_names`` does not contain three names.
        ValueError: If an angle is not scalar or one-dimensional, or if its
            length is incompatible with the input time dimension.
    """
    assert len(dim_names) == 3, "dim_names must be a tuple of three dimension names."
    x_name, y_name, z_name = dim_names

    if gas_data.chunks is not None:
        safe_chunks = {dim: -1 for dim in dim_names if dim in gas_data.dims}
        # Keep existing chunking for other dimensions (like 'time')
        for dim, chunks in gas_data.chunksizes.items():
            if dim not in safe_chunks:
                safe_chunks[dim] = chunks[0]
        gas_data = gas_data.chunk(safe_chunks)

    gas_data, density_data = xr.align(gas_data, density_data, join="exact")
    rho_ch4 = gas_data * density_data

    def _prepare_angle(
        angle: AngleInput, name: str, template: xr.DataArray
    ) -> xr.DataArray:
        """Convert an angle input to a scalar or time-indexed DataArray."""
        angle_values = np.asarray(angle, dtype=float)
        if angle_values.ndim == 0:
            angle_scalar = float(angle_values.item())
            if "time" in template.dims:
                angle_values = np.full(
                    template.sizes["time"], angle_scalar, dtype=float
                )
            else:
                angle_out = xr.DataArray(angle_scalar, name=name)
                angle_out.attrs = {
                    "units": "degrees",
                    "description": f"{name} used for slant-column calculation",
                }
                return angle_out

        if angle_values.ndim != 1:
            raise ValueError(
                f"{name} must be a scalar or one-dimensional list or array."
            )
        if "time" not in template.dims:
            raise ValueError(
                f"{name} was provided as a list or array, but gas_data has no "
                "time dimension."
            )
        if angle_values.size != template.sizes["time"]:
            raise ValueError(
                f"{name} length ({angle_values.size}) does not match the time "
                f"dimension ({template.sizes['time']})."
            )

        angle_out = xr.DataArray(
            angle_values,
            dims=("time",),
            coords={"time": template["time"]},
            name=name,
        )
        angle_out.attrs = {
            "units": "degrees",
            "description": f"{name} used for slant-column calculation",
        }
        return angle_out

    def _is_zero_zenith(zenith: xr.DataArray) -> bool:
        return bool(np.all(np.asarray(zenith.data) == 0.0))

    sza_calc = _prepare_angle(sza, "sza", rho_ch4)
    saa_calc = _prepare_angle(saa, "saa", rho_ch4)
    vza_calc = _prepare_angle(vza, "vza", rho_ch4)
    vaa_calc = _prepare_angle(vaa, "vaa", rho_ch4)

    def _compute_single_path(
        gas_dens: xr.DataArray,
        zenith: xr.DataArray,
        azimuth: xr.DataArray,
    ) -> xr.DataArray:
        """Integrate methane density along one displaced ray path."""
        if _is_zero_zenith(zenith):
            return gas_dens.integrate(z_name)

        zenith_rad = np.radians(zenith)
        azimuth_rad = np.radians(azimuth)

        # Calculate horizontal displacement per unit height
        dx_factor = np.tan(zenith_rad) * np.sin(azimuth_rad)
        dy_factor = np.tan(zenith_rad) * np.cos(azimuth_rad)

        # Vectorized coordinate creation. Time-indexed angles add the time
        # dimension; scalar angles remain scalar when the gas data has no time.
        x_target = gas_dens[x_name] + (gas_dens[z_name] * dx_factor)
        y_target = gas_dens[y_name] + (gas_dens[z_name] * dy_factor)

        da_shifted = gas_dens.interp(
            {
                x_name: x_target,
                y_name: y_target,
            },
            method="linear",
            kwargs={"fill_value": np.nan},
        )

        return da_shifted.integrate(z_name) / np.cos(zenith_rad)

    microhh_vertical_column = rho_ch4.integrate(z_name)
    solar_path_slant_column = _compute_single_path(rho_ch4, sza_calc, saa_calc)
    viewing_path_slant_column = _compute_single_path(rho_ch4, vza_calc, vaa_calc)
    two_way_slant_column = solar_path_slant_column + viewing_path_slant_column

    solar_air_mass_factor = 1.0 / np.cos(np.radians(sza_calc))
    viewing_air_mass_factor = 1.0 / np.cos(np.radians(vza_calc))
    two_way_geometric_air_mass_factor = solar_air_mass_factor + viewing_air_mass_factor

    solar_path_mf_contribution = (
        solar_path_slant_column / two_way_geometric_air_mass_factor
    )
    viewing_path_mf_contribution = (
        viewing_path_slant_column / two_way_geometric_air_mass_factor
    )
    mf_retrieved_column = solar_path_mf_contribution + viewing_path_mf_contribution

    shadow_branch_vcd_equivalent = solar_path_slant_column / solar_air_mass_factor
    plume_branch_vcd_equivalent = viewing_path_slant_column / viewing_air_mass_factor
    corrected_double_plume_branch_sum = (
        shadow_branch_vcd_equivalent + plume_branch_vcd_equivalent
    )

    shadow_branch_correction_factor = (
        two_way_geometric_air_mass_factor / solar_air_mass_factor
    )
    plume_branch_correction_factor = (
        two_way_geometric_air_mass_factor / viewing_air_mass_factor
    )

    variable_metadata = {
        "microhh_vertical_column": {
            "units": "kg m-2",
            "long_name": "True MicroHH vertical methane column",
            "description": (
                "Direct physical vertical column obtained by integrating "
                "MicroHH methane mass density through the atmosphere."
            ),
            "formula": "microhh_vertical_column = rho_ch4.integrate(z_name)",
        },
        "solar_path_slant_column": {
            "units": "kg m-2",
            "long_name": "Raw incoming solar-path methane slant column",
            "description": (
                "Raw physical slant column integrated along the incoming "
                "solar line of sight before any air-mass-factor normalization."
            ),
            "formula": "solar_path_slant_column = L_solar",
        },
        "viewing_path_slant_column": {
            "units": "kg m-2",
            "long_name": "Raw outgoing viewing-path methane slant column",
            "description": (
                "Raw physical slant column integrated along the outgoing "
                "viewing line of sight before any air-mass-factor normalization."
            ),
            "formula": "viewing_path_slant_column = L_viewing",
        },
        "two_way_slant_column": {
            "units": "kg m-2",
            "long_name": "Raw two-way physical methane slant column",
            "description": (
                "Physical slant column accumulated along the complete incoming "
                "solar plus outgoing viewing photon path."
            ),
            "formula": "two_way_slant_column = L_solar + L_viewing",
        },
        "solar_path_mf_contribution": {
            "units": "kg m-2",
            "long_name": (
                "Incoming-path contribution to the matched-filter "
                "vertical-column equivalent"
            ),
            "description": (
                "Incoming solar-path contribution to the AVIRIS-4-like "
                "matched-filter retrieved vertical-column equivalent, obtained "
                "by dividing the raw solar slant column by the assumed two-way "
                "geometric air-mass factor."
            ),
            "formula": "solar_path_mf_contribution = L_solar / m_two_way",
        },
        "viewing_path_mf_contribution": {
            "units": "kg m-2",
            "long_name": (
                "Outgoing-path contribution to the matched-filter "
                "vertical-column equivalent"
            ),
            "description": (
                "Outgoing viewing-path contribution to the AVIRIS-4-like "
                "matched-filter retrieved vertical-column equivalent, obtained "
                "by dividing the raw viewing slant column by the assumed two-way "
                "geometric air-mass factor."
            ),
            "formula": "viewing_path_mf_contribution = L_viewing / m_two_way",
        },
        "mf_retrieved_column": {
            "units": "kg m-2",
            "long_name": (
                "AVIRIS-4-like matched-filter retrieved vertical-column equivalent"
            ),
            "description": (
                "AVIRIS-4-like matched-filter-equivalent vertical column. It "
                "uses the assumed two-way geometric air-mass factor even when "
                "the two paths traverse the plume at different projected "
                "locations; it is not a physical slant column."
            ),
            "formula": ("mf_retrieved_column = (L_solar + L_viewing) / m_two_way"),
        },
        "shadow_branch_vcd_equivalent": {
            "units": "kg m-2",
            "long_name": "Corrected shadow-branch vertical-column equivalent",
            "description": (
                "Path-specific corrected vertical-column equivalent of the "
                "displaced plume-shadow branch, formed by normalizing the raw "
                "incoming solar-path column by its own air-mass factor."
            ),
            "formula": "shadow_branch_vcd_equivalent = L_solar / m_solar",
        },
        "plume_branch_vcd_equivalent": {
            "units": "kg m-2",
            "long_name": "Corrected plume-branch vertical-column equivalent",
            "description": (
                "Path-specific corrected vertical-column equivalent of the "
                "source-plume branch, formed by normalizing the raw outgoing "
                "viewing-path column by its own air-mass factor."
            ),
            "formula": "plume_branch_vcd_equivalent = L_viewing / m_viewing",
        },
        "corrected_double_plume_branch_sum": {
            "units": "kg m-2",
            "long_name": "Diagnostic sum of corrected plume and shadow branches",
            "description": (
                "Diagnostic image containing both individually corrected "
                "double-plume branches. It is not a physical SCD and not a "
                "single mass-conserving VCD: it sums two alternative "
                "projections of the same MicroHH plume and equals 2 * "
                "microhh_vertical_column for a horizontally homogeneous "
                "atmosphere."
            ),
            "formula": (
                "corrected_double_plume_branch_sum = "
                "shadow_branch_vcd_equivalent + plume_branch_vcd_equivalent"
            ),
        },
        "solar_air_mass_factor": {
            "units": "1",
            "long_name": "Solar geometric air-mass factor",
            "description": (
                "Geometric air-mass factor for the incoming solar path, equal "
                "to the secant of the solar zenith angle."
            ),
            "formula": "solar_air_mass_factor = sec(SZA)",
        },
        "viewing_air_mass_factor": {
            "units": "1",
            "long_name": "Viewing geometric air-mass factor",
            "description": (
                "Geometric air-mass factor for the outgoing viewing path, equal "
                "to the secant of the viewing zenith angle."
            ),
            "formula": "viewing_air_mass_factor = sec(VZA)",
        },
        "two_way_geometric_air_mass_factor": {
            "units": "1",
            "long_name": "Two-way geometric air-mass factor",
            "description": (
                "Assumed two-way geometric air-mass factor given by the sum of "
                "the incoming solar and outgoing viewing air-mass factors."
            ),
            "formula": (
                "two_way_geometric_air_mass_factor = "
                "solar_air_mass_factor + viewing_air_mass_factor"
            ),
        },
        "shadow_branch_correction_factor": {
            "units": "1",
            "long_name": "Shadow-branch correction factor",
            "description": (
                "Multiplicative Meier et al. correction that converts the "
                "incoming solar-path matched-filter contribution into the "
                "shadow-branch vertical-column equivalent."
            ),
            "formula": (
                "shadow_branch_correction_factor = "
                "two_way_geometric_air_mass_factor / solar_air_mass_factor"
            ),
        },
        "plume_branch_correction_factor": {
            "units": "1",
            "long_name": "Plume-branch correction factor",
            "description": (
                "Multiplicative Meier et al. correction that converts the "
                "outgoing viewing-path matched-filter contribution into the "
                "plume-branch vertical-column equivalent."
            ),
            "formula": (
                "plume_branch_correction_factor = "
                "two_way_geometric_air_mass_factor / viewing_air_mass_factor"
            ),
        },
    }

    canonical_variables = {
        "microhh_vertical_column": microhh_vertical_column,
        "solar_path_slant_column": solar_path_slant_column,
        "viewing_path_slant_column": viewing_path_slant_column,
        "two_way_slant_column": two_way_slant_column,
        "solar_path_mf_contribution": solar_path_mf_contribution,
        "viewing_path_mf_contribution": viewing_path_mf_contribution,
        "mf_retrieved_column": mf_retrieved_column,
        "shadow_branch_vcd_equivalent": shadow_branch_vcd_equivalent,
        "plume_branch_vcd_equivalent": plume_branch_vcd_equivalent,
        "corrected_double_plume_branch_sum": corrected_double_plume_branch_sum,
        "solar_air_mass_factor": solar_air_mass_factor,
        "viewing_air_mass_factor": viewing_air_mass_factor,
        "two_way_geometric_air_mass_factor": two_way_geometric_air_mass_factor,
        "shadow_branch_correction_factor": shadow_branch_correction_factor,
        "plume_branch_correction_factor": plume_branch_correction_factor,
    }
    for name, variable in canonical_variables.items():
        variable.name = name
        variable.attrs = variable_metadata[name]

    angle_metadata = {
        "sza": {
            "units": "degrees",
            "long_name": "Solar zenith angle",
            "description": (
                "Solar zenith angle used to define the incoming solar path and "
                "solar geometric air-mass factor."
            ),
        },
        "saa": {
            "units": "degrees",
            "long_name": "Solar azimuth angle",
            "description": (
                "Solar azimuth angle measured clockwise from North and used to "
                "define the horizontal displacement of the incoming solar path."
            ),
        },
        "vza": {
            "units": "degrees",
            "long_name": "Viewing zenith angle",
            "description": (
                "Viewing zenith angle used to define the outgoing viewing path "
                "and viewing geometric air-mass factor."
            ),
        },
        "vaa": {
            "units": "degrees",
            "long_name": "Viewing azimuth angle",
            "description": (
                "Viewing azimuth angle measured clockwise from North and used "
                "to define the horizontal displacement of the outgoing path."
            ),
        },
    }
    angle_variables = {
        "sza": sza_calc,
        "saa": saa_calc,
        "vza": vza_calc,
        "vaa": vaa_calc,
    }
    for name, variable in angle_variables.items():
        variable.name = name
        variable.attrs = angle_metadata[name]

    data_out = xr.Dataset(
        {
            **canonical_variables,
            **angle_variables,
        }
    )
    data_out = data_out.drop_vars([x_name, y_name], errors="ignore")
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
