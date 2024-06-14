import os
import skimage
import sys

from pathlib import Path
from typing import Union, Tuple


import cartopy.crs as ccrs
import netCDF4 as nc
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.join("/users", os.environ["USER"], "microhh", "python"))
sys.path.append(
    os.path.join("/users", os.environ["USER"], "microhh", "processing_chain", "src")
)
sys.path.append(os.path.join("/users", os.environ["USER"], "ddeq", "src"))
sys.path.append(os.path.join("/users", os.environ["USER"], "u-cat", "src"))

import microhh_tools as mht
import initialisation as init
import ddeq
import ucat


def estimate_emissions(
    data: xr.Dataset,
    wind: xr.Dataset,
    sources: xr.Dataset,
    filter_size: Union[int, float],
    q: float,
    sigma: int,
    background_dilation: int,
    crs: ccrs.epsg,
) -> Tuple[xr.Dataset, dict, dict]:
    """Estimate CH4 emissions from a simulated source in MicroHH using
    the Integrated Mass Enhancement (IME) method.

    Args:
        data (xr.Dataset):
            Dataset containing the trace gas columns.
        wind (xr.Dataset):
            Dataset containing wind information.
        sources (xr.Dataset):
            Dataset containing information about the source.
        filter_size (Union[int, float]):
            Size of filter user for computing the locam mean in pixels.
        q (float):
            probability for threshold z(q) that a pixel is significantly
            enhanced above the background used for the statistical z-test.
        sigma (int):
            Size of Gaussian kernel used for normalized convolution in pixels
        background_dilation (int):
            If not None add extra dilation of masked pixels using a disk with
            given radius given by `extra_dilation`.
        crs (ccrs.epsg):
            Coordinate reference system for the source.

    Raises:
        ValueError: Raise error if the wind dataset contains multiple time steps

    Returns:
        data (xr.Dataset):
            Dataset containing the trace gas columns and additional variables
            used for plume detection and quantification.
        curves (dict):
            Dictionary with ddeq.plume_coords.Poly2D curves fitted to
            all detected plumes.
        results (dict):
            Dictionary with xr.Datasets containing the estimated emissions
            and auxiliary variables for all detected sources.
    """

    if any("time" in dim for dim in wind.dims) or any(
        "time" in dim for dim in data.dims
    ):
        raise ValueError(
            "Either the wind dataset or the data dataset contains multiple time steps. "
            "Please select one time step."
        )

    # if available, add method of vertical integration of wind as attribute
    wind_method = wind.attrs.get("method", None)
    if wind_method is not None:
        wind.speed.attrs["method"] = wind_method

    data = ddeq.dplume.detect_plumes(
        data,
        sources,
        variable="CH4",
        variable_std="CH4_std",
        filter_type="gaussian",
        q=q,
        filter_size=filter_size,
        background=np.zeros_like(data["CH4"]),
    )

    data, curves = ddeq.plume_coords.compute_plume_line_and_coords(
        data, crs, radius=2, radius_units="px", plume_area="area", do_coords=True
    )

    # convert noise to correct units
    noise_units = data["CH4_std"].attrs.get("units", None)
    noise_level = ucat.convert_columns(
        np.nanmean(data["CH4_std"]),
        from_units=noise_units,
        to_units="kg m-2",
        molar_mass="CH4",
    )

    # convert data to correct units
    data_units = data["CH4"].attrs.get("units", None)
    ch4_mass = ucat.convert_columns(
        data["CH4"], from_units=data_units, to_units="kg m-2", molar_mass="CH4"
    )

    # put into array
    dims = data["CH4"].dims
    attrs = {"units": "kg m-2", "noise_level": noise_level}
    data["CH4_mass"] = xr.DataArray(ch4_mass, dims=dims, attrs=attrs)

    # normalized convolution (sigma px)
    data = ddeq.background.estimate(
        data,
        "CH4",
        sigma=sigma,
        mask_hits=False,
        extra_dilation=background_dilation,
    )

    # convert background to correct units
    background_units = data["CH4_estimated_background"].attrs.get("units", None)
    mass_bg = ucat.convert_columns(
        data["CH4_estimated_background"],
        from_units=background_units,
        to_units="kg m-2",
        molar_mass="CH4",
    )

    # put into array
    data["CH4_minus_estimated_background_mass"] = xr.DataArray(
        data.CH4_mass - mass_bg, dims=data.CH4.dims, attrs={"units": "kg m-2"}
    )

    # estimate emissions
    L_max = np.max(data.xp.values[data.detected_plume]) + 0.5 * np.sqrt(
        np.mean(data.pixel_area)
    )
    results = ddeq.ime.estimate_emissions(
        data,
        wind,
        sources,
        curves,
        L_max=L_max,
        gas="CH4",
        min_pixel_number=1,
        allowed_upstream_detected=10,
    )

    return data, curves, results


def compute_effective_wind_speed(
    data: xr.Dataset, true_emissions: Union[int, float]
) -> xr.Dataset:
    data["effective_wind_speed"] = (
        true_emissions * data.L_max
    ) / data.integrated_CH4_mass
    data["effective_wind_speed"].attrs["units"] = "m s-1"

    return data


def radial_ime(
    data,
    winds,
    sources,
    gas,
    radius,
    variable="{gas}_minus_estimated_background_mass",
    L_min=0,
    decay_time=np.nan,
):

    results = {}

    extra_variables = [
        ("L_min", "m"),
        ("L_max", "m"),
        ("L_std", "m"),
        ("wind_speed", "m s-1"),
        ("wind_speed_precision", "m s-1"),
        ("wind_direction", "°"),
        ("decay_time", "s"),
        ("emissions_scaling_factor", "1"),
        ("integrated_{gas}_mass", "kg"),
        ("integrated_{gas}_mass_precision", "kg"),
        ("{gas}_estimated_emissions", "kg s-1"),
        ("{gas}_estimated_emissions_precision", "kg s-1"),
    ]

    for name, source in sources.groupby("source", squeeze=False):

        results[name] = ddeq.misc.init_results_dataset(
            source,
            [gas],
            extra_vars=extra_variables,
            method="integrated mass enhancement",
        )

        if name not in data.source:
            continue

        this = data.sel(source=name)

        # interpolate missing data
        vcd = this[variable.format(gas=gas)].values

        # fill cloud- and data-gaps
        kernel = ddeq.dplume.gaussian_kernel(sigma=2)
        vcd = np.where(
            np.isfinite(vcd), vcd, ddeq.misc.normalized_convolution(vcd, kernel)
        )

        # wind speed and its uncertainty
        wind = winds.sel(source=name)

        U = float(wind["speed"])
        D = float(wind["direction"])
        U_std = float(wind["speed_precision"])

        # determine area for mass integration
        integration_area = disk_around_source(
            this, lon_o=source["lon_o"], lat_o=source["lat_o"], radius=radius
        )

        # compute integrated mass enhancement
        A = data["pixel_area"].values
        pixel_length = float(np.sqrt(np.mean(A[integration_area])))
        L = radius * pixel_length
        ime = np.nansum(vcd[integration_area] * A[integration_area])

        L_std = max(0.1 * L, 0.5 * pixel_length)

        flux = U / L * ime

        # uncertainty (considering IME, L and U)
        mass_std = this[f"{gas}_mass"].attrs.get("noise_level", np.nan)
        ime_std = (
            np.mean(A[integration_area]) * mass_std * np.sqrt(np.sum(integration_area))
        )
        flux_std = (
            1.0
            / L
            * np.sqrt(
                U**2 * ime_std**2 + ime**2 * U_std**2 + U**2 * ime**2 / L**2 * L_std**2
            )
        )

        variables_to_assign = {
            "L_min": L_min,
            "L_max": L,
            "L_std": L_std,
            "wind_speed": U,
            "wind_speed_precision": U_std,
            "wind_direction": D,
            f"integrated_{gas}_mass": ime,
            f"integrated_{gas}_mass_precision": ime_std,
            "decay_time": decay_time,
            f"{gas}_estimated_emissions": flux,
            f"{gas}_estimated_emissions_precision": flux_std,
        }

        for var_name, value in variables_to_assign.items():
            attrs = results[name][var_name].attrs if var_name in results[name] else {}
            results[name] = results[name].assign(
                **{var_name: (("source"), [value], attrs)}
            )

    return xr.concat(results.values(), dim="source")


def disk_around_source(
    data: xr.DataArray,
    lon_o: Union[float, xr.DataArray],
    lat_o: Union[float, xr.DataArray],
    radius: int,
    variable: str = "CH4",
):

    lon_o = lon_o.values if isinstance(lon_o, xr.DataArray) else lon_o
    lat_o = lat_o.values if isinstance(lat_o, xr.DataArray) else lat_o
    radius = radius.values if isinstance(radius, xr.DataArray) else radius

    # get the pixel which contains the source
    lat_i, lon_i, dist = ddeq.misc.find_closest(data, ("lon", "lat"), (lon_o, lat_o))

    # create empty field containing the source
    source_field = np.zeros_like(data[variable])
    source_field[lat_i, lon_i] = 1

    # add buffer around the source with radius in pixels
    footprint = skimage.morphology.disk(radius)
    dilated_source = skimage.morphology.binary_dilation(
        source_field, footprint=footprint
    )

    return dilated_source
