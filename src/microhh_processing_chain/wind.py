"""Wind diagnostics, similarity profiles, and turbulence spectra."""

import logging
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Any, Literal, Optional, Union, overload

import ls2d
import numpy as np
import scipy
import ucat
import xarray as xr
from pyproj import CRS
from scipy.optimize import fsolve
from scipy.signal import welch
from tqdm.auto import tqdm

import ddeq
import microhh_processing_chain.initialisation as init
import microhh_processing_chain.io as io
import microhh_processing_chain.preprocessing as pre
from microhh_processing_chain.constants import EARTH_ROTATION_RATE
from microhh_processing_chain.enums import WindIntegrationMethod

logger = logging.getLogger(__name__)


class WindProcessing:
    """Class used for post-processing MicroHH cases and handling the output.

    Args:
        case:
            Case to post-process.
    """

    def __init__(self, case: init.Case) -> None:
        import microhh_processing_chain.postprocessing as post

        self.post = post.PostProcessing(case)
        self.case = case
        self.directory = case.directory

    def create_wind_dataset(
        self,
        source: str,
        sources: xr.Dataset,
        radius: int,
        method: Union[WindIntegrationMethod, str],
        gas: Optional[str] = None,
        height: Optional[float] = None,
        save: bool = True,
        overwrite: bool = False,
    ) -> xr.Dataset:
        """Create a dataset containing the wind information around a source.

        Args:
            source:
                Source for which the wind dataset should be created.
            sources:
                Dataset with source information
            radius:
                Number of pixels around the source which should be considered
                for calculating wind properties.
            gas:
                Gas to be used for the vertical weighting of wind speed fields.
            method:
                Method of how the wind should be vertically integrated.
                "mean" "median", "height" or "concentration_weighted"
            height:
                Height at which the wind speed should be used if method = 'height'.
            save:
                Specify if the dataset should be saved to store.
            overwrite:
                Specify if the output file should be overwritten if it exists.

        Returns:
            Dataset containing the wind information around a source.
        """
        wind_data = self.create_2d_wind(method, gas, height, save, overwrite)

        selected_source = sources.sel(source=source)
        wind_data_subset = self._select_pixels_around_source(
            wind_data, selected_source, radius
        )

        global_attrs = wind_data_subset.attrs.copy()
        new_attrs = {
            "radius around source": f"{radius} pixels",
        }
        global_attrs.update(new_attrs)

        # combine in new dataset
        wind_src = xr.Dataset(
            coords={
                "source": (["source"], np.array([source], dtype=object)),
                "time": (["time"], wind_data_subset.time.values),
            },
            data_vars={
                "lon": (["source"], [selected_source.lon.values]),
                "lat": (["source"], [selected_source.lat.values]),
                "U": (
                    ["source", "time"],
                    [wind_data_subset["u"].median(dim=["x", "y"]).values],
                    wind_data_subset["u"].attrs,
                ),
                "V": (
                    ["source", "time"],
                    [wind_data_subset["v"].median(dim=["x", "y"]).values],
                    wind_data_subset["v"].attrs,
                ),
                "speed": (
                    ["source", "time"],
                    [wind_data_subset["speed"].median(dim=["x", "y"]).values],
                    wind_data_subset["speed"].attrs,
                ),
                "speed_precision": (
                    ["source", "time"],
                    [wind_data_subset["speed"].std(dim=["x", "y"]).values],
                    wind_data_subset["speed"].attrs,
                ),
                "direction": (
                    ["source", "time"],
                    [
                        self._circular_mean(
                            wind_data_subset.direction,
                            axes=[
                                list(wind_data_subset.dims).index("y"),
                                list(wind_data_subset.dims).index("x"),
                            ],
                        )
                    ],
                    wind_data_subset["direction"].attrs,
                ),
                "direction_precision": (
                    ["source", "time"],
                    [
                        self._circular_std(
                            wind_data_subset.direction,
                            axes=[
                                list(wind_data_subset.dims).index("y"),
                                list(wind_data_subset.dims).index("x"),
                            ],
                        )
                    ],
                    wind_data_subset["direction"].attrs,
                ),
            },
            attrs=global_attrs,
        )

        return wind_src

    def create_2d_wind(
        self,
        method: Union[WindIntegrationMethod, str],
        gas: Optional[str] = None,
        height: Optional[float] = None,
        save: bool = True,
        overwrite: bool = False,
        chunks: Optional[Union[str, dict]] = None,
    ) -> xr.Dataset:
        """Create a 2D wind dataset from 3D MicroHH wind fields.

        Args:
            method:
                Method to vertically integrate wind speeds. Available options are
                "mean", "median", "height", "concentration_weighted"
            gas:
                Gas to use if method = 'concentration_weighted' wind speeds.
            height:
                Height at which the wind speed should be used if method = 'height'.
            save:
                Specify if the resulting xarray dataset should be saved.
            overwrite:
                Specify if the output file should be overwritten if it exists.
            chunks:
                Chunking strategy for dask when loading the data.

        Returns:
            Dataset with vertically integrated wind.
        """
        wind_vars = self._get_wind_vars(method)

        # load wind.nc if it exists
        wind_nc_path = Path(self.directory, "wind_2D.nc")
        if wind_nc_path.exists():
            if overwrite:
                wind_nc_path.unlink()
                logger.info("Overwriting %s", wind_nc_path)
            else:
                return io.open_microhh_nc(self.case, variable="wind_2D", chunks=chunks)

        self._check_wind_integration(method, gas, height)

        wind_data = self.post.combine_prognostic_variables(
            variables=wind_vars, chunks=chunks
        )

        wind_data = self.vertically_integrate_wind(wind_data, method, gas, height)

        # compute wind speed and direction
        wind_data["speed"] = np.sqrt(wind_data.u**2 + wind_data.v**2)
        wind_data["direction"] = ddeq.wind.calculate_wind_direction(
            wind_data.u, wind_data.v
        )
        # add units
        for wind_var in wind_vars:
            wind_data[wind_var].attrs["units"] = "m s-1"
        wind_data["speed"].attrs["units"] = "m s-1"
        wind_data["direction"].attrs["units"] = "°"

        method_description = {
            "mean": "vertical mean of wind",
            "median": "vertical median of wind",
            "height": f"{height} m wind",
            "concentration_weighted": f"{gas} concentration weighted wind",
        }

        attributes = {
            "method": method_description[method],
        }

        wind_data.attrs.update(attributes)

        if save:
            wind_data.to_netcdf(wind_nc_path)

        return wind_data

    def _get_wind_vars(self, method: Union[WindIntegrationMethod, str]) -> list:
        """Get list of prognostic variables used for the given wind integration method.

        Args:
            method: Method to vertically integrate wind speeds. Available options are
                "mean", "median", "height", "concentration_weighted"

        Raises:
            NotImplementedError: Raise error if the method is not implemented.

        Returns:
            list: List of prognostic variables used for wind integration method.
        """
        method_val = (
            method.value
            if isinstance(method, WindIntegrationMethod)
            else str(method).lower()
        )

        if method_val in [
            WindIntegrationMethod.MEAN.value,
            WindIntegrationMethod.MEDIAN.value,
            WindIntegrationMethod.HEIGHT.value,
        ]:
            return ["u", "v"]
        elif method_val == WindIntegrationMethod.CONCENTRATION_WEIGHTED.value:
            return ["u", "v", "ch4"]
        else:
            raise NotImplementedError(f"Method {method} is not implemented.")

    def _check_wind_integration(
        self,
        method: Union[WindIntegrationMethod, str],
        gas: Optional[str],
        height: Optional[float],
    ) -> None:
        """Check if all required variables for wind integration method are given.

        Args:
            method:
                Method to vertically integrate wind speeds. Available options are
                "mean", "median", "height", "concentration_weighted"
            gas:
                Gas to use if method = 'concentration_weighted' wind speeds.
            height:
                Height at which the wind speed should be used if method = 'height'.
        """
        if method not in ["mean", "median", "height", "concentration_weighted"]:
            raise ValueError(
                "'method' must be 'mean', 'median', 'height' or "
                f"'concentration_weighted', not {method}."
            )
        if method == "height" and height is None:
            raise ValueError("Height must be an integer or float, not 'None'.")
        if method == "concentration_weighted" and gas is None:
            raise ValueError("'gas' must be a string, not 'None'.")

    def _select_pixels_around_source(
        self, wind_data: xr.Dataset, selected_source: xr.Dataset, radius: int
    ) -> xr.Dataset:
        """Select all pixels within a radius around a source.

        Args:
            wind_data:
                Dataset containing the wind information.
            selected_source:
                Source around which the pixels are to be selected.
            radius:
                Radius in pixels around the source.
        """
        source_x0 = selected_source.source_x0.values
        source_y0 = selected_source.source_y0.values

        x_res = wind_data.x.diff(dim="x").median()
        y_res = wind_data.y.diff(dim="y").median()

        wind_data_src = wind_data.where(
            (wind_data.x > source_x0 - x_res / 2 - radius)
            & (wind_data.x <= source_x0 + x_res / 2 + radius)
            & (wind_data.y > source_y0 - y_res / 2 - radius)
            & (wind_data.y <= source_y0 + y_res / 2 + radius),
            drop=True,
        )

        return wind_data_src

    def vertically_integrate_wind(
        self,
        wind_data: xr.Dataset,
        method: str,
        gas: Optional[str] = None,
        height: Optional[float] = None,
    ) -> xr.Dataset:
        """Vertically integrate 3D wind fields.

        Args:
            wind_data:
                3D wind dataset
            method:
                Method of how the wind should be vertically integrated.
                "mean" "median", "height" or "concentration_weighted"
            gas:
                Gas to be used for the vertical weighting of wind speed fields.
                Required if method = 'concentration_weighted'.
            height:
                Height at which the wind speed should be used if method = 'height'.
                Required if method = 'height'.

        Raises:
            NotImplementedError: Raise an error if an invalid method is provided.

        Returns:
            Vertically integrated wind dataset.
        """
        if method in ["mean", "median"]:
            return self.unweighted_mean_wind(wind_data, method)
        elif method == "height":
            return self.wind_at_height(wind_data, height)
        elif method == "concentration_weighted":
            return self.concentration_weighted_mean_wind(wind_data, gas)
        else:
            raise NotImplementedError

    def unweighted_mean_wind(self, data: xr.Dataset, method: str) -> xr.Dataset:
        """Calculate the unweighted mean or median wind speed from 3D wind fields.

        Args:
            data:
                Dataset containing the 3D wind fields.
            method:
                Method of how the wind should be vertically integrated.
                Either "mean" or "median".
        """
        if method == "mean":
            data_int = data.mean(dim="z")
        elif method == "median":
            data_int = data.median(dim="z")
        else:
            raise NotImplementedError

        data_int.attrs["METHOD"] = f"vertical {method} of wind"
        return data_int

    def wind_at_height(
        self,
        data: xr.Dataset,
        height: Optional[float],
    ) -> xr.Dataset:
        """Get the wind speed at a certain height.

        Args:
            data:
                Dataset containing the wind speed information.
            height:
                Height in m at which the wind speed should be taken from.

        """
        if height is None:
            raise ValueError("Height must be an integer or float, not 'None'.")

        # calculate the mean vertical resolution
        z_resolution = data.z.diff("z").mean().item()

        # determine the lower and upper bounds
        lower_bound = height - z_resolution / 2
        upper_bound = height + z_resolution / 2

        data_subset = data.sel(z=slice(lower_bound, upper_bound))

        data_int = data_subset.mean(dim="z")
        data_int.attrs["METHOD"] = f"wind at {height} m"

        return data_int

    def concentration_weighted_mean_wind(
        self,
        data: xr.Dataset,
        gas: Optional[str],
        wind_vars: Optional[Union[str, list]] = None,
        threshold: Optional[float] = None,
        threshold_unit: Optional[str] = None,
    ) -> xr.Dataset:
        """Compute the concentration weighted wind speed from the 3D fields.

        Calculate a weight profile based on the vertical distribution
        of 'gas' and the 3D wind fields with variable(s) 'wind_vars'.
        Remove data below a threshold of avoid division by 0.

        Args:
            data:
                Dataset containing the variables 'u', 'v' and {gas}.
            gas:
                Gas which should be used to weight the 3D wind fields.
            wind_vars:
                Variable name(s) of the wind components.
            threshold:
                Threshold below which the gas values are not used
                to compute the weight profile.
            threshold_unit:
                Unit of the threshold value. Required if threshold is given.

        Returns:
            Dataset containing the weighted wind.
        """
        if wind_vars is None:
            wind_vars = ["u", "v"]
        if gas is None:
            raise ValueError("'gas' must be a string, not 'None'.")

        if isinstance(wind_vars, str):
            wind_vars = [wind_vars]

        if not all(var in data for var in wind_vars):
            raise KeyError(
                f"Not all variables '{wind_vars}' are in the dataset. "
                f"Variables on the dataset include {list(data.data_vars)}"
            )

        # calculate weight profile
        data = self.calculate_weight_profile(
            data, gas=gas, threshold=threshold, threshold_unit=threshold_unit
        )

        for wind_component in wind_vars:
            data[f"{wind_component}_weighted"] = (
                data[f"{gas}_weight"] * data[wind_component]
            ).sum(dim="z")
            data[f"{wind_component}_weighted"].attrs["units"] = "m/s"
            data[f"{wind_component}_weighted"].attrs["description"] = (
                f"{gas} concentration weighted wind speed"
            )

        data.attrs["METHOD"] = f"{gas} concentration weighted wind"

        return data

    def calculate_weight_profile(
        self,
        data: xr.Dataset,
        gas: str,
        threshold: Optional[float] = None,
        threshold_unit: Optional[str] = "kg m-2",
    ) -> xr.Dataset:
        """Calculate a weight profile for 3D wind speed.

        Calculate a weight profile based on the concentration
        distribution of a gas. Remove data below 'threshold' to avoid division by 0.

        Args:
            data:
                Dataset to which the weight profile should be added.
            gas:
                Gas which is used to compute the weight profile.
            threshold:
                Threshold below which the gas values are not used
                to compute the weight profile to avoid division by 0.
            threshold_unit:
                Unit of the threshold value. Required if threshold is given.


        Returns:
            Dataset containing the weight profile.
        """
        if gas not in data.data_vars:
            raise KeyError(
                f"No variable named '{gas}'. "
                f"Variables on the dataset include {list(data.data_vars)}"
            )

        # get current unit of gas
        unit = data[gas].attrs.get("units", "a.u.")
        quantity, _, _ = ucat._unit2quantity(str(unit))

        # convert gas mixing ratios to concentrations to make it pressure dependent
        if quantity not in ["cm", "cM", "cn"]:
            # add pressure to convert mass mixing ratios to concentraions
            data = self.post.add_statistics_variable(data, "phydro")
            data = self.post.add_statistics_variable(data, "phydroh")
            data = self.post.convert_to_concentration(data, gas, to_unit="kg m-3")

        data_integrated = data[gas].integrate(coord="z")

        if threshold is None:
            threshold = float(np.nanquantile(data_integrated, q=0.8))
        else:
            if threshold_unit is None:
                raise ValueError(
                    "threshold_unit must be provided if threshold is given."
                )
            threshold = float(
                ucat.convert_columns(
                    threshold,
                    from_unit=threshold_unit,
                    to_unit=unit,
                    molar_mass=gas,
                )
            )

        # filter out values with low gas concentration to avoid erroneous weights
        data = data.where(data_integrated > threshold)

        # create weight profile
        data[f"{gas}_weight"] = data[gas] / data[gas].sum(dim="z")
        data[f"{gas}_weight"].attrs["units"] = "-"
        data[f"{gas}_weight"].attrs["long_name"] = (
            f"normalised {gas} concentration profile"
        )
        data[f"{gas}_weight"].attrs["threshold"] = f"{threshold} {threshold_unit}"

        return data

    def _circular_mean(
        self, a: Union[xr.DataArray, np.ndarray], axes: list[int]
    ) -> np.ndarray:
        """Calculate circular mean, e.g. of wind direction."""
        return scipy.stats.circmean(a, high=360, nan_policy="omit", axis=axes)

    def _circular_std(
        self, a: Union[xr.DataArray, np.ndarray], axes: list[int]
    ) -> np.ndarray:
        """Calculate circular standard deviation, e.g. of wind direction."""
        return scipy.stats.circstd(a, high=360, nan_policy="omit", axis=axes)


################ other functions ################
################################################################################


def wind_profile_from_given_wind(
    grid: Union[
        np.ndarray,
        ls2d.grid.Grid_equidist,
        ls2d.grid.Grid_linear_stretched,
        ls2d.grid.Grid_stretched,
        ls2d.grid.Grid_stretched_manual,
    ],
    wind_speed: float,
    height: float,
    surface_roughness: float,
    thl: float,
    e: float,
    sensible_heat_flux: float,
    threshold: float,
) -> np.ndarray:
    """Compute a wind profile from a given wind speed at a specific height.

    Args:
        grid: Array of heights above the ground level or LS2D grid object.
        wind_speed: Target wind speed at 'height' (m/s).
        height: Target height at which the wind speed should be 'wind_speed' (m).
        surface_roughness: Surface roughness length (m).
        thl: (Potential) temperature profile (K).
        e: Water vapor flux (kg/s).
        sensible_heat_flux: Sensible heat flux (W/m2).
        threshold: Maximum difference between target 'wind_speed' and
            calculated wind speed.

    Returns:
        Wind profile interpolated to the model grid.
    """
    if isinstance(grid, np.ndarray):
        model_profile = grid
    elif isinstance(
        grid,
        (
            ls2d.grid.Grid_equidist,
            ls2d.grid.Grid_linear_stretched,
            ls2d.grid.Grid_stretched,
            ls2d.grid.Grid_stretched_manual,
        ),
    ):
        model_profile = grid.z

    u_star = calculate_u_star(
        wind_speed=wind_speed,
        z=height,
        z0=surface_roughness,
        t=thl,
        e=e,
        h=sensible_heat_flux,
    )

    wind_profile = get_wind_profile(
        z=model_profile,
        z0=surface_roughness,
        u_star=u_star,
        t=thl,
        e=e,
        h=sensible_heat_flux,
    )

    # check if wind speed at height deviates too much from given value
    z_resolution = np.mean(np.diff(model_profile))
    lower_bound = height - z_resolution / 2
    upper_bound = height + z_resolution / 2

    # create a mask based on whether model_profile is within the bounds
    mask = np.where(
        (model_profile >= lower_bound) & (model_profile <= upper_bound),
        1,
        0,
    ).astype(bool)

    actual_wind_speed = np.mean(wind_profile[mask])

    if np.abs(wind_speed - actual_wind_speed) > threshold:
        logger.info(
            "Wind speed at %s m (%.2f m/s) deviates too much from prescribed "
            "wind speed (%s m/s).\nReturning vertically homogeneous wind profile.",
            height,
            actual_wind_speed,
            wind_speed,
        )
        wind_profile = pre.interpolate_to_model_profile(
            grid=grid, variable_value=wind_speed
        )

    return wind_profile


def get_wind_profile(
    z: Union[int, float, list, np.ndarray],
    z0: float,
    u_star: float,
    t: float,
    l_mo: Optional[float] = None,
    h: Optional[float] = None,
    h_turb: Optional[float] = None,
    e: float = 0,
    displacement_height: float = 0,
    stability_criterion: Literal["MO", "Ri"] = "Ri",
    kappa: float = 0.4,
) -> np.ndarray:
    """Calculate the wind profile based on the logarithmic wind profile equation.

    Give either the sensible heat flux H (in W/m2) or the turbulent sensible
    heat flux (in K m/s).

    Source: Bonan turbulent flux notes. Entries marked with ``*`` are from
    Wyngaard, Turbulence in the Atmosphere.

    Args:
        z: Array of heights above the ground level.
        z0: Roughness length.
        u_star: Friction velocity.
        t: (Potential) temperature (in K).
        l_mo: Monin-Obukhov length.
        h: Sensible heat flux.
        h_turb: Turbulent sensible heat flux.
        e: Water vapor flux.
        displacement_height: Zero-plane displacement height.
        stability_criterion: Stability criterion to use ('MO' or 'Ri').
        kappa: von Kármán constant.

    Returns:
        Wind speed at each height in z.
    """
    z = np.array(z)

    # compute displacement height:
    if displacement_height is None:
        displacement_height = 10 * z0 * (3 / 4)

    # calculate MO length if it is not given
    if l_mo is None:
        if (h is not None) and (h_turb is not None):
            raise ValueError(
                "Only one of sensible heat flux (h) or turbulent sensible heat "
                "flux (h_turb) should be provided, not both."
            )

        if h is not None:
            h_turb_value = calculate_virtual_turbulent_heat_flux(h, e, t=t)
        elif h_turb is not None:
            h_turb_value = h_turb
        else:
            raise ValueError(
                "Either sensible heat flux (h) or turbulent sensible heat "
                "flux (h_turb) should be provided."
            )

        l_mo = calculate_monin_obukhov_length(u_star, t, h_turb_value)

    if stability_criterion == "Ri":
        stability = _determine_stability_ri(float(np.nanmean(z) / l_mo))
    else:
        stability = _determine_stability_mo(l_mo)

    if stability == "stable":
        stability_correction = _correction_stable(z - displacement_height, l_mo)
    elif stability == "neutral":
        stability_correction = np.zeros_like(z, dtype=np.float64)
    elif stability == "unstable":
        stability_correction = _correction_unstable(z - displacement_height, l_mo)
    else:
        raise ValueError(f"Unknown stability: {stability}")

    uz = (u_star / kappa) * (
        np.log((z - displacement_height) / z0) - stability_correction
    )

    uz = _replace_by_majority_sign(uz)

    return uz


def calculate_virtual_temperature(
    t: Union[int, float, list, np.ndarray], q: Union[int, float, list, np.ndarray]
) -> np.ndarray:
    """Calculate the virtual (potential) temperature t_v.

    Args:
        t: (Potential) temperature (in K).
        q: Specific humidity (dimensionless, in kg/kg).

    Returns:
        Virtual temperature.
    """
    t = np.array(t)
    q = np.array(q)
    t_v = t * (1 + 0.608 * q)
    return t_v


def calculate_virtual_turbulent_heat_flux(
    h: float, e: float, t: float, rho: float = 1.225, cp: float = 1003.5
) -> float:
    """Calculate the virtual turbulent heat flux from sensible heat flux H.

    According to: https://doi.org/10.1175/1520-0469(2003)060%3C2372:OTMAVO%3E2.0.CO;2

    Args:
        h: Sensible heat flux (W/m^2).
        e: Water vapor flux (kg/s).
        t: Absolute temperature (K).
        rho: Density of air.
        cp: Specific heat capacity of air at constant pressure (J/(kg*K)).

    Returns:
        float: Virtual turbulent heat flux (K m/s).
    """
    return (1 / t) * (1 / rho) * ((h / cp) + 0.61 * e)


def calculate_monin_obukhov_length(
    u_star: float,
    t0: float,
    h_turb: float,
    kappa: float = 0.4,
    g: float = 9.81,
) -> float:
    """Calculate the Monin-Obukhov length (L).

    Args:
        u_star: Friction velocity (m/s).
        t0: Mean temperature (K).
        h_turb: Virtual turbulent heat flux (K m/s).
        kappa: von Kármán constant.
        g: Acceleration due to gravity.

    Returns:
        float: Monin-Obukhov length (m).
    """
    l_mo = -(u_star**3 * t0) / (kappa * g * h_turb)
    return l_mo


def _determine_stability_mo(l_mo: float, threshold: float = 500) -> str:
    """Determine the stability regime based on the Monin-Obukhov length (L).

    Based on published wind-lidar turbulence error classifications.
    Or from: https://onlinelibrary.wiley.com/doi/full/10.1002/we.1528

    Args:
        l_mo: Monin-Obukhov length (m).
        threshold: Threshold for neutral stability.

    Returns:
        str: Stability regime ("stable", "neutral", or "unstable").
    """
    if l_mo > 0 and threshold > l_mo:
        return "stable"
    elif l_mo < 0 and -threshold < l_mo:
        return "unstable"
    else:
        return "neutral"


def _determine_stability_ri(
    ri: float, upper_threshold: float = -0.1, lower_threshold: float = 0.05
) -> str:
    """Determine the stability regime based on the Richardson number (ri).

    Based on common Richardson-number dispersion-regime thresholds.

    Args:
        ri: Richardson number.
        upper_threshold: Upper threshold for stability classification.
        lower_threshold: Lower threshold for stability classification.

    Returns:
        str: Stability regime ("stable", "neutral", or "unstable").
    """
    stable = np.sum(ri < lower_threshold)
    unstable = np.sum(ri > upper_threshold)
    neutral = np.sum((ri >= lower_threshold) & (ri <= upper_threshold))

    if stable > unstable and stable > neutral:
        return "stable"
    elif unstable > stable and unstable > neutral:
        return "unstable"
    else:
        return "neutral"


def _correction_stable(
    z: Union[float, np.ndarray], l_mo: float, beta: float = 4.8
) -> Union[float, np.ndarray]:
    """Stability correction for stable conditions.

    * 4.8 in Wyngaard (2010) Turbulence in the Atmosphere.

    Args:
        z: Height above the ground level (m).
        l_mo: Monin-Obukhov length (m).
        beta: Stability correction parameter.

    Returns:
        Stability correction term.
    """
    return -beta * z / l_mo


def _correction_unstable(
    z: Union[float, np.ndarray], l_mo: float, alpha: float = 1 / 4, gamma: float = 15
) -> Union[float, np.ndarray]:
    """Stability correction for unstable conditions.

    Args:
        z: Height above the ground level (m).
        l_mo: Monin-Obukhov length (m).
        alpha: Stability correction parameter.
        gamma: Stability correction parameter.

    Returns:
        Stability correction term.
    """
    # according to Wyngaard (2010) Turbulence in the Atmosphere:
    gamma = 3.6
    a = 2 / 3
    b = -1 / 2
    x = (1 + gamma * np.abs(z / l_mo) ** a) ** b

    return x
    x = (1 - gamma * z / l_mo) ** alpha
    return (
        2 * np.log((1 + x) / 2)
        + np.log((1 + x**2) / 2)
        - (2 * np.arctan(x))
        + (np.pi / 2)
    )


def solve_wind_profile(
    u_star: float,
    wind_speed: float,
    z: float,
    z0: float,
    t: float,
    e: float,
    h: Optional[float] = None,
    h_turb: Optional[float] = None,
) -> np.ndarray:
    """Equation to solve for finding friction velocity (u_star).

    Args:
        u_star: Initial guess for friction velocity (m/s).
        wind_speed: Measured wind speed (m/s).
        z: Array of heights above the ground level (m).
        z0: Roughness length (m).
        t: (Potential) temperature profile (in K).
        e: Water vapor flux (kg/s).
        h: Sensible heat flux.
        h_turb: Turbulent sensible heat flux.

    Returns:
        Difference between calculated and measured wind speed.
    """
    return (
        get_wind_profile(z=z, z0=z0, u_star=u_star, t=t, e=e, h=h, h_turb=h_turb)
        - wind_speed
    )


def calculate_u_star(
    wind_speed: float,
    z: float,
    z0: float,
    t: float,
    e: float,
    h: Optional[float] = None,
    h_turb: Optional[float] = None,
    u_star_initial: Optional[float] = None,
) -> float:
    """Compute the friction velocity u* with correction term for stability numerically.

    Give either the sensible heat flux H (in W/m2) or the turbulent sensible
    heat flux (in K m/s).

    Args:
        wind_speed: Measured wind speed (m/s).
        z: Height above ground at which the wind speed was measured (m).
        z0: Roughness length (m).
        t: (Potential) temperature (in K).
        e: Water vapor flux (kg/s).
        h: Sensible heat flux.
        h_turb: Turbulent sensible heat flux.
        u_star_initial: Initial guess of u*. If not provided, it is
            calculated assuming neutrally stable conditions.
    """
    if u_star_initial is None:
        u_star_initial = calculate_u_star_stable(wind_speed, z, z0, kappa=0.4)

    u_star_solution = fsolve(
        solve_wind_profile,
        u_star_initial,
        args=(wind_speed, z, z0, t, e, h, h_turb),
    )
    return u_star_solution[0]


def calculate_u_star_stable(
    wind_speed: float,
    z: float,
    z0: float,
    kappa: float = 0.4,
) -> float:
    """Compute u* for neutrally stable situations.

    Compute u* without a correction term for stability analytically.

    Args:
        wind_speed: Measured wind speed (m/s).
        z: Height above ground at which the wind speed was measured (m).
        z0: Roughness length (m).
        kappa: von Kármán constant.

    Returns:
        float: U* for neutrally stable cases without a correction term for
        atmospheric stability.
    """
    # Calculate friction velocity for neutrally stable case
    u_star = (wind_speed * kappa) / np.log(z / z0)

    return u_star


def _replace_by_majority_sign(arr: np.ndarray) -> np.ndarray:
    """Replace values in an array with 0 if sign does not match the majority sign.

    Replace wind speed in array with 0 if it does not have the same sign as
    the majority of wind values in the array.

    Args:
        arr:
            Array containing the wind speed.

    """
    # Determine the signs of each element (-1 for negative, 1 for positive, 0 for zero)
    signs = np.sign(arr).astype(int)

    # Count the occurrences of each sign
    counts = np.bincount(signs + 1)  # +1 to shift range from [-1, 0, 1] to [0, 1, 2]

    # Determine the majority sign
    majority_sign_index = np.argmax(counts)  # 0 -> -1, 1 -> 0, 2 -> 1
    majority_sign = majority_sign_index - 1  # shift back to original range

    # Create a mask for values with the majority sign
    majority_mask = signs == majority_sign

    # Replace values with different sign than the majority with 0
    arr[~majority_mask] = 0

    return arr


def calculate_effective_wind_speed(
    results: xr.Dataset,
    true_emissions: float,
    gas: str,
    true_emissions_unit: str,
) -> xr.Dataset:
    """Add effective wind speed inferred from true and estimated emissions.

    Args:
        results: Dataset containing ``{gas}_emissions``,
            ``integrated_{gas}_mass``, and ``L_max``.
        true_emissions: Known source strength to compare against.
        gas: Gas name used in the emissions variables and unit conversion.
        true_emissions_unit: Unit of the known source strength.

    Returns:
        Dataset with ``effective_wind_speed`` added.

    Raises:
        ValueError: If estimated emissions do not define units.
    """
    est_emissions_units = results[f"{gas}_emissions"].attrs.get("units", None)
    if est_emissions_units is None:
        raise ValueError(f"No units found for estimated emissions of {gas}.")
    true_emissions_converted = ucat.convert_mass_per_time_unit(
        true_emissions,
        from_unit=true_emissions_unit,
        to_unit=est_emissions_units,
        gas_mass=gas,
    )

    results["effective_wind_speed"] = (
        true_emissions_converted * results["L_max"]
    ) / results[f"integrated_{gas}_mass"]
    results["effective_wind_speed"].attrs["units"] = "m s-1"

    return results


def compute_optical_flow_displacement(
    concentration: xr.DataArray,
    *,
    time_dim: Optional[str] = None,
    x_dim: Optional[str] = None,
    y_dim: Optional[str] = None,
    method: Literal["farneback", "lucas_kanade"] = "farneback",
    farneback_kwargs: Optional[Mapping[str, Any]] = None,
    lucas_kanade_kwargs: Optional[Mapping[str, Any]] = None,
) -> xr.Dataset:
    """Compute frame-to-frame concentration displacement with OpenCV.

    Args:
        concentration:
            Two-dimensional concentration field with a time dimension.
        time_dim:
            Name of the time dimension. Inferred when omitted.
        x_dim:
            Name of the column/zonal dimension. Inferred when omitted.
        y_dim:
            Name of the row/meridional dimension. Inferred when omitted.
        method:
            Optical-flow method. ``"farneback"`` returns dense vectors.
            ``"lucas_kanade"`` returns sparse vectors on a dense grid, with
            ``NaN`` away from successfully tracked feature points.
        farneback_kwargs:
            Optional overrides for ``cv2.calcOpticalFlowFarneback`` parameters.
        lucas_kanade_kwargs:
            Optional overrides for ``cv2.goodFeaturesToTrack`` and
            ``cv2.calcOpticalFlowPyrLK`` parameters.

    Returns:
        Dataset with pixel displacement fields for consecutive frame pairs.
    """
    cv2 = _import_cv2()
    _check_optical_flow_method(method)
    time_dim = _infer_time_dim(concentration, time_dim)
    x_dim, y_dim = _infer_horizontal_dims(concentration, time_dim, x_dim, y_dim)
    concentration_tyx = concentration.transpose(time_dim, y_dim, x_dim)

    if concentration_tyx.sizes[time_dim] < 2:
        raise ValueError("At least two time steps are required for optical flow.")

    farneback_params = _get_farneback_params(farneback_kwargs)
    lucas_kanade_params = _get_lucas_kanade_params(cv2, lucas_kanade_kwargs)
    values = concentration_tyx.values
    n_times = concentration_tyx.sizes[time_dim]
    y_size = concentration_tyx.sizes[y_dim]
    x_size = concentration_tyx.sizes[x_dim]
    displacement_x = np.full((n_times - 1, y_size, x_size), np.nan, dtype=np.float32)
    displacement_y = np.full((n_times - 1, y_size, x_size), np.nan, dtype=np.float32)

    for index in range(n_times - 1):
        previous = np.asarray(values[index], dtype=float)
        current = np.asarray(values[index + 1], dtype=float)
        valid = np.isfinite(previous) & np.isfinite(current)

        if not np.any(valid):
            continue

        previous_frame = _normalise_frame(previous)
        current_frame = _normalise_frame(current)
        if method == "farneback":
            _calculate_farneback_displacement(
                cv2,
                previous_frame,
                current_frame,
                valid,
                displacement_x[index],
                displacement_y[index],
                farneback_params,
            )
        else:
            _calculate_lucas_kanade_displacement(
                cv2,
                previous_frame,
                current_frame,
                valid,
                displacement_x[index],
                displacement_y[index],
                lucas_kanade_params,
            )

    time_values = concentration_tyx[time_dim].values
    time_start = time_values[:-1]
    time_end = time_values[1:]
    interval_time = _calculate_interval_midpoints(time_start, time_end)
    coords = _build_interval_spatial_coords(
        concentration_tyx, time_dim, x_dim, y_dim, interval_time
    )

    displacement = xr.Dataset(
        data_vars={
            "displacement_x": (
                [time_dim, y_dim, x_dim],
                displacement_x,
                {
                    "long_name": "Optical-flow displacement in column direction",
                    "units": "pixel",
                },
            ),
            "displacement_y": (
                [time_dim, y_dim, x_dim],
                displacement_y,
                {
                    "long_name": "Optical-flow displacement in row direction",
                    "units": "pixel",
                },
            ),
            "time_start": ([time_dim], time_start),
            "time_end": ([time_dim], time_end),
        },
        coords=coords,
        attrs={
            "method": f"OpenCV {method.replace('_', '-')} optical flow",
            "time_coordinate": "interval midpoint",
        },
    )
    displacement[time_dim].attrs.update(concentration_tyx[time_dim].attrs)
    displacement["displacement_x"].attrs["positive"] = (
        f"toward increasing {x_dim} index"
    )
    displacement["displacement_y"].attrs["positive"] = (
        f"toward increasing {y_dim} index"
    )

    return displacement


def compute_grid_resolution(
    data: Union[xr.DataArray, xr.Dataset],
    *,
    time_dim: Optional[str] = None,
    x_dim: Optional[str] = None,
    y_dim: Optional[str] = None,
) -> xr.Dataset:
    """Compute metric grid spacing and pixel area for a gridded field.

    Args:
        data:
            Gridded data containing metric ``x``/``y`` coordinates or projected
            coordinate variables.
        time_dim:
            Optional time dimension to exclude from horizontal-dimension
            inference.
        x_dim:
            Name of the column/zonal dimension. Inferred when omitted.
        y_dim:
            Name of the row/meridional dimension. Inferred when omitted.

    Returns:
        Dataset containing ``dx``, ``dy``, and ``pixel_area`` on the horizontal
        grid.
    """
    time_dim = _infer_time_dim(data, time_dim, required=False)
    x_dim, y_dim = _infer_horizontal_dims(data, time_dim, x_dim, y_dim)
    spatial = _as_spatial_dataset(data, time_dim, x_dim, y_dim)

    x_coord, y_coord = _get_projected_xy_coordinates(spatial, x_dim, y_dim)
    dx_values, dy_values = _calculate_directional_spacing(x_coord, y_coord)
    coords = _build_spatial_coords(spatial, x_dim, y_dim)

    resolution = xr.Dataset(
        data_vars={
            "dx": (
                [y_dim, x_dim],
                dx_values,
                {"long_name": "Grid spacing in column direction", "units": "m"},
            ),
            "dy": (
                [y_dim, x_dim],
                dy_values,
                {"long_name": "Grid spacing in row direction", "units": "m"},
            ),
        },
        coords=coords,
    )
    resolution["pixel_area"] = _get_pixel_area(
        spatial, x_dim, y_dim, dx_values, dy_values
    )

    return resolution


def calculate_optical_flow_effective_wind_speed(
    data: xr.Dataset,
    variable: str,
    *,
    time_dim: Optional[str] = None,
    x_dim: Optional[str] = None,
    y_dim: Optional[str] = None,
    method: Literal["farneback", "lucas_kanade"] = "farneback",
    farneback_kwargs: Optional[Mapping[str, Any]] = None,
    lucas_kanade_kwargs: Optional[Mapping[str, Any]] = None,
) -> xr.Dataset:
    """Calculate scalar effective wind speed from concentration optical flow.

    Args:
        data:
            Dataset containing a time series of concentration fields.
        variable:
            Name of the concentration data variable.
        time_dim:
            Name of the time dimension. Inferred when omitted.
        x_dim:
            Name of the column/zonal dimension. Inferred when omitted.
        y_dim:
            Name of the row/meridional dimension. Inferred when omitted.
        method:
            Optical-flow method passed to ``compute_optical_flow_displacement``.
        farneback_kwargs:
            Optional overrides for ``cv2.calcOpticalFlowFarneback`` parameters.
        lucas_kanade_kwargs:
            Optional overrides for ``cv2.goodFeaturesToTrack`` and
            ``cv2.calcOpticalFlowPyrLK`` parameters.

    Returns:
        Dataset with one concentration-weighted effective wind speed per
        consecutive time interval.
    """
    if variable not in data.data_vars:
        raise ValueError(f"Variable '{variable}' not found in dataset.")

    concentration = data[variable].load()
    time_dim = _infer_time_dim(concentration, time_dim)
    x_dim, y_dim = _infer_horizontal_dims(concentration, time_dim, x_dim, y_dim)
    flow = compute_optical_flow_displacement(
        concentration,
        time_dim=time_dim,
        x_dim=x_dim,
        y_dim=y_dim,
        method=method,
        farneback_kwargs=farneback_kwargs,
        lucas_kanade_kwargs=lucas_kanade_kwargs,
    )
    resolution = compute_grid_resolution(
        data.load(), time_dim=time_dim, x_dim=x_dim, y_dim=y_dim
    )

    concentration_tyx = concentration.transpose(time_dim, y_dim, x_dim)
    dt_values = _calculate_interval_seconds(concentration_tyx[time_dim].values)
    dt = xr.DataArray(dt_values, dims=[time_dim], coords={time_dim: flow[time_dim]})

    speed: xr.DataArray = (
        ((flow["displacement_x"] * resolution["dx"]) / dt) ** 2
        + ((flow["displacement_y"] * resolution["dy"]) / dt) ** 2
    ) ** 0.5
    speed.attrs.update({"long_name": "Optical-flow pixel speed", "units": "m s-1"})

    weights = _calculate_pair_weights(
        concentration_tyx, resolution["pixel_area"], time_dim, x_dim, y_dim, flow
    )
    valid: xr.DataArray = speed.notnull() & weights.notnull() & (weights > 0)
    weighted_speed = speed.where(valid) * weights.where(valid)
    weight_sum = weights.where(valid).sum(dim=[y_dim, x_dim], skipna=True)
    effective_speed = weighted_speed.sum(dim=[y_dim, x_dim], skipna=True) / weight_sum
    effective_speed = effective_speed.where(weight_sum > 0)
    effective_speed.attrs.update(
        {
            "long_name": "Concentration-weighted optical-flow effective wind speed",
            "units": "m s-1",
        }
    )

    result = xr.Dataset(
        data_vars={
            "effective_wind_speed": effective_speed,
            "dt": (
                [time_dim],
                dt_values,
                {"long_name": "Interval duration", "units": "s"},
            ),
            "time_start": flow["time_start"],
            "time_end": flow["time_end"],
            "valid_pixel_count": valid.sum(dim=[y_dim, x_dim]),
            "weight_sum": weight_sum,
        },
        coords={time_dim: flow[time_dim]},
        attrs={
            "method": (
                "Concentration-weighted speed from OpenCV "
                f"{method.replace('_', '-')} optical flow"
            ),
            "source_variable": variable,
            "weight": "positive finite pair-mean concentration times pixel area",
            "time_coordinate": "interval midpoint",
        },
    )
    result["weight_sum"].attrs["long_name"] = "Sum of valid concentration-area weights"
    result["valid_pixel_count"].attrs["long_name"] = "Number of valid weighted pixels"

    return result


def _check_optical_flow_method(method: str) -> None:
    """Validate the selected OpenCV optical-flow method."""
    if method not in {"farneback", "lucas_kanade"}:
        raise ValueError(
            "Unsupported optical-flow method. Use 'farneback' or 'lucas_kanade'."
        )


def _import_cv2() -> ModuleType:
    """Import OpenCV lazily for optional optical-flow support."""
    try:
        import cv2
    except ImportError as error:
        raise ImportError(
            "OpenCV is required for optical-flow wind calculations. "
            "Install it with `pip install -e '.[optical-flow]'`."
        ) from error

    return cv2


def _get_farneback_params(
    farneback_kwargs: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Return validated Farneback optical-flow parameters."""
    params: dict[str, Any] = {
        "pyr_scale": 0.5,
        "levels": 3,
        "winsize": 15,
        "iterations": 3,
        "poly_n": 5,
        "poly_sigma": 1.2,
        "flags": 0,
    }
    if farneback_kwargs is None:
        return params

    unknown = set(farneback_kwargs) - set(params)
    if unknown:
        raise ValueError(
            "Unsupported Farneback parameters: " + ", ".join(sorted(unknown))
        )

    params.update(farneback_kwargs)
    return params


def _calculate_farneback_displacement(
    cv2: ModuleType,
    previous_frame: np.ndarray,
    current_frame: np.ndarray,
    valid: np.ndarray,
    displacement_x: np.ndarray,
    displacement_y: np.ndarray,
    params: Mapping[str, Any],
) -> None:
    """Calculate dense Farneback displacement and write valid pixels."""
    flow = cv2.calcOpticalFlowFarneback(
        previous_frame,
        current_frame,
        None,
        params["pyr_scale"],
        params["levels"],
        params["winsize"],
        params["iterations"],
        params["poly_n"],
        params["poly_sigma"],
        params["flags"],
    )
    displacement_x[:] = np.where(valid, flow[..., 0], np.nan)
    displacement_y[:] = np.where(valid, flow[..., 1], np.nan)


def _get_lucas_kanade_params(
    cv2: ModuleType,
    lucas_kanade_kwargs: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Return validated Lucas-Kanade feature and tracking parameters."""
    params: dict[str, Any] = {
        "maxCorners": 500,
        "qualityLevel": 0.01,
        "minDistance": 3,
        "blockSize": 7,
        "winSize": (15, 15),
        "maxLevel": 2,
        "criteria": (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            10,
            0.03,
        ),
        "flags": 0,
        "minEigThreshold": 1e-4,
    }
    if lucas_kanade_kwargs is None:
        return params

    unknown = set(lucas_kanade_kwargs) - set(params)
    if unknown:
        raise ValueError(
            "Unsupported Lucas-Kanade parameters: " + ", ".join(sorted(unknown))
        )

    params.update(lucas_kanade_kwargs)
    return params


def _calculate_lucas_kanade_displacement(
    cv2: ModuleType,
    previous_frame: np.ndarray,
    current_frame: np.ndarray,
    valid: np.ndarray,
    displacement_x: np.ndarray,
    displacement_y: np.ndarray,
    params: Mapping[str, Any],
) -> None:
    """Track sparse feature points and write their pixel displacements."""
    previous_frame = _to_uint8_frame(previous_frame)
    current_frame = _to_uint8_frame(current_frame)
    feature_params = {
        name: params[name]
        for name in ("maxCorners", "qualityLevel", "minDistance", "blockSize")
    }
    lk_params = {
        name: params[name]
        for name in ("winSize", "maxLevel", "criteria", "flags", "minEigThreshold")
    }
    mask = np.where(valid, 255, 0).astype(np.uint8)
    previous_points = cv2.goodFeaturesToTrack(
        previous_frame, mask=mask, **feature_params
    )
    if previous_points is None:
        return

    current_points, status, _ = cv2.calcOpticalFlowPyrLK(
        previous_frame, current_frame, previous_points, None, **lk_params
    )
    if current_points is None or status is None:
        return

    previous_points_2d = previous_points.reshape(-1, 2)
    current_points_2d = current_points.reshape(-1, 2)
    tracked = status.reshape(-1).astype(bool)
    tracked &= np.all(np.isfinite(previous_points_2d), axis=1)
    tracked &= np.all(np.isfinite(current_points_2d), axis=1)
    if not np.any(tracked):
        return

    rows = np.rint(previous_points_2d[tracked, 1]).astype(int)
    cols = np.rint(previous_points_2d[tracked, 0]).astype(int)
    in_bounds = (
        (rows >= 0) & (rows < valid.shape[0]) & (cols >= 0) & (cols < valid.shape[1])
    )
    rows = rows[in_bounds]
    cols = cols[in_bounds]
    previous_tracked = previous_points_2d[tracked][in_bounds]
    current_tracked = current_points_2d[tracked][in_bounds]
    finite_source = valid[rows, cols]

    displacement_x[rows[finite_source], cols[finite_source]] = (
        current_tracked[finite_source, 0] - previous_tracked[finite_source, 0]
    )
    displacement_y[rows[finite_source], cols[finite_source]] = (
        current_tracked[finite_source, 1] - previous_tracked[finite_source, 1]
    )


def _normalise_frame(frame: np.ndarray) -> np.ndarray:
    """Scale one frame to finite float32 values in the interval [0, 1]."""
    finite = np.isfinite(frame)
    if not np.any(finite):
        return np.zeros(frame.shape, dtype=np.float32)

    min_value = float(np.nanmin(frame[finite]))
    max_value = float(np.nanmax(frame[finite]))
    if np.isclose(max_value, min_value):
        return np.zeros(frame.shape, dtype=np.float32)

    normalised = (frame - min_value) / (max_value - min_value)
    return np.where(finite, normalised, 0.0).astype(np.float32)


def _to_uint8_frame(frame: np.ndarray) -> np.ndarray:
    """Convert a normalized frame to the 8-bit format required by PyrLK."""
    frame_uint8 = np.clip(frame, 0.0, 1.0) * 255.0
    return frame_uint8.astype(np.uint8)


@overload
def _infer_time_dim(
    data: Union[xr.DataArray, xr.Dataset],
    time_dim: Optional[str],
    *,
    required: Literal[False],
) -> Optional[str]: ...


@overload
def _infer_time_dim(
    data: Union[xr.DataArray, xr.Dataset],
    time_dim: Optional[str],
    *,
    required: Literal[True] = True,
) -> str: ...


def _infer_time_dim(
    data: Union[xr.DataArray, xr.Dataset],
    time_dim: Optional[str],
    *,
    required: bool = True,
) -> Optional[str]:
    """Infer the time dimension from a DataArray or Dataset."""
    if time_dim is not None:
        if time_dim not in data.sizes:
            raise ValueError(f"Time dimension '{time_dim}' not found.")
        return time_dim

    candidates = [str(dim) for dim in data.sizes if "time" in str(dim).lower()]
    if len(candidates) == 1:
        return candidates[0]
    if not required:
        return None
    if len(candidates) > 1:
        raise ValueError(
            "Multiple possible time dimensions found. Pass time_dim explicitly."
        )
    raise ValueError("Could not infer time dimension. Pass time_dim explicitly.")


def _infer_horizontal_dims(
    data: Union[xr.DataArray, xr.Dataset],
    time_dim: Optional[str],
    x_dim: Optional[str],
    y_dim: Optional[str],
) -> tuple[str, str]:
    """Infer column and row dimensions."""
    if (x_dim is None) != (y_dim is None):
        raise ValueError("Pass both x_dim and y_dim, or neither.")

    if x_dim is not None and y_dim is not None:
        missing = [dim for dim in (x_dim, y_dim) if dim not in data.sizes]
        if missing:
            raise ValueError("Spatial dimensions not found: " + ", ".join(missing))
        return x_dim, y_dim

    candidates = (
        ("x", "y"),
        ("lon", "lat"),
        ("longitude", "latitude"),
        ("nobs", "nrows"),
    )
    for x_candidate, y_candidate in candidates:
        if x_candidate in data.sizes and y_candidate in data.sizes:
            return x_candidate, y_candidate

    horizontal_dims = [str(dim) for dim in data.sizes if dim != time_dim]
    if len(horizontal_dims) != 2:
        raise ValueError(
            "Could not infer two horizontal dimensions. Pass x_dim and y_dim."
        )

    return horizontal_dims[1], horizontal_dims[0]


def _build_interval_spatial_coords(
    concentration: xr.DataArray,
    time_dim: str,
    x_dim: str,
    y_dim: str,
    interval_time: np.ndarray,
) -> dict[str, Any]:
    """Build coordinates for dense interval output."""
    coords = _build_spatial_coords(concentration, x_dim, y_dim)
    coords[time_dim] = interval_time

    for coord_name, coord in concentration.coords.items():
        if not isinstance(coord_name, str):
            continue
        if coord_name in coords or time_dim in coord.dims:
            continue
        if set(coord.dims).issubset({y_dim, x_dim}):
            coords[coord_name] = coord

    return coords


def _build_spatial_coords(
    data: Union[xr.DataArray, xr.Dataset], x_dim: str, y_dim: str
) -> dict[str, Any]:
    """Build horizontal coordinates preserving dimension coordinates."""
    coords: dict[str, Any] = {}
    for dim in (y_dim, x_dim):
        if dim in data.coords:
            coords[dim] = data.coords[dim]
        else:
            coords[dim] = np.arange(data.sizes[dim])

    for coord_name in ("x", "y", "lon", "lat", "longitude", "latitude"):
        if coord_name in coords or coord_name not in data.coords:
            continue
        coord = data.coords[coord_name]
        if set(coord.dims).issubset({y_dim, x_dim}):
            coords[coord_name] = coord

    return coords


def _calculate_interval_midpoints(
    time_start: np.ndarray, time_end: np.ndarray
) -> np.ndarray:
    """Calculate midpoint labels for interval coordinates."""
    if np.issubdtype(time_start.dtype, np.datetime64):
        return time_start + (time_end - time_start) / 2
    return (time_start + time_end) / 2


def _calculate_interval_seconds(time_values: np.ndarray) -> np.ndarray:
    """Calculate consecutive time deltas in seconds."""
    if np.issubdtype(time_values.dtype, np.datetime64):
        return np.diff(time_values).astype("timedelta64[ns]").astype(float) / 1e9
    if np.issubdtype(time_values.dtype, np.timedelta64):
        return np.diff(time_values).astype("timedelta64[ns]").astype(float) / 1e9
    return np.diff(time_values).astype(float)


def _as_spatial_dataset(
    data: Union[xr.DataArray, xr.Dataset],
    time_dim: Optional[str],
    x_dim: str,
    y_dim: str,
) -> xr.Dataset:
    """Convert input to a two-dimensional spatial dataset view."""
    if isinstance(data, xr.DataArray):
        name = data.name or "_field"
        dataset = data.to_dataset(name=name)
    else:
        dataset = data

    if time_dim is None or time_dim not in dataset.sizes:
        return dataset

    return dataset.isel({time_dim: 0}, drop=True)


def _get_projected_xy_coordinates(
    data: xr.Dataset, x_dim: str, y_dim: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return projected x/y coordinate fields in meters."""
    if "x" in data and "y" in data and set(data["x"].dims) == {y_dim, x_dim}:
        return (
            data["x"].transpose(y_dim, x_dim).values.astype(float),
            data["y"].transpose(y_dim, x_dim).values.astype(float),
        )

    if x_dim == "x" and y_dim == "y" and "x" in data.coords and "y" in data.coords:
        x_values = data.coords["x"].values.astype(float)
        y_values = data.coords["y"].values.astype(float)
        x_grid, y_grid = np.meshgrid(x_values, y_values, indexing="xy")
        return x_grid, y_grid

    raise ValueError(
        "Metric projected x/y coordinates are required to convert optical-flow "
        "displacement to m s-1. Provide projected x/y variables or use x/y "
        "dimensions with meter coordinates."
    )


def _calculate_directional_spacing(
    x_coord: np.ndarray, y_coord: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate directional spacing from projected coordinate fields."""
    x_col_gradient = np.gradient(x_coord, axis=1)
    y_col_gradient = np.gradient(y_coord, axis=1)
    x_row_gradient = np.gradient(x_coord, axis=0)
    y_row_gradient = np.gradient(y_coord, axis=0)

    dx = np.hypot(x_col_gradient, y_col_gradient)
    dy = np.hypot(x_row_gradient, y_row_gradient)

    if not np.all(np.isfinite(dx)) or not np.all(np.isfinite(dy)):
        raise ValueError("Projected x/y coordinates produced non-finite resolution.")
    if np.nanmin(dx) <= 0 or np.nanmin(dy) <= 0:
        raise ValueError("Projected x/y coordinates must define positive spacing.")

    return dx, dy


def _get_pixel_area(
    data: xr.Dataset,
    x_dim: str,
    y_dim: str,
    dx_values: np.ndarray,
    dy_values: np.ndarray,
) -> xr.DataArray:
    """Return pixel area, using ddeq for georeferenced grids when possible."""
    if "pixel_area" in data:
        return data["pixel_area"].transpose(y_dim, x_dim)
    if "pixel_area" in data.coords:
        return data.coords["pixel_area"].transpose(y_dim, x_dim)

    if {"lon", "lat", "x", "y"}.issubset(data.variables):
        epsg = _get_projected_epsg(data)
        if epsg is not None:
            try:
                area_data = ddeq.plume_coords.compute_pixel_areas(
                    data.copy(), ccrs.epsg(epsg)
                )
                if "pixel_area" in area_data:
                    return area_data["pixel_area"].transpose(y_dim, x_dim)
            except Exception as error:
                logger.debug("Could not compute pixel areas with ddeq: %s", error)

    return xr.DataArray(
        dx_values * dy_values,
        dims=[y_dim, x_dim],
        coords=_build_spatial_coords(data, x_dim, y_dim),
        attrs={"long_name": "Pixel area", "units": "m2"},
    )


def _get_projected_epsg(data: xr.Dataset) -> Optional[int]:
    """Return projected EPSG code from x/y coordinate attributes."""
    for name in ("x", "y"):
        if name not in data:
            continue
        epsg = data[name].attrs.get("epsg")
        if epsg is None:
            continue
        crs = CRS.from_epsg(int(epsg))
        if crs.is_projected:
            return int(epsg)
    return None


def _calculate_pair_weights(
    concentration: xr.DataArray,
    pixel_area: xr.DataArray,
    time_dim: str,
    x_dim: str,
    y_dim: str,
    flow: xr.Dataset,
) -> xr.DataArray:
    """Calculate positive pair-mean concentration-area weights."""
    previous = concentration.isel({time_dim: slice(None, -1)})
    current = concentration.isel({time_dim: slice(1, None)})
    previous = previous.assign_coords({time_dim: flow[time_dim]})
    current = current.assign_coords({time_dim: flow[time_dim]})
    pair_mean = (previous + current) / 2
    finite_pair = np.isfinite(previous) & np.isfinite(current)
    weights = pair_mean.where(finite_pair & (pair_mean > 0)) * pixel_area
    weights.attrs.update(
        {
            "long_name": "Pair-mean concentration times pixel area",
            "description": "Only positive finite concentration pairs are retained.",
        }
    )
    return weights.transpose(time_dim, y_dim, x_dim)


def ramp(x: Union[int, float, np.ndarray]) -> Union[int, float, np.ndarray]:
    """Ramp function R(x): returns x if x > 0 else 0."""
    return np.maximum(x, 0)


def calculate_smagorinsky_constant(
    delta: Union[int, float, np.ndarray],
    l_mo: Union[int, float, np.ndarray],
    z: Union[int, float, np.ndarray],
    kappa: float = 0.4,
    c0: float = 0.135,
    alpha: float = 0.135,
    n: int = 3,
) -> Union[int, float, np.ndarray]:
    """Compute the Smagorinsky constant.

    Computes the Smagorinsky constant according to Eq. (17) in
    https://doi.org/10.1175/1520-0469(2003)060%3C2372:OTMAVO%3E2.0.CO;2.

    Args:
        delta: Characteristic scale (Δ). For simulations: resolution of the
            model, in m.
        l_mo: Obukhov length (l).
        z: Height above ground (in m).
        kappa: von Kármán constant.
        c0: Base concentration.
        alpha: Model parameter.
        n: Model exponent.

    Returns:
        Computed Smagorinsky constant c_s.
    """
    term1 = (1 + (c0 / alpha) * ramp(delta / l_mo)) ** -1
    term2 = (1 + (c0 * delta / (kappa * z)) ** n) ** (-1 / n)
    return c0 * term1 * term2


# ============================================================================
# Physics functions
# ============================================================================


def calculate_coriolis_force(lat: float) -> float:
    """Calculate the Coriolis force at a given latitude.

    Args:
        lat: Latitude in degrees.

    Returns:
        Coriolis parameter f_c (s^-1).
    """
    return 2 * EARTH_ROTATION_RATE * np.sin(np.radians(lat))


def calculate_richardson_number(
    stats: xr.Dataset,
    threshold: Optional[float] = 1e-10,
) -> xr.Dataset:
    """Calculate the Richardson flux number.

    The Richardson number is the ratio of buoyant production to shear
    production of turbulence kinetic energy.

    Args:
        stats: Dataset containing u_w, u_grad, v_w, v_grad, and b_w variables.
        threshold: Threshold for denominator to avoid division by near-zero.

    Returns:
        Dataset with Richardson number added as 'Ri' variable.
    """
    denom = stats.u_w * stats.u_grad + stats.v_w * stats.v_grad  # m^2 s^-3
    rf = stats.b_w / denom
    attrs = {
        "long_name": "Richardson number",
        "units": "1",
        "description": (
            "Ratio of buoyant production to shear production of "
            "turbulence kinetic energy"
        ),
    }

    da = xr.DataArray(rf, dims=rf.dims, coords=rf.coords, attrs=attrs)
    stats["Ri"] = da.where(np.abs(denom) > threshold)
    return stats


def calculate_spatial_psd_2d(
    field: np.ndarray, dx: float, dy: float, angular: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a variance-preserving 2D PSD using frequency/wavenumber coordinates.

    The returned psd2d satisfies approximately:

        psd2d.sum() == np.nanvar(field)

    after removal of the horizontal mean.

    Args:
        field: Two-dimensional input field.
        dx: Grid spacing in the x direction.
        dy: Grid spacing in the y direction.
        angular: Return angular wavenumbers instead of cycles per physical unit.

    Returns:
        Variance-preserving PSD and the x/y frequency coordinates.
    """
    from numpy.fft import fft2, fftfreq, fftshift

    field = np.asarray(field)
    nx, ny = field.shape

    field_prime = field - np.nanmean(field)
    field_prime = np.nan_to_num(field_prime, nan=0.0)

    field_fft = fft2(field_prime)

    # Variance-preserving normalization
    psd2d = np.abs(field_fft) ** 2 / (nx * ny) ** 2

    # Shift zero mode to center
    psd2d = fftshift(psd2d)

    # Physical frequency coordinates
    kx = fftshift(fftfreq(nx, d=dx))
    ky = fftshift(fftfreq(ny, d=dy))

    # Optional conversion from cycles/unit to radians/unit
    if angular:
        kx = 2.0 * np.pi * kx
        ky = 2.0 * np.pi * ky

    return psd2d, kx, ky


def radial_spectrum_from_spatial_psd(
    psd2d: np.ndarray,
    kx: np.ndarray,
    ky: np.ndarray,
    nbins: Optional[int] = None,
    bin_edges: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a variance-preserving radial spectrum from a 2D PSD.

    This function bins the 2D PSD into annuli in wavenumber space and sums
    the PSD values inside each annulus.

    Therefore:

        radial_psd.sum() ≈ psd2d.sum()

    assuming all Fourier modes are included in the radial bins.

    Args:
        psd2d: Variance-preserving two-dimensional PSD, usually shifted.
        kx: Frequency or wavenumber coordinates corresponding to the first axis.
        ky: Frequency or wavenumber coordinates corresponding to the second axis.
        nbins: Number of radial bins.
        bin_edges: Custom radial bin edges. Overrides ``nbins`` when provided.

    Returns:
        Radial bin centers, variance-preserving radial spectrum, and bin counts.

    Raises:
        ValueError: If coordinate lengths do not match the PSD shape.
    """
    psd2d = np.asarray(psd2d)
    kx = np.asarray(kx)
    ky = np.asarray(ky)

    nx, ny = psd2d.shape

    if kx.size != nx:
        raise ValueError("kx length must match psd2d.shape[0].")

    if ky.size != ny:
        raise ValueError("ky length must match psd2d.shape[1].")

    kx, ky = np.meshgrid(kx, ky, indexing="ij")
    k_radius = np.sqrt(kx**2 + ky**2)

    if bin_edges is None:
        n_bins = min(nx, ny) // 2 if nbins is None else nbins
        k_max = min(np.max(np.abs(kx)), np.max(np.abs(ky)))
        bin_edges_array = np.linspace(0.0, k_max, n_bins + 1)
    else:
        bin_edges_array = np.asarray(bin_edges)
        n_bins = len(bin_edges_array) - 1

    radial_psd = np.zeros(n_bins)
    counts = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (k_radius >= bin_edges_array[i]) & (
                k_radius <= bin_edges_array[i + 1]
            )
        else:
            mask = (k_radius >= bin_edges_array[i]) & (
                k_radius < bin_edges_array[i + 1]
            )

        radial_psd[i] = np.sum(psd2d[mask])
        counts[i] = np.count_nonzero(mask)

    k_radial = 0.5 * (bin_edges_array[:-1] + bin_edges_array[1:])

    return k_radial, radial_psd, counts


def _tke_spatial_spectrum_core(
    u_2d: np.ndarray,
    v_2d: np.ndarray,
    dx: float,
    dy: float,
    w_2d: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a radial TKE spectrum for a single two-dimensional plane.

    Receives raw NumPy arrays from xr.apply_ufunc.
    """
    # 1. Compute 2D PSDs
    psd2d_u, kx, ky = calculate_spatial_psd_2d(u_2d, dx, dy)
    psd2d_v, _, _ = calculate_spatial_psd_2d(v_2d, dx, dy)

    total_psd2d = psd2d_u + psd2d_v

    if w_2d is not None:
        psd2d_w, _, _ = calculate_spatial_psd_2d(w_2d, dx, dy)
        total_psd2d += psd2d_w

    tke_psd2d = 0.5 * total_psd2d

    # 2. Radially bin
    k_radial, radial_tke, _ = radial_spectrum_from_spatial_psd(tke_psd2d, kx, ky)
    return k_radial, radial_tke


def _tke_spatial_spectrum_core_array(
    u_2d: np.ndarray, v_2d: np.ndarray, *args: np.ndarray, dx: float, dy: float
) -> np.ndarray:
    """Return only radial TKE values for ``xr.apply_ufunc``."""
    w_2d: Optional[np.ndarray] = args[0] if args else None
    _, radial_tke = _tke_spatial_spectrum_core(u_2d, v_2d, dx, dy, w_2d=w_2d)
    return radial_tke


def calculate_tke_spatial_profile_dask(
    ds_u: xr.Dataset,
    ds_v: xr.Dataset,
    ds_w: Optional[xr.Dataset] = None,
    time_idx: int = -1,
) -> xr.Dataset:
    """Compute the TKE spatial spectrum for every vertical level using Dask.

    Assumes staggered-grid inputs:
        u(time, z,  xh, y)
        v(time, z,  x,  yh)
        w(time, zh, x,  y)

    The fields are collocated to the scalar grid:
        (z, x, y)

    before applying the 2D FFT spectrum.

    Args:
        ds_u: Dataset containing the u wind component.
        ds_v: Dataset containing the v wind component.
        ds_w: Optional dataset containing the w wind component.
        time_idx: Time index to analyse.

    Returns:
        Dataset containing the vertical profile of radial TKE spectra.
    """
    dx = float(np.median(ds_v.x.diff(dim="x").values))
    dy = float(np.median(ds_u.y.diff(dim="y").values))

    z_levels = ds_u.z.values

    u_slice = ds_u.u.isel(time=time_idx)
    v_slice = ds_v.v.isel(time=time_idx)

    # interpolate to same grid
    u_slice = 0.5 * (u_slice + u_slice.roll(xh=-1, roll_coords=False))
    u_slice = u_slice.assign_coords(xh=v_slice.x.values).rename({"xh": "x"})

    v_slice = 0.5 * (v_slice + v_slice.roll(yh=-1, roll_coords=False))
    v_slice = v_slice.assign_coords(yh=u_slice.y.values).rename({"yh": "y"})

    u_slice = u_slice.transpose("z", "x", "y")
    v_slice = v_slice.transpose("z", "x", "y")

    u_slice = u_slice.chunk({"z": 1, "x": -1, "y": -1})
    v_slice = v_slice.chunk({"z": 1, "x": -1, "y": -1})

    input_args = [u_slice, v_slice]
    input_core_dims = [["x", "y"], ["x", "y"]]

    # add w if provided
    if ds_w is not None:
        w_slice = ds_w.w.isel(time=time_idx)
        w_slice = (0.5 * (w_slice + w_slice.shift(zh=-1))).ffill(dim="zh")
        w_slice = w_slice.assign_coords(zh=u_slice.z.values).rename({"zh": "z"})

        w_slice = w_slice.transpose("z", "x", "y")
        w_slice = w_slice.chunk({"z": 1, "x": -1, "y": -1})

        input_args.append(w_slice)
        input_core_dims.append(["x", "y"])

    # Precompute output wavenumber coordinate
    dummy_nx = len(ds_v.x)
    dummy_ny = len(ds_u.y)

    _, dummy_kx, dummy_ky = calculate_spatial_psd_2d(
        np.zeros((dummy_nx, dummy_ny)), dx, dy
    )

    wavenumbers, _, _ = radial_spectrum_from_spatial_psd(
        np.zeros((dummy_nx, dummy_ny)), dummy_kx, dummy_ky
    )

    tke_da = xr.apply_ufunc(
        _tke_spatial_spectrum_core_array,
        *input_args,
        kwargs={"dx": dx, "dy": dy},
        input_core_dims=input_core_dims,
        output_core_dims=[["wavenumber"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[u_slice.dtype],
        dask_gufunc_kwargs={
            "output_sizes": {"wavenumber": len(wavenumbers)},
            "allow_rechunk": True,
        },
    )

    # Assign output coordinates
    tke_da = tke_da.rename("tke_spectrum")
    tke_da = tke_da.assign_coords(
        z=z_levels,
        wavenumber=wavenumbers,
    )
    tke_da.attrs.update(
        {
            "long_name": "TKE spatial spectrum",
            "units": "m2 s-2",
            "description": (
                "Variance-preserving radial TKE spectrum; "
                "sum over wavenumber bins gives TKE variance."
            ),
        }
    )
    tke_da["z"].attrs["units"] = "m"
    tke_da["wavenumber"].attrs.update(
        {
            "long_name": "Wavenumber",
            "units": "m-1",
            "description": "Spatial frequency in cycles per meter.",
        }
    )
    tke_ds = tke_da.to_dataset(name="tke_spectrum")

    # add metadata
    tke_ds.attrs["dx"] = dx
    tke_ds.attrs["dy"] = dy
    tke_ds.attrs["components"] = "u, v" + (", w" if ds_w is not None else "")

    return tke_ds


def calculate_temporal_variance_psd(
    signal: np.ndarray,
    fs: float,
    nperseg: int = 256,
    noverlap: Optional[int] = None,
    detrend: str = "linear",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a variance-preserving temporal PSD using Welch's method.

    The returned psd satisfies approximately:

        psd.sum() * (f[1] - f[0]) ≈ np.var(signal)

    Args:
        signal: Time series at a single point.
        fs: Sampling frequency in Hz.
        nperseg: Length of each Welch segment.
        noverlap: Number of overlapping samples between segments.
        detrend: Detrending applied to each segment.

    Returns:
        Frequency bins and variance-preserving PSD.

    Raises:
        ValueError: If the signal is shorter than one Welch segment.
    """
    signal = np.asarray(signal, dtype=float)

    # Remove NaNs
    signal = signal[~np.isnan(signal)]

    if len(signal) < nperseg:
        raise ValueError(
            f"Signal length {len(signal)} is shorter than nperseg={nperseg}."
        )

    f, psd = welch(
        signal,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=detrend,
        scaling="density",  # integral over f gives variance
    )

    return f, psd


def _tke_temporal_spectrum_core(
    u_t: np.ndarray,
    v_t: np.ndarray,
    fs: float,
    w_t: Optional[np.ndarray] = None,
    nperseg: int = 256,
    noverlap: Optional[int] = None,
    detrend: str = "linear",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a TKE temporal spectrum from one-dimensional time series.

    This function computes the temporal PSD of the velocity time series
    at a single (x, y, z) point.

    Returns:
        Frequency bins and TKE temporal PSD.
    """
    f, p_uu = calculate_temporal_variance_psd(u_t, fs, nperseg, noverlap, detrend)
    _, p_vv = calculate_temporal_variance_psd(v_t, fs, nperseg, noverlap, detrend)

    total_psd = p_uu + p_vv

    if w_t is not None:
        _, p_ww = calculate_temporal_variance_psd(w_t, fs, nperseg, noverlap, detrend)
        total_psd += p_ww

    tke_psd = 0.5 * total_psd

    return f, tke_psd


def calculate_tke_temporal_profile(
    ds_u: xr.Dataset,
    ds_v: xr.Dataset,
    ds_w: Optional[xr.Dataset] = None,
    x_idx: int = 0,
    y_idx: int = 0,
    fs: float = 1.0,
    nperseg: int = 256,
    noverlap: Optional[int] = None,
    detrend: str = "linear",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the TKE temporal spectrum at every vertical level.

    The calculation uses velocity time series for a single (x, y) point in
    the domain.

    This is the temporal analogue of calculate_tke_profile:
    instead of a 2D spatial FFT over (x, y) at fixed time,
    it applies Welch's method over the time axis at fixed (x, y).

    Args:
        ds_u: Dataset containing the u wind component.
        ds_v: Dataset containing the v wind component.
        ds_w: Optional dataset containing the w wind component.
        x_idx: Index along the x axis to extract the time series from.
        y_idx: Index along the y axis to extract the time series from.
        fs: Sampling frequency in Hz.
        nperseg: Welch segment length.
        noverlap: Overlap between Welch segments.
        detrend: Detrending mode passed to Welch's method.

    Returns:
        Height levels, frequency bins, and TKE temporal PSD by height.

    Raises:
        ValueError: If no vertical levels are found.
    """
    z_levels = ds_u.z.values
    energy_spectra = []
    frequencies = None

    for i in tqdm(range(len(z_levels)), desc="Processing levels"):
        u_t = ds_u.u.isel(z=i, xh=x_idx, y=y_idx).values
        v_t = ds_v.v.isel(z=i, x=x_idx, yh=y_idx).values
        w_t = ds_w.w.isel(zh=i, x=x_idx, y=y_idx).values if ds_w is not None else None

        f, tke_psd = _tke_temporal_spectrum_core(
            u_t, v_t, fs, w_t=w_t, nperseg=nperseg, noverlap=noverlap, detrend=detrend
        )

        if frequencies is None:
            frequencies = f

        energy_spectra.append(tke_psd)

    if frequencies is None:
        raise ValueError("No vertical levels found in ds_u.")

    return z_levels, frequencies, np.array(energy_spectra)


def _tke_temporal_spectrum_core_array(
    u_t: np.ndarray,
    v_t: np.ndarray,
    *args: Optional[np.ndarray],
    fs: float,
    nperseg: int,
    noverlap: Optional[int],
    detrend: str,
) -> np.ndarray:
    """Wrapper around _tke_temporal_spectrum_core for use with xr.apply_ufunc.

    Accepts raw NumPy arrays; w_t is optional via *args.
    """
    w_t = args[0] if args else None
    _, tke_psd = _tke_temporal_spectrum_core(
        u_t, v_t, fs, w_t=w_t, nperseg=nperseg, noverlap=noverlap, detrend=detrend
    )
    return tke_psd


def calculate_tke_temporal_profile_dask(
    ds_u: xr.Dataset,
    ds_v: xr.Dataset,
    ds_w: Optional[xr.Dataset] = None,
    x_idx: Optional[int] = None,
    y_idx: Optional[int] = None,
    fs: float = 1.0,
    nperseg: int = 256,
    noverlap: Optional[int] = None,
    detrend: str = "linear",
) -> xr.Dataset:
    """Compute TKE temporal spectra by vertical level using Dask.

    Applies Welch's method over the time axis at a fixed (x, y) point
    for every vertical level in parallel.

    Args:
        ds_u: Dataset containing the u wind component.
        ds_v: Dataset containing the v wind component.
        ds_w: Optional dataset containing the w wind component.
        x_idx: Optional x index. Must be paired with ``y_idx``.
        y_idx: Optional y index. Must be paired with ``x_idx``.
        fs: Sampling frequency in Hz.
        nperseg: Welch segment length.
        noverlap: Overlap between Welch segments.
        detrend: Detrending mode passed to Welch's method.

    Returns:
        Dataset containing TKE temporal PSD at each vertical level.

    Raises:
        ValueError: If exactly one of ``x_idx`` and ``y_idx`` is provided.
    """
    z_levels = ds_u.z.values

    if (x_idx is None) != (y_idx is None):
        raise ValueError("x_idx and y_idx must both be provided, or both be None.")

    if x_idx is None or y_idx is None:
        u_col = ds_u.u.transpose("z", "time")
        v_col = ds_v.v.transpose("z", "time")
    else:
        u_col = ds_u.u.isel(xh=x_idx, y=y_idx).transpose("z", "time")
        v_col = ds_v.v.isel(x=x_idx, yh=y_idx).transpose("z", "time")

    u_col = u_col.chunk({"z": 1, "time": -1})
    v_col = v_col.chunk({"z": 1, "time": -1})

    input_args = [u_col, v_col]
    input_core_dims = [["time"], ["time"]]

    if ds_w is not None:
        if "zh" in ds_w.dims:
            ds_w = ds_w.rename({"zh": "z"})
        if x_idx is None or y_idx is None:
            w_col = ds_w.w.transpose("z", "time")
        else:
            w_col = ds_w.w.isel(x=x_idx, y=y_idx)

        # Align zh -> z coordinate so apply_ufunc sees consistent dims
        w_col = w_col.assign_coords(z=z_levels)
        w_col = w_col.transpose("z", "time")
        w_col = w_col.chunk({"z": 1, "time": -1})
        input_args.append(w_col)
        input_core_dims.append(["time"])

    # Precompute output frequency coordinate using a dummy signal
    dummy_signal = np.zeros(len(ds_u.time))
    frequency, _ = calculate_temporal_variance_psd(
        dummy_signal, fs, nperseg=nperseg, noverlap=noverlap, detrend=detrend
    )
    n_freqs = len(frequency)

    tke_da = xr.apply_ufunc(
        _tke_temporal_spectrum_core_array,
        *input_args,
        kwargs={
            "fs": fs,
            "nperseg": nperseg,
            "noverlap": noverlap,
            "detrend": detrend,
        },
        input_core_dims=input_core_dims,
        output_core_dims=[["frequency"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[u_col.dtype],
        dask_gufunc_kwargs={
            "output_sizes": {"frequency": n_freqs},
            "allow_rechunk": True,
        },
    )

    tke_da = tke_da.rename("tke_temporal_spectrum")
    tke_da = tke_da.assign_coords(
        z=z_levels,
        frequency=frequency,
    )
    tke_da.attrs.update(
        {
            "long_name": "TKE temporal spectrum",
            "units": "m2 s-1",
            "description": (
                "TKE temporal power spectral density from Welch's method; "
                "integral over frequency gives TKE variance."
            ),
        }
    )
    tke_da["z"].attrs["units"] = "m"
    tke_da["frequency"].attrs.update(
        {
            "long_name": "Frequency",
            "units": "Hz",
        }
    )

    tke_ds = tke_da.to_dataset(name="tke_temporal_spectrum")

    # add metadata to indicate Welch parameters used
    tke_ds.attrs["welch_fs"] = fs if fs is not None else "-"
    tke_ds.attrs["welch_nperseg"] = nperseg if nperseg is not None else "-"
    tke_ds.attrs["welch_noverlap"] = noverlap if noverlap is not None else "-"
    tke_ds.attrs["welch_detrend"] = detrend if detrend is not None else "-"

    return tke_ds


__all__ = [
    "WindProcessing",
    "calculate_coriolis_force",
    "calculate_effective_wind_speed",
    "calculate_monin_obukhov_length",
    "calculate_optical_flow_effective_wind_speed",
    "calculate_richardson_number",
    "calculate_smagorinsky_constant",
    "calculate_spatial_psd_2d",
    "calculate_temporal_variance_psd",
    "calculate_tke_spatial_profile_dask",
    "calculate_tke_temporal_profile",
    "calculate_tke_temporal_profile_dask",
    "calculate_u_star",
    "calculate_u_star_stable",
    "calculate_virtual_temperature",
    "calculate_virtual_turbulent_heat_flux",
    "compute_grid_resolution",
    "compute_optical_flow_displacement",
    "get_wind_profile",
    "radial_spectrum_from_spatial_psd",
    "ramp",
    "solve_wind_profile",
    "wind_profile_from_given_wind",
]
