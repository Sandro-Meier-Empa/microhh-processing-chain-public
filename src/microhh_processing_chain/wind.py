import logging
import os
from pathlib import Path
from typing import Literal, Optional, Union

import ls2d
import microhh_processing_chain.initialisation as init
import microhh_processing_chain.io as io
import microhh_processing_chain.preprocessing as pre
import numpy as np
import scipy
import ucat
import xarray as xr
from microhh_processing_chain.constants import EARTH_ROTATION_RATE
from microhh_processing_chain.enums import WindIntegrationMethod
from scipy.optimize import fsolve
from scipy.signal import welch
from tqdm.auto import tqdm

import ddeq

logger = logging.getLogger(__name__)


class WindProcessing:
    def __init__(self, case: init.Case) -> None:
        """Class used for post-processing MicroHH cases and handling the output.

        Args:
            case (init.Case):
                Case to post-process.
        """
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
        chunks: Optional[dict] = None,
    ) -> xr.Dataset:
        """Create a dataset containing the wind information around a source
        which is used for estimating emissions in 'ddeq'.

        Args:
            source (str):
                Source for which the wind dataset should be created.
            sources (xr.Dataset):
                Dataset with source information
            radius (int):
                Number of pixels around the source which should be considered
                for calculating wind properties.
            gas (str):
                Gas to be used for the vertical weighting of wind speed fields.
            method (str):
                Method of how the wind should be vertically integrated.
                "mean" "median", "height" or "concentration_weighted"
            save (bool, optional. Default: True):
                Specify if the dataset should be saved to store.

        Raises:
            e: Raise an error if the datasets of the wind components u and v
            could not be found or processed.

        Returns:
            xr.Dataset: Dataset containing the wind information around a source.
        """
        wind_data = self.create_2D_wind(method, gas, height, save, overwrite)

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

    def create_2D_wind(
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
            method (str):
                Method to vertically integrate wind speeds. Available options are
                "mean", "median", "height", "concentration_weighted"
            gas (str, optional):
                Gas to use if method = 'concentration_weighted' wind speeds.
                Defaults to None.
            height (float, optional):
                Height at which the wind speed should be used if method = 'height'.
                Defaults to None.
            save (bool, optional):
                Specify if the resulting xarray dataset should be saved.
                Defaults to True.
            overwrite (bool, optional):
                Specify if the output file should be overwritten if it exists.
                Defaults to False.

        Returns:
            xr.Dataset: Dataset with vertically integrated wind.
        """
        wind_vars = self._get_wind_vars(method)

        # load wind.nc if it exists
        wind_nc_path = Path(self.directory, "wind_2D.nc")
        if os.path.exists(wind_nc_path):
            if overwrite:
                os.remove(wind_nc_path)
                logger.info(f"Overwriting {wind_nc_path}")
            else:
                return io.open_microhh_nc(self.case, variable="wind_2D", chunks=chunks)

        self._check_wind_integration(method, gas, height)

        wind_data = self.post.combine_prognostic_variables(
            vars=wind_vars, chunks=chunks
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
            list: List of prognostic variables used for the given wind integration method.
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
        """Check if all required variables for a given wind integration method are given.

        Args:
            method (str):
                Method to vertically integrate wind speeds. Available options are
                "mean", "median", "height", "concentration_weighted"
            gas (str):
                Gas to use if method = 'concentration_weighted' wind speeds.
            height (float):
                Height at which the wind speed should be used if method = 'height'.
        """

        if method not in ["mean", "median", "height", "concentration_weighted"]:
            raise ValueError(
                f"'method' must be 'mean', 'median', 'height' or 'concentration_weighted', not {method}."
            )
        if method == "height" and height is None:
            raise ValueError("Height must be an integer or float, not 'None'.")
        if method == "concentration_weighted" and gas is None:
            raise ValueError("'gas' must be a string, not 'None'.")

    def _select_pixels_around_source(
        self, wind_data: xr.Dataset, selected_source: xr.Dataset, radius: int
    ) -> xr.Dataset:
        """Select all pixels within a radius around a source

        Args:
            wind_data (xr.Dataset):
                Dataset containing the wind information.
            selected_source (xr.Dataset):
                Source around which the pixels are to be selected.
            radius (int):
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
            wind_data (xr.Dataset):
                3D wind dataset
            method (str):
                Method of how the wind should be vertically integrated.
                "mean" "median", "height" or "concentration_weighted"

        Raises:
            NotImplementedError: Raise an error if an invalid method is provided.

        Returns:
            xr.Dataset: Vertically integrated wind dataset.
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
            data (xr.Dataset):
                Dataset containing the wind speed information.
            height (float):
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
        wind_vars: Union[str, list] = ["u", "v"],
        threshold: Optional[float] = None,
        threshold_unit: Optional[str] = None,
    ) -> xr.Dataset:
        """Compute the concentration weighted wind speed from the 3D fields
        of 'gas' and the 3D wind fields with variable(s) 'wind_vars'.
        Remove data below a threshold of avoid division by 0.

        Args:
            data (xr.Dataset):
                Dataset containing the variables 'u', 'v' and {gas}.
            gas (str, optional):
                Gas which should be used to weight the 3D wind fields.
                Defaults to "ch4".
            wind_vars (Union[str, list], optional):
                Variable name(s) of the wind components. Defaults to ["u", "v"].
            threshold (float, optional):
                Threshold below which the gas values are not used
                to compute the weight profile. Defaults to None.
            threshold_unit (str, optional):
                Unit of the threshold value. Required if threshold is given.

        Returns:
            xr.Dataset: Dataset containing the weighted wind.
        """
        if gas is None:
            raise ValueError("'gas' must be a string, not 'None'.")

        if isinstance(wind_vars, str):
            wind_vars = [wind_vars]

        if not all(var in data for var in wind_vars):
            raise KeyError(
                f"Not all variables '{wind_vars}' are in the dataset. Variables on the dataset include {list(data.data_vars)}"
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
        """Calculate a weight profile for 3D wind speed based on the concentration
        distribution of a gas. Remove data below 'threshold' to avoid division by 0.

        Args:
            data (xr.Dataset):
                Dataset to which the weight profile should be added.
            gas (str, optional):
                Gas which is used to compute the weight profile.
            threshold (float, optional):
                Threshold below which the gas values are not used
                to compute the weight profile to avoid division by 0.
            threshold_unit (str, optional):
                Unit of the threshold value. Required if threshold is given.


        Returns:
            xr.Dataset: Dataset containing the weight profile.
        """
        if gas not in data.data_vars:
            raise KeyError(
                f"No variable named '{gas}'. Variables on the dataset include {list(data.data_vars)}"
            )

        # get current unit of gas
        unit = data[gas].attrs.get("units", "a.u.")
        quantity, _, _ = ucat._unit2quantity(str(unit))

        # convert gas mixing ratios to concentrations to make it pressure dependent
        if quantity not in ["cm", "cM", "cn"]:
            # add pressure which is needed to convert mass mixing ratios to concentraions
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
        """Calculate circular mean, e.g. of wind direction"""
        return scipy.stats.circmean(a, high=360, nan_policy="omit", axis=axes)

    def _circular_std(
        self, a: Union[xr.DataArray, np.ndarray], axes: list[int]
    ) -> np.ndarray:
        """Calculate circular standard deviation, e.g. of wind direction"""
        return scipy.stats.circstd(a, high=360, nan_policy="omit", axis=axes)


class EffectiveWind:
    # TODO
    def _init__(self) -> None:
        return None


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
    E: float,
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
        E: Water vapor flux (kg/s).
        sensible_heat_flux: Sensible heat flux (W/m2).
        threshold: Maximum difference between target 'wind_speed' and
            calculated wind speed.

    Returns:
        np.ndarray: Wind profile interpolated to the model grid.
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
        wind_speed, height, surface_roughness, thl, E=E, H=sensible_heat_flux
    )

    wind_profile = get_wind_profile(
        z=model_profile,
        z0=surface_roughness,
        u_star=u_star,
        T=thl,
        E=E,
        H=sensible_heat_flux,
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
            f"Wind speed at {height} m ({actual_wind_speed:.2f} m/s) deviates too much from prescribed wind speed ({wind_speed} m/s).\n"
            "Returning vertically homogeneous wind profile."
        )
        wind_profile = pre.interpolate_to_model_profile(
            grid=grid, variable_value=wind_speed
        )

    return wind_profile


def get_wind_profile(
    z: Union[int, float, list, np.ndarray],
    z0: float,
    u_star: float,
    T: float,
    L: Optional[float] = None,
    H: Optional[float] = None,
    H_turb: Optional[float] = None,
    E: float = 0,
    displacement_height: float = 0,
    stability_criterion: Literal["MO", "Ri"] = "Ri",
    kappa: float = 0.4,
) -> np.ndarray:
    """Calculate the wind profile based on the logarithmic wind profile equation.

    Give either the sensible heat flux H (in W/m2) or the turbulent sensible
    heat flux (in K m/s).

    Source: https://denning.atmos.colostate.edu/ats761/BonanBookFigures/Scans/Turbulent_fluxes_14.pdf

    Marked with *:
    From https://www.cambridge.org/core/books/turbulence-in-the-atmosphere/60C43BE9C14A0511C0B03ACD42731068

    Args:
        z: Array of heights above the ground level.
        z0: Roughness length.
        u_star: Friction velocity.
        T: (Potential) temperature (in K).
        L: Monin-Obukhov length (m). Defaults to None.
        H: Sensible heat flux (W/m^2). Defaults to None.
        H_turb: Turbulent sensible heat flux (K m/s). Defaults to None.
        E: Water vapor flux (kg/s). Defaults to 0.
        displacement_height: Zero-plane displacement height (m). Defaults to 0.
        stability_criterion: Stability criterion to use ('MO' or 'Ri').
            Defaults to 'Ri'.
        kappa: von Kármán constant. Defaults to 0.4.

    Returns:
        np.ndarray: Wind speed at each height in z.
    """
    z = np.array(z)

    # compute displacement height:
    if displacement_height is None:
        displacement_height = 10 * z0 * (3 / 4)

    # calculate MO length if it is not given
    if L is None:
        if (H is not None) and (H_turb is not None):
            raise ValueError(
                "Only one of sensible heat flux (H) or turbulent sensible heat flux (H_turb) should be provided, not both."
            )

        if H is not None:
            h_turb_value = calculate_virtual_turbulent_heat_flux(H, E, T=T)
        elif H_turb is not None:
            h_turb_value = H_turb
        else:
            raise ValueError(
                "Either sensible heat flux (H) or turbulent sensible heat flux (H_turb) should be provided."
            )

        L = calculate_monin_obukhov_length(u_star, T, h_turb_value)

    if stability_criterion == "Ri":
        stability = _determine_stability_Ri(float(np.nanmean(z) / L))
    else:
        stability = _determine_stability_MO(L)

    if stability == "stable":
        stability_correction = _correction_stable(z - displacement_height, L)
    elif stability == "neutral":
        stability_correction = np.zeros_like(z, dtype=np.float64)
    elif stability == "unstable":
        stability_correction = _correction_unstable(z - displacement_height, L)
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
    """Calculate the virtual (potential) temperature T_v.

    Args:
        t: (Potential) temperature (in K).
        q: Specific humidity (dimensionless, in kg/kg).

    Returns:
        np.ndarray: Virtual temperature (in K).
    """
    t = np.array(t)
    q = np.array(q)
    T_v = t * (1 + 0.608 * q)
    return T_v


def calculate_virtual_turbulent_heat_flux(
    H: float, E: float, T: float, rho: float = 1.225, cp: float = 1003.5
) -> float:
    """Calculate the virtual turbulent heat flux from sensible heat flux H.

    According to: https://doi.org/10.1175/1520-0469(2003)060%3C2372:OTMAVO%3E2.0.CO;2

    Args:
        H: Sensible heat flux (W/m^2).
        E: Water vapor flux (kg/s).
        T: Absolute temperature (K).
        rho: Density of air (kg/m^3). Defaults to 1.225.
        cp: Specific heat capacity of air at constant pressure (J/(kg*K)).
            Defaults to 1003.5.

    Returns:
        float: Virtual turbulent heat flux (K m/s).
    """
    return (1 / T) * (1 / rho) * ((H / cp) + 0.61 * E)


def calculate_monin_obukhov_length(
    u_star: float,
    T0: float,
    H_turb: float,
    kappa: float = 0.4,
    g: float = 9.81,
) -> float:
    """Calculate the Monin-Obukhov length (L).

    Args:
        u_star: Friction velocity (m/s).
        T0: Mean temperature (K).
        H_turb: Virtual turbulent heat flux (K m/s).
        kappa: von Kármán constant (dimensionless). Defaults to 0.4.
        g: Acceleration due to gravity (m/s^2). Defaults to 9.81.

    Returns:
        float: Monin-Obukhov length (m).
    """

    L = -(u_star**3 * T0) / (kappa * g * H_turb)
    return L


def _determine_stability_MO(L: float, threshold: float = 500) -> str:
    """Determine the stability regime based on the Monin-Obukhov length (L).

    From: https://www.researchgate.net/publication/266043126_Estimating_the_systematic_er-_rors_in_turbulence_sensed_by_wind_lidars
    Or from: https://onlinelibrary.wiley.com/doi/full/10.1002/we.1528

    Args:
        L: Monin-Obukhov length (m).
        threshold: Threshold for neutral stability. Defaults to 500.

    Returns:
        str: Stability regime ("stable", "neutral", or "unstable").
    """
    if L > 0 and L < threshold:
        return "stable"
    elif L < 0 and L > -threshold:
        return "unstable"
    else:
        return "neutral"


def _determine_stability_Ri(
    Ri: float, upper_threshold: float = -0.1, lower_threshold: float = 0.05
) -> str:
    """Determine the stability regime based on the Richardson number (Ri).

    From: https://www.researchgate.net/publication/269105054_Dispersion_modeling_of_air_pollutants_in_the_atmosphere_a_review

    Args:
        Ri: Richardson number.
        upper_threshold: Upper threshold for stability classification.
            Defaults to -0.1.
        lower_threshold: Lower threshold for stability classification.
            Defaults to 0.05.

    Returns:
        str: Stability regime ("stable", "neutral", or "unstable").
    """
    stable = np.sum(Ri < lower_threshold)
    unstable = np.sum(Ri > upper_threshold)
    neutral = np.sum((Ri >= lower_threshold) & (Ri <= upper_threshold))

    if stable > unstable and stable > neutral:
        return "stable"
    elif unstable > stable and unstable > neutral:
        return "unstable"
    else:
        return "neutral"


def _correction_stable(
    z: Union[float, np.ndarray], L: float, beta: float = 4.8
) -> Union[float, np.ndarray]:
    """Stability correction for stable conditions.

    * 4.8 in Wyngaard (2010) Turbulence in the Atmosphere.

    Args:
        z: Height above the ground level (m).
        L: Monin-Obukhov length (m).
        beta: Stability correction parameter. Defaults to 4.8.

    Returns:
        Stability correction term.
    """
    return -beta * z / L


def _correction_unstable(
    z: Union[float, np.ndarray], L: float, alpha: float = 1 / 4, gamma: float = 15
) -> Union[float, np.ndarray]:
    """Stability correction for unstable conditions.

    Args:
        z: Height above the ground level (m).
        L: Monin-Obukhov length (m).
        alpha: Stability correction parameter. Defaults to 1/4.
        gamma: Stability correction parameter. Defaults to 15.

    Returns:
        Stability correction term.
    """
    # according to Wyngaard (2010) Turbulence in the Atmosphere:
    gamma = 3.6
    a = 2 / 3
    b = -1 / 2
    x = (1 + gamma * np.abs(z / L) ** a) ** b

    return x
    x = (1 - gamma * z / L) ** alpha
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
    T: float,
    E: float,
    H: Optional[float] = None,
    H_turb: Optional[float] = None,
) -> np.ndarray:
    """Equation to solve for finding friction velocity (u_star).

    Args:
        u_star: Initial guess for friction velocity (m/s).
        wind_speed: Measured wind speed (m/s).
        z: Array of heights above the ground level (m).
        z0: Roughness length (m).
        T: (Potential) temperature profile (in K).
        E: Water vapor flux (kg/s).
        H: Sensible heat flux (W/m^2). Defaults to None.
        H_turb: Turbulent sensible heat flux (K m/s). Defaults to None.

    Returns:
        np.ndarray: Difference between calculated and measured wind speed.
    """
    return (
        get_wind_profile(z=z, z0=z0, u_star=u_star, T=T, E=E, H=H, H_turb=H_turb)
        - wind_speed
    )


def calculate_u_star(
    wind_speed: float,
    z: float,
    z0: float,
    T: float,
    E: float,
    H: Optional[float] = None,
    H_turb: Optional[float] = None,
    u_star_initial: Optional[float] = None,
) -> float:
    """Compute the friction velocity u* with a correction term for stability numerically.

    Give either the sensible heat flux H (in W/m2) or the turbulent sensible
    heat flux (in K m/s).

    Args:
        wind_speed: Measured wind speed (m/s).
        z: Height above ground at which the wind speed was measured (m).
        z0: Roughness length (m).
        T: (Potential) temperature (in K).
        E: Water vapor flux (kg/s).
        H: Sensible heat flux (W/m^2). Defaults to None.
        H_turb: Turbulent sensible heat flux (K m/s). Defaults to None.
        u_star_initial: Initial guess of u*. If not provided, it is
            calculated assuming neutrally stable conditions. Defaults to None.
    """
    if u_star_initial is None:
        u_star_initial = calculate_u_star_stable(wind_speed, z, z0, kappa=0.4)

    u_star_solution = fsolve(
        solve_wind_profile,
        u_star_initial,
        args=(wind_speed, z, z0, T, E, H, H_turb),
    )
    return u_star_solution[0]


def calculate_u_star_stable(
    wind_speed: float,
    z: float,
    z0: float,
    kappa: float = 0.4,
) -> float:
    """Compute u* for neutrally stable situations without a correction term for stability analytically.

    Args:
        wind_speed (float): Measured wind speed (m/s).
        z (float): Height above ground at which the wind speed was measured (m).
        z0 (float): Roughness length (m).
        kappa (float, optional): von Kármán constant (dimensionless). Defaults to 0.4.

    Returns:
        float: U* for neutrally stable cases without a correction term for atmospheric stability.
    """

    # Calculate friction velocity for neutrally stable case
    u_star = (wind_speed * kappa) / np.log(z / z0)

    return u_star


def _replace_by_majority_sign(arr: np.ndarray) -> np.ndarray:
    """Replace wind speed in array with 0 if it does not have the same sign as
    the majority of wind values in the array.

    Args:
        arr (np.ndarray):
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


def ramp(x: Union[int, float, np.ndarray]) -> Union[int, float, np.ndarray]:
    """Ramp function R(x): returns x if x > 0 else 0."""
    return np.maximum(x, 0)


def calculate_smagorinsky_constant(
    delta: Union[int, float, np.ndarray],
    L: Union[int, float, np.ndarray],
    z: Union[int, float, np.ndarray],
    kappa: float = 0.4,
    c0: float = 0.135,
    alpha: float = 0.135,
    n: int = 3,
) -> Union[int, float, np.ndarray]:
    """Computes Smagorinsky constant according to Eq. (17) in
    https://doi.org/10.1175/1520-0469(2003)060%3C2372:OTMAVO%3E2.0.CO;2

    Args:
        delta: Characteristic scale (Δ). For simulations: resolution of the
            model, in m.
        L: Obukhov length (L).
        z: Height above ground (in m).
        kappa: von Kármán constant (κ). Defaults to 0.4.
        c0: Base concentration (c₀). Defaults to 0.135.
        alpha: Model parameter (α). Defaults to 0.135.
        n: Model exponent (n). Defaults to 3.

    Returns:
        Computed Smagorinsky constant c_s.
    """
    term1 = (1 + (c0 / alpha) * ramp(delta / L)) ** -1
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
    Rf = stats.b_w / denom
    attrs = {
        "long_name": "Richardson number",
        "units": "1",
        "description": "Ratio of buoyant production to shear production of turbulence kinetic energy",
    }

    da = xr.DataArray(Rf, dims=Rf.dims, coords=Rf.coords, attrs=attrs)
    stats["Ri"] = da.where(np.abs(denom) > threshold)
    return stats


def calculate_spatial_psd_2d(
    field: np.ndarray, dx: float, dy: float, angular: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute a variance-preserving 2D PSD using physical frequency/wavenumber coordinates.

    The returned psd2d satisfies approximately:

        psd2d.sum() == np.nanvar(field)

    after removal of the horizontal mean.

    Parameters
    ----------
    field : 2D array, shape [nx, ny]
        Input field.
    dx : float
        Grid spacing in the x direction.
    dy : float
        Grid spacing in the y direction.
    angular : bool, optional
        If False, return frequencies in cycles per physical unit.
        If True, return angular wavenumbers in radians per physical unit.

    Returns
    -------
    psd2d : 2D array, shape [nx, ny]
        Variance-preserving 2D PSD, shifted so the zero mode is in the center.
    kx : 1D array, shape [nx]
        Physical frequency or angular wavenumber coordinates in x.
    ky : 1D array, shape [ny]
        Physical frequency or angular wavenumber coordinates in y.
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
    """
    Compute a variance-preserving radial spectrum from a 2D PSD.

    This function bins the 2D PSD into annuli in wavenumber space and sums
    the PSD values inside each annulus.

    Therefore:

        radial_psd.sum() ≈ psd2d.sum()

    assuming all Fourier modes are included in the radial bins.

    Parameters
    ----------
    psd2d : 2D array, shape [nx, ny]
        Variance-preserving 2D PSD, usually shifted.
    kx, ky : 1D arrays
        Physical frequency or wavenumber coordinates corresponding to psd2d.
        These should be shifted in the same way as psd2d.
    nbins : int, optional
        Number of radial bins. If None, defaults to min(nx, ny) // 2.
    bin_edges : 1D array, optional
        Custom radial bin edges. If provided, nbins is ignored.

    Returns
    -------
    k_radial : 1D array
        Radial bin centers.
    radial_psd : 1D array
        Variance-preserving radial spectrum.
    counts : 1D array
        Number of Fourier modes in each radial bin.
    """

    psd2d = np.asarray(psd2d)
    kx = np.asarray(kx)
    ky = np.asarray(ky)

    nx, ny = psd2d.shape

    if kx.size != nx:
        raise ValueError("kx length must match psd2d.shape[0].")

    if ky.size != ny:
        raise ValueError("ky length must match psd2d.shape[1].")

    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    k_radius = np.sqrt(KX**2 + KY**2)

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


def _tke_spatial_spectrum_core(u_2d, v_2d, dx, dy, w_2d=None):
    """
    Core function that handles a single 2D plane of data.
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


def _tke_spatial_spectrum_core_array(u_2d, v_2d, *args, dx, dy):
    w_2d = args[0] if args else None
    _, radial_tke = _tke_spatial_spectrum_core(u_2d, v_2d, dx, dy, w_2d=w_2d)
    return radial_tke


def calculate_tke_spatial_profile_dask(
    ds_u, ds_v, ds_w=None, time_idx=-1
) -> xr.Dataset:
    """
    Computes the 2D or 3D TKE spectrum for every vertical level using Dask.

    Assumes staggered-grid inputs:
        u(time, z,  xh, y)
        v(time, z,  x,  yh)
        w(time, zh, x,  y)

    The fields are collocated to the scalar grid:
        (z, x, y)

    before applying the 2D FFT spectrum.
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
        w_slice = 0.5 * (w_slice + w_slice.roll(zh=-1, roll_coords=False))
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
            "description": "Variance-preserving radial TKE spectrum; sum over wavenumber bins gives TKE variance.",
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
    """
    Compute a variance-preserving temporal PSD using Welch's method.

    The returned psd satisfies approximately:

        psd.sum() * (f[1] - f[0]) ≈ np.var(signal)

    Parameters
    ----------
    signal : 1D array
        Time series at a single point.
    fs : float
        Sampling frequency [Hz].
    nperseg : int, optional
        Length of each Welch segment. Defaults to 256.
    noverlap : int, optional
        Number of overlapping samples between segments.
        If None, defaults to nperseg // 2 (scipy default).
    detrend : str, optional
        Detrending applied to each segment. 'linear' removes a linear trend;
        'constant' removes the mean only. Matches the spatial code's mean removal.

    Returns
    -------
    f : 1D array
        Frequency bins [Hz].
    psd : 1D array
        Variance-preserving PSD [units² / Hz].
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
    """
    Core function: compute TKE temporal spectrum from 1D time series
    at a single (x, y, z) point.

    Returns
    -------
    f : 1D array
        Frequency bins [Hz].
    tke_psd : 1D array
        TKE temporal PSD, i.e. 0.5 * (P_uu + P_vv [+ P_ww]) [m² s⁻² / Hz].
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
    """
    Compute the TKE temporal spectrum at every vertical level,
    for a single (x, y) point in the domain.

    This is the temporal analogue of calculate_tke_profile:
    instead of a 2D spatial FFT over (x, y) at fixed time,
    it applies Welch's method over the time axis at fixed (x, y).

    Parameters
    ----------
    ds_u, ds_v, ds_w : /
        Datasets containing velocity components. Expected dims:
            u(time, z,  xh, y)
            v(time, z,  x,  yh)
            w(time, zh, x,  y)   [optional]
    x_idx : int
        Index along the x axis to extract the time series from.
    y_idx : int
        Index along the y axis to extract the time series from.
    fs : float
        Sampling frequency [Hz]. Must match the temporal resolution of the data.
    nperseg : int
        Welch segment length.
    noverlap : int, optional
        Overlap between Welch segments.
    detrend : str
        Detrending per segment ('linear' or 'constant').

    Returns
    -------
    energy_spectra : 2D array, shape [z, f]
        TKE temporal PSD at each vertical level.
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
):
    """
    Wrapper around _tke_temporal_spectrum_core for use with xr.apply_ufunc.
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
    """
    Dask-parallelised version of calculate_tke_temporal_profile.

    Applies Welch's method over the time axis at a fixed (x, y) point
    for every vertical level in parallel.

    Parameters
    ----------
    (same as calculate_tke_temporal_profile)

    Returns
    -------
    tke_ds : xr.DataArray, dims (z, frequency)
        TKE temporal PSD at each vertical level.
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
            "description": "TKE temporal power spectral density from Welch's method; integral over frequency gives TKE variance.",
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
    "get_wind_profile",
    "calculate_virtual_temperature",
    "calculate_virtual_turbulent_heat_flux",
    "calculate_monin_obukhov_length",
    "solve_wind_profile",
    "calculate_u_star",
    "calculate_u_star_stable",
    "calculate_effective_wind_speed",
    "calculate_smagorinsky_constant",
    "calculate_coriolis_force",
    "calculate_richardson_number",
]
