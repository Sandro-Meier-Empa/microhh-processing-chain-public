import os
import scipy
import sys

from scipy.optimize import fsolve
from typing import Union
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.append(os.path.join("/users", os.environ["USER"], "ddeq", "src"))
sys.path.append(os.path.join("/users", os.environ["USER"], "processing_chain", "src"))
sys.path.append(os.path.join("/users", os.environ["USER"], "u-cat", "src"))
sys.path.append(os.path.join("/users", os.environ["USER"], "LS2D"))

import initialisation as init
import ddeq
import ucat
import ls2d


class WindProcessing:
    def __init__(self, case: init.Case):
        """Class used for post-processing MicroHH cases and handling the output.

        Args:
            case (init.Case):
                Case to post-process.
        """
        import processing as proc

        self.post = proc.PostProcessing(case)
        self.file_directory = case.file_directory

    def create_wind_dataset(
        self,
        source: str,
        sources: xr.Dataset,
        radius: int,
        method: str,
        gas: str = None,
        height: Union[int, float] = None,
        save: bool = True,
        overwrite: bool = False,
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
            xr.Dataset: Dataset containg the wind information around a source.
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
                "lon": (["source"], [selected_source.lon_o.values]),
                "lat": (["source"], [selected_source.lat_o.values]),
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
        method: str,
        gas: str = None,
        height: Union[int, float] = None,
        save: bool = True,
        overwrite: bool = False,
    ) -> xr.Dataset:
        """Create a 2D wind dataset from 3D MicroHH wind fields.

        Args:
            method (str):
                Method to vertically integrate wind speeds. Available options are
                "mean", "median", "height", "concentration_weighted"
            gas (str, optional):
                Gas to use if method = 'concentration_weighted' wind speeds.
                Defaults to None.
            height (Union[int, float], optional):
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
        wind_nc_path = Path(self.file_directory, "wind_2D.nc")
        if os.path.exists(wind_nc_path):
            if overwrite:
                os.remove(wind_nc_path)
                print(f"Overwriting {wind_nc_path}")
            else:
                return self.post.open_microhh_nc(wind_nc_path)

        self._check_wind_integration(method, gas, height)

        wind_data = self.post.combine_prognostic_variables(vars=wind_vars)

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

    def _get_wind_vars(self, method: str) -> list:
        """Get list of prognostic variables used for the given wind integration method.

        Args:
            method (str):
                Method to vertically integrate wind speeds. Available options are
                "mean", "median", "height", "concentration_weighted"

        Raises:
            NotImplementedError: Raise error if the method is not implemented.

        Returns:
            list: List of prognostic variables used for the given wind integration method.
        """
        if method in ["mean", "median", "height"]:
            return ["u", "v"]
        elif method == "concentration_weighted":
            return ["u", "v", "ch4"]
        else:
            raise NotImplementedError(f"Method {method} is not implemented.")

    def _check_wind_integration(self, method: str, gas: str, height: Union[int, float]):
        """Check if all required variables for a given wind integration method are given.

        Args:
            method (str):
                Method to vertically integrate wind speeds. Available options are
                "mean", "median", "height", "concentration_weighted"
            gas (str, optional):
                Gas to use if method = 'concentration_weighted' wind speeds.
            height (Union[int, float], optional):
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
                Dataset containg the wind information.
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
        gas: str = None,
        height: Union[int, float] = None,
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

    def unweighted_mean_wind(self, data, method):
        if method == "mean":
            return data.mean(dim="z")
        elif method == "median":
            return data.median(dim="z")
        else:
            raise NotImplementedError

    def wind_at_height(
        self,
        data: xr.Dataset,
        height: Union[int, float],
    ):
        """Get the wind speed at a certain height.

        Args:
            data (xr.Dataset):
                Dataset containing the wind speed information.
            height (Union[int, float]):
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

        return data_subset.mean(dim="z")

    def concentration_weighted_mean_wind(
        self, data: xr.Dataset, gas: str = "ch4", mixing_ratio_threshold: float = None
    ) -> xr.Dataset:
        """_summary_

        Args:
            data (xr.Dataset):
                Dataset containing the wind information.
            gas (str, optional):
                Gas which should be used to weight the 3D wind fields.
                Defaults to "ch4".
            mixing_ratio_threshold (float, optional):
                Threshold in mol mol-1 below which the gas values are not used
                to compute the weight profile. Defaults to None.

        Returns:
            xr.Dataset: Dataset containing the weighted wind.
        """

        wind_components = ["u", "v"]

        # calculate weight profile
        data = self.calculate_weight_profile(data, gas, mixing_ratio_threshold)

        for wind_component in wind_components:
            data[wind_component] = data[f"{gas}_weight"] * data[wind_component]

        data = data.integrate(coord="z")

        return data[wind_components]

    def calculate_weight_profile(
        self, data: xr.Dataset, gas: str = "ch4", mixing_ratio_threshold: float = None
    ) -> xr.Dataset:
        """Calculate a weight profile for 3D wind speed based on the concentration
        distribution of a gas.

        Args:
            data (xr.Dataset):
                Dataset to which the weight profile should be added.
            gas (str, optional):
                Gas which is used to compute the weight profile. Defaults to "ch4".
            mixing_ratio_threshold (float, optional):
                Threshold in mol mol-1 below which the gas values are not used
                to compute the weight profile. Defaults to None.

        Returns:
            xr.Dataset: Dataset containing the weight profile.
        """
        if gas not in data.data_vars:
            raise KeyError(
                f"No variable named '{gas}'. Variables on the dataset include {list(data.data_vars)}"
            )

        # get current unit of gas
        unit = data[gas].attrs.get("units", None)
        quantity, _ = ucat._unit2quantity(unit)

        # convert gas mixing ratios to concentrations to make it pressure dependent
        if quantity not in ["cm", "cM", "cn"]:
            # add pressure which is needed to convert mass mixing ratios to concentraions
            data = self.post.add_statistics_variable(data, "phydro")
            data = self.post.convert_to_concentration(data, gas, to_unit="kg m-3")

        # calculate threshold. Default is 1 ppb
        mixing_ratio_threshold = (
            1e-9 if mixing_ratio_threshold is None else mixing_ratio_threshold
        )
        threshold = ucat.convert_points(
            mixing_ratio_threshold,
            from_unit="mol mol-1",
            to_unit=unit,
            molar_mass=gas,
        )

        # filter out values with low gas concentration to avoid erroneous weights
        data = data.where(data[gas].integrate(coord="z") > threshold)

        # create weight profile
        data[f"{gas}_weight"] = data[f"{gas}"] / data[f"{gas}"].integrate(coord="z")
        data[f"{gas}_weight"].attrs["units"] = "-"
        data[f"{gas}_weight"].attrs[
            "long_name"
        ] = f"normalised {gas} concentration profile"
        data[f"{gas}_weight"].attrs["threshold"] = f"{mixing_ratio_threshold} mol mol-1"

        return data

    def _circular_mean(self, a, axes):
        """Calculate circular mean, e.g. of wind direction"""
        return scipy.stats.circmean(a, high=360, nan_policy="omit", axis=axes)

    def _circular_std(self, a, axes):
        """Calculate circular standard deviation, e.g. of wind direction"""
        return scipy.stats.circstd(a, high=360, nan_policy="omit", axis=axes)


def wind_profile_from_given_wind(
    grid: Union[
        np.ndarray,
        ls2d.grid.Grid_equidist,
        ls2d.grid.Grid_linear_stretched,
        ls2d.grid.Grid_stretched,
        ls2d.grid.Grid_stretched_manual,
    ],
    wind_speed: Union[int, float],
    height: Union[int, float],
    surface_roughness: Union[int, float],
    thl: Union[int, float, list, np.ndarray],
    qt: Union[int, float, list, np.ndarray],
    sensible_heat_flux: Union[int, float],
    threshold: float = None,
) -> xr.Dataset:
    """_summary_

    Args:
        grid (Union[ np.ndarray, ls2d.grid.Grid_equidist, ls2d.grid.Grid_linear_stretched,
        ls2d.grid.Grid_stretched, ls2d.grid.Grid_stretched_manual]):
            Array of heights above the ground level.
        wind_speed (Union[int, float]):
            Target wind speed at 'height', (in m).
        height (Union[int, float]):
            Target height at which the wind speed should be 'wind_speed'.
        surface_roughness (Union[int, float]):
            Surface roughness length, (in m).
        thl (Union[int, float, list, np.ndarray]):
            (Potential) temperature profile, (in K).
        qt (Union[int, float, list, np.ndarray]):
            Specific humidity profile (dimensionless, in kg/kg)
        sensible_heat_flux (Union[int, float]):
            Sensible heat flux (W/m2).
        threshold (float, optional):
            Maximum difference between target 'wind_speed' and calculated wind speed. Defaults to None.

    Returns:
        xr.Dataset: _description_
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

    u_star = compute_u_star(
        wind_speed, height, surface_roughness, thl, qt, H=sensible_heat_flux
    )

    wind_profile = get_wind_profile(
        z=model_profile,
        z0=surface_roughness,
        u_star=u_star,
        t=thl,
        q=qt,
        H=sensible_heat_flux,
    )

    # check if wind speed at height deviates too much from given value
    z_resolution = np.mean(np.diff(model_profile))
    lower_bound = height - z_resolution / 2
    upper_bound = height + z_resolution / 2

    # Ucreate a mask based on whether model_profile is within the bounds
    mask = np.where(
        (model_profile >= lower_bound) & (model_profile <= upper_bound),
        1,
        0,
    ).astype(bool)

    actual_wind_speed = np.mean(wind_profile[mask])

    if np.abs(wind_speed - actual_wind_speed) > threshold:
        print(
            f"Wind speed at {height} m ({actual_wind_speed:.2f} m/s) deviates too much from prescribed wind speed ({wind_speed} m/s).\n"
            "Returning vertically homogenous wind profile."
        )
        wind_profile = init.interpolate_to_model_profile(
            grid=grid, variable_profile=wind_speed
        )

    return wind_profile


def get_wind_profile(
    z: Union[int, float, list, np.ndarray],
    z0: Union[int, float],
    u_star: Union[int, float],
    t: Union[int, float, list, np.ndarray],
    q: Union[int, float, list, np.ndarray],
    H: Union[int, float] = None,
    H_turb: Union[int, float] = None,
    displacement_height: Union[int, float] = 0,
    kappa: float = 0.4,
    alpha: float = 1 / 4,
    beta: float = 4.7,
    gamma: float = 15.0,
) -> np.ndarray:
    """
    Calculate the wind profile based on the logarithmic wind profile equation.
    Give either the sensible heat flux H (in W/m2) or the turbulent sensible heat flux (in K m/s).
    Source: https://denning.atmos.colostate.edu/ats761/BonanBookFigures/Scans/Turbulent_fluxes_14.pdf

    Parameters:
    z (Union[int, float, list, np.ndarray]): Array of heights above the ground level.
    z0 (Union[int, float]): Roughness length.
    u_star (Union[int, float]): Friction velocity.
    t (Union[int, float, list, np.ndarray]): (Potential) temperature profile (in K).
    q (Union[int, float, list, np.ndarray]): Specific humidity profile (dimensionless, in kg/kg).
    H (Union[int, float]): Sensible heat flux (W/m^2).
    H_turb (Union[int, float]): Turbulent sensible heat flux (K m/s).
    displacement_height (Union[int, float], optional): Zero-plane displacement height (m). Defaults to 0.
    kappa (float, optional): von Kármán constant. Defaults to 0.4.
    alpha (float, optional): Parameter for stability correction in unstable conditions. Defaults to 1/4.
    beta (float, optional): Parameter for stability correction in stable conditions. Defaults to 4.7.
    gamma (float, optional): Parameter for stability correction in unstable conditions. Defaults to 15.0.

    Returns:
    np.ndarray: Wind speed at each height in z.
    """
    z = np.array(z)

    # compute displacement height:
    # displacement_height = 10 * z0 * (3 / 4)

    displacement_height = 0

    # raise error if both H and H_turb are provided
    if (H is not None) and (H_turb is not None):
        raise ValueError(
            "Only one of sensible heat flux (H) or turbulent sensible heat flux (H_turb) should be provided, not both."
        )

    # convert sensible heat flux H to turbulent sensible heat flux H_turb, if H is provided
    elif H is not None:
        H_turb = calculate_virtual_turbulent_heat_flux(H)

    # raise error if both H and H_turb are not provided
    elif H_turb is None:
        raise ValueError(
            "Either sensible heat flux (H) or turbulent sensible heat flux (H_turb) should be provided."
        )

    T_v_mean = calculate_virtual_temperature(t, q).mean()
    L = calculate_monin_obukhov_length(u_star, T_v_mean, H_turb)

    stability = determine_stability(L)

    if stability == "stable":
        stability_correction = _correction_stable(z - displacement_height, L, beta)
    elif stability == "neutral":
        stability_correction = 0
    elif stability == "unstable":
        stability_correction = _correction_unstable(
            z - displacement_height, L, alpha, gamma
        )

    uz = (u_star / kappa) * (
        np.log((z - displacement_height) / z0) - stability_correction
    )

    uz = replace_by_majority_sign(uz)

    return uz


def calculate_virtual_temperature(
    t: Union[int, float, list, np.ndarray], q: Union[int, float, list, np.ndarray]
) -> np.ndarray:
    """
    Calculate the virtual temperature T_v.

    Parameters:
    t (Union[int, float, list, np.ndarray]): (Potential) temperature (in K).
    q (Union[int, float, list, np.ndarray]): Specific humidity (dimensionless, in kg/kg).

    Returns:
    np.ndarray: Virtual temperature (in K).
    """
    t = np.array(t)
    q = np.array(q)
    T_v = t * (1 + 0.608 * q)
    return T_v


def calculate_virtual_turbulent_heat_flux(
    H: Union[int, float], rho: float = 1.225, cp: float = 1003.5
) -> float:
    """
    Calculate the virtual turbulent heat flux from sensible heat flux H.

    Parameters:
    H (Union[int, float]): Sensible heat flux (W/m^2).
    rho (float, optional): Density of air (kg/m^3). Defaults to 1.225.
    cp (float, optional): Specific heat capacity of air at constant pressure (J/(kg*K)). Defaults to 1003.5.

    Returns:
    float: Virtual turbulent heat flux (K m/s).
    """
    return H / (rho * cp)


def calculate_monin_obukhov_length(
    u_star: Union[int, float],
    tv_mean: Union[int, float],
    H_turb: Union[int, float],
    kappa: float = 0.4,
    g: float = 9.81,
) -> float:
    """
    Calculate the Monin-Obukhov length (L).

    Parameters:
    u_star (Union[int, float]): Friction velocity (m/s).
    tv_mean (Union[int, float]): Mean (potential) virtual temperature (K).
    H_turb (Union[int, float]): Virtual turbulent heat flux (K m/s).
    kappa (float, optional): von Kármán constant (dimensionless). Defaults to 0.4.
    g (float, optional): Acceleration due to gravity (m/s^2). Defaults to 9.81.

    Returns:
    float: Monin-Obukhov length (m).
    """

    L = -(u_star**3 * tv_mean) / (kappa * g * H_turb)
    return L


def determine_stability(L: float, threshold: float = 500) -> str:
    """
    Determine the stability regime based on the Monin-Obukhov length (L).

    Parameters:
    L (float): Monin-Obukhov length (m).
    threshold (float, optional): Threshold for neutral stability. Defaults to 1e5.

    Returns:
    str: Stability regime ("stable", "neutral", or "unstable").
    """
    if L > 0 and L < threshold:
        return "stable"
    elif L < 0 and L > -threshold:
        return "unstable"
    else:
        return "neutral"


def _correction_stable(
    z: Union[int, float, np.ndarray], L: float, beta: float = 4.7
) -> Union[int, float, np.ndarray]:
    """
    Stability correction for stable conditions.

    Parameters:
    z (Union[int, float, np.ndarray]): Height above the ground level (m).
    L (float): Monin-Obukhov length (m).
    beta (float, optional): Stability correction parameter. Defaults to 4.7.

    Returns:
    Union[int, float, np.ndarray]: Stability correction term.
    """
    return -beta * z / L


def _correction_unstable(
    z: Union[int, float, np.ndarray], L: float, alpha: float = 1 / 4, gamma: float = 15
) -> Union[int, float, np.ndarray]:
    """
    Stability correction for unstable conditions.

    Parameters:
    z (Union[int, float, np.ndarray]): Height above the ground level (m).
    L (float): Monin-Obukhov length (m).
    alpha (float, optional): Stability correction parameter. Defaults to 1/4.
    gamma (float, optional): Stability correction parameter. Defaults to 15.

    Returns:
    Union[int, float, np.ndarray]: Stability correction term.
    """
    x = (1 - gamma * z / L) ** alpha
    return (
        2 * np.log((1 + x) / 2)
        + np.log((1 + x**2) / 2)
        - (2 * np.arctan(x))
        + (np.pi / 2)
    )


def solve_wind_profile(
    u_star: Union[int, float],
    wind_speed: Union[int, float],
    z: Union[int, float, list, np.ndarray],
    z0: Union[int, float],
    t: Union[int, float, list, np.ndarray],
    q: Union[int, float, list, np.ndarray],
    H: Union[int, float] = None,
    H_turb: Union[int, float] = None,
) -> np.ndarray:
    """
    Equation to solve for finding friction velocity (u_star).

    Parameters:
    u_star (Union[int, float]): Initial guess for friction velocity (m/s).
    z (Union[int, float, list, np.ndarray]): Array of heights above the ground level (m).
    z0 (Union[int, float]): Roughness length (m).
    t (Union[int, float, list, np.ndarray]): (Potential) temperature profile (in K).
    q (Union[int, float, list, np.ndarray]): Specific humidity profile (dimensionless, kg/kg).
    H (Union[int, float]): Sensible heat flux (W/m^2).
    wind_speed (Union[int, float]): Measured wind speed (m/s).

    Returns:
    np.ndarray: Difference between calculated and measured wind speed.
    """
    return get_wind_profile(z, z0, u_star, t, q, H, H_turb) - wind_speed


def compute_u_star(
    wind_speed: Union[int, float],
    z: Union[int, float],
    z0: Union[int, float],
    t: Union[int, float],
    q: Union[int, float],
    H: Union[int, float] = None,
    H_turb: Union[int, float] = None,
    u_star_initial: Union[int, float] = None,
) -> float:
    """
    Compute the friction velocity u* with a correction term for stability numerically.
    Give either the sensible heat flux H (in W/m2) or the turbulent sensible heat flux (in K m/s).

    Parameters:
    wind_speed (Union[int, float]): Measured wind speed (m/s).
    z (Union[int, float]): Height above ground at which the wind speed was measured (m).
    z0 (Union[int, float]): Roughness length (m).
    t (Union[int, float]): (Potential) temperature (in K).
    q (Union[int, float]): Specific humidity (dimensionless, kg/kg).
    H (Union[int, float]): Sensible heat flux (W/m^2).
    H_turb (Union[int, float]): Turbulent sensible heat flux (K m/s).
    u_star_initial (Union[int, float]): Initial guess of u*. If not provided, it is calculated assuming neutrally stable conditions.
    """
    if u_star_initial is None:
        u_star_initial = compute_u_star_stable(wind_speed, z, z0, kappa=0.4)

    u_star_solution = fsolve(
        solve_wind_profile,
        u_star_initial,
        args=(wind_speed, z, z0, t, q, H, H_turb),
    )
    return u_star_solution[0]


def compute_u_star_stable(
    wind_speed: Union[int, float],
    z: Union[int, float],
    z0: Union[int, float],
    kappa: float = 0.4,
) -> float:
    """Compute u* for neutrally stable situations without a correction term for stability analytically.

    Args:
        wind_speed (Union[int, float]): Measured wind speed (m/s).
        z (Union[int, float]): Height above ground at which the wind speed was measured (m).
        z0 (Union[int, float]): Roughness length (m).
        kappa (float, optional): von Kármán constant (dimensionless). Defaults to 0.4.

    Returns:
        float: U* for neutrally stable cases without a correction term for atmospheric stability.
    """

    # Calculate friction velocity for neutrally stable case
    u_star = (wind_speed * kappa) / np.log(z / z0)

    return u_star


def replace_by_majority_sign(arr: np.ndarray) -> np.ndarray:
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
