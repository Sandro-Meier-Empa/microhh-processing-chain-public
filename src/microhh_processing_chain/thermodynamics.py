"""Thermodynamic conversions for atmospheric model data."""

from typing import Optional, Union

import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

ThermodynamicInput = Union[
    xr.DataArray,
    np.ndarray,
    list[float],
    tuple[float, ...],
    int,
    float,
]
ThermodynamicOutput = Union[xr.DataArray, np.ndarray, float]

_DRY_AIR_GAS_CONSTANT = 287.05  # J kg-1 K-1
_DRY_AIR_HEAT_CAPACITY = 1004.0  # J kg-1 K-1
_VIRTUAL_TEMPERATURE_COEFFICIENT = 0.61


class PotentialTemperatureProfile:
    """Generate an initial potential-temperature profile.

    Args:
        z: Heights [m] at which to evaluate the profile.
        theta_ref: Potential temperature [K] at ground level.
        z_i: Height [m] of the initial inversion center.
        delta_theta: Potential-temperature increase [K] across the inversion.
        gamma_bl: Vertical gradient [K m-1] below the inversion (boundary layer).
        gamma_ft: Vertical gradient [K m-1] above the inversion (free troposphere).
        inversion_width: Thickness [m] of the inversion layer. A value of 0.0
            gives a sharp transition, while larger values give a smoother transition.
    """

    def __init__(
        self,
        z: Union[np.ndarray, list[float], tuple[float, ...]],
        theta_ref: float,
        z_i: float,
        delta_theta: float,
        gamma_bl: float,
        gamma_ft: float,
        inversion_width: float = 0.0,
    ) -> None:
        self.z = np.asarray(z, dtype=float)
        self.theta_ref = theta_ref
        self.z_i = z_i
        self.delta_theta = delta_theta
        self.gamma_bl = gamma_bl
        self.gamma_ft = gamma_ft
        self.inversion_width = inversion_width

    @property
    def stability(self) -> str:
        """Return a string describing the stability of the profile."""
        if self.gamma_bl < 0:
            return "unstable"
        elif self.gamma_bl == 0:
            return "neutral"
        else:
            return "stable"

    @property
    def temperature_profile(self) -> np.ndarray:
        """Return potential temperature [K] at each height in z."""
        z = np.asarray(self.z, dtype=float)

        if self.inversion_width < 0:
            raise ValueError("inversion_width must be nonnegative.")

        if self.z_i < self.inversion_width / 2:
            raise ValueError("The inversion must lie above the ground.")

        z_bottom = self.z_i - self.inversion_width / 2
        z_top = self.z_i + self.inversion_width / 2

        if self.inversion_width > 0:
            inversion_fraction = np.clip(
                (z - z_bottom) / self.inversion_width,
                0.0,
                1.0,
            )
        else:
            inversion_fraction = (z >= self.z_i).astype(float)

        return (
            self.theta_ref
            + self.gamma_bl * np.minimum(z, z_bottom)
            + self.delta_theta * inversion_fraction
            + self.gamma_ft * np.maximum(z - z_top, 0.0)
        )

    def plot(self, ax: Optional[Axes] = None, **kwargs: object) -> Axes:
        """Plot the potential temperature profile."""
        if ax is None:
            _, ax = plt.subplots()

        (line,) = ax.plot(self.temperature_profile, self.z, label=self.stability)
        line.set(**kwargs)
        ax.set_xlabel("Potential Temperature [K]")
        ax.set_ylabel("Height [m]")
        ax.legend()

        return ax


def potential_temperature(
    temperature: ThermodynamicInput,
    pressure: ThermodynamicInput,
    specific_humidity: Optional[ThermodynamicInput] = None,
    reference_pressure: ThermodynamicInput = 100000.0,
) -> ThermodynamicOutput:
    """Calculate potential or virtual potential temperature.

    Args:
        temperature: Absolute temperature in K.
        pressure: Air pressure in Pa.
        specific_humidity: Specific humidity in kg/kg. When provided, return
            virtual potential temperature.
        reference_pressure: Reference pressure in Pa.

    Returns:
        Potential temperature in K, optionally corrected for humidity.
    """
    temperature_data = _as_numeric(temperature)
    pressure_data = _as_numeric(pressure)
    reference_pressure_data = _as_numeric(reference_pressure)
    exponent = _DRY_AIR_GAS_CONSTANT / _DRY_AIR_HEAT_CAPACITY

    result = temperature_data * (reference_pressure_data / pressure_data) ** exponent
    long_name = "potential_temperature"

    if specific_humidity is not None:
        humidity_data = _as_numeric(specific_humidity)
        result = result * (1 + _VIRTUAL_TEMPERATURE_COEFFICIENT * humidity_data)
        long_name = "virtual_potential_temperature"

    return _finalize_result(result, long_name)


def absolute_temperature(
    potential_temperature: ThermodynamicInput,
    pressure: ThermodynamicInput,
    reference_pressure: ThermodynamicInput = 100000.0,
) -> ThermodynamicOutput:
    """Calculate absolute temperature from potential temperature and pressure.

    Args:
        potential_temperature: Potential temperature in K.
        pressure: Air pressure in Pa.
        reference_pressure: Reference pressure in Pa.

    Returns:
        Absolute temperature in K.
    """
    potential_temperature_data = _as_numeric(potential_temperature)
    pressure_data = _as_numeric(pressure)
    reference_pressure_data = _as_numeric(reference_pressure)
    exponent = _DRY_AIR_GAS_CONSTANT / _DRY_AIR_HEAT_CAPACITY

    result = (
        potential_temperature_data
        * (pressure_data / reference_pressure_data) ** exponent
    )
    return _finalize_result(result, "absolute_temperature")


def virtual_temperature(
    temperature: ThermodynamicInput,
    specific_humidity: ThermodynamicInput,
) -> ThermodynamicOutput:
    """Calculate virtual temperature from temperature and specific humidity.

    Args:
        temperature: Absolute or potential temperature in K.
        specific_humidity: Specific humidity in kg/kg.

    Returns:
        Virtual temperature in K.
    """
    temperature_data = _as_numeric(temperature)
    humidity_data = _as_numeric(specific_humidity)
    result = temperature_data * (1 + _VIRTUAL_TEMPERATURE_COEFFICIENT * humidity_data)
    return _finalize_result(result, "virtual_temperature")


def _as_numeric(value: ThermodynamicInput) -> Union[xr.DataArray, np.ndarray]:
    """Preserve DataArrays and convert other supported inputs to arrays."""
    if isinstance(value, xr.DataArray):
        return value
    return np.asarray(value)


def _finalize_result(
    result: Union[xr.DataArray, np.ndarray, np.generic], long_name: str
) -> ThermodynamicOutput:
    """Attach DataArray metadata and restore scalar outputs."""
    if isinstance(result, xr.DataArray):
        result.attrs.update({"units": "K", "long_name": long_name})
        return result

    if result.ndim == 0:
        return float(result.item())

    return result


__all__ = [
    "ThermodynamicInput",
    "ThermodynamicOutput",
    "absolute_temperature",
    "potential_temperature",
    "virtual_temperature",
]
