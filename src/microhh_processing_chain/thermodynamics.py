"""Thermodynamic conversions for atmospheric model data."""

from typing import Optional, Union

import numpy as np
import xarray as xr

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
