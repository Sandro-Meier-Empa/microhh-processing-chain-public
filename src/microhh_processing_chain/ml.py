"""Utilities for simple array transformations used in model experiments."""

import xarray as xr


def flip_data(data: xr.Dataset, flip_x: bool, flip_y: bool) -> xr.Dataset:
    """Flip all 2D variables in a dataset along the x and/or y axis.

    Args:
        data: Dataset containing variables to flip.
        flip_x: Flip variables along the x-axis.
        flip_y: Flip variables along the y-axis.

    Returns:
        Dataset with all x/y variables flipped as requested.
    """
    # get all variables with dims "x" and "y"
    vars_to_flip = [
        var for var in data.data_vars if {"x", "y"}.issubset(data[var].dims)
    ]

    data_flipped = data.copy(deep=True)
    for var in vars_to_flip:
        data_flipped[var] = flip_variable(data[var], flip_x, flip_y)

    return data_flipped


def flip_variable(arr: xr.DataArray, flip_x: bool, flip_y: bool) -> xr.DataArray:
    """Flip a 2D variable along the x and/or y axis.

    Args:
        arr: Variable to flip. Must have x and y dimensions.
        flip_x: Flip the variable along the x-axis.
        flip_y: Flip the variable along the y-axis.

    Returns:
        Flipped variable with the original dimension order and coordinates.
    """
    arr_flipped = arr.copy(deep=True)
    if flip_x:
        arr_flipped = arr_flipped.isel(x=slice(None, None, -1))
    if flip_y:
        arr_flipped = arr_flipped.isel(y=slice(None, None, -1))

    new_arr = xr.DataArray(
        arr_flipped.values,
        dims=arr.dims,
        coords=arr.coords,
        attrs=arr.attrs,
    )

    return new_arr


def scale_emissions(
    data: xr.Dataset, scale_factor: float, gas: str = "CH4"
) -> xr.Dataset:
    """Scale a gas emission variable in a dataset."""
    data = data.copy(deep=True)
    if gas in data.data_vars:
        data[gas] *= scale_factor
    return data


__all__ = [
    "flip_data",
    "flip_variable",
    "scale_emissions",
]
