"""Utility functions for the microhh_processing_chain package."""

import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from numbers import Real
from pathlib import Path
from typing import Any, Optional, Union, cast

import numpy as np
import xarray as xr
from scipy.optimize import brentq


def calculate_linear_stretched_alpha(zsize: float, ktot: int, dz0: float) -> float:
    """Calculate the stretching factor for an LS2D linear-stretched grid.

    Args:
        zsize: Target vertical domain size.
        ktot: Number of vertical grid cells.
        dz0: Thickness of the lowest grid cell, in the same units as
            ``zsize``.

    Returns:
        Nonnegative ``alpha`` for
        ``ls2d.grid.Grid_linear_stretched(kmax=ktot, dz0=dz0, alpha=alpha)``.

    Raises:
        TypeError: If an argument has an invalid type.
        ValueError: If an argument is not finite and positive, ``ktot`` is less
            than two, or the target domain is smaller than an equidistant grid.
    """
    if isinstance(ktot, bool) or not isinstance(ktot, (int, np.integer)):
        raise TypeError("ktot must be an integer.")
    if ktot < 2:
        raise ValueError("ktot must be at least 2.")

    for name, value in (("zsize", zsize), ("dz0", dz0)):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{name} must be a real number.")
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive.")

    zsize = float(zsize)
    dz0 = float(dz0)
    equidistant_zsize = ktot * dz0
    if zsize < equidistant_zsize:
        raise ValueError("zsize must be at least ktot * dz0 for a nonnegative alpha.")
    if zsize == equidistant_zsize:
        return 0.0

    target_grid_size = zsize / dz0

    def grid_size_difference(alpha: float) -> float:
        if alpha == 0.0:
            return float(ktot - target_grid_size)
        grid_size = np.expm1(ktot * np.log1p(alpha)) / alpha
        return float(grid_size - target_grid_size)

    # At this upper bound, the final cell alone spans the target domain.
    upper_bound = np.expm1(np.log(target_grid_size) / (ktot - 1))
    return cast(float, brentq(grid_size_difference, 0.0, upper_bound))


def open_in_vscode(file_path: Union[str, Path]) -> None:
    """Open a file in VS Code.

    Args:
        file_path: Path of the file to open in VS Code.
    """
    file_path = Path(file_path)
    subprocess.run(["code", str(file_path)], cwd=file_path.parent)


def object_to_list(var: Any) -> list:  # noqa: ANN401
    """Convert various input types to a list.

    Handles strings (comma-separated), xarray DataArrays, numpy arrays,
    and single values.

    Args:
        var: Input variable to convert to list.

    Returns:
        List representation of the input.

    Raises:
        ValueError: If input is None.
    """
    if var is None:
        raise ValueError("Input should not be 'None'")

    if isinstance(var, str):
        return _parse_string_to_list(var)
    elif isinstance(var, xr.DataArray):
        return var.values.tolist()
    elif isinstance(var, np.ndarray):
        return var.tolist()
    elif isinstance(var, list):
        return var
    else:
        return [var]


def _parse_string_to_list(var: str) -> list:
    """Parse a comma-separated string into a list with type conversion.

    Args:
        var: Comma-separated string to parse.

    Returns:
        List with items converted to int/float where possible.
    """
    items = [s.strip() for s in var.split(",")]

    converted_items = []
    for item in items:
        try:
            converted_items.append(int(item))
        except ValueError:
            try:
                converted_items.append(float(item))
            except ValueError:
                converted_items.append(item)

    return converted_items


def to_comma_separated_list(
    lst: Union[list, np.ndarray], n: int = 1, nsignificant: Optional[int] = None
) -> str:
    """Convert a Python iterable to a comma-separated string.

    Floats are formatted with significant digits, so small values such as
    source strengths in kg s-1 are never rounded to zero. Float precision
    artifacts are hidden (e.g. 209.20000000000002 -> 209.2, float32 0.1 -> 0.1).
    Very small or large values are written in exponent notation (e.g.
    2.5e-05), which both MicroHH and ``microhh_py`` parse.

    Args:
        lst: List or array of items to convert.
        n: Number of times to repeat the list.
        nsignificant: Number of significant digits for floats. When omitted,
            the full precision of the float type is used (15 digits for
            float64, 6 for float32).

    Returns:
        Comma-separated string representation.

    Raises:
        ValueError: If *nsignificant* is smaller than 1.
    """
    if nsignificant is not None and nsignificant < 1:
        raise ValueError(f"nsignificant must be at least 1, got {nsignificant}.")

    def clean_fmt(x: object) -> str:
        """Format numbers without exposing floating-point precision artifacts."""
        if isinstance(x, (float, np.floating)):
            if nsignificant is not None:
                digits = nsignificant
            elif isinstance(x, np.floating):
                digits = np.finfo(x.dtype).precision
            else:
                digits = np.finfo(np.float64).precision
            return f"{x:.{digits}g}"
        return str(x)

    clean_items = [clean_fmt(x) for x in lst]
    repeated_items = clean_items * n
    return ",".join(repeated_items)


@contextmanager
def confirm_action(prompt: str) -> Iterator[bool]:
    """Context manager for user confirmation prompts.

    Args:
        prompt: Message to display to the user.

    Yields:
        True if user confirmed, False otherwise.
    """
    user_input = input(f"{prompt} (y/n): ")
    yield user_input.lower() == "y"


__all__ = [
    "calculate_linear_stretched_alpha",
    "confirm_action",
    "object_to_list",
    "open_in_vscode",
    "to_comma_separated_list",
]
