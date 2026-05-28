"""Utility functions for the microhh_processing_chain package."""

import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import xarray as xr


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
    lst: Union[list, np.ndarray], n: int = 1, ndecimals: Optional[int] = None
) -> str:
    """Convert a Python iterable to a comma-separated string.

    Handles float precision artifacts (e.g. 209.20000000000002 -> 209.2)
    while preserving small numbers (e.g. 0.00000123).

    Args:
        lst: List or array of items to convert.
        n: Number of times to repeat the list.
        ndecimals: Number of decimal places for floats. Uses significant
            digits when omitted.

    Returns:
        Comma-separated string representation.
    """

    def clean_fmt(x: Union[float, np.floating]) -> str:
        """Format numbers without exposing floating-point precision artifacts."""
        if isinstance(x, (float, np.floating)):
            if ndecimals is not None:
                return f"{x:.{ndecimals}f}"
            return f"{x:.15g}"
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
    "confirm_action",
    "object_to_list",
    "open_in_vscode",
    "to_comma_separated_list",
]
