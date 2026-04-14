"""Utility functions for the microhh_processing package."""

import subprocess
from contextlib import contextmanager
from collections.abc import Iterator
from pathlib import Path
from typing import Union, List, Any, Optional

import numpy as np
import xarray as xr


def open_in_vscode(file_path: Union[str, Path]) -> None:
    """Open a file in VS Code.

    Args:
        file_path: Path of the file to open in VS Code.
    """
    file_path = Path(file_path)
    subprocess.run(["code", str(file_path)], cwd=file_path.parent)


def object_to_list(var: Any) -> List:
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

    match var:
        case str():
            return _parse_string_to_list(var)
        case xr.DataArray():
            return var.values.tolist()
        case np.ndarray():
            return var.tolist()
        case list():
            return var
        case _:
            return [var]


def _parse_string_to_list(var: str) -> List:
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
    lst: Union[List, np.ndarray], n: int = 1, ndecimals: Optional[int] = None
) -> str:
    """Convert a Python iterable to a comma-separated string.

    Handles float precision artifacts (e.g. 209.20000000000002 -> 209.2)
    while preserving small numbers (e.g. 0.00000123).

    Args:
        lst: List or array of items to convert.
        n: Number of times to repeat the list (default is 1).
        ndecimals: Number of decimal places for floats. If None, uses
            up to 15 significant digits.

    Returns:
        Comma-separated string representation.
    """

    def clean_fmt(x: Any) -> str:
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
    "open_in_vscode",
    "object_to_list",
    "to_comma_separated_list",
    "confirm_action",
]
