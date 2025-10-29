import subprocess

import xarray as xr
import numpy as np
import netCDF4 as nc4

from pathlib import Path
from typing import Union, Literal, Optional


def calculate_coriolis_force(lat: float) -> float:
    """Calculate the Coriolis force at a given latitude."""
    fc = 2 * 7.2921159e-5 * np.sin(np.radians(lat))
    return fc
