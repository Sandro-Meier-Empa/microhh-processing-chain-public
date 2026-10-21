"""Miscellaneous utility functions."""

import datetime
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import units
from astropy.coordinates import AltAz, EarthLocation, SkyCoord, get_sun
from astropy.time import Time
from matplotlib.colors import LinearSegmentedColormap
from netCDF4 import Dataset, Group


def make_white_inferno(
    n: int = 256, white_fraction: float = 0.25, inferno_start: float = 0.10
) -> LinearSegmentedColormap:
    """Create an inferno-like colormap with a smooth transition from white.

    Parameters
    ----------
    n : int
        Number of colors in the colormap.
    white_fraction : float
        Fraction of the colormap used for the white-to-inferno transition.
        Larger values make the transition softer.
    inferno_start : float
        Where to start sampling inferno. Higher values skip the very dark colors.
    """
    inferno = plt.get_cmap("inferno_r")

    n_white = int(n * white_fraction)
    n_inferno = n - n_white

    # First inferno color used after the white transition
    target = inferno(inferno_start)

    # Smooth interpolation from white to the first inferno color
    white = np.array([1, 1, 1, 1])
    ramp = np.linspace(0, 1, n_white)[:, None]
    transition = white * (1 - ramp) + np.array(target) * ramp

    # Then continue through inferno
    inferno_colors = inferno(np.linspace(inferno_start, 1, n_inferno))

    colors = np.vstack([transition, inferno_colors])

    return LinearSegmentedColormap.from_list("white_inferno_soft", colors)


def make_white_plasma(
    n: int = 256, white_fraction: float = 0.25, plasma_start: float = 0.10
) -> LinearSegmentedColormap:
    """Create an inferno-like colormap with a smooth transition from white.

    Parameters
    ----------
    n : int
        Number of colors in the colormap.
    white_fraction : float
        Fraction of the colormap used for the white-to-inferno transition.
        Larger values make the transition softer.
    inferno_start : float
        Where to start sampling inferno. Higher values skip the very dark colors.
    """
    inferno = plt.get_cmap("plasma_r")

    n_white = int(n * white_fraction)
    n_inferno = n - n_white

    # First inferno color used after the white transition
    target = inferno(plasma_start)

    # Smooth interpolation from white to the first inferno color
    white = np.array([1, 1, 1, 1])
    ramp = np.linspace(0, 1, n_white)[:, None]
    transition = white * (1 - ramp) + np.array(target) * ramp

    # Then continue through inferno
    inferno_colors = inferno(np.linspace(plasma_start, 1, n_inferno))

    colors = np.vstack([transition, inferno_colors])

    return LinearSegmentedColormap.from_list("white_plasma_soft", colors)


def _collect_group_variables(
    group: Union[Dataset, Group],
) -> dict[str, set[str]]:
    """Collect locally defined variable names by absolute group path."""
    group_variables = {group.path: set(group.variables)}

    for subgroup in group.groups.values():
        group_variables.update(_collect_group_variables(subgroup))

    return group_variables


def compare_netcdf_structure(
    nc_path_1: Union[str, Path],
    nc_path_2: Union[str, Path],
) -> None:
    """Print differences between group paths and variable names in NetCDF files.

    The comparison includes the root group and recursively nested groups. It does
    not inspect variable values, dimensions, data types, or attributes.

    Args:
        nc_path_1: First NetCDF file to compare.
        nc_path_2: Second NetCDF file to compare.
    """
    with Dataset(nc_path_1, "r") as ds1, Dataset(nc_path_2, "r") as ds2:
        group_variables1 = _collect_group_variables(ds1)
        group_variables2 = _collect_group_variables(ds2)

        groups1 = set(group_variables1)
        groups2 = set(group_variables2)

        only_groups_1 = sorted(groups1 - groups2)
        only_groups_2 = sorted(groups2 - groups1)
        common_groups = sorted(groups1 & groups2)

        print("=" * 80)
        print("GROUP COMPARISON")
        print("=" * 80)

        print(f"\nGroups only in {nc_path_1}:")
        print("\n".join(f"  {g}" for g in only_groups_1) or "  None")

        print(f"\nGroups only in {nc_path_2}:")
        print("\n".join(f"  {g}" for g in only_groups_2) or "  None")

        print("\nGroups in both files:")
        print("\n".join(f"  {g}" for g in common_groups) or "  None")

        for group in sorted(groups1 | groups2):
            vars1 = group_variables1.get(group, set())
            vars2 = group_variables2.get(group, set())

            only_vars_1 = sorted(vars1 - vars2)
            only_vars_2 = sorted(vars2 - vars1)
            common_vars = sorted(vars1 & vars2)

            print("\n" + "=" * 80)
            print(f"VARIABLE COMPARISON FOR GROUP {group}")
            print("=" * 80)

            print(f"\nOnly in {nc_path_1}:")
            print("\n".join(f"  {v}" for v in only_vars_1) or "  None")

            print(f"\nOnly in {nc_path_2}:")
            print("\n".join(f"  {v}" for v in only_vars_2) or "  None")

            print("\n" + "In both:")
            print("\n".join(f"  {v}" for v in common_vars) or "  None")


def calculate_sun_position(
    lat: float,
    lon: float,
    time_utc: Union[datetime.datetime, np.datetime64, pd.Timestamp],
    height: float = 0,
) -> SkyCoord:
    """Calculate the Solar Zenith Angle (SZA) for a location at a specific time.

    Parameters:
    lat (float): Latitude of the location.
    lon (float): Longitude of the location.
    time_utc (Union[datetime.datetime, np.datetime64, pd.Timestamp]): UTC time.
    height (float, optional): Height of the location in meters. Default is 0.

    Returns:
    SkyCoord: The position of the sun in AltAz coordinates.
    """
    # Create an EarthLocation object for the given latitude and longitude
    location = EarthLocation(
        lat=lat * units.deg, lon=lon * units.deg, height=height * units.m
    )

    # Convert the UTC time to an Astropy Time object
    time = Time(time_utc, scale="utc")

    # Get the position of the sun at the given time and location
    altaz_frame = AltAz(obstime=time, location=location)
    sun_altaz = get_sun(time).transform_to(altaz_frame)

    return sun_altaz


def _as_weights(
    species: Union[list[str], dict[str, float]],
) -> dict[str, float]:
    """Return `species` as a species-to-weight mapping (default weight 1.0)."""
    if isinstance(species, dict):
        return species
    return dict.fromkeys(species, 1.0)


class CamsGlobInventoryPaths:
    """Class to manage paths to CAMS global inventory NetCDF files for a specific year.

    Parameters:
    ----------
    data_path : str or Path
        The base path where the CAMS global inventory data is stored.
    year : int
        The year for which to retrieve the inventory files.
    lumping_co2 : list of str or dict[str, float], optional
        List of CO2 species to lump together (weight 1.0 each) or dictionary
        mapping species to weights. Default is None, which uses a predefined dictionary.
    lumping_co : list of str or dict[str, float], optional
        List of CO species to lump together (weight 1.0 each) or dictionary
        mapping species to weights. Default is None, which uses a predefined dictionary.
    lumping_nox : list of str or dict[str, float], optional
        List of NOx species to lump together (weight 1.0 each) or dictionary
        mapping species to weights. Default is None, which uses a predefined dictionary.
    lumping_vocs : list of str or dict[str, float], optional
        List of VOC species to lump together (weight 1.0 each) or dictionary
        mapping species to weights. Default is None, which uses a predefined dictionary.
    source_cams_glob : list of str, optional
        List of source categories to include. Default is None, which uses a
        predefined list.

    """

    def __init__(
        self,
        data_path: Union[str, Path],
        year: int,
        lumping_co2: Optional[Union[list[str], dict[str, float]]] = None,
        lumping_co: Optional[Union[list[str], dict[str, float]]] = None,
        lumping_nox: Optional[Union[list[str], dict[str, float]]] = None,
        lumping_vocs: Optional[Union[list[str], dict[str, float]]] = None,
        source_cams_glob: Optional[Union[list[str], dict[str, float]]] = None,
    ) -> None:
        self.path = Path(data_path) / str(year)
        self.year = year

        self.nc_files = []  # list(self.path.rglob("*.nc"))

        self.source_cams_glob = (
            [
                "anthropogenic",
                "biogenic",
                "oceanic",
                "shipping",
            ]
            if source_cams_glob is None
            else source_cams_glob
        )

        for source in self.source_cams_glob:
            path_sector = self.path / source
            if path_sector.exists():
                self.nc_files.extend(list(path_sector.rglob("*.nc")))

        self.lumping_co2 = (
            {
                "co2-excl-short": 1.0,
                "co2-short-cycle": 1.0,
                "carbon-dioxide": 1.0,
            }
            if lumping_co2 is None
            else _as_weights(lumping_co2)
        )

        self.lumping_co = (
            {
                "carbon-monoxide": 1.0,
            }
            if lumping_co is None
            else _as_weights(lumping_co)
        )

        self.lumping_nox = (
            {
                "nitrogen-oxides": 1.0,
            }
            if lumping_nox is None
            else _as_weights(lumping_nox)
        )

        ### Values are from the ACTRIS data portal (https://data.eurochamp.org/data-access/kin/#/data)
        k_oh = (
            2.44e-11  # propane OH reaction rate constant at 298 K in cm3 molecule-1 s-1
        )
        self.lumping_vocs = (
            {
                "acetylene": 7.5e-13 / k_oh,
                "alcohols": 8.78e-13 / k_oh,
                "benzene": 1.2e-12 / k_oh,
                "butanes": 2.38e-12 / k_oh,
                # "chlorinated-hydrocarbons": , # TODO
                "esters": 8.75e-13 / k_oh,
                "ethane": 2.49e-13 / k_oh,
                "ethene": 8.52e-12 / k_oh,
                "ethers": 2.8e-12 / k_oh,
                # "formaldehyde": 1.0, there is its own variable
                "hexanes": 4.97e-12 / k_oh,
                "isoprene": 1.0e-10 / k_oh,
                "monoterpenes": 5.3e-11 / k_oh,
                # "non-methane-vocs": 1.0, # drop this for now
                "other-aldehydes": 1.5e-11 / k_oh,
                "other-alkenes-alkynes": 3.1e-11 / k_oh,
                "other-aromatics": 7e-12 / k_oh,
                # "other-vocs": 1.0, # TODO
                "pentanes": 3.76e-12 / k_oh,
                "propane": 1.1e-12 / k_oh,
                "propene": 2.44e-11 / k_oh,  # the reference value for k_oh
                "toluene": 5.6e-12 / k_oh,
                "total-acids": 1.87e-13 / k_oh,
                "total-ketones": 1.87e-13 / k_oh,
                "trimethylbenzenes": 3.3e-11 / k_oh,
                "xylenes": 1.7e-11 / k_oh,
                # "dimethyl-sulphide": 1.0, oceanic
                # "tribromomethane": 1.0, oceanic
                # "vocs-all": 1.0,
            }
            if lumping_vocs is None
            else _as_weights(lumping_vocs)
        )

        self.formaldehyde = ["formaldehyde"]

    def co2(self) -> list[Path]:
        """Return a list of NetCDF files for CO2 species."""
        return [
            file
            for file in self.nc_files
            if any(
                file.name.endswith(f"_{species}_{self.year}.nc")
                for species, weight in self.lumping_co2.items()
            )
        ]

    def co(self) -> list[Path]:
        """Return a list of NetCDF files for CO species."""
        return [
            file
            for file in self.nc_files
            if any(
                file.name.endswith(f"_{species}_{self.year}.nc")
                for species, weight in self.lumping_co.items()
            )
        ]

    def nox(self) -> list[Path]:
        """Return a list of NetCDF files for NOx species."""
        return [
            file
            for file in self.nc_files
            if any(
                file.name.endswith(f"_{species}_{self.year}.nc")
                for species, weight in self.lumping_nox.items()
            )
        ]

    def vocs(self) -> list[Path]:
        """Return a list of NetCDF files for VOCs species."""
        return [
            file
            for file in self.nc_files
            if any(
                file.name.endswith(f"_{species}_{self.year}.nc")
                for species, weight in self.lumping_vocs.items()
            )
        ]

    def vocs_weights(self) -> dict[str, float]:
        """Return a dictionary mapping VOC species to their weights."""
        return self.lumping_vocs

    def hcho(self) -> list[Path]:
        """Return a list of NetCDF files for formaldehyde (HCHO)."""
        return [
            file
            for file in self.nc_files
            if file.name.endswith(f"_formaldehyde_{self.year}.nc")
        ]


__all__ = [
    "CamsGlobInventoryPaths",
    "calculate_sun_position",
    "compare_netcdf_structure",
    "make_white_inferno",
    "make_white_plasma",
]
