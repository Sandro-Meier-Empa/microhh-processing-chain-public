"""Spatial coordinate transformations for MicroHH domains."""

from typing import Literal, Optional, Union, cast

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from cartopy.mpl.geoaxes import GeoAxes
from matplotlib.figure import Figure
from pyproj import Transformer
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info

import ddeq


class Domain:
    """Represent a projected MicroHH domain and its lon/lat grid.

    Adapted from https://github.com/microhh/microhhpy

    Args:
        xsize: Domain extent in the x-direction in meters.
        ysize: Domain extent in the y-direction in meters.
        itot: Number of grid points in the x-direction.
        jtot: Number of grid points in the y-direction.
        lon: Longitude of the reference point.
        lat: Latitude of the reference point.
        anchor: Anchor point for the reference coordinates, either a named
            anchor or x/y offsets in meters from the reference point.
        epsg_code: Coordinate reference system used for metric coordinates.
            When omitted, a matching UTM zone is inferred from the reference point.
    """

    def __init__(
        self,
        xsize: int,
        ysize: int,
        itot: int,
        jtot: int,
        lon: float,
        lat: float,
        anchor: Union[
            Literal[
                "center",
                "lower left",
                "lower right",
                "upper left",
                "upper right",
                "lower center",
                "upper center",
                "center left",
                "center right",
            ],
            tuple[float, float],
        ] = "lower left",
        epsg_code: Optional[str] = None,
    ) -> None:
        if epsg_code is None:
            epsg_code_str = get_epsg_from_pyproj(lat, lon)
            epsg_code = f"epsg:{epsg_code_str}"

        self.transformer = Transformer.from_crs("epsg:4326", epsg_code, always_xy=True)

        self.itot, self.jtot = itot, jtot

        self.dx, self.dy = xsize / itot, ysize / jtot
        self.xsize, self.ysize = xsize, ysize

        # Grid definitions (Arakawa C-grid style)
        self.x = np.linspace(self.dx / 2, xsize - self.dx / 2, itot)  # Centers
        self.xh = np.linspace(0, xsize, itot + 1)  # Edges/Faces

        self.y = np.linspace(self.dy / 2, ysize - self.dy / 2, jtot)
        self.yh = np.linspace(0, ysize, jtot + 1)

        self.xx, self.yy = np.meshgrid(
            self.x, self.y, indexing="xy"
        )  # Meshgrid for centers

        # Handle Offsets
        self.ref_lon, self.ref_lat = lon, lat
        ref_x, ref_y = self.transformer.transform(self.ref_lon, self.ref_lat)

        self.x_offset, self.y_offset = self._get_offsets(ref_x, ref_y, anchor)

        # Compute lon/lat arrays
        self.lon, self.lat = self.to_lonlat(self.x, self.y, meshgrid=True)
        self.lon_h, self.lat_h = self.to_lonlat(self.xh, self.yh, meshgrid=True)

        self.bbox = (
            float(self.lon.min()),
            float(self.lon.max()),
            float(self.lat.min()),
            float(self.lat.max()),
        )

    def _get_offsets(
        self,
        ref_x: float,
        ref_y: float,
        anchor: Union[
            Literal[
                "center",
                "lower left",
                "lower right",
                "upper left",
                "upper right",
                "lower center",
                "upper center",
                "center left",
                "center right",
            ],
            tuple[float, float],
        ],
    ) -> tuple[float, float]:
        """Return projected offsets for the configured domain anchor."""
        if isinstance(anchor, tuple):
            x_rel, y_rel = anchor
            return ref_x - x_rel, ref_y - y_rel
        elif anchor == "center":
            return ref_x - self.xsize / 2.0, ref_y - self.ysize / 2.0
        elif anchor == "lower left":
            return ref_x, ref_y
        elif anchor == "lower right":
            return ref_x - self.xsize, ref_y
        elif anchor == "upper left":
            return ref_x, ref_y - self.ysize
        elif anchor == "upper right":
            return ref_x - self.xsize, ref_y - self.ysize
        elif anchor == "lower center":
            return ref_x - self.xsize / 2.0, ref_y
        elif anchor == "upper center":
            return ref_x - self.xsize / 2.0, ref_y - self.ysize
        elif anchor == "center left":
            return ref_x, ref_y - self.ysize / 2.0
        elif anchor == "center right":
            return ref_x - self.xsize, ref_y - self.ysize / 2.0
        else:
            raise ValueError("Invalid anchor!")

    def to_lonlat(
        self,
        x: Union[float, list[float], np.ndarray],
        y: Union[float, list[float], np.ndarray],
        meshgrid: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert x/y coordinates to longitude and latitude.

        Args:
            x: Coordinates in meters along the model x-axis.
            y: Coordinates in meters along the model y-axis.
            meshgrid: Return coordinate grids instead of coordinate vectors.

        Returns:
            Longitude and latitude arrays.
        """
        x = np.array(x)
        y = np.array(y)

        if meshgrid:
            # indexing='ij' is often safer for matrix-based CFD data
            x, y = np.meshgrid(x, y, indexing="xy")

        return self.transformer.transform(
            x + self.x_offset, y + self.y_offset, direction="INVERSE"
        )

    def to_xy(
        self,
        lon: Union[float, list[float], np.ndarray],
        lat: Union[float, list[float], np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert longitude and latitude to x/y coordinates.

        Args:
            lon: Longitude values.
            lat: Latitude values.

        Returns:
            Model x and y coordinates in meters.
        """
        lon = np.array(lon)
        lat = np.array(lat)

        tx, ty = self.transformer.transform(lon, lat)
        return tx - self.x_offset, ty - self.y_offset

    def plot(
        self,
        lon: Optional[Union[float, np.ndarray]] = None,
        lat: Optional[Union[float, np.ndarray]] = None,
        figsize: tuple[int, int] = (10, 8),
    ) -> Figure:
        """Plot the domain boundaries and reference point.

        Args:
            lon: Optional longitude values of points to plot.
            lat: Optional latitude values of points to plot.
            figsize: Figure size in inches.

        Returns:
            Figure containing the domain plot.
        """
        fig, ax = plt.subplots(
            figsize=figsize,
            subplot_kw={"projection": ccrs.PlateCarree()},
        )
        geo_ax: GeoAxes = cast(GeoAxes, ax)

        # Plot the reference point
        geo_ax.plot(
            self.ref_lon, self.ref_lat, "ko", markersize=5, label="Reference Point"
        )

        # Plot additional points if provided
        if lon is not None and lat is not None:
            geo_ax.plot(lon, lat, "ro", markersize=3, label="Points")

        # Plot the domain boundaries
        geo_ax.plot(self.lon_h[0, :], self.lat_h[0, :], "b-")  # Bottom edge
        geo_ax.plot(self.lon_h[-1, :], self.lat_h[-1, :], "b-")  # Top edge
        geo_ax.plot(self.lon_h[:, 0], self.lat_h[:, 0], "b-")  # Left edge
        geo_ax.plot(self.lon_h[:, -1], self.lat_h[:, -1], "b-")  # Right edge

        geo_ax.set_xlabel("Longitude [°]")
        geo_ax.set_ylabel("Latitude [°]")
        geo_ax.legend(loc="upper right")

        gl = geo_ax.gridlines(
            crs=ccrs.PlateCarree(),
            draw_labels=True,
            linewidth=0.8,
            linestyle="--",
            alpha=0.6,
        )
        gl.top_labels = False
        gl.right_labels = False

        return fig

    def create_grid_nc(self) -> xr.Dataset:
        """Create an xarray Dataset containing the grid information.

        The dataset can be used to remap data to the same grid as the
        MicroHH simulation.

        Returns:
            Dataset with cell centers and cell bounds.
        """
        lon_grid, lat_grid = self.to_lonlat(self.x, self.y, meshgrid=True)

        data = xr.Dataset(
            coords={
                "lon": (
                    ["nrows", "nobs"],
                    lon_grid,
                    {
                        "units": "degrees_east",
                        "standard_name": "longitude",
                        "bounds": "lonc",
                    },
                ),
                "lat": (
                    ["nrows", "nobs"],
                    lat_grid,
                    {
                        "units": "degrees_north",
                        "standard_name": "latitude",
                        "bounds": "latc",
                    },
                ),
            },
            data_vars={
                "dummy": (["nrows", "nobs"], np.zeros_like(lon_grid)),
            },
            attrs={},
        )
        data_corner = ddeq.misc.add_corners(data)
        data_corner["lonc"].attrs = {"units": "degrees_east"}
        data_corner["latc"].attrs = {"units": "degrees_north"}

        if "vertices" not in data_corner.dims and "corners" in data_corner.dims:
            data_corner = data_corner.rename({"corners": "vertices"})

        return data_corner


def get_epsg_from_pyproj(lat: float, lon: float) -> str:
    """Return the UTM EPSG code containing the given coordinate."""
    # Query for UTM CRS info based on a single point (AOI)
    utm_crs_list = query_utm_crs_info(
        datum_name="WGS 84",
        area_of_interest=AreaOfInterest(
            west_lon_degree=lon,
            south_lat_degree=lat,
            east_lon_degree=lon,
            north_lat_degree=lat,
        ),
    )

    return utm_crs_list[0].code


__all__ = [
    "Domain",
    "get_epsg_from_pyproj",
]
