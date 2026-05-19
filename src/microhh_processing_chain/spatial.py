import numpy as np

from pyproj import Transformer
from typing import Optional, Union, Literal, Tuple, cast
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from cartopy.mpl.geoaxes import GeoAxes
import cartopy.crs as ccrs
import xarray as xr

import ddeq

__all__ = [
    "Domain",
    "get_epsg_from_pyproj",
]


class Domain:
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
        """Initialize the Domain object.
        Adapted from https://github.com/microhh/microhhpy

        Args:
            xsize (int): Size of the domain in the x-direction (meters).
            ysize (int): Size of the domain in the y-direction (meters).
            itot (int): Number of grid points in the x-direction.
            jtot (int): Number of grid points in the y-direction.
            lon (float): Longitude of the reference point (degrees). Must be in the
                same coordinate reference system as the provided EPSG code (if any).
            lat (float): Latitude of the reference point (degrees). Must be in the
                same coordinate reference system as the provided EPSG code (if any).
            anchor (str or tuple): Anchor point for the reference coordinates. Can be a string
                ("center", "lower left", "lower right", "upper left", "upper right", "lower center",
                "upper center", "center left", "center right") or a tuple of relative offsets
                (x_rel, y_rel) in meters from the reference point.
            epsg_code (str, optional): EPSG code for the coordinate reference system. If None,
                it will be determined automatically based on the provided latitude and longitude.
        """

        if epsg_code is None:
            epsg_code_str = get_epsg_from_pyproj(lat, lon)
            epsg_code = f"epsg:{epsg_code_str}"

        self.transformer = Transformer.from_crs("epsg:4326", epsg_code, always_xy=True)

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
        lon = np.array(lon)
        lat = np.array(lat)

        tx, ty = self.transformer.transform(lon, lat)
        return tx - self.x_offset, ty - self.y_offset

    def plot(
        self,
        lon: Optional[Union[float, np.ndarray]] = None,
        lat: Optional[Union[float, np.ndarray]] = None,
        figsize: Tuple[int, int] = (10, 8),
    ) -> Figure:

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
        The dataset can be used to remap data to the same grid as the MicroHH simulation.
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
