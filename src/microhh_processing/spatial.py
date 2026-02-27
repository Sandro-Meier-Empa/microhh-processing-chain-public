import numpy as np

from pyproj import Transformer
from typing import Union, Literal, Tuple
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info

import matplotlib.pyplot as plt
import cartopy.crs as ccrs

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
        epsg_code: str = None,
    ):
        """Initialize the Domain object.
        Adapted from https://github.com/microhh/microhhpy
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
        ref_x: int,
        ref_y: int,
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

    def to_lonlat(self, x, y, meshgrid=False):
        x = np.array(x)
        y = np.array(y)

        if meshgrid:
            # indexing='ij' is often safer for matrix-based CFD data
            x, y = np.meshgrid(x, y, indexing="xy")

        return self.transformer.transform(
            x + self.x_offset, y + self.y_offset, direction="INVERSE"
        )

    def to_xy(self, lon, lat):
        lon = np.array(lon)
        lat = np.array(lat)

        tx, ty = self.transformer.transform(lon, lat)
        return tx - self.x_offset, ty - self.y_offset

    def plot(
        self,
        lon: Union[float, np.ndarray] = None,
        lat: Union[float, np.ndarray] = None,
        figsize: Tuple[int, int] = (10, 8),
    ) -> None:

        fig, ax = plt.subplots(
            figsize=figsize,
            subplot_kw={"projection": ccrs.PlateCarree()},
        )

        # Plot the reference point
        ax.plot(self.ref_lon, self.ref_lat, "ko", markersize=5, label="Reference Point")

        # Plot additional points if provided
        if lon is not None and lat is not None:
            ax.plot(lon, lat, "ro", markersize=3, label="Points")

        # Plot the domain boundaries
        ax.plot(self.lon_h[0, :], self.lat_h[0, :], "b-")  # Bottom edge
        ax.plot(self.lon_h[-1, :], self.lat_h[-1, :], "b-")  # Top edge
        ax.plot(self.lon_h[:, 0], self.lat_h[:, 0], "b-")  # Left edge
        ax.plot(self.lon_h[:, -1], self.lat_h[:, -1], "b-")  # Right edge

        ax.set_xlabel("Longitude [°]")
        ax.set_ylabel("Latitude [°]")
        ax.legend(loc="upper right")

        gl = ax.gridlines(
            crs=ccrs.PlateCarree(),
            draw_labels=True,
            linewidth=0.8,
            linestyle="--",
            alpha=0.6,
        )
        gl.top_labels = False
        gl.right_labels = False

        return fig


def get_epsg_from_pyproj(lat, lon):
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
