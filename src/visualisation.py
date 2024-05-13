import cartopy
import matplotlib.figure
import os
import scipy.stats
import sys

from pathlib import Path
from typing import Union

import netCDF4 as nc
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(f'/users/{os.environ["USER"]}/microhh/python')
import microhh_tools as mht

sys.path.append(f'/users/{os.environ["USER"]}/processing_chain/src/')
sys.path.append(os.path.join("/users", os.environ["USER"], "ddeq", "src"))
import processing as proc
import initialisation as init
import ddeq as ddeq


class Visualisation:

    wind_dims = {
        "u": ["xh", "y", "z"],
        "v": ["x", "yh", "z"],
        "w": ["x", "y", "zh"],
    }

    def __init__(self, case: init.Case):
        """Visualise the model output

        Args:
            case (init.Case):
                Case to visualise.
            source (str) ['scratch', 'store']:
                Source from where the case should be read.

        """
        self.case = case
        self.case_name = case.case_name
        self.suffix = case.suffix
        self.case_path = case.case_path
        self.scratch_path = case.scratch_path
        self.store_path = case.store_path

        self.post = proc.PostProcessing(self.case)

    def plot_initial_conditions(self, case: init.Case, ncols=4):
        """Plot the initial conditions in separate plots for the different groups.

        Args:
            case (init.Case): _description_
            ncols (int, optional): _description_. Defaults to 4.
        """

        fname = case.get_file_location(file="_input.nc")

        groups = list(nc.Dataset(fname).groups.keys())

        # loop over all groups to get the dimensions of all variables
        for group in groups:
            data_group = xr.open_dataset(fname, group=group, decode_times=False)
            dims = []
            for var in data_group.data_vars:
                dims.append(data_group[var].dims)

            # collect all variables within one group which share the dimension
            for sub_dims in list(set(dims)):
                vars = []
                for var in data_group.data_vars:
                    if data_group[var].dims == sub_dims:
                        vars.append(var)

                fig, axs = plt.subplots(
                    ncols=ncols,
                    nrows=np.ceil(len(vars) / ncols).astype(int),
                    sharey=True,
                    layout="constrained",
                )

                fig.suptitle(f"{group=}")

                for i, ax in enumerate(axs.flat):
                    if i < len(vars):
                        if len(sub_dims) == 0:
                            ax.plot(
                                np.full(
                                    len(data_group["z"]), data_group[vars[i]].values
                                ),
                                data_group["z"],
                            )
                        elif len(sub_dims) == 1 and "z" in sub_dims:
                            ax.plot(data_group[vars[i]], data_group["z"])
                        elif len(sub_dims) == 1 and "time_surface" in sub_dims:
                            ax.plot(data_group["time_surface"], data_group[vars[i]])
                        elif len(sub_dims) == 2:
                            ax.pcolormesh(
                                data_group[sub_dims[0]],
                                data_group[sub_dims[1]],
                                data_group[vars[i]].T,
                            )
                        ax.set_title(vars[i])
                    else:
                        ax.axis("off")

                if "z" in sub_dims:
                    fig.supylabel("Height [m]")
                if any("time" in s for s in sub_dims):
                    fig.supxlabel("Time [s]")

                plt.show()

    def plot_side_views(
        self,
        variable: str,
        timestep: int,
        fig_width: Union[int, float] = 15,
        cbar_width: Union[int, float] = 0.01,
    ) -> matplotlib.figure.Figure:
        """Plot 3D Model output in a figure with 3 subplots.
            1. YZ-View (integrated along x axis)
            2. XY-View (integrated along z axis)
            3. XZ-View (integrated along y axis)
            Added text information of the model resolution and wind properties.

        Args:
            variable (str):
                Name of the variable to be plotted.
            timestep (int):
                Time step to be plotted.
            fig_width (Union[int, float], optional. Default: 15):
                Width of figure.
            cbar_width (Union[int, float], optional: Default 0.01.):
                Width of colorbar.

        Returns:
            matplotlib.figure.Figure: Figure with subplots and text
        """

        variable = variable.upper()

        try:
            data = self.post.open_3d_netCDF(f"{self.case_name}_output_vars")
        except:
            data = self.post.open_3d_netCDF(variable)

        xdim, ydim, zdim = (
            self.wind_dims[variable]
            if variable in self.wind_dims.keys()
            else ["x", "y", "z"]
        )

        if data.time.size > 1 and "time" in data.dims:
            data = data.isel(time=timestep)

        x_max = data[xdim].max().values
        y_max = data[ydim].max().values
        z_max = data[zdim].max().values

        if variable in self.wind_dims.keys():
            dd_yz = data.mean(dim=xdim)
            dd_xy = data.mean(dim=zdim)
            dd_xz = data.mean(dim=ydim)
        else:
            dd_yz = data.sum(dim=xdim)
            dd_xy = data.sum(dim=zdim)
            dd_xz = data.sum(dim=ydim)

        aspect_ratio = (z_max + x_max) / (z_max + y_max)
        fig_height = fig_width / aspect_ratio

        fig, axs = plt.subplots(
            2,
            2,
            figsize=(fig_width, fig_height),
            gridspec_kw={
                "width_ratios": [1, x_max / z_max],
                "height_ratios": [y_max / z_max, 1],
            },
        )

        axs[1, 0].remove()

        # Side view along X

        im = axs[0, 0].pcolormesh(dd_yz[zdim], dd_yz[ydim], dd_yz[variable].T)
        axs[0, 0].set_xlabel(f"{zdim} [{data[zdim].attrs.get('units', '-')}]")
        axs[0, 0].set_ylabel(f"{ydim} [{data[ydim].attrs.get('units', '-')}]")

        # Top-down view
        im = axs[0, 1].pcolormesh(dd_xy[xdim], dd_xy[ydim], dd_xy[variable])
        axs[0, 1].set_xticklabels([])
        axs[0, 1].set_yticklabels([])

        # Side view along Y
        im = axs[1, 1].pcolormesh(dd_xz[xdim], dd_xz[zdim], dd_xz[variable])
        axs[1, 1].set_xlabel(f"{xdim} [{data[xdim].attrs.get('units', '-')}]")
        axs[1, 1].set_ylabel(f"{zdim} [{data[zdim].attrs.get('units', '-')}]")

        # Get the position of the rightmost subplot
        right_bottom = axs[1, 1].get_position()
        left_top = axs[0, 0].get_position()

        # Set the position and size of the colorbar
        cbar_ax = fig.add_axes(
            [
                left_top.xmax - cbar_width,
                right_bottom.ymin,
                cbar_width,
                right_bottom.ymax - right_bottom.ymin,
            ]
        )
        cb = fig.colorbar(
            im,
            cax=cbar_ax,
            label=f"{variable} " f"[{data[{variable}].attrs.get('units', '-')}]",
            ticklocation="left",
        )

        text = self.add_text(data, timestep, xdim, ydim, zdim)

        fig.text(
            left_top.xmin,
            right_bottom.ymax,
            text,
            horizontalalignment="left",
            verticalalignment="top",
        )

        return fig

    def add_text(
        self, data: xr.Dataset, timestep: int, xdim: str, ydim: str, zdim: str
    ) -> str:
        """Format text information of the model resolution and wind properties
        for 'plot_side_views'

        Args:
            data (xr.Dataset):
                Dataset from which one time step should be plotted.
            timestep (int):
                Time steop which should be plotted.

        Returns:
            str: Formatted text.
        """

        text = []

        # Information about time
        if pd.api.types.is_numeric_dtype(data["time"]):
            text.append(f"Timestep: {timestep}, " f"Time: {int(data['time'].values)} s")
        elif pd.api.types.is_datetime64_any_dtype(data["time"]):
            formatted_time = data["time"].dt.strftime("%H:%M:%S %d.%m.%Y").values[0]
            text.append(f"Timestep: {timestep}, " f"Time: {formatted_time}")
        else:
            text.append("Time: Unsupported data type")

        # Information about resolution

        text.append(f"\nGrid resolution")
        text.append(f"x: {data[xdim].diff(xdim).mean().values:.1f} m")
        text.append(f"y: {data[ydim].diff(ydim).mean().values:.1f} m")
        text.append(f"z: {data[zdim].diff(zdim).mean().values:.1f} m")

        # Information about wind
        try:
            text.append(f"\nMean wind")
            for wind_component in ["u", "v"]:
                text.append(
                    f"{wind_component}: " f"{data[wind_component].mean():.1f} m/s"
                )
        except:
            text.append("Wind data not found.")

        # Information about sources
        try:
            source_strength = self.post.model_emissions_to_kg_per_h()
            source_height = self.post.ini_file["source"]["source_z0"]
            if len(source_strength) > 1:
                # Multiple sources
                text.append(f"\nSources")
                formatted_strength = ", ".join(f"{x:.2f}" for x in source_strength)
                formatted_height = ", ".join(f"{x:.2f}" for x in source_height)
            else:
                # Single source
                text.append(f"\nSource")
                formatted_strength = f"{source_strength[0]:.2f}"
                formatted_height = f"{source_height:.0f}"

            text.append(f"source strength: {formatted_strength} kg/h")
            text.append(f"source height: {formatted_height} m")

        except RuntimeError:
            pass

        return "\n".join(text)


def plot_ime_result(
    gas: str,
    data: xr.Dataset,
    winds: xr.Dataset,
    line_densities: xr.Dataset,
    curves: ddeq.plume_coords.Poly2D,
    source: str,
    domain=None,
    crs=None,
    scaling: Union[int, float] = None,
    vmin: Union[int, float] = None,
    vmax: Union[int, float] = None,
    title=None,
):

    if domain is None:
        domain = ddeq.misc.Domain(
            name="",
            startlat=data.lat.min() - 0.2,
            stoplat=data.lat.max() + 0.2,
            startlon=data.lon.min() - 0.2,
            stoplon=data.lon.max() + 0.2,
        )

    with plt.style.context({"font.size": 10}):
        fig_scaling = 1.0
        figsize = (13.94 * fig_scaling, 7.45 * fig_scaling)

        # create canvas for the subfigures
        fig = plt.figure(figsize=figsize)
        subfigs = fig.subfigures(1, 2, width_ratios=[2, 1])

        vmin = None if vmin is None else vmin
        vmax = None if vmax is None else vmax
        scaling = 1.0 if scaling is None else scaling

        # set titles
        subfigs[0].suptitle("a")
        subfigs[1].suptitle("b")

        # plot satellite image(s)
        axs_left = subfigs[0].add_subplot(111, projection=domain.proj)

        axs_left.set_aspect("equal", adjustable="box")
        axs_left.set_xlim(domain.startlon, domain.stoplon)
        axs_left.set_ylim(domain.startlat, domain.stoplat)
        ddeq.vis.add_lines(axs_left)

        with xr.set_options(keep_attrs=True):
            scaled_data = scaling * data[gas]

        # plot map with detected plume
        sat = ddeq.vis.show_detected_plumes(
            data,
            curves,
            scaled_data,
            gas=gas,
            add_polygons=False,
            winds=winds,
            domain=domain,
            zoom_on=source,
            do_zoom=False,
            ax=axs_left,
            cax=None,
            vmin=vmin,
            vmax=vmax,
            crs=crs,
        )

        # adjust margins
        # subfigs[0].subplots_adjust(left=0.0,
        #                            right=0.93,
        #                            bottom=ax_bottom,
        #                            top=ax_top)

        # add retrieval information
        text = add_ime_text(gas, data.sel(source=source), line_densities, title=title)
        subfigs[1].text(0.0, 0.75, text, va="top", linespacing=1.1)

        return fig


def add_ime_text(
    gas: str, data: xr.Dataset, result: xr.Dataset, title: str = None
) -> list:
    """Add retrieval information

    Args:
        gas (_type_): _description_
        data (_type_): _description_
        result (_type_): _description_
        title (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """

    time = pd.Timestamp(data.time.values)
    plume_size = np.nansum(data.detected_plume)

    text = []
    detection_gas = data.attrs["trace_gas"]

    if title is not None:
        text.append(title)

    text.append("\nInstrument:")
    text.append(f'• Time: {time.round(freq="S")}')
    text.append(
        f'• Resolution: {np.sqrt(np.mean(data.pixel_area.values)):.1f} {data.pixel_area.attrs.get("units", "-")[:-1]}'
    )

    noise = data[gas].attrs["noise_level"]
    units = data[gas].attrs["units"]
    text.append(f"• {ddeq.vis.sub_numbers(gas)} noise: {noise:g} {units}")

    text.append("")
    text.append("Plume detection:")
    text.append(f"• Trace gas: {ddeq.vis.sub_numbers(detection_gas)}")
    text.append(f'• z-value: {data.attrs.get("probability for z-value", "-"):g}')
    text.append(f'• Filter size: {data.attrs.get("filter_size", "-"):g}')
    text.append(f"• Plume size: {plume_size:d} px")
    text.append(
        f'• Plume length: {result.L_max.values[0]:.1f} {result.L_max.attrs.get("units", "-")}'
    )

    text.append("\nWind:")
    text.append(f"• Speed: {np.mean(result.wind_speed):.1f} m/s")
    text.append(
        f"• Direction: {scipy.stats.circmean(result.wind_direction, high=360):.0f}°"
    )

    text.append(f"\n{ddeq.vis.sub_numbers(gas)} emissions:")

    estimated_emissions = result[f"{gas}_estimated_emissions"].values[0] * 3600
    estimated_emissions_unc = (
        result[f"{gas}_estimated_emissions_precision"].values[0] * 3600
    )

    if "strength" in result.data_vars:
        true_emissions = result.strength.values[0]
        text.append(f"• Reported: {true_emissions:.3g} kg/h")
    text.append(
        f"• Estimated: {estimated_emissions:.3g} ± {estimated_emissions_unc:.2g} kg/h"
    )

    return "\n".join(text)
