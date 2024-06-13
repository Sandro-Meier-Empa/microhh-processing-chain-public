import cartopy
import matplotlib.figure
import os
import scipy.stats
import sys

from pathlib import Path
from typing import Union
from matplotlib.lines import Line2D
from matplotlib import cm
from matplotlib.colors import Normalize

import netCDF4 as nc
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.join("/users", os.environ["USER"], "microhh", "python"))
sys.path.append(os.path.join("/users", os.environ["USER"], "processing_chain", "src"))
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

        """
        self.case = case
        self.case_name = case.case_name
        self.suffix = case.suffix
        self.case_path = case.case_path
        self.file_directory = case.file_directory

        self.post = proc.PostProcessing(self.case)

    def plot_initial_conditions(self, case: init.Case, ncols=4):
        """Plot the initial conditions in separate plots for the different groups.

        Args:
            case (init.Case):
                TODO
            ncols (int, optional):
                TODO. Defaults to 4.
        """

        file_path = Path(self.file_directory, f"{case.case_name}_input.nc")

        groups = list(nc.Dataset(file_path).groups.keys())

        # loop over all groups to get the dimensions of all variables
        for group in groups:
            data_group = xr.open_dataset(file_path, group=group, decode_times=False)
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

    def plot_prognostic_variables(
        self,
        data: xr.Dataset = None,
        variables: list = ["u", "v", "thl", "qt"],
        n_lines: int = 5,
        colorful: bool = True,
        time_format: str = "h",
    ):
        variables = proc.vars_to_list(variables)

        # define factor for time format
        if time_format == "s":
            time_factor = 1
        elif time_format == "min":
            time_factor = 60
        elif time_format == "h":
            time_factor = 60**2
        else:
            raise ValueError(
                f"Time format {time_format} is not supported. Use 's', 'min' or 'h' instead."
            )
        # read in simulated data
        if data is None:
            post_processing = proc.PostProcessing(self.case)
            data = post_processing.handle_prognostic_variables(
                ["u", "v", "w", "thl", "ch4"]
            )

        # read in initial conditions
        data_initial = init.read_initial_cond(self.case)

        fig, ax = plt.subplots(ncols=len(variables), sharey=True)

        # define step size for plot
        t_step = int(data.time.size / n_lines)
        x_step = int(data.x.size / n_lines)
        y_step = int(data.y.size / n_lines)

        # Normalize the time values to range [0, 1] for colormap
        norm = Normalize(vmin=0, vmax=data.time.max() / time_factor)

        # Create a ScalarMappable and set the colormap
        sm = plt.cm.ScalarMappable(cmap=cm.viridis, norm=norm)
        sm.set_array([])  # Only needed for the colorbar to work

        # plot simulated values
        for t in np.arange(data.time.size, step=t_step):
            color_sim = (
                cm.viridis(norm(data.time[t] / time_factor)) if colorful else "gray"
            )
            color_init = "C3" if colorful else "C0"

            # loop over x dimension
            for x in np.arange(
                int(data.x.min().item()), int(data.x.max().item()), x_step
            ):
                # loop over y dimension
                for y in np.arange(
                    int(data.y.min().item()), int(data.y.max().item()), y_step
                ):
                    subset = data.isel(time=t, x=x, y=y)
                    for c, component in enumerate(variables):
                        ax[c].plot(subset[component], subset.z, color=color_sim)

        # plot initial conditions
        for c, component in enumerate(variables):
            ax[c].plot(
                data_initial[component],
                data_initial.z,
                label="prescribed",
                color=color_init,
            )
            ax[c].set_xlabel(f"{component} [{data[component].attrs['units']}]")

        ax[0].set_ylabel("Height [m]")

        if colorful:
            # create colorbar
            cbar = fig.colorbar(sm, ax=ax)
            cbar.set_label(f"Time [{time_format}]")
            ax[-1].legend(loc="upper right")
        else:
            # create legend manually
            lines = [Line2D([0], [0], color=c) for c in [color_init, color_sim]]
            labels = ["initial", "simulated"]
            ax[-1].legend(lines, labels, loc="upper right")

        return fig, ax

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
            data = self.post.open_3d_prognostic_variable(
                f"{self.case_name}_output_vars"
            )
        except:
            data = self.post.open_3d_prognostic_variable(variable)

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
        gas (_type_): TODO
        data (_type_): TODO
        result (_type_): TODO
        title (_type_, optional): TODO. Defaults to None.

    Returns:
        _type_: TODO
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

    text.append("\nMean wind:")
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


def plot_gas_weight(
    data: xr.Dataset, gas: str = "ch4", fig_width: Union[int, float] = 15
):

    if "time" in data.sizes:
        raise ValueError("Select only one time step.")

    if f"{gas}_weight" not in data.data_vars:
        raise KeyError(
            f"'{gas}_weight' is not available in dataset. Use 'calculate_weight_profile' from module 'wind' first."
        )

    x_max = data["x"].max().values
    y_max = data["y"].max().values
    aspect_ratio = int(x_max / y_max) + 1

    fig_height = fig_width / aspect_ratio

    fig, ax = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(fig_width, fig_height),
        gridspec_kw={
            "width_ratios": [4, 1],
        },
    )

    # First subplot for pcolormesh
    cax = ax[0].pcolormesh(
        data.x,
        data.y,
        data[gas]
        .integrate(coord="z")
        .where(data[f"{gas}_weight"].integrate(coord="z")),
    )
    ax[0].set_aspect("equal")
    ax[0].set_xlabel("x [m]")
    ax[0].set_ylabel("y [m]")
    weight = f"{gas}_weight"
    ax[0].legend(
        [],
        [],
        title=f"{gas.upper()} column > {data[weight].attrs.get('threshold', '-')}",
    )

    fig.colorbar(
        cax,
        ax=ax[0],
        label=f"{gas.upper()} [{data[gas].attrs.get('units', '-')}]",
    )

    # Calculate statistics
    weight = f"{gas}_weight"
    gas_median = data[weight].median(dim=["x", "y"])
    gas_min = data[weight].min(dim=["x", "y"])
    gas_max = data[weight].max(dim=["x", "y"])

    # Plot weight profile with min-max range
    ax[1].plot(gas_median, data.z, label="mean")  # Plotting the mean
    ax[1].fill_betweenx(
        data.z, gas_min, gas_max, label="value range", color="gray", alpha=0.5
    )

    ax[1].set_xlabel("Weight [-]")
    ax[1].set_ylabel("Height [m]")
    ax[1].legend()

    plt.tight_layout()
    plt.show()
