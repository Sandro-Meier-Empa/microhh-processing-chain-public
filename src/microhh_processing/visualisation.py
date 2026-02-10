# import cartopy
import matplotlib.figure
import os
import scipy.stats
import sys

from pathlib import Path
from typing import List, Literal, Sequence, Tuple, Union
from matplotlib.lines import Line2D
from matplotlib import cm
from matplotlib.colors import Normalize, TwoSlopeNorm
from IPython.display import display
from mpl_toolkits.axes_grid1 import make_axes_locatable

import netCDF4 as nc
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import ipywidgets as widgets

import microhh_processing.preprocessing as pre
import microhh_processing.postprocessing as post
import microhh_processing.initialisation as init
import microhh_processing.misc as misc
import microhh_processing.io as io
import microhh_processing.wind as wind_proc
import microhh_processing.sources as src

import ddeq

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

    def plot_initial_conditions(
        self, ncols: int = 4, figsize: tuple[int, int] = (15, 10)
    ) -> None:
        """
        Visualize initial conditions for different groups in a NetCDF dataset.

        This method creates subplots for variables within each group, handling
        different dimensional structures and plotting strategies.

        Args:
            case (init.Case): Case containing the initial conditions to visualize.
            ncols (int, optional): Number of columns in the subplot grid. Defaults to 4.
            figsize (tuple[int, int], optional): Figure size (width, height) in inches.
                Defaults to (15, 10).

        Raises:
            FileNotFoundError: If the NetCDF file for the case cannot be found.
            ValueError: If no data groups are found in the file.
        """

        file_path = Path(self.case.file_directory) / f"{self.case.name}_input.nc"

        if not file_path.exists():
            raise FileNotFoundError(f"NetCDF file not found: {file_path}")

        # Open dataset with context manager for proper resource management
        with nc.Dataset(file_path) as nc_dataset:
            groups = list(nc_dataset.groups.keys())

            if not groups:
                raise ValueError(f"No data groups found in {file_path}")

        # Process each group
        for group in groups:
            self._plot_group_variables(file_path, group, ncols, figsize)

    def _plot_group_variables(
        self, file_path: Path, group: str, ncols: int, figsize: tuple[int, int]
    ) -> None:
        """
        Plot variables for a specific group with intelligent subplot generation.

        Args:
            file_path (Path): Path to the NetCDF file.
            group (str): Name of the group to process.
            ncols (int): Number of columns in the subplot grid.
            figsize (tuple[int, int]): Figure size.
        """
        if group in ["timedep"]:
            sharey = False
        else:
            sharey = True

        # Use xarray for more convenient data handling
        with xr.open_dataset(file_path, group=group, decode_times=False) as data_group:
            # Group variables by their dimensions
            vars_by_dims = {}
            for var in data_group.data_vars:
                dims = data_group[var].dims
                if dims not in vars_by_dims:
                    vars_by_dims[dims] = []
                vars_by_dims[dims].append(var)

            # Plot each dimensional group
            for dims, vars_list in vars_by_dims.items():
                nrows = int(np.ceil(len(vars_list) / ncols))
                fig, axs = plt.subplots(
                    ncols=ncols,
                    nrows=nrows,
                    figsize=figsize,
                    sharey=sharey,
                    layout="constrained",
                )

                fig.suptitle(f"Group: {group}")

                # Determine auxiliary dimension for vertical plotting
                aux_dim = next((d for d in ["z", "lay"] if d in data_group.dims), None)

                # Flatten axs for consistent iteration
                axs_flat = axs.flat if nrows > 1 else [axs] if ncols == 1 else axs

                for i, (ax, var) in enumerate(zip(axs_flat, vars_list)):
                    self._plot_variable(data_group, var, dims, ax, aux_dim)
                    ax.set_title(var)

                # Hide unused subplots
                for ax in axs_flat[len(vars_list) :]:
                    ax.axis("off")

                # Label axes based on dimensions
                if aux_dim:
                    fig.supylabel(f"{aux_dim.upper()} [m]")

                if any("time" in str(d) for d in dims):
                    fig.supxlabel("Time [s]")

                plt.show()

    def _plot_variable(
        self,
        data_group: xr.Dataset,
        var: str,
        dims: tuple,
        ax: plt.Axes,
        aux_dim: str,
    ) -> None:
        """
        Intelligently plot a single variable based on its dimensions.

        Args:
            data_group (xr.Dataset): Dataset group containing the variable.
            var (str): Variable name to plot.
            dims (tuple): Dimensions of the variable.
            ax (plt.Axes): Matplotlib axis to plot on.
            aux_dim (str | None): Auxiliary dimension for vertical plots.
        """
        var_data = data_group[var]

        if len(dims) == 0:
            if aux_dim:
                ax.plot(
                    np.full(len(data_group[aux_dim]), var_data.values),
                    data_group[aux_dim],
                )
            else:
                # Fallback if no auxiliary dimension
                ax.plot([var_data.values])

        elif len(dims) == 1:
            # get dimension name that contains "time_"
            time_dim = next((d for d in dims if "time_" in d), None)

            if time_dim in dims:
                ax.plot(data_group[time_dim], var_data, marker="x")
            elif aux_dim in dims:
                ax.plot(var_data, data_group[aux_dim], marker="x")
            else:
                ax.plot(var_data, marker="x")

        elif len(dims) == 2:
            # 2D heatmap
            im = ax.pcolormesh(data_group[dims[0]], data_group[dims[1]], var_data.T)

            # add colorbar to bottom
            cbar = plt.colorbar(im, ax=ax, orientation="horizontal")

    def plot_prognostic_variables(
        self,
        variables: list[str],
        data: xr.Dataset = None,
        n_lines: int = 5,
        colorful: bool = True,
        time_format: str = "h",
        plot_format: Literal["line", "shaded"] = "line",
        color_init: str = "k",
    ):
        """
        Plot the prognostic variables
        """
        no_data = True if data is None else False
        variables = misc.object_to_list(variables)
        time_factor = self._get_time_factor(time_format)

        # read in initial conditions
        data_initial = io.open_initial_cond(self.case)

        fig, ax = plt.subplots(ncols=len(variables), sharey=True)

        for c, component in enumerate(variables):
            # if data is None, read data
            if no_data:
                data = io.open_microhh_nc(self.case, variable=component)

            # get dim names from variable
            time_dim, z_dim, y_dim, x_dim = data[component].dims
            t_step = int(data[time_dim].size / n_lines)

            # Normalize the time values to range [0, 1] for colormap
            norm = Normalize(vmin=0, vmax=data.time.max() / time_factor)

            # Create a ScalarMappable and set the colormap
            sm = plt.cm.ScalarMappable(cmap=cm.viridis, norm=norm)
            sm.set_array([])  # Only needed for the colorbar to work

            for t in np.arange(data[time_dim].size, step=t_step):
                subset_t = data.isel({time_dim: t})
                color = (
                    sm.to_rgba(subset_t[time_dim].values / time_factor)
                    if colorful
                    else "gray"
                )
                if plot_format == "line":
                    self._plot_lines(
                        ax=ax[c],
                        data=subset_t,
                        variable=component,
                        n_lines=n_lines,
                        x_dim=x_dim,
                        y_dim=y_dim,
                        z_dim=z_dim,
                        color=color,
                    )

                elif plot_format == "shaded":
                    self._plot_shaded(
                        ax=ax[c],
                        data=subset_t,
                        variable=component,
                        x_dim=x_dim,
                        y_dim=y_dim,
                        z_dim=z_dim,
                        color=color,
                    )

            # plot initial conditions
            ax[c].plot(
                data_initial[component],
                data_initial.z,
                label="prescribed",
                color=color_init,
            )
            ax[c].set_xlabel(f"{component} [{data[component].attrs.get('units', '-')}]")

        ax[0].set_ylabel("Height [m]")

        if colorful:
            # create colorbar
            cbar = fig.colorbar(sm, ax=ax)
            cbar.set_label(f"Time [{time_format}]")
            ax[-1].legend(loc="upper right")
        else:
            # create legend manually
            lines = [Line2D([0], [0], color=c) for c in [color_init, color]]
            labels = ["initial", "simulated"]
            ax[-1].legend(lines, labels, loc="upper right")

        return fig

    def _get_time_factor(self, time_format: str) -> int:
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

        return time_factor

    def _plot_lines(
        self,
        ax: plt.Axes,
        data: xr.Dataset,
        variable: str,
        n_lines: int,
        x_dim: str,
        y_dim: str,
        z_dim: str,
        color: Union[str, tuple],
    ):
        # define step size for plot
        x_step = int(data[x_dim].size / n_lines)
        y_step = int(data[y_dim].size / n_lines)

        # loop over x dimension
        for x in np.arange(data[x_dim].size, step=x_step):
            # loop over y dimension
            for y in np.arange(data[y_dim].size, step=y_step):
                subset = data.isel({x_dim: x, y_dim: y})
                ax.plot(subset[variable], subset[z_dim], color=color)

    def _plot_shaded(
        self,
        ax: plt.Axes,
        data: xr.Dataset,
        variable: str,
        x_dim: str,
        y_dim: str,
        z_dim: str,
        color: Union[str, tuple],
        alpha: float = 0.3,
    ):
        # integrate over x and y dimensions: Mean and std
        integrated = data[variable].mean(dim=[x_dim, y_dim])
        integrated_std = data[variable].std(dim=[x_dim, y_dim])
        ax.fill_betweenx(
            data[z_dim],
            integrated - integrated_std,
            integrated + integrated_std,
            color=color,
            alpha=alpha,
        )

    def plot_side_views(
        self,
        data: xr.Dataset,
        variable: str,
        timestep: int,
        fig_width: Union[int, float] = 15,
        cbar_width: Union[int, float] = 0.01,
        cmap: str = "viridis",
    ) -> matplotlib.figure.Figure:
        """Plot 3D Model output in a figure with 3 subplots.
            1. YZ-View (integrated along x axis)
            2. XY-View (integrated along z axis)
            3. XZ-View (integrated along y axis)
            Added text information of the model resolution and wind properties.

        Args:
            data (xr.Dataset):
                Dataset containing the variable
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

        time_dim, zdim, ydim, xdim = data[variable].dims
        data = data.isel({time_dim: timestep})

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

        im = axs[0, 0].pcolormesh(
            dd_yz[zdim], dd_yz[ydim], dd_yz[variable].T, cmap=cmap
        )
        axs[0, 0].set_xlabel(f"{zdim} [{data[zdim].attrs.get('units', '-')}]")
        axs[0, 0].set_ylabel(f"{ydim} [{data[ydim].attrs.get('units', '-')}]")

        # Top-down view
        im = axs[0, 1].pcolormesh(dd_xy[xdim], dd_xy[ydim], dd_xy[variable], cmap=cmap)
        axs[0, 1].set_xticklabels([])
        axs[0, 1].set_yticklabels([])

        # Side view along Y
        im = axs[1, 1].pcolormesh(dd_xz[xdim], dd_xz[zdim], dd_xz[variable], cmap=cmap)
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
            source_strength = src.get_source_strength(self.case, to_unit="kg/h")
            source_height = np.atleast_1d(self.case.ini_file["source"]["source_z0"])
            if len(source_strength) > 1:
                # Multiple sources
                text.append(f"\nSources")
                formatted_strength = ", ".join(f"{x:.2f}" for x in source_strength)
                formatted_height = ", ".join(f"{x:.2f}" for x in source_height)
            else:
                # Single source
                text.append(f"\nSource")
                formatted_strength = f"{source_strength[0]:.2f}"
                formatted_height = f"{source_height[0]:.0f}"

            text.append(f"source strength: {formatted_strength} kg/h")
            text.append(f"source height: {formatted_height} m")

        except RuntimeError:
            pass

        return "\n".join(text)

    def plot_stats(self, variables: Union[str, list]):
        """
        Plot variables from the statistics netcdf file.

        Args:
            variables (str, list(str)):
                Variables within the statistics file to plot.
        """
        stats = io.open_statistics(self.case)

        if isinstance(variables, str):
            variables = list(variables)

        for var in variables:
            fig, axs = plt.subplots(1, 1, figsize=(12, 6))

            if "z" in stats[var].dims:
                ydim = "z"
            elif "zh" in stats[var].dims:
                ydim = "zh"
            else:
                ydim = None

            if ydim is not None:
                stats[var].plot(ax=axs, y=ydim)
            else:
                stats[var].plot(ax=axs)

            plt.tight_layout()
            plt.show()

    def compare_nudge(self, cmap="RdBu_r", figsize=(15, 6), percent=False):

        stats = io.open_statistics(self.case)
        initial_cond = io.open_initial_cond(self.case)

        # get all variables that contain "_nudge" in their name
        nudge_vars = [var for var in initial_cond.data_vars if "_nudge" in var]
        if len(nudge_vars) == 0:
            raise ValueError("No nudge variables found in initial condition file.")

        # get all time coordinates in initial condition file
        time_vars = [c for c in initial_cond.coords if "time" in c]
        if len(time_vars) == 0:
            raise ValueError("No time coordinate found in initial condition file.")

        ic_time_values = [initial_cond[time_var].max() for time_var in time_vars]
        min_time = int(np.nanmin([stats.time.max(), *ic_time_values]))

        # subset to min time (apply to every time coord)
        stats_sub = stats.sel(time=slice(0, min_time))
        initial_cond_sub = initial_cond.sel({c: slice(0, min_time) for c in time_vars})

        # interpolate initial conditions to stats times
        interp_dict = {t: stats.time for t in time_vars}
        initial_cond_sub_interp = initial_cond_sub.interp(
            interp_dict,
            kwargs={
                "fill_value": "extrapolate"
            },  # allow extrapolation because time values may have numerical differences
        )

        fig, axs = plt.subplots(ncols=len(nudge_vars), figsize=figsize, sharey=True)
        for c, nudge_var in enumerate(nudge_vars):
            stat_var = nudge_var.replace("_nudge", "")
            values = stats_sub[stat_var] - initial_cond_sub_interp[nudge_var]

            if percent:
                values = 100 * values / initial_cond_sub_interp[nudge_var]
                units = "%"
            else:
                units = stats[stat_var].attrs.get("units", "-")

            vmax = np.nanquantile(np.abs(values), 0.99)
            norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

            values.T.plot(
                ax=axs[c],
                cmap=cmap,
                norm=norm,
                cbar_kwargs={
                    "label": f"Simulated - Nudged [{units}]",
                    "orientation": "horizontal",
                },
            )

        plt.tight_layout()
        plt.show()

    def compare_wind_profiles(self, timestep: int):

        initial_cond = io.open_initial_cond(self.case)
        stats = io.open_statistics(self.case)

        ustar = stats.ustar.isel(time=timestep).values
        obuk = stats.obuk.isel(time=timestep).values

        u_prof = wind_proc.get_wind_profile(
            z=initial_cond.z.values,
            z0=self.case.ini_file["boundary"]["z0m"],
            u_star=ustar,
            T=None,
            L=obuk,
        )

        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(8, 6))

        ax.plot(u_prof, initial_cond.z.values, label="theory")
        ax.plot(stats.isel(time=timestep).u.values, stats.z.values, label="simulation")
        ax.set_xlabel("Wind speed [m/s]")
        ax.set_ylabel("Height [m]")
        ax.legend()

        return fig


class VariableViewer:
    """
    Class to visualise variables from a microHH dataset.
    """

    def __init__(
        self,
        data: xr.Dataset,
        variable: str,
        time_dim: str = "time",
        figsize: tuple = (10, 6),
        fixed_cbar: bool = True,
        cmap: str = "viridis",
        vmin: float = None,
        vmax: float = None,
        time_format: Literal["s", "min", "h"] = "s",
    ):
        """
        Initialise the VariableViewer with a microHH dataset.

        Args:
            data (xr.Dataset):
                Dataset containing microHH model output.
        """
        self.data = data
        self.variable = variable
        self.time_dim = time_dim

        self.fig_output = widgets.Output()
        self.fig = None
        self.ax = None
        self.figsize = figsize
        self.fixed_cbar = fixed_cbar
        self.cmap = cmap
        self.vmin = vmin
        self.vmax = vmax
        self.time_format = time_format

        self.time_step = 0

        self._check_data_vars()
        self._check_data_shape()
        self._check_time_dim()
        self._handle_vlim()
        self._create_widgets()
        self._init_plot()
        self._handle_cbar()
        self._wrap_widgets()
        self._connect_events()

    def _check_data_vars(self):
        """Check if variable is in data."""
        if self.variable not in self.data.data_vars:
            raise ValueError(f"Variable '{self.variable}' not found in dataset.")

    def _check_data_shape(self):
        """Check the shape of the data."""
        if self.data[self.variable].ndim > 3:
            raise ValueError(
                "Data has more than 3 dimensions. Please integrate or select dimensions accordingly."
            )

    def _check_time_dim(self):
        """Check if time dimension is in data."""
        if self.time_dim not in self.data[self.variable].dims:
            raise ValueError(
                f"Time dimension '{self.time_dim}' not found in variable dimensions."
            )

    def _init_plot(self):
        """Initialise the plot."""
        self.fig, self.ax = plt.subplots(nrows=1, ncols=1, figsize=self.figsize)
        self.fig.canvas.header_visible = False

        data_init = self.data[self.variable].isel(time=0)
        ydim, xdim = data_init.dims

        with self.fig_output:
            self.ax.set_aspect("equal")

            self.mesh = self.ax.pcolormesh(
                self.data[xdim].values,
                self.data[ydim].values,
                data_init,
                cmap=self.cmap,
            )

            # set labels
            self.ax.set_xlabel(f"{xdim} [{self.data[xdim].attrs.get('units', '-')}]")
            self.ax.set_ylabel(f"{ydim} [{self.data[ydim].attrs.get('units', '-')}]")

            # create cax
            divider = make_axes_locatable(self.ax)
            cax = divider.append_axes("right", size="3%", pad=0.1)
            units = self.data[self.variable].attrs.get("units", "-")
            cbar = self.fig.colorbar(
                self.mesh,
                cax=cax,
                label=f"{self.variable} [{units}]",
            )

            # set title
            self.ax.set_title(self._get_title())

            # add pixel value on hover
            self.ax.format_coord = self._format_coord

            plt.show()

    def _format_coord(self, x, y):
        """Custom formatter to show x, y, and the data value."""
        try:
            data = self.data[self.variable].isel(time=self.time_slider.value)
            val = data.sel(x=x, y=y, method="nearest").values

            return f"x={x:.1f}, y={y:.1f}, val={val:.3g}"
        except Exception:
            return f"x={x:.1f}, y={y:.1f}"

    def _handle_cbar(self):
        """Handle fixed colorbar limits."""
        if self.fixed_cbar:
            data_min = self.vmin
            data_max = self.vmax
            self.mesh.set_clim(data_min, data_max)

    def _get_title(self) -> str:
        time = (
            self.data[self.time_dim]
            .isel({self.time_dim: self.time_slider.value})
            .values
        )
        if self.time_format == "s":
            return f"Time: {time:.0f} {self.time_format}"
        elif self.time_format == "min":
            time = time / 60
        elif self.time_format == "h":
            time = time / 3600
        return f"Time: {time:.1f} {self.time_format}"

    def _update_plot(self, change: dict):
        """Update the plot based on the time slider."""
        timestep = change["new"]

        data_subset = self.data[self.variable].isel({self.time_dim: timestep}).values

        # set the new data
        self.mesh.set_array(data_subset.ravel())

        # update cbar
        if not self.fixed_cbar:
            self.mesh.set_clim(np.nanmin(data_subset), np.nanmax(data_subset))

        # update title
        self.ax.set_title(self._get_title())

        self.fig.canvas.draw_idle()

    def _update_vlim(self, change: dict):
        """Update the colorbar limits."""
        vmin = self.vmin_slider.value
        vmax = self.vmax_slider.value

        self.mesh.set_clim(vmin, vmax)
        self.fig.canvas.draw_idle()

    def _handle_vlim(self):
        if self.vmin is None:
            self.vmin = np.nanquantile(self.data[self.variable].values, 0.001)

        if self.vmax is None:
            self.vmax = np.nanquantile(self.data[self.variable].values, 0.999)

        self.data_min = np.nanmin(self.data[self.variable].values)
        self.data_max = np.nanmax(self.data[self.variable].values)

        self.vstep = (self.data_max - self.data_min) / 100

    def _create_widgets(self):
        self.time_slider = widgets.IntSlider(
            value=0,
            min=0,
            max=self.data.sizes[self.time_dim] - 1,
            step=1,
            description="Time Step:",
            continuous_update=True,
        )

        self.vmin_slider = widgets.FloatSlider(
            value=self.vmin,
            min=self.data_min,
            max=self.data_max,
            step=self.vstep,
            description="Min:",
            continuous_update=True,
        )
        self.vmin_slider.readout_format = ddeq.gui_utils.determine_readout_format(
            self.vstep
        )

        self.vmax_slider = widgets.FloatSlider(
            value=self.vmax,
            min=self.data_min,
            max=self.data_max,
            step=self.vstep,
            description="Max:",
            continuous_update=True,
        )
        self.vmax_slider.readout_format = ddeq.gui_utils.determine_readout_format(
            self.vstep
        )

    def _wrap_widgets(self):
        vlim_box = widgets.VBox([self.vmin_slider, self.vmax_slider])
        widget_box = widgets.HBox([self.time_slider, vlim_box])
        self.ui = widgets.VBox([widget_box, self.fig_output])

    def _connect_events(self):
        self.time_slider.observe(self._update_plot, names="value")
        self.vmin_slider.observe(self._update_vlim, names="value")
        self.vmax_slider.observe(self._update_vlim, names="value")

    def display(self):
        """Display the variable."""
        display(self.ui)


def plot_gas_weight(
    data: xr.Dataset, gas: str = "CH4", fig_width: Union[int, float] = 15
):
    """
    Plot the vertical distribution of a gas as well as the 2D plume that
    contributes to it.
    """

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


def plot_case_comparison(
    cases: Sequence[init.Case],
    variables: List[str],
    difference: bool = True,
    default_case_index: int = 0,
    cmap: str = "RdBu_r",
    figsize: Tuple[int, int] = (12, 5),
):
    """
    Plot a comparison of one or more variables from multiple cases.

    Parameters
    ----------
    cases : sequence[init.Case] or init.Case
        One or more case objects.
    variables : sequence[str] or str
        One or more variable names to plot.
    difference : bool, optional
        If True, plot differences to the default case. If False, plot
        absolute values. Default is True.
    default_case_index : int, optional
        Index of the default case used when `difference=True`.
    cmap : str, optional
        Colormap name passed to `pcolormesh`.
    figsize : tuple[int, int], optional
        Figure size passed to `plt.subplots`.
    """
    cases = misc.object_to_list(cases)
    variables = misc.object_to_list(variables)

    # read in stats from all cases
    case_stats = [io.open_statistics(case) for case in cases]

    # Case with the most timesteps
    max_time_case_idx = int(np.argmax([stats.sizes["time"] for stats in case_stats]))
    ref_time = case_stats[max_time_case_idx].time
    # Reindex all cases to the reference time dimension
    case_stats_reindexed = [
        stats.reindex(time=ref_time, method="nearest", tolerance=1e-6)
        for stats in case_stats
    ]

    # If plotting differences, select default case but do not mutate lists
    if difference:
        default_case_stats = case_stats_reindexed[default_case_index]
        default_case = cases[default_case_index]
        # All other (case, stats) pairs
        comp_cases_stats = [
            (case, stats)
            for i, (case, stats) in enumerate(zip(cases, case_stats_reindexed))
            if i != default_case_index
        ]
    else:
        default_case_stats = None  # for type checkers only
        default_case = None  # unused when difference=False
        comp_cases_stats = list(zip(cases, case_stats_reindexed))

    n_plots = len(comp_cases_stats)
    if n_plots == 0:
        raise ValueError("No cases to compare after excluding default case.")

    for var in variables:
        fig, axs = plt.subplots(1, ncols=n_plots, figsize=figsize, sharey=True)
        axs = np.atleast_1d(axs)
        ax_flat = axs.flatten()

        # Colorbar axis attached to the last subplot
        divider = make_axes_locatable(ax_flat[-1])
        cax = divider.append_axes("right", size="3%", pad=0.1)

        # Use the first stats object as reference for metadata
        ref_stats = comp_cases_stats[0][1]
        data_array = ref_stats[var]
        units = data_array.attrs.get("units", "-")
        dim_names = data_array.dims

        # Prepare values (differences or absolute)
        vals: List[np.ndarray] = []
        if difference:
            default_values = default_case_stats[var].values
            for _, stats in comp_cases_stats:
                vals.append(stats[var].values - default_values)
            clabel = f"{var} difference to default case [{units}]"
        else:
            for _, stats in comp_cases_stats:
                vals.append(stats[var].values)
            clabel = f"{var} [{units}]"

        # Determine color scale from all values
        concatenated = np.concatenate([v.ravel() for v in vals])
        vmin = np.nanquantile(concatenated, 0.001)
        vmax = np.nanquantile(concatenated, 0.999)

        center = 0.0 if difference else (vmin + vmax) / 2.0
        norm = TwoSlopeNorm(vcenter=center, vmin=vmin, vmax=vmax)

        # Plot each case
        for ax, val, (case, stats) in zip(ax_flat, vals, comp_cases_stats):
            x_coord = stats[dim_names[0]]
            y_coord = stats[dim_names[1]]

            cm = ax.pcolormesh(
                x_coord.values,
                y_coord.values,
                val.T,
                shading="auto",
                cmap=cmap,
                norm=norm,
            )

            x_units = x_coord.attrs.get("units", "-")
            x_label = x_coord.attrs.get("long_name", "")
            ax.set_xlabel(f"{x_label} [{x_units}]")
            ax.set_title(case.suffix)

        # Shared y-label from reference stats
        y_coord = ref_stats[dim_names[1]]
        y_units = y_coord.attrs.get("units", "-")
        y_label = y_coord.attrs.get("long_name", "")
        ax_flat[0].set_ylabel(f"{y_label} [{y_units}]")

        fig.colorbar(cm, cax=cax, label=clabel)

    plt.show()
