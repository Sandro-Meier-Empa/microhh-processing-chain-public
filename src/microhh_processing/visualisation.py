# import cartopy
import matplotlib.figure
import os
import scipy.stats
import sys

from pathlib import Path
from typing import Literal, Union
from matplotlib.lines import Line2D
from matplotlib import cm
from matplotlib.colors import Normalize
from IPython.display import display
from mpl_toolkits.axes_grid1 import make_axes_locatable

import netCDF4 as nc
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import ipywidgets as widgets

import microhh_processing.processing as proc
import microhh_processing.initialisation as init
import microhh_processing.misc as misc
import microhh_processing.io as io
import microhh_processing.wind as wind_proc


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
        self.name = case.name
        self.suffix = case.suffix
        self.name_spec = case.name_spec
        self.file_directory = case.file_directory

        self.post = proc.PostProcessing(self.case)

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

        file_path = Path(self.file_directory) / f"{self.name}_input.nc"

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
                    sharey=True,
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
            if "time_surface" in dims:
                ax.plot(data_group["time_surface"], var_data)
            elif aux_dim in dims:
                ax.plot(var_data, data_group[aux_dim])
            else:
                ax.plot(var_data)

        elif len(dims) == 2:
            # 2D heatmap
            ax.pcolormesh(data_group[dims[0]], data_group[dims[1]], var_data.T)

    def plot_prognostic_variables(
        self,
        data: xr.Dataset,
        variables: list = ["u", "v", "th"],
        n_lines: int = 5,
        colorful: bool = True,
        time_format: str = "h",
    ):
        """
        Plot the prognostic variables
        """

        variables = proc.object_to_list(variables)

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

        # read in initial conditions
        data_initial = io.open_initial_cond(self.case)

        fig, ax = plt.subplots(ncols=len(variables), sharey=True)

        # Normalize the time values to range [0, 1] for colormap
        norm = Normalize(vmin=0, vmax=data.time.max() / time_factor)

        # Create a ScalarMappable and set the colormap
        sm = plt.cm.ScalarMappable(cmap=cm.viridis, norm=norm)
        sm.set_array([])  # Only needed for the colorbar to work

        for c, component in enumerate(variables):
            # get dim names from variable
            time_dim, z_dim, y_dim, x_dim = data[component].dims

            # define step size for plot
            t_step = int(data[time_dim].size / n_lines)
            x_step = int(data[x_dim].size / n_lines)
            y_step = int(data[y_dim].size / n_lines)

            # plot simulated values
            for t in np.arange(data[time_dim].size, step=t_step):
                color_sim = (
                    cm.viridis(norm(data[time_dim][t] / time_factor))
                    if colorful
                    else "gray"
                )
                color_init = "C3" if colorful else "C0"

                # loop over x dimension
                for x in np.arange(data[x_dim].size, step=x_step):
                    # loop over y dimension
                    for y in np.arange(data[y_dim].size, step=y_step):
                        subset = data.isel(time=t, x=x, y=y)
                        for c, component in enumerate(variables):
                            ax[c].plot(
                                subset[component], subset[z_dim], color=color_sim
                            )

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

        return fig

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
            source_strength = self.post.model_emissions_to_kg_per_h()
            source_height = self.case.ini_file["source"]["source_z0"]
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

    def compare_wind_profiles(self, timestep: int):

        initial_cond = io.open_initial_cond(self.case)
        stats = io.open_statistics(self.case)

        ustar = stats.ustar.isel(time=timestep).values
        obuk = stats.obuk.isel(time=timestep).values

        u_prof = wind_proc.get_wind_profile(
            z=initial_cond.z.values,
            z0=self.case.ini_file["boundary"]["z0m"],
            u_star=ustar,
            T=initial_cond.isel(z=0).th.values,
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

            plt.show()

    def _handle_cbar(self):
        """Handle fixed colorbar limits."""
        if self.vmin is None:
            self.vmin = np.nanquantile(self.data[self.variable].values, 0.001)

        if self.vmax is None:
            self.vmax = np.nanquantile(self.data[self.variable].values, 0.999)

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
        if self.time_format == "min":
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

    def _create_widgets(self):
        self.time_slider = widgets.IntSlider(
            value=0,
            min=0,
            max=self.data.sizes[self.time_dim] - 1,
            step=1,
            description="Time Step:",
            continuous_update=True,
        )

    def _wrap_widgets(self):
        self.ui = widgets.VBox([self.time_slider, self.fig_output])

    def _connect_events(self):
        self.time_slider.observe(self._update_plot, names="value")

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


def plot_tke(
    case,
    tke_threshold: bool = False,
    tke_derivative_threshold: bool = False,
    threshold_value=0,
):
    """
    Plot the behaviour of the TKE over time to determine the spin-up time of turbulence.
    """
    stats = init.read_statistics(case)
    _, threshold_time = proc.get_2D_tke_threshold_time(
        stats, tke_threshold, tke_derivative_threshold, threshold_value
    )

    # Calculate the median of tke over the 'z' dimension
    tke_median = stats.tke.median(dim="z")

    # Calculate the derivative of tke_median with respect to time
    tke_derivative = tke_median.differentiate(coord="time")

    # Calculate the relative derivative
    tke_median_overall = stats.tke.median(dim=["z", "time"])
    relative_derivative = tke_derivative / tke_median_overall

    # Create subplots
    fig, ax = plt.subplots(2, 2, sharex=True, figsize=(14, 10))

    # Plot the original 2D TKE as the first subplot
    cax = ax[0, 0].pcolormesh(stats["time"], stats["z"], stats["tke"].T, cmap="viridis")
    ax[0, 0].set_title("2D TKE")
    ax[0, 0].set_xlabel("Time")
    ax[0, 0].set_ylabel("Height [m]")
    ax[0, 0].axvspan(stats["time"].min(), threshold_time, color="white", alpha=0.3)

    # Add the colorbar above the first subplot
    cbar = fig.colorbar(cax, ax=ax[0, 0], orientation="horizontal", pad=0.15)
    cbar.set_label(f"TKE [{stats.tke.attrs['units']}]")

    # Plot median tke on the second subplot
    ax[0, 1].plot(
        tke_median["time"],
        tke_median,
        linestyle="dotted",
        color="C0",
    )
    ax[0, 1].plot(
        tke_median["time"].where(tke_median["time"] >= threshold_time, drop=True),
        tke_median.where(tke_median["time"] >= threshold_time, drop=True),
        linestyle="solid",
        color="C0",
    )
    ax[0, 1].set_title("Median TKE over Time")
    ax[0, 1].set_xlabel("Time")
    ax[0, 1].set_ylabel(f"Median TKE [{stats.tke.attrs['units']}]")

    if tke_threshold:
        ax[0, 1].axhline(
            y=threshold_value * tke_median.max(),
            color="black",
            label=f"Threshold: {int(threshold_value*100)}% of max TKE",
        )
        ax[0, 1].legend()

    # Plot the derivative of tke on the third subplot
    ax[1, 0].plot(
        tke_derivative["time"],
        tke_derivative,
        linestyle="dotted",
        color="orange",
    )
    ax[1, 0].plot(
        tke_derivative["time"].where(
            tke_derivative["time"] >= threshold_time, drop=True
        ),
        tke_derivative.where(tke_derivative["time"] >= threshold_time, drop=True),
        linestyle="solid",
        color="orange",
    )
    ax[1, 0].axhline(0, color="black", linestyle="--", linewidth=0.7)
    ax[1, 0].set_title("Derivative of Median TKE over Time")
    ax[1, 0].set_xlabel("Time")
    ax[1, 0].set_ylabel("Derivative of Median TKE")

    # Plot the relative derivative on the fourth subplot
    ax[1, 1].plot(
        relative_derivative["time"],
        relative_derivative * 100,
        linestyle="dotted",
        color="green",
    )
    ax[1, 1].plot(
        relative_derivative["time"].where(
            relative_derivative["time"] >= threshold_time, drop=True
        ),
        relative_derivative.where(
            relative_derivative["time"] >= threshold_time, drop=True
        )
        * 100,
        linestyle="solid",
        color="green",
    )
    ax[1, 1].axhline(0, color="black", linestyle="--", linewidth=0.7)
    ax[1, 1].set_title("Relative Derivative of Median TKE over Time [%]")
    ax[1, 1].set_xlabel("Time")

    if tke_derivative_threshold:
        ax[1, 1].axhline(
            y=threshold_value * 100,
            color="black",
            label=f"Threshold: {int(threshold_value*100)}% of second derivative of TKE",
        )
        ax[1, 1].legend()

    # Adjust layout
    plt.tight_layout()
    plt.show()
