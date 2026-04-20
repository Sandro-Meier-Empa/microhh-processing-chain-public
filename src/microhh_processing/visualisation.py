# import cartopy
import logging
import re
import matplotlib.figure

from pathlib import Path
from typing import Any, List, Literal, Optional, Sequence, Tuple, Union, cast
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable, get_cmap
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize, TwoSlopeNorm
from IPython.display import display
from mpl_toolkits.axes_grid1 import make_axes_locatable

import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import ipywidgets as widgets

import microhh_processing.initialisation as init
import microhh_processing.sources as src
import microhh_processing.wind as wind_proc
from microhh_processing import io, utils

from ddeq.gui import interactions

logger = logging.getLogger(__name__)

__all__ = [
    "CaseViewer",
    "VariableViewer",
    "plot_gas_weight",
    "plot_case_comparison",
]


class CaseViewer:
    wind_dims = {
        "u": ["xh", "y", "z"],
        "v": ["x", "yh", "z"],
        "w": ["x", "y", "zh"],
    }

    def __init__(self, case: init.Case) -> None:
        """Visualise the model output

        Args:
            case (init.Case):
                Case to visualise.

        """
        self.case = case

    def plot_initial_conditions(
        self, ncols: int = 4, figsize: tuple = (15, 10), cmap: str = "viridis"
    ) -> None:
        """Plot all variables from the initial conditions file, grouped by their dimensions.

        Args:
            ncols (int, optional): Number of columns in the subplot grid. Defaults to 4.
            figsize (tuple, optional): Size of the figure. Defaults to (15, 10).
            cmap (str, optional): Colormap to use for 2D variables. Defaults to "viridis".
        """
        file_path = Path(self.case.file_directory) / f"{self.case.name}_input.nc"

        if not file_path.exists():
            raise FileNotFoundError(f"NetCDF file not found: {file_path}")

        # Use netCDF4 just to get group names, then switch to xarray
        data_tree = xr.open_datatree(file_path)

        if not data_tree.groups:
            raise ValueError(f"No data groups found in {file_path}")

        for node in data_tree.subtree:
            group = node.path.strip("/")
            group_ds = node.dataset
            sharey = group != "timedep"
            self._plot_group_variables(group_ds, ncols, figsize, cmap, sharey=sharey)

    def _plot_group_variables(
        self,
        ds: xr.Dataset,
        ncols: int,
        figsize: tuple[int, int],
        cmap: str,
        sharey: bool = True,
    ) -> None:

        # Group variables by their dimension signature
        dim_groups = {}
        for var in ds.data_vars:
            dim_groups.setdefault(ds[var].dims, []).append(var)

        for dims, vars_list in dim_groups.items():
            n_vars = len(vars_list)
            nrows = int(np.ceil(n_vars / ncols))

            # squeeze=False ensures axs is always a 2D array
            fig, axs = plt.subplots(
                nrows,
                ncols,
                figsize=figsize,
                sharey=sharey,
                layout="constrained",
                squeeze=False,
            )

            for i, var_name in enumerate(vars_list):
                ax = axs.flat[i]
                self._plot_variable(ds[var_name], ax, cmap)
                ax.set_title(var_name)

            # Hide empty subplots
            for ax in axs.flat[n_vars:]:
                ax.axis("off")

            plt.show()

    def _plot_variable(self, data: xr.DataArray, ax: Axes, cmap: str) -> None:
        dims = data.dims
        # Check if it is a height profile
        aux_dim = next((d for d in ["z", "lay"] if d in dims), None)
        # Check if it is a time series
        time_dim = next((str(d) for d in dims if "time_" in str(d)), None)

        if len(dims) == 0:
            ax.axhline(float(data.values), linestyle="--")

        elif len(dims) == 1:
            if aux_dim:
                ax.plot(data.values, data[aux_dim], marker="x")
                ax.set_ylabel(aux_dim)
            elif time_dim:
                data.plot(ax=ax, marker="x")  # pyright: ignore
            else:
                data.plot(ax=ax)  # pyright: ignore

        elif len(dims) == 2:
            data.T.plot(  # pyright: ignore
                ax=ax, cmap=cmap, cbar_kwargs={"orientation": "horizontal", "pad": 0.1}
            )

    def plot_prognostic_variables(
        self,
        variables: list[str],
        data: Optional[xr.Dataset] = None,
        n_lines: int = 5,
        colorful: bool = True,
        time_format: str = "h",
        plot_format: Literal["line", "shaded"] = "line",
        color_init: str = "k",
    ) -> matplotlib.figure.Figure:
        """Plot the prognostic variables."""
        variables = [str(variable) for variable in utils.object_to_list(variables)]
        time_factor = self._get_time_factor(time_format)

        # read in initial conditions
        data_initial = io.open_initial_cond(self.case)

        fig, ax = plt.subplots(ncols=len(variables), sharey=True)
        axes = np.atleast_1d(ax)
        sm: Optional[ScalarMappable] = None
        color: Union[str, tuple[float, float, float, float]] = color_init

        for c, component in enumerate(variables):
            current_data = (
                data
                if data is not None
                else io.open_microhh_nc(self.case, variable=component)
            )

            # get dim names from variable
            time_dim, z_dim, y_dim, x_dim = [
                str(dim) for dim in current_data[component].dims
            ]
            t_step = max(1, int(current_data[time_dim].size / n_lines))

            # Normalize the time values to range [0, 1] for colormap
            norm = Normalize(
                vmin=0,
                vmax=float(np.asarray(current_data[time_dim].max().values).item())
                / time_factor,
            )

            # Create a ScalarMappable and set the colormap
            sm = ScalarMappable(cmap=get_cmap("viridis"), norm=norm)
            sm.set_array([])  # Only needed for the colorbar to work

            for t in np.arange(current_data[time_dim].size, step=t_step):
                subset_t = current_data.isel({time_dim: int(t)})
                if colorful:
                    norm_value = (
                        float(np.asarray(subset_t[time_dim].values).item())
                        / time_factor
                    )
                    rgba = tuple(
                        float(value) for value in get_cmap("viridis")(norm(norm_value))
                    )
                    color = cast(tuple[float, float, float, float], rgba)
                else:
                    color = "gray"
                if plot_format == "line":
                    self._plot_lines(
                        ax=axes[c],
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
                        ax=axes[c],
                        data=subset_t,
                        variable=component,
                        x_dim=x_dim,
                        y_dim=y_dim,
                        z_dim=z_dim,
                        color=color,
                    )

            # plot initial conditions
            axes[c].plot(
                data_initial[component],
                data_initial.z,
                label="prescribed",
                color=color_init,
            )
            axes[c].set_xlabel(
                f"{component} [{current_data[component].attrs.get('units', '-')}]"
            )

        axes[0].set_ylabel("Height [m]")

        if colorful and sm is not None:
            # create colorbar
            cbar = fig.colorbar(sm, ax=axes)
            cbar.set_label(f"Time [{time_format}]")
            axes[-1].legend(loc="upper right")
        else:
            # create legend manually
            lines = [Line2D([0], [0], color=c) for c in [color_init, color]]
            labels = ["initial", "simulated"]
            axes[-1].legend(lines, labels, loc="upper right")

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
        ax: Axes,
        data: xr.Dataset,
        variable: str,
        n_lines: int,
        x_dim: str,
        y_dim: str,
        z_dim: str,
        color: Union[str, tuple[float, float, float, float]],
    ) -> None:
        # define step size for plot
        x_step = max(1, int(data[x_dim].size / n_lines))
        y_step = max(1, int(data[y_dim].size / n_lines))

        # loop over x dimension
        for x in np.arange(data[x_dim].size, step=x_step):
            # loop over y dimension
            for y in np.arange(data[y_dim].size, step=y_step):
                subset = data.isel({x_dim: x, y_dim: y})
                ax.plot(subset[variable], subset[z_dim], color=color)

    def _plot_shaded(
        self,
        ax: Axes,
        data: xr.Dataset,
        variable: str,
        x_dim: str,
        y_dim: str,
        z_dim: str,
        color: Union[str, tuple[float, float, float, float]],
        alpha: float = 0.3,
    ) -> None:
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
        fig_width: float = 15,
        cbar_width: float = 0.01,
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
            fig_width (float, optional. Default: 15):
                Width of figure.
            cbar_width (float, optional: Default 0.01.):
                Width of colorbar.

        Returns:
            matplotlib.figure.Figure: Figure with subplots and text
        """

        time_dim, zdim, ydim, xdim = [str(dim) for dim in data[variable].dims]
        data = data.isel({time_dim: timestep})

        x_max = data[xdim].max().values
        y_max = data[ydim].max().values
        z_max = data[zdim].max().values

        if variable in self.wind_dims.keys():
            dd_yz = data.mean(dim=str(xdim))
            dd_xy = data.mean(dim=str(zdim))
            dd_xz = data.mean(dim=str(ydim))
        else:
            dd_yz = data.sum(dim=str(xdim))
            dd_xy = data.sum(dim=str(zdim))
            dd_xz = data.sum(dim=str(ydim))

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
            (
                float(left_top.xmax - cbar_width),
                float(right_bottom.ymin),
                float(cbar_width),
                float(right_bottom.ymax - right_bottom.ymin),
            )
        )
        fig.colorbar(
            im,
            cax=cbar_ax,
            label=f"{variable} [{data[{variable}].attrs.get('units', '-')}]",
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
            text.append(f"Timestep: {timestep}, Time: {int(data['time'].values)} s")
        elif pd.api.types.is_datetime64_any_dtype(data["time"]):
            formatted_time = data["time"].dt.strftime("%H:%M:%S %d.%m.%Y").values[0]
            text.append(f"Timestep: {timestep}, Time: {formatted_time}")
        else:
            text.append("Time: Unsupported data type")

        # Information about resolution

        text.append("\nGrid resolution")
        text.append(f"x: {data[xdim].diff(xdim).mean().values:.1f} m")
        text.append(f"y: {data[ydim].diff(ydim).mean().values:.1f} m")
        text.append(f"z: {data[zdim].diff(zdim).mean().values:.1f} m")

        # Information about wind
        try:
            text.append("\nMean wind")
            for wind_component in ["u", "v"]:
                text.append(f"{wind_component}: {data[wind_component].mean():.1f} m/s")
        except Exception:
            text.append("Wind data not found.")

        # Information about sources
        try:
            source_strength = src.get_source_strength(self.case, to_unit="kg/h")
            source_height = np.atleast_1d(self.case.ini_file["source"]["source_z0"])
            if len(source_strength) > 1:
                # Multiple sources
                text.append("\nSources")
                formatted_strength = ", ".join(f"{x:.2f}" for x in source_strength)
                formatted_height = ", ".join(f"{x:.2f}" for x in source_height)
            else:
                # Single source
                text.append("\nSource")
                formatted_strength = f"{source_strength[0]:.2f}"
                formatted_height = f"{source_height[0]:.0f}"

            text.append(f"source strength: {formatted_strength} kg/h")
            text.append(f"source height: {formatted_height} m")

        except RuntimeError:
            pass

        return "\n".join(text)

    def plot_stats(self, variables: Union[str, list], cmap: str = "viridis") -> None:
        """
        Plot variables from the statistics netcdf file.

        Args:
            variables (str, list(str)):
                Variables within the statistics file to plot.
            cmap: str:
                Colormap to use for plotting.
        """
        stats = io.open_statistics(self.case)

        if isinstance(variables, str):
            variables = list(variables)

        for var in variables:
            fig, axs = plt.subplots(1, 1, figsize=(12, 6))

            dims = stats[var].dims
            ydim = next((d for d in dims if re.fullmatch(r"zh|z", d)), None)
            kwargs = {"cmap": cmap} if stats[var].ndim == 2 else {}

            if ydim is not None:
                stats[var].plot(ax=axs, y=ydim, **kwargs)
            else:
                stats[var].plot(ax=axs, **kwargs)

            plt.tight_layout()
            plt.show()

    def compare_nudge(
        self,
        cmap: str = "RdBu_r",
        figsize: tuple[int, int] = (15, 6),
        percent: bool = False,
    ) -> None:

        stats = io.open_statistics(self.case)
        initial_cond = io.open_initial_cond(self.case)

        # get all variables that contain "_nudge" in their name
        nudge_vars = [
            str(var) for var in initial_cond.data_vars if "_nudge" in str(var)
        ]
        if len(nudge_vars) == 0:
            raise ValueError("No nudge variables found in initial condition file.")

        # get all time coordinates in initial condition file
        time_vars = [
            str(coord) for coord in initial_cond.coords if "time" in str(coord)
        ]
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

            values.T.plot(  # pyright: ignore
                ax=axs[c],
                cmap=cmap,
                norm=norm,
                cbar_kwargs={
                    "label": f"Simulated - Nudged [{units}]",
                    "orientation": "horizontal",
                },
            )
            axs[c].set_title(stat_var)

        plt.tight_layout()
        plt.show()

    def compare_wind_profiles(self, timestep: int) -> matplotlib.figure.Figure:

        initial_cond = io.open_initial_cond(self.case)
        stats = io.open_statistics(self.case)

        ustar = float(stats.ustar.isel(time=timestep).values)
        obuk = float(stats.obuk.isel(time=timestep).values)
        mean_temp = float(stats.th.isel(time=timestep).mean().values)

        u_prof = wind_proc.get_wind_profile(
            z=initial_cond.z.values,
            z0=self.case.ini_file["boundary"]["z0m"],
            u_star=ustar,
            T=mean_temp,
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
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        time_format: Literal["s", "min", "h"] = "s",
    ) -> None:
        """Initialise the VariableViewer with a microHH dataset.

        Args:
            data (xr.Dataset):
                Dataset containing microHH model output.
        """
        self.data = data
        self.variable = variable
        self.time_dim = time_dim

        self.fig_output = widgets.Output()
        self.fig: Any = None
        self.ax: Any = None
        self.mesh: Any = None
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

    def _check_data_vars(self) -> None:
        """Check if variable is in data."""
        if self.variable not in self.data.data_vars:
            raise ValueError(f"Variable '{self.variable}' not found in dataset.")

    def _check_data_shape(self) -> None:
        """Check the shape of the data."""
        if self.data[self.variable].ndim > 3:
            raise ValueError(
                "Data has more than 3 dimensions. Please integrate or select dimensions accordingly."
            )

    def _check_time_dim(self) -> None:
        """Check if time dimension is in data."""
        if self.time_dim not in self.data[self.variable].dims:
            raise ValueError(
                f"Time dimension '{self.time_dim}' not found in variable dimensions."
            )

    def _init_plot(self) -> None:
        """Initialise the plot."""
        self.fig, self.ax = plt.subplots(nrows=1, ncols=1, figsize=self.figsize)
        setattr(self.fig.canvas, "header_visible", False)

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
            self.fig.colorbar(
                self.mesh,
                cax=cax,
                label=f"{self.variable} [{units}]",
            )

            # set title
            self.ax.set_title(self._get_title())

            # add pixel value on hover
            self.ax.format_coord = self._format_coord

            plt.show()

    def _format_coord(self, x: float, y: float) -> str:
        """Custom formatter to show x, y, and the data value."""
        try:
            data = self.data[self.variable].isel(time=self.time_slider.value)
            val = data.sel(x=x, y=y, method="nearest").values

            return f"x={x:.1f}, y={y:.1f}, val={val:.3g}"
        except Exception:
            return f"x={x:.1f}, y={y:.1f}"

    def _handle_cbar(self) -> None:
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

    def _update_plot(self, change: dict[str, Any]) -> None:
        """Update the plot based on the time slider."""
        if self.ax is None or self.fig is None or self.mesh is None:
            return
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

    def _update_vlim(self, change: dict[str, Any]) -> None:
        """Update the colorbar limits."""
        if self.fig is None or self.mesh is None:
            return
        vmin = self.vmin_slider.value
        vmax = self.vmax_slider.value

        self.mesh.set_clim(vmin, vmax)
        self.fig.canvas.draw_idle()

    def _handle_vlim(self) -> None:
        if self.vmin is None:
            self.vmin = np.nanquantile(self.data[self.variable].values, 0.001)

        if self.vmax is None:
            self.vmax = np.nanquantile(self.data[self.variable].values, 0.999)

        self.data_min = np.nanmin(self.data[self.variable].values)
        self.data_max = np.nanmax(self.data[self.variable].values)

        self.vstep = (self.data_max - self.data_min) / 100

    def _create_widgets(self) -> None:
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
        self.vmin_slider.readout_format = interactions.determine_readout_format(
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
        self.vmax_slider.readout_format = interactions.determine_readout_format(
            self.vstep
        )

    def _wrap_widgets(self) -> None:
        vlim_box = widgets.VBox([self.vmin_slider, self.vmax_slider])
        widget_box = widgets.HBox([self.time_slider, vlim_box])
        self.ui = widgets.VBox([widget_box, self.fig_output])

    def _connect_events(self) -> None:
        self.time_slider.observe(self._update_plot, names="value")
        self.vmin_slider.observe(self._update_vlim, names="value")
        self.vmax_slider.observe(self._update_vlim, names="value")

    def display(self) -> None:
        """Display the variable."""
        display(self.ui)


def plot_gas_weight(data: xr.Dataset, gas: str = "CH4", fig_width: float = 15) -> None:
    """Plot the vertical distribution of a gas as well as the 2D plume that
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
) -> None:
    """Plot a comparison of one or more variables from multiple cases.

    Args:
        cases: One or more case objects.
        variables: One or more variable names to plot.
        difference: If True, plot differences to the default case.
            If False, plot absolute values. Defaults to True.
        default_case_index: Index of the default case used when
            ``difference=True``. Defaults to 0.
        cmap: Colormap name passed to ``pcolormesh``. Defaults to "RdBu_r".
        figsize: Figure size passed to ``plt.subplots``. Defaults to (12, 5).
    """
    cases = utils.object_to_list(cases)
    variables = utils.object_to_list(variables)

    # Load and interpolate stats to common time axis
    case_stats = [io.open_statistics(case) for case in cases]
    ref_time = case_stats[int(np.argmax([s.sizes["time"] for s in case_stats]))].time

    logger.info(
        f"Interpolating statistics to common time axis with {len(ref_time)} time steps."
    )
    case_stats = [stats.interp(time=ref_time) for stats in case_stats]

    # Prepare cases for comparison
    default_stats: Optional[xr.Dataset] = None
    if difference:
        default_stats = case_stats[default_case_index]
        comp_cases_stats = [
            (case, stats)
            for i, (case, stats) in enumerate(zip(cases, case_stats))
            if i != default_case_index
        ]
    else:
        comp_cases_stats = list(zip(cases, case_stats))

    if not comp_cases_stats:
        raise ValueError("No cases to compare after excluding default case.")

    for var in variables:
        # Extract values and metadata
        ref_stats = comp_cases_stats[0][1]
        units = ref_stats[var].attrs.get("units", "-")
        dims = [str(dim) for dim in ref_stats[var].dims]

        # Prepare data values
        if difference and default_stats is not None:
            default_values = default_stats[var].values
            vals = [stats[var].values - default_values for _, stats in comp_cases_stats]
            clabel = f"{var} difference [{units}]"
        else:
            vals = [stats[var].values for _, stats in comp_cases_stats]
            clabel = f"{var} [{units}]"

        # Determine color normalization
        concatenated = np.concatenate([v.ravel() for v in vals])
        vmin, vmax = np.nanquantile(concatenated, [0.001, 0.999])
        norm = _get_norm(vmin, vmax, difference)

        # Create subplots
        fig, axs = plt.subplots(1, len(comp_cases_stats), figsize=figsize, sharey=True)
        axs = np.atleast_1d(axs)

        # Plot data
        y_coord: Optional[xr.DataArray] = None
        for ax, val, (case, stats) in zip(axs, vals, comp_cases_stats):
            x_coord, y_coord = stats[dims[0]], stats[dims[1]]
            ax.pcolormesh(
                x_coord.values,
                y_coord.values,
                val.T,
                shading="auto",
                cmap=cmap,
                norm=norm,
            )
            ax.set_xlabel(
                f"{x_coord.attrs.get('long_name', '')} [{x_coord.attrs.get('units', '-')}]"
            )
            ax.set_title(case.suffix)

        if y_coord is not None:
            axs[0].set_ylabel(
                f"{y_coord.attrs.get('long_name', '')} [{y_coord.attrs.get('units', '-')}]"
            )

        # Add colorbar
        divider = make_axes_locatable(axs[-1])
        cax = divider.append_axes("right", size="3%", pad=0.1)
        fig.colorbar(axs[-1].collections[0], cax=cax, label=clabel)

        plt.tight_layout()
        plt.show()


def _get_norm(vmin: float, vmax: float, difference: bool) -> Normalize:
    """Determine appropriate normalization for colormap."""
    if not difference:
        return Normalize(vmin=vmin, vmax=vmax)

    if vmin < 0 < vmax:
        return TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax)
    elif vmax <= 0:
        return Normalize(vmin=vmin, vmax=0.0)
    else:
        return Normalize(vmin=0.0, vmax=vmax)


__all__ = ["CaseViewer", "VariableViewer", "plot_gas_weight", "plot_case_comparison"]
