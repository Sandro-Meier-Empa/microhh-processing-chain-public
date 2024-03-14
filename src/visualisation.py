import glob
import importlib
import matplotlib.figure
import os
import shutil
import subprocess
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
import processing as proc
import initialisation as init



class Visualisation():

    wind_dims = {
            "u"    : ["xh", "y", "z"],
            "v"    : ["x", "yh", "z"],
            "w"    : ["x", "y", "zh"],
        }

    def __init__(self, case: init.Case, source: str):
        """Visualise the model output

        Args:
            case (init.Case):
                Case to visualise.
            source (str) ['scratch', 'store']:
                Source from where the case should be read.

        """

        case.read_case(source)

        self.case = case
        self.case_name = case.case_name
        self.suffix = case.suffix
        self.case_path = case.case_path
        self.scratch_path = case.scratch_path
        self.store_path = case.store_path
        self.ini_path = case.ini_path
        self.ini_file = mht.Read_namelist(self.ini_path)


    def plot_side_views(self, variable: str, timestep: int,
                        fig_width: Union[int, float]=15,
                        cbar_width: Union[int, float]=0.01
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

        post_processing = proc.PostProcessing(self.case)

        data = post_processing.open_3d_netCDF(variable)

        xdim, ydim, zdim = self.wind_dims[variable] if variable in self.wind_dims.keys() else ["x", "y", "z"]
        
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

        aspect_ratio = (z_max + x_max)/(z_max + y_max)
        fig_height = fig_width / aspect_ratio

        fig, axs = plt.subplots(2, 2, figsize=(fig_width, fig_height), 
                                gridspec_kw={'width_ratios': [1, x_max/z_max], 
                                            'height_ratios': [y_max/z_max, 1]})

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
        cbar_ax = fig.add_axes([left_top.xmax-cbar_width,
                                right_bottom.ymin,
                                cbar_width,
                                right_bottom.ymax-right_bottom.ymin])
        cb = fig.colorbar(im,
                          cax=cbar_ax,
                          label=f"{variable} "
                          f"[{data[{variable}].attrs.get('units', '-')}]",
                          ticklocation="left")
        
        text = self.add_text(data, timestep,
                             xdim, ydim, zdim)

        fig.text(left_top.xmin,
                right_bottom.ymax,
                text,
                horizontalalignment='left',
                verticalalignment='top')
        
        return fig


    def add_text(self, data: xr.Dataset, timestep: int,
                 xdim: str, ydim: str, zdim: str) -> str:
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
        if pd.api.types.is_numeric_dtype(data['time']):
            text.append(f"Timestep: {timestep}, "
                        f"Time: {int(data['time'].values)} s")
        elif pd.api.types.is_datetime64_any_dtype(data['time']):
            formatted_time = data['time'].dt.strftime("%H:%M:%S %d.%m.%Y").values[0]
            text.append(f"Timestep: {timestep}, "
                        f"Time: {formatted_time}")
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
                    f"{wind_component}: "
                    f"{data[wind_component].mean():.1f} m/s")
        except:
            text.append("Wind data not found.")

        # Information about sources
        post_processing = proc.PostProcessing(self.case)
        try:
            source_strength = post_processing.model_emissions_to_kg_per_h()
            source_height = self.ini_file['source']['source_z0']
            if len(source_strength) > 1:
                # Multiple sources
                text.append(f"\nSources")
                formatted_strength = ', '.join(f'{x:.2f}' for x in source_strength)
                formatted_height = ', '.join(f'{x:.2f}' for x in source_height)
            else:
                # Single source
                text.append(f"\nSource")
                formatted_strength = f'{source_strength[0]:.2f}'
                formatted_height = f'{source_height:.0f}'

            text.append(f"source strength: {formatted_strength} kg/h")
            text.append(f"source height: {formatted_height} m")

        except RuntimeError:
            pass

        return '\n'.join(text)
    
