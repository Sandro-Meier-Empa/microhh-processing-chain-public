import glob
import os
import shutil
import subprocess
import sys
import pyproj
import scipy
import ucat

from pathlib import Path
from typing import Union

import netCDF4 as nc
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.join('/users', os.environ["USER"], 'microhh', 'python'))
sys.path.append(os.path.join('/users', os.environ["USER"], 'microhh', 'processing_chain', 'src'))
sys.path.append(os.path.join('/users', os.environ["USER"], 'ddeq', 'src'))

import microhh_tools as mht
import initialisation as init
import ddeq as ddeq

# molar mass (g/mol)
gas_mass = {
    'air': 28.9664,  # dry air
    'co':  28.01,
    'co2': 44.01,
    'h2o': 18.01528,
    'no2': 46.0055,
    'no':  30.01,
    'o3':  48.00,
    'so2': 64.066,
    'ch4': 16.0425,
}



class PreProcessing:
    
    # define number of cores used on daint multicore (cpu) and hybrid (gpu)
    PROCESSOR_VERSIONS = {
        "cpu": 36,
        "gpu": 12
    }

    PARTITION_CONDITIONS = {
        "debug": {"time_limit": 30 * 60, "max_nodes": 10},
        "normal": {"time_limit": 24 * 60 * 60, "max_nodes": 480},
        "prepost": {"time_limit": 30 * 60, "max_nodes": 1},
        "low": {"time_limit": 24 * 30 * 60, "max_nodes": 124},
    }

    def __init__(self, case: init.Case, 
                 processor: str, mpi: bool,
                 run_time: str="00:30:00", nodes: int=1, ntasks_per_core: int=1,
                 ntasks_per_node: int=None, cpus_per_task: int=1,
                 large_memory: bool=False, partition: str="debug"):
        """Class used for pre-processing and running MicroHH cases

        Args:
            case (init.Case):
                Case to pre-process and run.
            processor (str) ['cpu', 'gpu']:
                Processor version to be used to run the model.
            mpi (bool):
                Indicates if mpi should be enabled.
            run_time (str, optional. Default: "00:30:00"):
                Specify the job duration. Determines the queuing time.
                Format: "HH:MM:SS".
            nodes (int, optional. Default: 1):
                Number of nodes to use for running MicroHH.
            ntasks_per_core (int, optional. Default: 1):
                Number of tasks per core. If > 1, hyperthreading is turned on.
            ntasks_per_node (int, optional):
                Number of tasks per node.
                If not specified, the default for 'cpu' is set to
                36 and 12 for 'gpu'. If 'mpi = False', it is set to 1.
            cpus_per_task (int, optional. Default: 1):
                Number of cpus per task.
                Specify the number of cpus per task.
                Defines the number of OpenMP threads per MPI rank.
                The maximum value depends on the number of tasks per node.
            large_memory (bool, optional. Defaults False):
                Specifies if nodes with large memory should be used. 
            partition (str, optional. Default: 'debug')
                ['debug', 'normal', 'prepost', 'low']:
                Queue to be used on daint.
        """
        
        case.read_case("scratch")
        
        self.user = case.user
        self.case_name = case.case_name
        self.suffix = case.suffix
        self.case_path = case.case_path
        self.scratch_path = case.scratch_path
        self.store_path = case.store_path
        self.ini_path = case.ini_path
        self.ini_file = case.ini_file
        self.processor = processor.lower()
        self.mpi = mpi
        self.run_time = run_time
        self.nodes = nodes
        self.ntasks_per_core = ntasks_per_core
        self.cpus_per_task = cpus_per_task
        self.large_memory = large_memory
        self.partition = partition
        self.ntasks_per_node = ntasks_per_node or self._get_default_ntasks_per_node()

        self._check_build_exists()
        self._check_cscs_setup_valid()
        self._check_ini_setup_valid()


    def _get_default_ntasks_per_node(self) -> int:
        """Determines the default number of tasks per node
        if not specified in __init__

        Returns:
            ntasks_per_node (int):
                Number of tasks per node to be used.
                Set to 36 for 'cpu' and 12 for 'gpu' on daint.
                If 'mpi' is disabled, the tasks per node are set to 1.
        """
        if self.mpi:
            return self.PROCESSOR_VERSIONS[self.processor] * self.ntasks_per_core if self.processor in self.PROCESSOR_VERSIONS else 1
        else:
            return 1

    def _check_cscs_setup_valid(self):
        """Checks if the setup is valid to be run on daint based
        on the selected queue, wall time and number of nodes.

        Raises:
            ValueError: Raise error if the wall time is
                too long for the selected queue.
            ValueError: Raise error if the number of nodes is   
                too large for the selected queue.
        """
        # Check if CSCS setup is valid
        run_time_delta = self._parse_run_time(self.run_time)
        conditions = self.PARTITION_CONDITIONS.get(self.partition, {})
        if conditions.get("time_limit") and run_time_delta.total_seconds() > conditions["time_limit"]:
            raise ValueError("Invalid time limit for the selected partition.")
        if conditions.get("max_nodes") and self.nodes > conditions["max_nodes"]:
            raise ValueError("Invalid number of nodes for the selected partition.")

    def _check_build_exists(self):
        """Check if the selected model build exists.

        Raises:
            ValueError: Raise error if the model build does not exist.
        """
        # Check if build exists
        if self.processor == "gpu" and self.mpi:
            raise ValueError("MPI is not enabled for GPU accelerated version.")
    
    def _check_ini_setup_valid(self):
        """
        Check if the setup specified in the .ini file is valid based on the selected number of nodes and the model
        segregation and number of grid cells.

        Raises:
            ValueError: If the specified grid and model segregation do not match in x direction or y direction.
            ValueError: If the segregation and number of nodes do not match.
        """
        self._adjust_mpi_settings()

        self._check_grid_match()
        self._check_nodes_match()

        self._check_source_position()
        self._check_buffer_height()

    def _adjust_mpi_settings(self):
        """Adjust MPI settings if MPI is disabled."""
        if not self.mpi:
            for dim in ["x", "y"]:
                self.ini_file.set_value(group="master", variable=f"np{dim}", value=1)
            self.ini_file.save(self.ini_path, allow_overwrite=True)
            print("Overwriting 'npx' and 'npy' because MPI is disabled.")

    def _check_grid_match(self):
        """Check if the grid size matches the number of processes."""
        itot, jtot = self.ini_file['grid']['itot'], self.ini_file['grid']['jtot']
        npx, npy = self.ini_file['master']['npx'], self.ini_file['master']['npy']

        if itot % npx != 0:
            raise ValueError(f"itot ({itot}) is not divisible by npx ({npx})")
        elif jtot % npy != 0:
            raise ValueError(f"jtot ({jtot}) is not divisible by npy ({npy})")

    def _check_nodes_match(self):
        """Check if the number of nodes matches the expected value."""
        expected_nodes = self.ini_file['master']['npx'] * self.ini_file['master']['npy']
        if expected_nodes != self.ntasks_per_node:
            raise ValueError(
                f"Task per node ({self.ntasks_per_node}) does "
                f"not match the expected value.")

    def _check_source_position(self):
        """Check if the source position conflicts with the domain size."""
        if "source" in self.ini_file.groups:
            for coord in ["x", "y", "z"]:
                source_ext = self.ini_file['source'][f'source_{coord}0'] + \
                             2 * self.ini_file['source'][f'sigma_{coord}']

                if source_ext > self.ini_file['grid'][f'{coord}size']:
                    raise ValueError(
                        f"{coord}-positioning conflicts with the domain size.")

    def _check_buffer_height(self):
        """Check if the buffer height lies within the domain."""
        if "buffer" in self.ini_file.groups:
            zstart = self.ini_file['buffer'].get('zstart', 0)
            zsize = self.ini_file['grid']['zsize']
            if zstart > zsize:
                raise ValueError(
                    f"Buffer height={zstart} lies outside of the domain "
                    f" with zsize={zsize}.")            

    def _parse_run_time(self, run_time: str) -> pd.Timedelta:
        """Parse the run_time string and return a timedelta object

        Args:
            run_time (str): 
                Specify the job duration. Determines the queuing time.
                Format: "HH:MM:SS".

        Raises:
            ValueError: Raise error if the format is not "HH:MM:SS".

        Returns:
            pd.Timedelta: Timedelta of the wall clock time.
        """
        
        try:
            return pd.to_timedelta(run_time)
        except ValueError:
            raise ValueError("Invalid run_time format. Use HH:MM:SS")

    def create_batch_script(self, start_jobs: bool=False, wait: bool=False):
        """Create batch script to run a simulation.

        Args:
            start_jobs (bool, optional. Default: False):
                Specify if the job should be started.
            wait (bool, optional. Default: False):
                Specify if the next job should only be submitted
                after the previous one has stopped.

        Raises:
            FileNotFoundError: Raise error if the path on scratch does not exist.
        """

        multithread = "multithread" if self.ntasks_per_core > 1 else "nomultithread"
        constraint = "mc" if self.processor == "cpu" else "gpu"
        self.job_file = os.path.join(self.scratch_path, "bash_script.sbash")

        if not os.path.exists(self.scratch_path):
            raise FileNotFoundError(
                f"Directory {self.scratch_path} does not exist yet."
                )

        with open(self.job_file, "w") as bash_script:
            header = "#!/bin/bash -l \n\n"
            
            header += f"#SBATCH --job-name={self.case_path} \n"
            header += "#SBATCH --account=em05\n"
            header += "#SBATCH --mail-type=ALL\n"
            header += "#SBATCH --mail-user=sandro.meier@empa.ch\n"
            header += f"#SBATCH --time={self.run_time}\n"
            header += f"#SBATCH --nodes={self.nodes}\n"
            header += f"#SBATCH --ntasks-per-core={self.ntasks_per_core}\n"
            header += f"#SBATCH --ntasks-per-node={self.ntasks_per_node}\n"
            header += f"#SBATCH --cpus-per-task={self.cpus_per_task}\n"
            if self.large_memory:
                header += "#SBATCH --mem=120GB"
            header += f"#SBATCH --partition={self.partition} \n"
            header += f"#SBATCH --constraint={constraint}\n"
            header += f"#SBATCH --hint={multithread}\n"
            header += f"#SBATCH --chdir={str(self.scratch_path)}\n\n"

            header += "export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK\n"
            if self.processor == "gpu":
                header += "export CRAY_CUDA_MPS=1\n\n"
            else:
                header += "\n"

            header += f"srun {self.scratch_path}/microhh init {self.case_name}\n"
            header += f"srun {self.scratch_path}/microhh run {self.case_name}"

            bash_script.write(header)
            
        if start_jobs:
            # Launch the job on daint
            self.run_sim(self.job_file, wait)


    def run_sim(self, job_file: Path, wait: bool=True):
        """Run a simulation using sbatch
        
        Args:
            job_file (Path):
                Specify the path of the batch script which should be run.
            wait (bool, optional. Default: False):
                Specify if the next job should only be submitted
                after the previous one has stopped.
        """

        print(f"Starting {job_file=}")
        os.system(f"sbatch {'--wait' if wait else ''} {job_file}")

    def run_preprocessing(self, method: str,
                          start_jobs: bool=False,
                          wait: bool=False):
        """ Prepare MicroHH for running a simulation and executes the simulation.
        - Create or uses the {case_name}_input.nc
        - Copy the specified build of MicroHH to scratch
        - Create batch script and execute it to run the model
    
        Args:
            method (str) ['default', 'new']:
                Specify if the {case_name}_input.nc file should be
                generated by running the {case_name}_input.py file or
                by using the initialsiation.Init class.
            start_jobs (bool, optional. Default: False):
                Specify if the job should be started.
            wait (bool, optional. Default: False):
                Specify if the next job should only be submitted
                after the previous one has stopped.

        Raises:
            FileNotFoundError: Raise error if the .ini file cannot be found on scatch.
        """
        
        # Create nc file for initial conditions
        nc_file_name = f"{self.case_name}_input.nc"

        if method == "default":
            # default: use {case_name}_input.py to create {case_name}_input.nc
            case_path = os.path.join('/users', self.user, 'microhh', 'cases', self.case_name)
            py_input_files = list(Path(case_path).glob(f"{self.case_name}_input.py"))

            # Check if there are any matching input files
            if py_input_files:
                # Run the Python script which creates the {case_name}_input.nc file on scratch
                subprocess.run(['python', str(py_input_files[0])],
                               cwd=self.scratch_path)
            else:
                print(f"No file '{self.case_name}_input.py' found in {case_path}.")

        elif method == "new":
            # default: use initialisation.Initialisation to create {case_name}_input.nc
            file_path = os.path.join(self.scratch_path, nc_file_name)
            if not os.path.exists(file_path):
                raise FileNotFoundError(
                    f"No file '{nc_file_name}' found in {self.scratch_path}. \n"
                    f"If you want to use a Python script "
                    f"'{self.case_name}_input.py' to create the file, "
                    f"use 'method=default'. \n"
                    f"Otherwise, create '{nc_file_name} using "
                    f"the class 'Initialisation'."
                    )

        # Copy microhh executable
        processor = self.processor
        version = "mpi" if self.mpi else "nompi"

        microhh_executable_path = os.path.join('/users', self.user, 'microhh', f'build.{processor}.{version}', 'microhh')
        shutil.copy(microhh_executable_path, self.scratch_path)

        # Create bash script
        self.create_batch_script(start_jobs, wait)


class PostProcessing:
    def __init__(self, case: init.Case, vars: Union[str, list]):
        """Class used for post-processing MicroHH cases and handling the output.

        Args:
            case (init.Case):
                Case to post-process.
            vars (Union[str, list]):
                Modelled variables which should be processed.
                E.g. ["u", "v", "w", "ch4", "qt", "thl"]
        """
        case.read_case("scratch")

        self.user = case.user
        self.case_name = case.case_name
        self.suffix = case.suffix
        self.case_path = case.case_path
        self.scratch_path = case.scratch_path
        self.store_path = case.store_path
        self.ini_path = case.ini_path
        self.ini_file = case.ini_file

    def vars_to_list(self, vars):

        if not isinstance(vars, list):
            vars = [vars]

        return vars

    def run_postprocessing(self, vars,
                           overwrite: bool=False,
                           precision: str="double",
                           start_time=None, end_time=None):
        """Run all post-processing steps.

        Args:
            overwrite (bool, optional. Default: False):
                Specify if existing files on 'scratch' and 
                'store' should be overwritten.
            precision (str, optional. Default "double"):
                Precision used for the created .nc file.
            start_time (int, optional. Default: None):
                Start time in s of the data.
            end_time (int, optional. Default: None):
                End time in s of the data.
        """

        self.create_3d_output(
            vars=vars,
            overwrite=overwrite,
            precision=precision,
            start_time=start_time,
            end_time=end_time
            )
        self.move_to_store(
            vars=vars,
            overwrite=overwrite
            )


    def create_3d_output(self, vars,
                         overwrite: bool=False,
                         precision: str="double",
                         start_time=None, end_time=None):
        """Create 3D netCDF files for the variables specified

        Args:
            overwrite (bool, option. Default: False):
                Specify if existing files on 'scratch' and 
                'store' should be overwritten.
            precision (str, optional. Default: "double"):
                Precision used for the created .nc file.
            start_time (int, optional. Default: None):
                Start time in s of the data.
            end_time (int, optional. Default: None):
                End time in s of the data.
        """
        vars = self.vars_to_list(vars)

        if start_time is None:
            start_time = self.ini_file['time']['starttime']
        if end_time is None:
            end_time = self.ini_file['time']['endtime']
        sample_time = self.ini_file['time']['savetime']

        for var in vars:
            nc_file_path = os.path.join(self.scratch_path, f"{var}.nc")

            if os.path.exists(nc_file_path):
                if overwrite:
                    os.remove(nc_file_path)
                    print(f"Overwriting {nc_file_path}")
                else:
                    return

        post_command = (
            f"python 3d_to_nc.py -d {self.scratch_path} "
            f"-f {self.case_name}.ini -v {' '.join(vars)} "
            f"-p {precision} -t0 {start_time} -t1 {end_time} "
            f"-tstep {sample_time}"
        )
        subprocess.run(
            post_command,
            shell=True,
            cwd=os.path.join('/users', self.user, 'microhh', 'python')
            )
        
        
    def _nc_assign_names_and_attributes(self, data):
        """Assign long_name and unit attributes to variables in a .nc file.
    
        Args:
            nc_file_path (str): Path to the .nc file.
        """
        # Define the dictionary with variable attributes using a dictionary comprehension
        variables = {var: (f"{var} mixing ratio", "mol mol-1") for var in gas_mass.keys()}
        variables.update({
            'thl': ('liquid water potential temperature', 'K'),
            'qt': ('total specific humidity', 'kg kg-1'),
            'u': ('zonal wind component', 'm s-1'),
            'v': ('meridional wind component', 'm s-1'),
            'w': ('vertical wind component', 'm s-1'),
            'p': ('air pressure', 'Pa'),
        })

        # Assign attributes to variables
        for var in data.variables:
            if var in variables:
                long_name, unit = variables[var]
                data[var].attrs.update({'long_name': long_name, 'units': unit})

        return data


    def interpolate_to_grid_midpoint(self, nc_file: xr.Dataset):
        dim = self._get_dims_at_grid_boundaries(nc_file)

        if dim is None:
            return nc_file
        
        print(f"Interpolating {dim} to {dim[:-1]}")

        midpoints = nc_file[dim][:-1].values + np.diff(nc_file[dim].values)/2
        new_coords = {dim: midpoints}
        nc_file_interpolated = nc_file.interp(**new_coords)

        # Create a new dimension name without the 'h' at the end
        new_dim_name = dim[:-1]
        # Rename the dimension in the Dataset
        nc_file_interpolated = nc_file_interpolated.rename({dim: new_dim_name})

        return nc_file_interpolated
    
    def _get_dims_at_grid_boundaries(self, nc_file):
        # Iterate over the dimensions of the Dataset
        for dim in list(nc_file.dims):
            # Check if the dimension name ends with 'h'
            if dim.endswith('h'):
                return dim

        return None
    
    def combine_nc_files(self, vars):
        vars = self.vars_to_list(vars)

        # Create a list to store individual xarray.Dataset objects
        datasets = []

        # Iterate through each variable
        for var in vars:
            # Construct the file path for the current variable
            nc_file_path = os.path.join(self.store_path, f"{var}.nc")

            # Open the NetCDF file as an xarray.Dataset
            try:
                dataset = xr.open_dataset(nc_file_path)
            except ValueError:
                dataset = xr.open_dataset(nc_file_path, decode_times=False)

            # homogenise dimensions
            dataset = self.interpolate_to_grid_midpoint(dataset)

            # Append the dataset to the list
            datasets.append(dataset)

        # Merge all datasets into one
        combined_dataset = xr.merge(datasets)

        # Save the combined dataset to a new NetCDF file
        output_filename = os.path.join(
            self.store_path,
            f"{self.case_name}_output_vars.nc"
            )
        combined_dataset.to_netcdf(output_filename)

        # Close all datasets
        for dataset in datasets:
            dataset.close()

        return combined_dataset

    def open_3d_netCDF(self, variable: str) -> xr.Dataset:
        """Open 3D output netCDF file of 'variable' from model output.

        Args:
            variable (str):
                Variable name of dataset to open.

        Returns:
            xr.Dataset: 3D netCDF file from model output.
        """

        fname = f"{self.store_path}/{variable.lower()}.nc"
        try:
            data = xr.open_dataset(fname)
        except ValueError:
            data = xr.open_dataset(fname, decode_times=False)

        data = self._nc_assign_names_and_attributes(data)
        data = self._capitalise_variable_names(data)

        return data
    
    def _format_column_unit(self, to_unit):

        if '/' in to_unit:
            n, d = map(str.strip, to_unit.split('/'))
            if d[-1].isdigit():
                d = "-".join([d[:-1], d[-1]])
            
            formatted_unit = " ".join([n,d])

        else:
            formatted_unit = to_unit

        return formatted_unit
    
    def _format_volume_unit(self, to_unit):

        # vertical integration assumes units to be in m
        if " " in to_unit: 
            n, d = to_unit.split()
            if d == "m":
                formatted_unit = n
            if d == "m-2":
                formatted_unit = " ".join([n, "m-3"])
            
            return formatted_unit
        
        else:
            raise ValueError(f"{to_unit} is a unit not supported")

    def convert_to_columns(self, variable, to_unit):
        data = self.open_3d_netCDF(variable.lower())
        data = self._nc_assign_names_and_attributes(data)

        new_var_unit = self._format_volume_unit(to_unit)

        variable = variable.upper()
        
        # convert to correct volumentric unit first.
        # default MicroHH output is mol mol-1
        data[variable] = (
            data[variable].dims,
            ucat.convert_points(
                data[variable],
                data[variable].attrs.get("units", None),
                new_var_unit,
                molar_mass=variable.upper())
                )

        # integrate vertically
        df_column = data.integrate(coord="z")

        # adjust units due to integration
        df_column[variable].attrs["units"] = self._format_column_unit(to_unit)
        df_column[variable].attrs["description"] = f"vertically integrated {variable} columns"

        return df_column

    def add_gaussian_noise(self, data: xr.Dataset, var: str, std: float) -> xr.Dataset:
        """
        Add Gaussian noise to a specified variable in the dataset.

        Args:
            data (xr.Dataset):
                The dataset containing the variable to which noise will be added.
            var (str):  
                The name of the variable to which Gaussian noise will be added.
            std (float):
                The standard deviation of the Gaussian noise distribution.

        Returns:
            xr.Dataset: The dataset with the specified variable modified to include Gaussian noise.
        """
        var = var.upper()

        # Create a deep copy of the dataset to avoid modifying the original in-place
        data_copy = data.copy(deep=True)

        # assume no bias (center = 0) but noise (= std)
        noise = np.random.normal(0, std, data_copy[var].size)
        noise = noise.reshape(data_copy[var].shape)

        data_copy[var] = data_copy[var] + noise
        data_copy[var].attrs.update(data[var].attrs)
        data_copy[var].attrs["noise_level"] = std

        return data_copy
    

    def degrade_resolution(self,
                           data: xr.Dataset,
                           zonal_res: int,
                           meridional_res: int) -> xr.Dataset:
        """Degrade resolution of 3D or 2D MicroHH output by aggregating pixels.

        Args:
            data (xr.Dataset): 3D or 2D data
            zonal_res (int):
                Number of pixels in zonal direction which should be aggregated.
            meridional_res (int):
                Number of pixels in meridional direction which should be aggregated.

        Returns:
            xr.Dataset: Data with coarser resolution.
        """
        spatial_dims = {
            ("x", "y"): ("x", "y"),
            ("lon", "lat"): ("lon", "lat"),
            ("longitude", "latitude"): ("longitude", "latitude")
        }

        for dims, (zonal_dim, meridional_dim) in spatial_dims.items():
            if all(dim in data.dims for dim in dims):
                degraded_data = data.coarsen({
                    zonal_dim: zonal_res,
                    meridional_dim: meridional_res},
                    boundary="trim").mean()
                return degraded_data

        raise ValueError(f"Unknown spatial dimensions: {data.dims}")


    def create_sources_dataset(self,
                           lat, lon,
                           source_names=None,
                           source_labels=None,
                           source_radius=None):
    
        def to_list(value):
            return value if isinstance(value, (list, np.ndarray)) else [value]
    
        source_list = self.ini_file['source']['sourcelist']
        nr_sources = len(to_list(source_list))
    
        source_names = source_names or source_labels or [f"source{i+1}" for i in range(nr_sources)]
        source_labels = source_labels or source_names
    
        if source_radius is None:
            pixel_size = np.max([
                self.ini_file['grid']['xsize'] / self.ini_file['grid']['itot'],
                self.ini_file['grid']['ysize'] / self.ini_file['grid']['jtot']
            ])
            source_radius = np.full(nr_sources, pixel_size / 1e3) # in km
    
        data = {
            'lon_o': (['source'], np.array(to_list(lon), dtype=float)),
            'lat_o': (['source'], np.array(to_list(lat), dtype=float)),
            'grid_x': (['source'], np.array(to_list(self.ini_file['source']['source_x0']), dtype=float)),
            'grid_y': (['source'], np.array(to_list(self.ini_file['source']['source_y0']), dtype=float)),
            'type': (['source'], np.full(nr_sources, "power plant", dtype=object)),
            'label': (['source'], np.array(to_list(source_labels), dtype=object)),
            'radius': (['source'], np.array(to_list(source_radius), dtype=float)),
            'strength': (['source'], self.model_emissions_to_kg_per_h()),
        }
    
        coords = {'source': np.array(to_list(source_names), dtype=object)}
    
        sources = xr.Dataset(data, coords=coords)
        sources.lon_o.attrs['units'] = 'longitude of source'
        sources.lat_o.attrs['units'] = 'latitude of source'
        sources.radius.attrs['units'] = 'km'
        sources.strength.attrs['units'] = 'kg h-1'
    
        for dim in ["x", "y"]:
            sources[f"grid_{dim}"].attrs['units'] = 'm'
            sources[f"grid_{dim}"].attrs['description'] = f'{dim} position in MicroHH grid'
    
        return sources
    

    def create_wind_dataset(self, source, sources, radius):
        vars = ["u", "v"]
    
        try:
            self.run_postprocessing(vars)
            wind_data = self.combine_nc_files(vars)
        except Exception as e:
            raise e
    
        wind_data['speed'] = np.sqrt(wind_data.u**2 + wind_data.v**2)
        wind_data['direction'] = ddeq.wind.calculate_wind_direction(wind_data.u, wind_data.v)
    
        selected_source = sources.sel(source=source)
        grid_x = selected_source.grid_x.values
        grid_y = selected_source.grid_y.values
    
        x_res = wind_data.x.diff(dim="x").median()
        y_res = wind_data.y.diff(dim="y").median()
    
        wind_data_subset = wind_data.where(
            (wind_data.x > grid_x - x_res/2 - radius) & 
            (wind_data.x <= grid_x + x_res/2 + radius) &
            (wind_data.y > grid_y - y_res/2 - radius) & 
            (wind_data.y <= grid_y + y_res/2 + radius),
            drop=True
        )

        wind_data_subset = self.vertically_integrate_wind(wind_data_subset, method="mean")
    
        wind_data_subset['u_median'] = wind_data_subset.u.median(
            dim=["x", "y"])
        wind_data_subset['v_median'] = wind_data_subset.v.median(
            dim=["x", "y"])
        wind_data_subset['speed_median'] = wind_data_subset.speed.median(
            dim=["x", "y"])
        wind_data_subset['speed_std'] = wind_data_subset.speed.std(
            dim=["x", "y"])
        wind_data_subset['direction_mean'] = self._circular_mean(
            wind_data_subset.direction, axes=[list(wind_data_subset.dims).index('y'),
                                              list(wind_data_subset.dims).index('x')])
        wind_data_subset['direction_std'] = self._circular_std(
            wind_data_subset.direction, axes=[list(wind_data_subset.dims).index('y'),
                                              list(wind_data_subset.dims).index('x')])
    
        wind_src = xr.Dataset(
            coords={
                "source": (["source"], np.array([source], dtype=object)),
                "time": (["time"], wind_data_subset.time.values)
            },
            data_vars={
                "lon": (
                    ["source"],
                    [selected_source.lon_o.values]),
                "lat": (
                    ["source"],
                    [selected_source.lat_o.values]),
                "U": (
                    ["source", "time"],
                    [wind_data_subset.u_median.values],
                    {"units": "m s-1"}),
                "V": (
                    ["source", "time"],
                    [wind_data_subset.v_median.values],
                    {"units": "m s-1"}),
                "speed": (
                    ["source", "time"],
                    [wind_data_subset.speed_median.values],
                    {"units": "m s-1"}),
                "speed_precision": (
                    ["source", "time"],
                    [wind_data_subset.speed_std.values],
                    {"units": "m s-1"}),
                "direction": (
                    ["source", "time"],
                    [wind_data_subset.direction_mean.values],
                    {"units": "°"}),
                "direction_precision": (
                    ["source", "time"],
                    [wind_data_subset.direction_std.values],
                    {"units": "°"}),
            },
            attrs=dict(radius=f"{radius} pixels around determined source pixel")
        )
    
        return wind_src
    

    def _circular_mean(self, a, axes):
            return scipy.stats.circmean(a, high=360, nan_policy="omit", axis=axes)
    

    def _circular_std(self, a, axes):
            return scipy.stats.circstd(a, high=360, nan_policy="omit", axis=axes)

    
    def vertically_integrate_wind(self, wind_data, method):

        if method == "mean":
            return wind_data.mean(dim="z")
        else:
            # FIXME: implement other ways of vertical averaging
            raise NotImplementedError
    

    def convert_microhh_grid_to_coords(self, lat_o, lon_o, data, sources,
                                       target_crs = 'epsg:4326',
                                       microhh_crs = 'epsg:32632'):

        # convert lat and lon into a metric CRS which is valid for Europe
        # 'epsg:32632' Coordinate system in meters which covers Europe
        # 'epsg:4326' Global lat-lon coordinate system

        # convert coordinates to metric crs
        latlon_to_metric = pyproj.Transformer.from_crs(target_crs, microhh_crs)
        x_o, y_o = latlon_to_metric.transform(lat_o, lon_o)

        xx, yy = np.meshgrid(data['x'], data['y'])

        # shift grid of x and y values so that the source is at the location
        # given by lat_o and lon_o
        shifted_grid_x = xx-sources.grid_x.values+x_o
        shifted_grid_y = yy-sources.grid_y.values+y_o

        # transform shifted grid to lat / lon
        lat_grid, lon_grid = latlon_to_metric.transform(
            shifted_grid_x,
            shifted_grid_y,
            direction="INVERSE")

        # capitalise gas names
        data = self._capitalise_variable_names(data)

        data = data.rename({
            "y": "nrows",
            "x": "nobs"
            })

        # define the coordinates
        coords = {
            'time': (['time'], data.time.data),
            'lon': (['nrows', 'nobs'], lon_grid),
            'lat': (['nrows', 'nobs'], lat_grid)
            }

        # extract only the variables that are not already in the dataset
        vars_to_include = {var: data[var] for var in data.variables if var not in data.dims}

        # create the new dataset
        data = xr.Dataset(vars_to_include, coords=coords).drop_vars(["nrows", "nobs"])

        return data


    def _capitalise_gas_key(self, variable_name):
        for key in gas_mass.keys():
            if key in variable_name:
                return key.upper()
        return variable_name
    

    def _capitalise_variable_names(self, dataset: xr.Dataset) -> xr.Dataset:
        """Capitalize variable names in an xarray dataset if they match keys in the gas_mass dictionary.
    
        Args:
            dataset (xr.Dataset): The xarray dataset to process.
    
        Returns:
            xr.Dataset: The dataset with variable names capitalized if they match keys in the gas_mass dictionary.
        """
        for var_name in list(dataset.variables):
            new_var_name = self._capitalise_gas_key(var_name)
            if new_var_name != var_name:
                dataset = dataset.rename({var_name: new_var_name})
        return dataset


    def convert_microhh_timestep_to_date(self,
                                         start_date: Union[str, pd.Timestamp],
                                         data: xr.Dataset) -> xr.Dataset:
        """Convert time into a timestamp to make the
        dataset digestible by `ddeq`

        Args:
            start_date (Union[str, Timestamp]):
                The start date as a string or a pandas Timestamp object.
            data (xr.Dataset):
                The dataset to convert.

        Returns:
            xr.Dataset:
                The dataset with the time converted to a timestamp.
        """
        data['time'] = pd.to_timedelta(data['time'], unit='s')

        # Check if start_date is a string and convert it to a Timestamp if necessary
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)

        # Add Timedelta to the start date
        data['time'] = start_date + data['time']

        return data

    def add_corners(self, d, var=('lon', 'lat'), dims=('nrows', 'nobs')):
        """
        Add corner coordinates for `var`.
        """
        ic = np.arange(d[dims[1]].size+1) - 0.5
        jc = np.arange(d[dims[0]].size+1) - 0.5

        bbox=[-0.5, d[dims[0]].size, -0.5, d[dims[1]].size]

        for name in var:
            interpolator = scipy.interpolate.RectBivariateSpline(
                d[dims[0]], d[dims[1]], d[name], kx=1, ky=1, bbox=bbox
            )
            xc = interpolator(jc, ic)
            xc = np.moveaxis([xc[:-1,:-1], xc[:-1,1:], xc[1:,1:], xc[1:,:-1]], 0, 2)
            d[f'{name}c'] = xr.DataArray(xc, dims=dims + ('corners', ))
            
        return d


    def model_emissions_to_kg_per_h(self) -> np.array:
        """
        Convert the prescribed emissions from the .ini file to kg/h

        Returns:
            np.array: Emitted quantities in kg/h
        """
        sourcelist = self.ini_file['source']['sourcelist']
        swvmr = self.ini_file['source']['swvmr']
        strength = self.ini_file['source']['strength']

        if not isinstance(sourcelist, list):
            sourcelist = np.array([sourcelist])
            swvmr = np.array([swvmr])
            strength = np.array([strength])

        molar_mass = np.array([gas_mass[gas] for gas in sourcelist])

        if all(swvmr): # it true, units are kmol s-1
            strength *= molar_mass * 3600 # convert to kg/h
        else: # units are kg kg s-1, seems to be an error
            strength = np.full(len(swvmr), np.nan)

        return strength


    def move_to_store(self, vars, overwrite: bool):
        """Move files used for running MicroHH to store for reproducibility.

        Args:
            overwrite (bool, optional):
                Specify if existing files on 'store' should be overwritten.

        Raises:
            FileExistsError: Raise error if the folder
            already exists and 'overwrite=False'.
        """
        vars = self.vars_to_list(vars)

        # Handle target path
        if not os.path.exists(self.store_path):
            os.makedirs(self.store_path)
        elif overwrite:
            shutil.rmtree(self.store_path)
            print(f"Overwriting {self.store_path}.")
            os.makedirs(self.store_path)

        # Copy files for reproducibility
        file_patterns = ['*'+self.case_name+'.ini', '*'+self.case_name+'_input.nc', 'slurm-*.out']
        for pattern in file_patterns:
            for file in glob.glob(os.path.join(self.scratch_path, pattern)):
                dest_file = os.path.join(self.store_path, os.path.basename(file))
                if not os.path.exists(dest_file) or overwrite:
                    shutil.copy(file, self.store_path)
                    print(f"Successfully copied '{file}' to '{self.store_path}'.")
                else:
                    print(f"'{file}' already exists in '{self.store_path}'. Skipping.")

        # Copy 3D .nc files of variables
        for var in vars:
            var_file = os.path.join(self.scratch_path, f"{var}.nc")
            if os.path.exists(var_file):
                dest_file = os.path.join(self.store_path, os.path.basename(var_file))
                if not os.path.exists(dest_file) or overwrite:
                    shutil.copy(var_file, self.store_path)
                    print(f"Successfully copied '{var_file}' to '{self.store_path}'.")
                else:
                    print(f"'{var_file}' already exists in '{self.store_path}'. Skipping.")
            else:
                print(f"'{var_file}' does not exist.")


    def parse_slurm_output(self, slurm_source: str) -> pd.DataFrame:
        """Parse the output of the 'slurm.out' script to get the job information.

        Args:
            slurm_source (str) ['scratch', 'store']:
                Specify the source from where the 'slurm.out' script should be read.

        Raises:
            ValueError: Raise error if 'slurm_source' is invalid.
            FileNotFoundError: Raise error if the 'slurm.out' cannot be found.

        Returns:
            pd.DataFrame: Data frame containing the job information.
        """
        counter = 0
        job_info = False
        headers = []
        values = []

        if slurm_source == "scratch":
            source_path = self.scratch_path
        elif slurm_source == "store":
            source_path = self.store_path
        else:
            raise ValueError(
                f"Invalid source: '{slurm_source}'. Use 'scratch' or 'store'."
                )

        out_files = list(Path(source_path).glob("slurm-*.out*"))

        if not out_files:
            raise FileNotFoundError(f"No slurm output script found in {source_path}")
        
        for out_file in out_files:
            with open(out_file, 'r') as f:
                for line in f:
                    # use the lines after "job information"
                    if "Job information" in line:
                        counter = 0
                        job_info = True
                        continue

                    if counter == 1 and job_info: # first line defines the header
                        header = line
                    elif counter == 2 and job_info: # second line defines the length of the content
                        separator = line
                        sep_len = [len(i) for i in separator.split()]
                    elif counter == 3 and job_info: # third line defines the value
                        value = line
                    elif counter == 4 and job_info: # reset for next job info
                        job_info = False
                        startIndex = 0

                        # parse header and values
                        for l in sep_len:
                            headers.append(header[startIndex: startIndex + l].strip())
                            values.append(value[startIndex: startIndex + l].strip())
                            startIndex += l+1
                            df = pd.DataFrame([values], columns=headers)

                    counter += 1
        
        # Format columns
        timedelta_cols = ['Elapsed', 'Time limit']
        datetime_cols = ['Submit', 'Eligible', 'Start', 'End']
        df[timedelta_cols] = df[timedelta_cols].apply(pd.to_timedelta)
        df[datetime_cols] = df[datetime_cols].apply(pd.to_datetime)

        # Extracting integer part and unit part from usage and memory
        df[['Energy', 'Energy unit']] = df['Energy'].str.split(expand=True)
        df['Energy'] = df['Energy'].astype(float)

        try: # only available for simulations on GPU
            df['Execution time'] = df['Execution time'].apply(pd.to_timedelta)

            df[['Usage', 'Usage unit']] = df['Usage'].str.split(expand=True)
            df[['Max mem', 'Max mem unit']] = df['Max mem'].str.split(expand=True)
            df[['Usage', 'Max mem']] = df[['Usage', 'Max mem']].astype(float)
        except:
            pass

        return df


############# other functions #############

def jobs_in_queue(omit: list=[
    "MIN_CPUS", "MIN_TMP_DISK", "GROUP", "OVER_SUBSCRIBE",
    "COMMENT", "MIN_MEMORY", "REQ_NODES",
    "COMMAND", "PRIORITY", "QOS", "ST",
    "RESERVATION", "WCKEY", "EXC_NODES",
    "NICE", "S:C:T", "EXEC_HOST", "DEPENDENCY",
    "ARRAY_JOB_ID", "SOCKETS_PER_NODE", "CORES_PER_SOCKET",
    "THREADS_PER_CORE", "ARRAY_TASK_ID", "CONTIGUOUS",
    "NODELIST(REASON)", "UID", "LICENSES",
    "CORE_SPEC", "SCHEDNODES"
    ]) -> pd.DataFrame:
    """Get information about jobs in the queue.

    Args:
        omit (list, optional): 
            Specify which columns of the output should be omitted.

    Returns:
        pd.DataFrame: Information about jobs in the queue.
    """

    # get the user name
    user = os.environ["USER"]

    # call squeue returning only all variables of the jobs
    squeue = subprocess.Popen(
        ["squeue", "-u", user, "-o", "%all"], stdout=subprocess.PIPE
    )

    # get the output
    output = squeue.communicate()[0]

    # decode the output
    output = output.decode("utf-8")

    # parse data
    data = output.split("\n")
    headers = data[0].split('|')
    rows = [x for x in data[1:] if x]

    # Create the DataFrame
    df = pd.DataFrame([row.split('|') for row in rows], columns=headers)

    # remove omitted columns
    df = df.drop(columns=omit)

    return df