import glob
import os
import pyproj
import re
import scipy
import shutil
import subprocess
import sys
import warnings
import zipfile

from pathlib import Path
from typing import Union, Tuple

import pandas as pd
import xarray as xr
import numpy as np

import ucat

sys.path.append(os.path.join("/users", os.environ["USER"], "microhh", "python"))
import microhh_tools as mht

sys.path.append(os.path.join("/users", os.environ["USER"]))
import processing_chain.src.initialisation as init
import processing_chain.src.cscs as cscs
import processing_chain.src.wind as wind_proc

# molar mass (g/mol)
gas_mass = {
    "air": 28.9664,  # dry air
    "co": 28.01,
    "co2": 44.01,
    "h2o": 18.01528,
    "no2": 46.0055,
    "no": 30.01,
    "o3": 48.00,
    "so2": 64.066,
    "ch4": 16.0425,
}


class PreProcessing:
    def __init__(
        self,
        case: init.Case,
        alps_setup: cscs.AlpsSetup,
    ):
        """Class used for pre-processing and running MicroHH cases

        Args:
            case (init.Case):
                Case to pre-process and run.
            alps_setup (cscs.AlpsSetup):
                Object containing the setup variables used for creating a bash script to run the MicroHH simulations.

        """
        self.case = case
        self.user = case.user
        self.case_name = case.case_name
        self.suffix = case.suffix
        self.case_path = case.case_path
        self.scratch_path = case.scratch_path
        self.store_path = case.store_path
        self.microhh_executable_path = case.microhh_executable_path
        self.ini_path = case.ini_path
        self.ini_file = mht.Read_namelist(self.ini_path)
        self.processor = alps_setup.processor
        self.alps_setup = alps_setup

        self._check_microhh_build_exists()

    def _check_microhh_build_exists(self):
        """Check if the selected model build exists.

        Raises:
            ValueError: Raise error if the model build does not exist.
        """
        # Check if build exists
        if self.alps_setup.processor == "gpu" and self.alps_setup.mpi:
            raise ValueError("MPI is not enabled for GPU accelerated version.")

    def run_simulation(
        self,
        start_jobs: bool = False,
        wait: bool = False,
    ):
        """Prepare MicroHH for running a simulation and executes the simulation.
        - Copy the initial conditions {case_name}_input.nc
        - Copy the specified build of MicroHH to scratch
        - Create bash script and execute it to run the model

        Args:
            start_jobs (bool, optional. Default: False):
                Specify if the job should be started.
            wait (bool, optional. Default: False):
                Specify if the next job should only be submitted
                after the previous one has stopped.

        Raises:
            FileNotFoundError: Raise error if oen of the files cannot be found on scatch.
        """

        # copy microHH to directory
        self.copy_executable()

        # check if all files are present
        files_required = [
            f"{self.case_name}_input.nc",
            f"{self.case_name}.ini",
            "microhh",
        ]

        for filename in files_required:
            file_path = Path(self.scratch_path, filename)
            if not file_path.exists():
                raise FileNotFoundError(
                    f"Required file '{filename}' not found in '{self.scratch_path}'."
                )

        # Create bash script if it does not exist
        job_file = self.create_run_script()

        # check if the .ini setup is valid
        self._check_ini_setup_valid()

        if start_jobs:
            # Launch the job on alps
            self.alps_setup.execute_bash_script(job_file, wait)

    def restart_simulation(
        self,
        reuse_bash_script: bool = True,
        start_jobs: bool = False,
        wait: bool = False,
    ):

        # List of files to keep
        files_to_keep = [
            f"{self.case_name}.ini",
            f"{self.case_name}_input.nc",
            "microhh",
        ]

        # create bash script if needed
        if reuse_bash_script:
            files_to_keep.append("run_microhh.sbash")
        else:
            job_file = self.create_run_script()

        # delete existing files except {case_name}.ini, {case_name}_input.nc and executable
        for file_path in self.scratch_path.iterdir():
            if file_path.name not in files_to_keep:
                file_path.unlink()
                print(f"Deleted {file_path}")

        # check if the .ini setup is valid
        self._check_ini_setup_valid()

        if start_jobs:
            # Launch the job on alps
            self.alps_setup.execute_bash_script(job_file, wait)

    def rerun_simulation(
        self,
        reuse_bash_script: bool = True,
        reuse_microhh: bool = True,
        start_jobs: bool = False,
        wait: bool = False,
    ):
        # Re-run simulation based on run_microhh.sbash, {case_name}.ini and {case_name}_input.nc.

        # List of files to copy
        files_to_copy = [
            f"{self.case_name}.ini",
            f"{self.case_name}_input.nc",
        ]

        if reuse_bash_script:
            files_to_copy.append("run_microhh.sbash")
        else:
            job_file = self.create_run_script()

        if reuse_microhh:
            files_to_copy.append("microhh")
        else:
            self.copy_executable()

        # copy files from store to scratch
        for filename in files_to_copy:
            for file in glob.glob(Path(self.store_path, filename)):
                shutil.copy(file, self.scratch_path)
                print(f"Successfully copied '{file}' to '{self.scratch_path}'.")

        # check if the .ini setup is valid
        self._check_ini_setup_valid()

        if start_jobs:
            # Launch the job on alps
            self.alps_setup.execute_bash_script(job_file, wait)

    def create_run_script(self) -> Path:
        job_file = Path(self.scratch_path, "run_microhh.sbash")
        if not job_file.exists():
            job_file = self.alps_setup.create_bash_script(
                commands=[
                    f"srun {self.scratch_path}/microhh init {self.case_name}",
                    f"srun {self.scratch_path}/microhh run {self.case_name}",
                ],
                job_name="run_microhh",
                file_path=job_file,
                scratch_path=self.scratch_path,
            )
        else:
            print(f"Reading existing bash script {job_file}")

        return job_file

    def _check_ini_setup_valid(self):
        """
        Check if the setup specified in the .ini file is valid based on the daint setup and the parameters specified in the .ini file.
        """
        self._check_ini_content()
        self._adjust_mpi_settings()
        self._check_grid_match()
        self._check_nodes_match()
        self._check_source_position()
        self._check_source_sigma()
        self._check_buffer_height()
        self._check_surface_roughness_height()

    def _check_ini_content(self):
        """check if the .ini file is not empty"""
        if len(self.ini_file.groups) == 0:
            raise ValueError(f"{self.case_name}.ini is empty.")

    def _adjust_mpi_settings(self):
        """Adjust MPI settings if MPI is disabled."""
        if (
            not self.alps_setup.mpi
            and not self.ini_file["master"]["npx"] == 1
            and not self.ini_file["master"]["npy"] == 1
        ):
            for dim in ["x", "y"]:
                self.ini_file.set_value(group="master", variable=f"np{dim}", value=1)
            self.ini_file.save(self.ini_path, allow_overwrite=True)
            print("Overwriting 'npx' and 'npy' because MPI is disabled.")

    def _check_grid_match(self):
        """Check if the grid size matches the number of processes."""
        itot, jtot, ktot = (
            self.ini_file["grid"]["itot"],
            self.ini_file["grid"]["jtot"],
            self.ini_file["grid"]["ktot"],
        )
        npx, npy = self.ini_file["master"]["npx"], self.ini_file["master"]["npy"]

        if itot % npx != 0:
            raise ValueError(f"itot ({itot}) is not divisible by npx ({npx})")
        if itot % npy != 0:
            raise ValueError(f"itot ({itot}) is not divisible by npy ({npy})")
        if jtot % npx != 0:
            raise ValueError(f"jtot ({jtot}) is not divisible by npx ({npx})")
        if jtot % npy != 0:
            raise ValueError(f"jtot ({jtot}) is not divisible by npy ({npy})")
        if ktot % npx != 0:
            raise ValueError(f"ktot ({ktot}) is not divisible by npx ({npx})")

    def _check_nodes_match(self):
        """Check if the number of nodes matches the expected value."""
        expected_nodes = self.ini_file["master"]["npx"] * self.ini_file["master"]["npy"]
        if expected_nodes != self.alps_setup.ntasks_per_node:
            raise ValueError(
                f"Task per node ({self.alps_setup.ntasks_per_node}) does "
                f"not match the expected value."
            )

    def _check_source_position(self):
        """Check if the source position conflicts with the domain size."""
        if "source" in self.ini_file.groups:
            for coord in ["x", "y", "z"]:
                source_ext = (
                    self.ini_file["source"][f"source_{coord}0"]
                    + 2 * self.ini_file["source"][f"sigma_{coord}"]
                )

                if source_ext > self.ini_file["grid"][f"{coord}size"]:
                    raise ValueError(
                        f"{coord}-positioning conflicts with the domain size."
                    )

    def _check_source_sigma(self):
        """Check if the source std is greater than zero."""
        if "source" in self.ini_file.groups:
            for coord in ["x", "y", "z"]:
                if self.ini_file["source"][f"sigma_{coord}"] == 0:
                    raise ValueError(f"'sigma_{coord}' cannot be 0.")

    def _check_buffer_height(self):
        """Check if the buffer height lies within the domain."""
        if "buffer" in self.ini_file.groups:
            zstart = self.ini_file["buffer"].get("zstart", 0)
            zsize = self.ini_file["grid"]["zsize"]
            if zstart > zsize:
                raise ValueError(
                    f"Buffer height={zstart} lies outside of the domain "
                    f" with zsize={zsize}."
                )

    def _check_surface_roughness_height(self):
        """Check if the surface roughness is smaller than the lowest vertical grid level."""
        # read in {case}_input.nc file:
        initial_cond = init.read_initial_cond(self.case)
        grid_values = initial_cond.z.values
        non_neg_grid_values = [value for value in grid_values if value >= 0]
        lowest_grid_level = min(non_neg_grid_values)
        warning_threshold = 0.9 * lowest_grid_level

        if "boundary" in self.ini_file.groups:
            for roughness_type in ["z0m", "z0h"]:
                if roughness_type in self.ini_file["boundary"]:
                    surface_roughness = self.ini_file["boundary"][roughness_type]
                    if surface_roughness >= lowest_grid_level:
                        warnings.warn(
                            f"Surface roughness ({roughness_type} = {surface_roughness} m) "
                            f"is larger than the lowest level of the grid (={lowest_grid_level} m). "
                            "This may cause the model to crash."
                        )
                    elif surface_roughness >= warning_threshold:
                        warnings.warn(
                            f"Surface roughness ({roughness_type} = {surface_roughness} m) "
                            f"similar to the lowest grid level (={lowest_grid_level} m). "
                            "This may cause instability in the model."
                        )

    def copy_executable(self):
        """Copy MicroHH executable to scratch"""
        processor = self.processor
        version = "mpi" if self.alps_setup.mpi else "nompi"

        microhh_executable_path = self.microhh_executable_path / "microhh"

        shutil.copy(microhh_executable_path, self.scratch_path)


class PostProcessing:
    def __init__(self, case: init.Case):
        """Class used for post-processing MicroHH cases and handling the output.

        Args:
            case (init.Case):
                Case to post-process.
        """
        self.case = case
        self.user = case.user
        self.case_name = case.case_name
        self.suffix = case.suffix
        self.case_path = case.case_path
        self.scratch_path = case.scratch_path
        self.store_path = case.store_path
        self.ini_path = case.ini_path
        self.ini_file = mht.Read_namelist(self.ini_path)
        self.file_directory = case.file_directory

    def run_postprocessing(
        self,
        gas: str,
        precision: str = "double",
        start_time=None,
        end_time=None,
        to_unit: str = "kg m-2",
        default_p_and_t: bool = False,
        method: str = "height",
        height: Union[int, float] = None,
        overwrite: bool = False,
        save: bool = True,
        chunks: Union[str, dict] = None,
    ):
        """Run all post-processing steps to...
            ... create netCDF files from grid output files
            ... convert gas mixing ratios to concentrations
            ... vertically integrate the concentrations
            ... vertically integrate wind data

        Args:
            gas (str):
                Gas which should be processed to concentrations and vertically integrated.
            precision (str, optional):
                Precision used for the created .nc file. Defaults to "double".
            start_time (int, optional. Default: None):
                Start time in s of the data to process.
            end_time (int, optional. Default: None):
                End time in s of the data to process.
            to_unit (str, optional):
                Units of the trace gas columns. Defaults to "kg m-2".
            default_p_and_t (bool, optional):
                Indicate if the standard values for pressure and temperature should be
                used to convert mixing ratios to concentrations or if the
                modelled pressure and temperature should be used. Defaults to False.
            method (str, optional):
                Method to vertically integrate wind speeds. Available options are
                "mean", "median", "height", "concentration_weighted". Defaults to "height".
            height (Union[int, float], optional):
                Height at which the wind speed should be used if method = 'height'. Defaults to None.
            overwrite (bool, optional):
                Specify if the resulting dataset should be overwritten if it exists. Defaults to False.
            save (bool, optional):
                Specify if the resulting dataset should be saved. Defaults to True.
        """
        vars = ["u", "v", "thl", gas]

        print(f"########## START PROCESSING OF {self.case_path} ##########")

        # skip post-processing if simulation failed at start
        current_time_step, max_time_step = cscs.get_simulation_progress(self.case)
        if current_time_step == 0:
            print("ERROR: Simulation failed. Skipped post-processing.")
            return

        self.create_3d_output(
            vars=vars,
            overwrite=overwrite,
            precision=precision,
            start_time=start_time,
            end_time=end_time,
        )
        gas_2D = self.convert_to_columns(
            variable=gas,
            to_unit=to_unit,
            default_p_and_t=default_p_and_t,
            save=save,
            overwrite=overwrite,
            chunks=chunks,
        )
        wind_processing = wind_proc.WindProcessing(self.case)
        wind_2D = wind_processing.create_2D_wind(
            method=method,
            gas=gas,
            height=height,
            save=save,
            overwrite=overwrite,
            chunks=chunks,
        )
        print(f"########## FINISH PROCESSING OF {self.case_path} ##########")

    def create_3d_output(
        self,
        vars: Union[str, list],
        overwrite: bool = False,
        precision: str = "double",
        start_time=None,
        end_time=None,
    ):
        """Create 3D netCDF files for the variables specified

        Args:
            vars (Union[str, list]):
                Modelled variables which should be processed.
                E.g. ["u", "v", "w", "ch4", "qt", "thl"]
            overwrite (bool, option. Default: False):
                Specify if existing files on 'scratch' and
                'store' should be overwritten.
            precision (str, optional. Default: "double"):
                Precision used for the created .nc file.
            start_time (int, optional. Default: None):
                Start time in s of the data to process.
            end_time (int, optional. Default: None):
                End time in s of the data to process.
        """
        vars = object_to_list(vars)
        vars_to_process = vars.copy()

        if start_time is None:
            start_time = self.ini_file["time"]["starttime"]
        if end_time is None:
            end_time = self.ini_file["time"]["endtime"]
        sample_time = self.ini_file["time"]["savetime"]

        for var in vars:
            nc_file_path = Path(self.scratch_path, f"{var}.nc")

            if nc_file_path.exists():
                if overwrite:
                    os.remove(nc_file_path)
                    print(f"Overwriting {nc_file_path}")
                else:
                    vars_to_process.remove(var)

        # only process missing variables
        if vars_to_process:
            post_command = (
                f"{sys.executable} 3d_to_nc.py -d {self.scratch_path} "
                f"-f {self.case_name}.ini -v {' '.join(vars_to_process)} "
                f"-p {precision} -t0 {start_time} -t1 {end_time} "
                f"-tstep {sample_time}"
            )
            subprocess.run(
                post_command,
                shell=True,
                cwd=Path("/users", self.user, "microhh", "python"),
            )

    def move_to_store(
        self,
        overwrite: bool = False,
        move_var_files: bool = False,
        vars: Union[str, list] = None,
        move_processed_files: bool = True,
    ):
        """Move files used for running MicroHH to store for reproducibility.

        Args:
            overwrite (bool, optional):
                Specify if existing files on 'store' should be overwritten.
            move_var_files (bool, optional):
                Specify if the .nc files of the variable files should be moved as well.
                Defaults to False.
            vars (Union[str, list], optional):
                Names of the variables to move if 'move_var_files' is True. Defaults to None.
            move_processed_files (bool, optional):
                Specify if the processed files (e.g. {gas}_2D.nc) should be moved. Defaults to True.
        """

        # Handle target path
        if not self.store_path.exists():
            self.store_path.mkdir(parents=True, exist_ok=True)
        elif overwrite:
            shutil.rmtree(self.store_path)
            print(f"Overwriting {self.store_path}.")
            self.store_path.mkdir(parents=True)

        # Copy files for reproducibility
        file_patterns = [
            "*" + self.case_name + ".ini",
            "*" + self.case_name + "_input.nc",
            "*" + self.case_name + ".out",
            "*" + self.case_name + ".default.*.nc",
            "run_microhh_*.out",
            "run_microhh.sbash",
            "microhh",
        ]
        for pattern in file_patterns:
            self.move_pattern_to_store(pattern, overwrite)

        if move_var_files:
            vars = object_to_list(vars)
            for var in vars:
                pattern = f"{var}.nc"
                self.move_pattern_to_store(pattern, overwrite)

        if move_processed_files:
            for pattern in ["*_2D.nc"]:
                self.move_pattern_to_store(pattern, overwrite)

    def move_pattern_to_store(self, pattern: str, overwrite: bool):
        """Move files whith a certain pattern to store.

        Args:
            pattern (str):
                Pattern of files which should be moved.
            overwrite (bool):
                Specify if existing files should be overwritten.
        """

        for file in self.scratch_path.glob(pattern):
            dest_file = Path(self.store_path, file.name)
            if not dest_file.exists() or overwrite:
                shutil.copy(file, self.store_path)
                print(f"Successfully copied '{file}' to '{self.store_path}'.")
            else:
                print(f"'{file}' already exists in '{self.store_path}'. Skipping.")

    def zip_grid_files(self, remove_files: bool = False):
        """Zip all binary grid files together.

        Args:
            remove_files (bool, optional):
                Specify if the grid files should be removed after zipping them together.
                Defaults to False.
        """
        zip_filename = Path(self.scratch_path, "binary_grid_files.zip")

        if zip_filename.exists():
            print(f"{zip_filename} already exists. Skipping...")
            return

        pattern = re.compile(r"^.*\.\d{7}$")
        matching_files = [
            file for file in self.scratch_path.iterdir() if pattern.match(file.name)
        ]

        with zipfile.ZipFile(zip_filename, "w") as zip_file:
            for file in matching_files:
                zip_file.write(file, file.name)

        print(f"{len(matching_files)} files were zipped to '{zip_filename}'.")

        if remove_files:
            for file in matching_files:
                file.unlink()

    def unzip_grid_files(self, remove_zip_file: bool = False):
        """Unzip the binay grid files to scratch"""
        zip_filename = Path(self.scratch_path, "binary_grid_files.zip")

        if not zip_filename.exists():
            print(f"{zip_filename} does not exist. Skipping...")
            return

        with zipfile.ZipFile(zip_filename, "r") as zip_ref:
            zip_ref.extractall(self.scratch_path)

        print(
            f"Files from '{zip_filename}' successfully extracted to '{self.scratch_path}'"
        )

        if remove_zip_file:
            zip_filename.unlink()

    def open_3d_prognostic_variable(
        self, variable: str, chunks: Union[str, dict] = None
    ) -> xr.Dataset:
        """Open 3D output netCDF file of 'variable' from model output.

        Args:
            variable (str):
                Variable name of dataset to open.

        Returns:
            xr.Dataset: 3D netCDF file from model output.
        """
        file_path = Path(self.file_directory, f"{variable.lower()}.nc")

        if not file_path.exists():
            self.create_3d_output(vars=variable.lower())

        data = self.open_microhh_nc(file_path, chunks=chunks)

        data = self._nc_assign_names_and_attributes(data)
        data = self._capitalise_variable_names(data)

        return data

    def open_microhh_nc(
        self, file_path: Union[str, Path], chunks: Union[str, dict] = None
    ) -> xr.Dataset:
        """Open a .nc file related to MicroHH output.

        Args:
            file_path (Union[str, Path]):
                Path of the file including file name.
        """
        try:
            data = xr.open_dataset(file_path, chunks=chunks)
        except ValueError:
            data = xr.open_dataset(file_path, chunks=chunks, decode_times=False)

        return data

    def _nc_assign_names_and_attributes(self, data: xr.Dataset) -> xr.Dataset:
        """Assign long_name and unit attributes to variables in a .nc file.

        Args:
            data (xr.Dataset):
                MicroHH dataset to which the attributes and units should be assigned.
        """
        # Define the dictionary with variable attributes using a dictionary comprehension
        variables = {
            var: (f"{var} mixing ratio", "mol mol-1") for var in gas_mass.keys()
        }
        variables.update(
            {
                "thl": ("liquid water potential temperature", "K"),
                "qt": ("total water mixing ratio", "kg kg-1"),
                "u": ("zonal wind component", "m s-1"),
                "v": ("meridional wind component", "m s-1"),
                "w": ("vertical wind component", "m s-1"),
                "p": ("air pressure", "Pa"),
            }
        )

        # Assign attributes to variables
        for var in data.variables:
            if var.lower() in variables:
                long_name, unit = variables[var.lower()]
                data[var].attrs.update({"long_name": long_name, "units": unit})

        return data

    def _capitalise_variable_names(self, dataset: xr.Dataset) -> xr.Dataset:
        """Capitalize variable names in an xr.Dataset if they match
        keys in the gas_mass dictionary.

        Args:
            dataset (xr.Dataset): The xarray dataset to process.

        Returns:
            xr.Dataset: The dataset with variable names capitalized if they match
            keys in the gas_mass dictionary.
        """
        for var_name in list(dataset.variables):
            new_var_name = self._capitalise_gas_key(var_name)
            if new_var_name != var_name:
                dataset = dataset.rename({var_name: new_var_name})
        return dataset

    def _capitalise_gas_key(self, variable_name: str) -> str:
        """Capitalise gases to make them consistent with 'ddeq'"""
        for key in gas_mass.keys():
            if key in variable_name:
                return key.upper()
        return variable_name

    def handle_prognostic_variables(
        self,
        vars: Union[str, list],
        filename: str,
        overwrite: bool = False,
        save: bool = True,
        chunks: Union[str, dict] = None,
    ) -> xr.Dataset:
        """Combine .nc files of different variables to one file.

        Args:
            vars (Union[str, list]):
                Modelled variables which should be processed.
                E.g. ["u", "v", "w", "ch4", "qt", "thl"]
            filename (str):
                Name of the file without path or file extension.
            overwrite (bool. Default: False):
                Indicate if an existing file should be overwritten.
            save (bool. Default: True):
                Indicate if the created file should be saved.

        Returns:
            xr.Dataset: Dataset with combined variables.
        """
        file_path = Path(self.file_directory, f"{filename}.nc")

        # Check if the output file already exists
        if file_path.exists() and not overwrite:
            # If the file exists, open and return it
            return self.open_microhh_nc(file_path, chunks=chunks)

        # combine the prognostic variables into one dataset
        combined_dataset = self.combine_prognostic_variables(vars=vars, chunks=chunks)

        # Save the combined dataset to a new NetCDF file
        if save:
            print(f"Saving {file_path}")
            combined_dataset.to_netcdf(file_path)

        return combined_dataset

    def combine_prognostic_variables(
        self,
        vars: Union[str, list],
        data: xr.Dataset = None,
        chunks: Union[str, dict] = None,
    ) -> xr.Dataset:
        """Combine multiple prognostic variables of the MicroHH output to one
        dataset. Specify either the variables to combine or give a dataset to
        which the new variables should be added.

        Args:
            vars (Union[str, list]):
                Variable(s) to add together or to assign to the given dataset.
            data (xr.Dataset):
                If given, the variable(s) are added to this dataset


        Returns:
            xr.Dataset:
                Dataset containing the combined variables.
        """
        vars = object_to_list(vars)
        datasets = [] if data is None else [data]

        for var in vars:
            dataset = self.open_3d_prognostic_variable(var, chunks=chunks)

            # homogenise dimensions
            dataset = self.interpolate_to_grid_midpoint(dataset)

            datasets.append(dataset)

        # Merge all datasets into one
        combined_dataset = xr.merge(datasets, combine_attrs="no_conflicts")

        return combined_dataset

    def interpolate_to_grid_midpoint(self, nc_file: xr.Dataset) -> xr.Dataset:
        """Interpolate variables at grid edges 'h' to the middle of the grid.

        Args:
            nc_file (xr.Dataset): Dataset with the variables to interpolate.

        Returns:
            xr.Dataset: Dataset with the interpolated variables
        """
        dim = self._get_dims_at_grid_boundaries(nc_file)

        if dim is None:
            return nc_file

        print(f"Interpolating {dim} to {dim[:-1]}")

        midpoints = nc_file[dim].values + np.nanmedian(np.diff(nc_file[dim].values) / 2)
        new_coords = {dim: midpoints}
        nc_file_interpolated = nc_file.interp(**new_coords, kwargs={"fill_value": "extrapolate"})

        # Create a new dimension name without the 'h' at the end
        new_dim_name = dim[:-1]
        # Rename the dimension in the Dataset
        nc_file_interpolated = nc_file_interpolated.rename({dim: new_dim_name})

        return nc_file_interpolated

    def _get_dims_at_grid_boundaries(self, data: xr.Dataset) -> str:
        """Get the name of the dimension that is sampled at the boundaries of the
        MicroHH grid

        Args:
            data (xr.Dataset):
                Dataset which contains a dimension which is sampled at the boundaries
                of the grid.

        Returns:
            str: Name of dimension which is sampled at the grid boundaries.
        """
        # iterate over the dimensions of the Dataset
        for dim in list(data.dims):
            # check if the dimension name ends with 'h'
            if dim.endswith("h"):
                return dim

        return None

    def convert_to_columns(
        self,
        variable: str,
        to_unit: str,
        default_p_and_t: bool = True,
        save: bool = True,
        overwrite: bool = False,
        chunks: Union[str, dict] = None,
    ) -> xr.Dataset:
        """Convert 3D MicroHH model output to vertically integrated columns.
        Assumes that the vertical dimension is in units m.

        Args:
            variable (str):
                Variable which should be vertically integrated.
            to_unit (str):
                Target unit of the columns.
            default_p_and_t (bool. Default: False):
                Indicate if the standard values for pressure and temperature should
                be used to convert mixing ratios to concentrations or if the
                modelled pressure and temperature should be used.
            save (bool. Default: True):
                Indicate if the created file should be saved.
            overwrite (bool. Default: False):
                Indicate if an existing file should be overwritten.

        Returns:
            xr.Dataset: Vertically integrated trace gas dataset.
        """

        # read in file if it exists:
        file_path = Path(self.file_directory, f"{variable.lower()}_2D.nc")
        if file_path.exists():
            if overwrite:
                os.remove(file_path)
                print(f"Overwriting {file_path}")
            else:
                return self.open_microhh_nc(file_path, chunks=chunks)

        data = self.open_3d_prognostic_variable(variable.lower(), chunks=chunks)

        to_unit = ucat.format_unit(to_unit, sep=" ")
        volume_unit = ucat.col_unit2vol_unit(to_unit)

        variable = variable.upper()

        # convert to correct volumentric unit first.
        # default MicroHH output is mol mol-1
        # use default pressure and temperature for computational efficiency
        if default_p_and_t:
            data[variable] = (
                ucat.convert_columns(
                    data[variable],
                    from_unit=data[variable].attrs.get("units", None),
                    to_unit=volume_unit,
                    molar_mass=variable,
                ),
            )
        else:
            data = self.convert_to_concentration(data, variable, volume_unit)

        # integrate vertically
        data_2D = data.integrate(coord="z")

        # adjust units due to integration
        data_2D[variable].attrs["units"] = ucat.format_unit(to_unit, sep=" ")
        data_2D[variable].attrs[
            "description"
        ] = f"vertically integrated {variable} columns"

        # remove other variables
        data_2D = data_2D[variable]

        if save:
            data_2D.to_netcdf(file_path)

        return data_2D

    def add_gaussian_noise(
        self, data: xr.Dataset, var: str, std: float, noise_units: str
    ) -> xr.Dataset:
        """
        Add Gaussian noise to a specified variable in the dataset.

        Args:
            data (xr.Dataset):
                The dataset containing the variable to which noise will be added.
            var (str):
                The name of the variable to which Gaussian noise will be added.
            std (float):
                The standard deviation of the Gaussian noise distribution.
            unit (str):
                Unit of the noise added.

        Returns:
            xr.Dataset: The dataset with the specified variable modified to include Gaussian noise.
        """
        var = var.upper()

        # Create a deep copy of the dataset to avoid modifying the original in-place
        data_copy = data.copy(deep=True)

        # convert noise to same unit as data
        data_units = data[var].attrs.get("units", None)
        std_conv = ucat.convert_columns(
            std, from_unit=noise_units, to_unit=data_units, molar_mass=var
        )

        # assume no bias (center = 0) but noise (= std_conv)
        noise = np.random.normal(0, std_conv, data_copy[var].size)
        noise = noise.reshape(data_copy[var].shape)

        data_copy[var] = data[var] + noise
        data_copy[var].attrs.update(data[var].attrs)
        data_copy[var].attrs["noise_level"] = std_conv

        # add std_conv for variable based on artificial noise
        data_copy[f"{var}_std"] = (data[var].dims, np.full_like(data[var], std_conv))
        data_copy[f"{var}_std"].attrs["units"] = data_units

        return data_copy

    def degrade_resolution(
        self, data: xr.Dataset, zonal_res: int, meridional_res: int
    ) -> xr.Dataset:
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
            ("nrows", "nobs"): ("nrows", "nobs"),
            ("longitude", "latitude"): ("longitude", "latitude"),
        }

        for dims, (zonal_dim, meridional_dim) in spatial_dims.items():
            if all(dim in data.dims for dim in dims):
                degraded_data = data.coarsen(
                    {zonal_dim: zonal_res, meridional_dim: meridional_res},
                    boundary="trim",
                ).mean()
                return degraded_data

        raise ValueError(f"Unknown spatial dimensions: {data.dims}")

    def create_sources_dataset(
        self,
        lat: Union[list, np.ndarray],
        lon: Union[list, np.ndarray],
        source_names: Union[list, np.ndarray] = None,
        source_labels: Union[list, np.ndarray] = None,
        source_diameter: Union[list, np.ndarray] = None,
    ) -> xr.Dataset:
        """Create a sources dataset used for the plume detection and in 'ddeq'.

        Args:
            lat (Union[list, np.ndarray]):
                List of latitudes for the locations of a source.
            lon (Union[list, np.ndarray]):
                List of longitudes for the locations of a source.
            source_names (Union[list, np.ndarray], optional. Default: None):
                List of names of the sources. Should not contain any special
                characters. If not specified, the sources will be numbered
                starting with source1 etc.
            source_labels (Union[list, np.ndarray], optional. Default: None):
                List of names of the sources used for plotting.
                If not specified, the sources will be numbered
                starting with source1 etc.
            source_diameter (Union[list, np.ndarray], optional. Default: None):
                Diameter (in pixels) around the soruce which are considered
                to assign a plume to a source. If not specified, 5 * pixel_size
                is used.

        Returns:
            xr.Dataset: Dataset with source information.
        """

        source_list = self.ini_file["source"]["sourcelist"]
        nr_sources = len(object_to_list(source_list))

        source_names = (
            source_names or source_labels or [f"source{i+1}" for i in range(nr_sources)]
        )
        source_labels = source_labels or source_names

        if source_diameter is None:
            pixel_size = np.max(
                [
                    self.ini_file["grid"]["xsize"] / self.ini_file["grid"]["itot"],
                    self.ini_file["grid"]["ysize"] / self.ini_file["grid"]["jtot"],
                ]
            )
            source_diameter = np.full(nr_sources, 5 * pixel_size)  # in m

        data = {
            "lon": (["source"], np.array(object_to_list(lon), dtype=float)),
            "lat": (["source"], np.array(object_to_list(lat), dtype=float)),
            "source_x0": (
                ["source"],
                np.array(
                    object_to_list(self.ini_file["source"]["source_x0"]), dtype=float
                ),
            ),
            "source_y0": (
                ["source"],
                np.array(
                    object_to_list(self.ini_file["source"]["source_y0"]), dtype=float
                ),
            ),
            "source_z0": (
                ["source"],
                np.array(
                    object_to_list(self.ini_file["source"]["source_z0"]), dtype=float
                ),
            ),
            "label": (
                ["source"],
                np.array(object_to_list(source_labels), dtype=object),
            ),
            "diameter": (
                ["source"],
                np.array(object_to_list(source_diameter), dtype=float),
            ),
            "strength": (["source"], self.model_emissions_to_kg_per_h()),
        }

        coords = {"source": np.array(object_to_list(source_names), dtype=object)}

        sources = xr.Dataset(data, coords=coords)
        sources.lon.attrs["units"] = "longitude of source"
        sources.lat.attrs["units"] = "latitude of source"
        sources.diameter.attrs["units"] = "m"
        sources.strength.attrs["units"] = "kg h-1"

        for dim in ["x", "y", "z"]:
            sources[f"source_{dim}0"].attrs["units"] = "m"
            sources[f"source_{dim}0"].attrs[
                "description"
            ] = f"{dim} position in MicroHH grid"

        return sources

    def convert_microhh_grid_to_coords(
        self,
        data: xr.Dataset,
        source: str = None,
        sources: xr.Dataset = None,
        target_crs: str = "epsg:4326",
        microhh_crs: str = "epsg:32632",
    ) -> xr.Dataset:
        """Convert the modelled MicroHH grid into a lat/lon grid.
        If lat and lon are specified in the .ini file, these values are
        used as domain centres.
        If 'source' and 'sources' are provided, the lat and lon of the source
        is used to convert the grid such that the source is located at the
        latitude and longitude specified in the 'sources' dataset.

        Args:
            data (xr.Dataset):
                Dataset for which the modelled grid should be transformed to
                a lat/lon grid.
            source (str, optional):
                Source simulated in the dataset 'data'.
            sources (xr.Dataset, optional):
                Dataset which contains information about the modelled source.
            target_crs (str, optional. Default: 'epsg:4326'):
                Coordinate reference system to which the modelled data should
                be transformed to.
            microhh_crs (str, optional. Default: 'epsg:32632'):
                Metric coordinate reference system which matches the desired
                lat/lon location of the source.
                Default 'epsg:32632' is suited for sources in Europe.

        Returns:
            xr.Dataset: Dataset with transformed coordinates.
        """

        # if lat/lon is defined in .ini file: use these values to convert grid
        if "lat" in self.ini_file["grid"] and source is None and sources is None:
            lat = self.ini_file["grid"]["lat"]
            lon = self.ini_file["grid"]["lon"]
            grid_x = self.ini_file["grid"]["xsize"] / 2
            grid_y = self.ini_file["grid"]["ysize"] / 2
        elif source is not None and sources is not None:
            source_data = sources.sel(source=source)
            lat = source_data.lat.values
            lon = source_data.lon.values
            grid_x = source_data.source_x0.values
            grid_y = source_data.source_y0.values
        else:
            raise ValueError(
                f"Provide 'lat' and 'lon' in the {self.case_name} "
                f"file or 'source' and 'sources' as arguments"
            )

        # convert coordinates to metric crs
        transformer = pyproj.Transformer.from_crs(target_crs, microhh_crs)
        x_o, y_o = transformer.transform(lat, lon)

        # create 2D grid with x and y values
        xx, yy = np.meshgrid(data["x"], data["y"])

        # shift grid of x and y values so that the source or domain center
        # is at the location given by lat and lon
        shifted_grid_x = xx - grid_x + x_o
        shifted_grid_y = yy - grid_y + y_o

        # transform shifted grid to lat / lon
        lat_grid, lon_grid = transformer.transform(
            shifted_grid_x, shifted_grid_y, direction="INVERSE"
        )

        data = data.rename({"y": "nrows", "x": "nobs"})

        # define the coordinates
        coords = {
            "time": (["time"], data.time.data),
            "lon": (["nrows", "nobs"], lon_grid),
            "lat": (["nrows", "nobs"], lat_grid),
        }

        # extract only the variables that are not a dimension
        vars_to_include = {
            var: data[var] for var in data.variables if var not in data.dims
        }

        # create the new dataset
        data = xr.Dataset(vars_to_include, coords=coords).drop_vars(["nrows", "nobs"])

        return data

    def convert_microhh_timestep_to_date(
        self,
        data: xr.Dataset,
        start_date: Union[str, pd.Timestamp] = None,
        format="%Y-%m-%d %H:%M:%S",
    ) -> xr.Dataset:
        """Convert time into a timestamp to make the
        dataset digestible by `ddeq`.
        Use 'datetime_utc' if available in the .ini file.

        Args:
            data (xr.Dataset):
                The dataset to convert.
            start_date (Union[str, Timestamp], optional):
                The start date as a string or a pandas Timestamp object.

        Returns:
            xr.Dataset:
                The dataset with the time converted to a timestamp.
        """
        # use 'datetime_utc' if available in the .ini file
        if "datetime_utc" in self.ini_file["time"] and start_date is None:
            start_date = self.ini_file["time"]["datetime_utc"]
        elif "datetime_utc" in self.ini_file["time"] and start_date is not None:
            start_date = start_date
            print(
                f"Using {start_date} to overwrite {self.ini_file['time']['datetime_utc']} "
                f"in .ini file"
            )
        else:
            start_date = start_date

        # Check if start_date is a string and convert it to a Timestamp if necessary
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date, format=format)

        # check if the time coordinate is alreay given as datetime:
        if data["time"].dtype == "<M8[ns]":
            # overwrite datetime
            time_diff = pd.Timestamp(data["time"][0].values) - start_date
            data["time"] = data["time"] - time_diff

        elif data["time"].dtype in ["int64", "float64"]:
            # convert time to timedelta
            data["time"] = pd.to_timedelta(data["time"], unit="s")
            # add Timedelta to the start date
            data["time"] = start_date + data["time"]

        else:
            raise ValueError("Data type of 'time' was not recognized.")

        return data

    @staticmethod
    def add_corners(
        data: xr.Dataset,
        var: Tuple[str, str] = ("lon", "lat"),
        dims: Tuple[str, str] = ("nrows", "nobs"),
    ) -> xr.Dataset:
        """
        Add corner coordinates for 'var'.
        """
        ic = np.arange(data[dims[1]].size + 1) - 0.5
        jc = np.arange(data[dims[0]].size + 1) - 0.5

        bbox = [-0.5, data[dims[0]].size, -0.5, data[dims[1]].size]

        for name in var:
            interpolator = scipy.interpolate.RectBivariateSpline(
                data[dims[0]], data[dims[1]], data[name], kx=1, ky=1, bbox=bbox
            )
            xc = interpolator(jc, ic)
            xc = np.moveaxis([xc[:-1, :-1], xc[:-1, 1:], xc[1:, 1:], xc[1:, :-1]], 0, 2)
            data[f"{name}c"] = xr.DataArray(xc, dims=dims + ("corners",))

        return data

    def model_emissions_to_kg_per_h(self) -> np.ndarray:
        """
        Convert the prescribed emissions from the .ini file to kg/h

        Returns:
            np.array: Emitted quantities in kg/h
        """
        sourcelist = self.ini_file["source"]["sourcelist"]
        swvmr = self.ini_file["source"]["swvmr"]
        strength = self.ini_file["source"]["strength"]

        if not isinstance(sourcelist, list):
            sourcelist = np.array([sourcelist])
            swvmr = np.array([swvmr])
            strength = np.array([strength])

        molar_mass = np.array([gas_mass[gas] for gas in sourcelist])

        if all(swvmr):  # it true, units are kmol s-1
            strength *= molar_mass * 60**2  # convert to kg/h
        else:  # units are kg kg s-1, seems to be an error
            strength = np.full(len(swvmr), np.nan)

        return strength

    def convert_to_concentration(
        self,
        data: xr.Dataset,
        gas: str = "ch4",
        to_unit: str = "kg m-3",
        chunks: Union[str, dict] = None,
    ) -> xr.Dataset:
        """Convert the gas given in data to concentrations using modelled pressure
        and potential temperature.

        Args:
            data (xr.Dataset):
                Dataset containing the variable which should be converted to concentrations.
            gas (str, optional):
                Gas for which the columns should be computed. Defaults to "ch4".
            to_unit (str, optional):
                Concentration units to which the gas should be converted.
                Defaults to "kg m-3".
        """
        required_prognostic_vars = [gas, "thl"]

        # check if all required prognostic variables are in the dataset
        for var in required_prognostic_vars:
            if var not in data.data_vars:
                print(f"Adding '{var}' to the dataset.")
                data = self.combine_prognostic_variables(
                    vars=var, data=data, chunks=chunks
                )

        # check if pressure is in the dataset
        if "phydro" not in data.data_vars:
            print(f"Adding 'phydro' to the dataset.")
            data = self.add_statistics_variable(data, "phydro")

        # convert potential temperature to absolute temperature
        if "ta" not in data.data_vars:
            print(f"Calculating absolute temperature.")
            data = self.add_absolute_temperature(data)

        from_unit = data[gas].attrs.get("units", None)

        print(
            f"Converting {gas} mixing ratios [{from_unit}] to concentrations [{to_unit}].",
            "This might take a while.",
        )

        data[gas] = ucat.convert_points(
            data[gas],
            from_unit=from_unit,
            to_unit=to_unit,
            molar_mass=gas,
            p=data.phydro,
            T=data.ta,
        )

        return data

    def add_statistics_variable(self, data: xr.Dataset, variable: str) -> xr.Dataset:
        # read in statistics file
        stats = init.read_statistics(self.case)

        # check if variable is in statistics file
        if variable not in stats.data_vars:
            raise KeyError(f"No variable named '{variable}'.")

        # check if target file has the necessary dimensions:
        for dim in stats.sizes:
            if dim not in data.dims or dim not in data.coords:
                data = data.assign_coords({dim: stats[dim]})

        # get dask chunks
        chunks = dict(data.chunks)

        if "time" in stats[variable].dims and "z" in stats[variable].dims:
            variable_values = np.tile(
                stats[variable].values[:, :, np.newaxis, np.newaxis],
                (1, 1, data.y.size, data.x.size),
            )
            data[variable] = (
                ("time", "z", "y", "x"),
                variable_values,
                stats[variable].attrs,
            )
        if "time" in stats[variable].dims and "zh" in stats[variable].dims:
            variable_values = np.tile(
                stats[variable].values[:, :, np.newaxis, np.newaxis],
                (1, 1, data.y.size, data.x.size),
            )
            data[variable] = (
                ("time", "zh", "y", "x"),
                variable_values,
                stats[variable].attrs,
            )
            if "z" in chunks:
                chunks["zh"] = (chunks["z"][0] + 1,)  # zh has dim z + 1
                chunks.pop("z")

        data[variable] = data[variable].chunk(chunks)

        return data

    def add_absolute_temperature(self, data: xr.Dataset) -> xr.Dataset:
        """FIXME

        Args:
            data (xr.Dataset): _description_

        Returns:
            xr.Dataset: _description_
        """
        required_vars = ["phydro", "phydroh"]

        for var in required_vars:
            if var not in data.data_vars:
                print(f"Adding '{var}' to the dataset.")
                data = self.add_statistics_variable(data, variable=var)

        # Calculate absolute temperature
        data["ta"] = PostProcessing.calculate_absolute_temperature(
            theta=data.thl, p=data.phydro, p0=data.phydroh.sel(zh=0)
        )

        # Add attributes to 'ta' variable
        data["ta"].attrs["long_name"] = "Absolute temperature"
        data["ta"].attrs["units"] = "K"

        return data

    @staticmethod
    def calculate_absolute_temperature(theta, p, p0):
        """
        Calculate absolute temperature from potential temperature and pressure.

        FIXME
        """
        # Constants
        R = scipy.constants.R  # Specific gas constant for dry air in J/(kg·K)
        cp = 1004  # Specific heat capacity at constant pressure for dry air in J/(kg·K)

        # Calculate absolute temperature
        absolute_temperature = theta * (p / p0) ** (R / cp)

        return absolute_temperature

    def calculate_3D_tke(
        self, data: xr.Dataset, chunks: Union[str, dict] = None
    ) -> xr.Dataset:
        """Calculate the TKE based on the 3D prognostic variables of MicroHH

        Args:
            data (xr.Dataset):
                Dataset containing the required variables. If it does not contain
                all variables, they are added.

        Returns:
            xr.Dataset:
                Dataset containing TKE as a variable.
        """
        wind_components = ["u", "v", "w"]

        for var in wind_components:
            if var not in data.data_vars:
                data = self.combine_prognostic_variables(
                    vars=var, data=data, chunks=chunks
                )

        # compute temporal deviation for each wind component
        for wind_component in wind_components:
            data[f"{wind_component}_std"] = data[wind_component] - data[
                wind_component
            ].mean(dim="time")
            data[f"{wind_component}_std"].attrs["units"] = data[wind_component].attrs[
                "units"
            ]
            data[f"{wind_component}_std"].attrs[
                "description"
            ] = f"deviation of {wind_component} wind speed from mean"

        # compute TKE
        data["TKE"] = 0.5 * (
            data["u_std"] ** 2 + data["v_std"] ** 2 + data["w_std"] ** 2
        ).mean(dim="time")
        data[f"{wind_component}_std"].attrs["units"] = "m2 s-2"
        data[f"{wind_component}_std"].attrs["long_name"] = "turbulent kinetic energy"

        return data

    def parse_slurm_output(self) -> pd.DataFrame:
        """Parse the output of the 'slurm.out' script to get the job information.

        Returns:
            pd.DataFrame: Data frame containing the job information.
        """
        counter = 0
        job_info = False
        headers = []
        values = []

        out_files = list(self.file_directory.glob("run_microhh_*.out"))

        if not out_files:
            raise FileNotFoundError(
                f"No slurm output script found in {self.file_directory}"
            )

        # FIXME: Handle case with multiple files

        for out_file in out_files:
            with open(out_file, "r") as f:
                for line in f:
                    # use the lines after "job information"
                    if "Job information" in line:
                        counter = 0
                        job_info = True
                        continue

                    if counter == 1 and job_info:  # first line defines the header
                        header = line
                    elif (
                        counter == 2 and job_info
                    ):  # second line defines the length of the content
                        separator = line
                        sep_len = [len(i) for i in separator.split()]
                    elif counter == 3 and job_info:  # third line defines the value
                        value = line
                    elif counter == 4 and job_info:  # reset for next job info
                        job_info = False
                        startIndex = 0

                        # parse header and values
                        for l in sep_len:
                            headers.append(header[startIndex : startIndex + l].strip())
                            values.append(value[startIndex : startIndex + l].strip())
                            startIndex += l + 1
                            df = pd.DataFrame([values], columns=headers)

                    counter += 1

        # Format columns
        timedelta_cols = ["Elapsed", "Time limit"]
        datetime_cols = ["Submit", "Eligible", "Start", "End"]
        df[timedelta_cols] = df[timedelta_cols].apply(pd.to_timedelta)
        df[datetime_cols] = df[datetime_cols].apply(pd.to_datetime)

        # Extracting integer part and unit part from usage and memory
        df[["Energy", "Energy unit"]] = df["Energy"].str.split(expand=True)
        df["Energy"] = df["Energy"].astype(float)

        try:  # only available for simulations on GPU
            df["Execution time"] = df["Execution time"].apply(pd.to_timedelta)
            df[["Usage", "Usage unit"]] = df["Usage"].str.split(expand=True)
            df[["Max mem", "Max mem unit"]] = df["Max mem"].str.split(expand=True)
            df[["Usage", "Max mem"]] = df[["Usage", "Max mem"]].astype(float)
        except:
            pass

        return df


############# other functions #############
def object_to_list(var) -> list:
    if var is None:
        raise ValueError("Input should not be 'None'")
    if isinstance(var, xr.DataArray):
        var = var.values.tolist()
    elif isinstance(var, np.ndarray):
        var = var.tolist()
    elif not isinstance(var, list):
        var = [var]
    return var


def get_2D_tke_threshold_time(
    stats: xr.Dataset,
    tke_threshold: bool = False,
    tke_der_threshold: bool = False,
    threshold_value=0,
) -> Tuple[int, Union[int, np.datetime64]]:

    if not tke_threshold and not tke_der_threshold:
        raise ValueError("Either 'tke_threshold' or 'tke_der_treshold' must be true.")
    if tke_threshold and tke_der_threshold:
        raise ValueError(
            "Only one of 'tke_threshold' or 'tke_der_treshold' should be true."
        )

    # Calculate the median of tke over the 'z' dimension
    tke_median = stats.tke.median(dim="z")

    if tke_threshold:
        threshold_time = tke_median.where(
            tke_median > threshold_value * tke_median.max()
        ).dropna(dim="time")
    elif tke_der_threshold:
        # Calculate the derivative of tke_median with respect to time
        tke_derivative = tke_median.differentiate(coord="time")

        # Calculate the relative derivative
        tke_median_overall = stats.tke.median(dim=["z", "time"])
        relative_derivative = tke_derivative / tke_median_overall

        # Find the time where the threshold is exceeded for the first time
        threshold_time = relative_derivative.where(
            relative_derivative > threshold_value
        ).dropna(dim="time")
    else:
        raise ValueError

    if threshold_time.size > 0:
        time_value = threshold_time.time[0].item()
        time_ind = stats["time"].where(stats["time"] == time_value).argmax().item()
        return time_ind, time_value
    else:
        raise ValueError(f"No data for TKE threshold = {threshold_value}.")
