"""Post-processing utilities for MicroHH model output."""

import logging
import re
import shlex
import shutil
import zipfile
from pathlib import Path
from typing import Literal, Optional, Union, cast

import cscs
import microhh_py.convert_3d_to_nc as convert_3d_to_nc
import microhh_py.cross_to_nc as cross_to_nc
import microhh_py.microhh_tools as mht
import netCDF4
import numpy as np
import pandas as pd
import scipy
import ucat
import xarray as xr
from tqdm.auto import tqdm

import microhh_processing_chain.initialisation as init
import microhh_processing_chain.io as io
import microhh_processing_chain.sources as src
import microhh_processing_chain.spatial as spatial
from microhh_processing_chain import utils
from microhh_processing_chain.constants import GAS_MOLAR_MASS
from microhh_processing_chain.enums import Precision

logger = logging.getLogger(__name__)


def convert_microhh_grid_to_coords(
    case: init.Case,
    data: xr.Dataset,
    source: Optional[xr.Dataset] = None,
    target_crs: str = "epsg:4326",
    microhh_crs: str = "epsg:32632",
) -> xr.Dataset:
    """Convert the modelled MicroHH grid into a lat/lon grid.

    If lat and lon are specified in the .ini file, these values are
    used as domain centres.
    If 'source' is provided, the lat and lon of the source
    is used to convert the grid such that the source is located at the
    latitude and longitude specified in the 'sources' dataset.

    Args:
        case:
            MicroHH case containing the grid configuration.
        data:
            Dataset for which the modelled grid should be transformed to
            a lat/lon grid.
        source:
            Source metadata used as the grid anchor.
        target_crs:
            Coordinate reference system to which the modelled data should
            be transformed to.
        microhh_crs:
            Metric coordinate reference system which matches the desired
            lat/lon location of the source.

    Returns:
        Dataset with transformed coordinates.
    """
    # if lat/lon is defined in .ini file: use these values to convert grid
    if "lat" in case.config["grid"] and "lon" in case.config["grid"] and source is None:
        lat_ref = case.config["grid"]["lat"]
        lon_ref = case.config["grid"]["lon"]
        grid_x_ref = case.config["grid"]["xsize"] / 2
        grid_y_ref = case.config["grid"]["ysize"] / 2
    elif source is not None:
        lat_ref = source.lat.values
        lon_ref = source.lon.values
        grid_x_ref = source.source_x0.values
        grid_y_ref = source.source_y0.values
    else:
        raise ValueError(
            f"Provide 'lat' and 'lon' in the {case.name} "
            f"file or 'source' and 'sources' as arguments"
        )

    # Create domain
    domain = spatial.Domain(
        xsize=case.config["grid"]["xsize"],
        ysize=case.config["grid"]["ysize"],
        itot=case.config["grid"]["itot"],
        jtot=case.config["grid"]["jtot"],
        lon=lon_ref,
        lat=lat_ref,
        anchor=(grid_x_ref, grid_y_ref),
        epsg_code=microhh_crs,
    )
    lon_grid, lat_grid = domain.to_lonlat(data.x.values, data.y.values, meshgrid=True)

    data = data.rename({"x": "nobs", "y": "nrows"})
    data = data.assign_coords(
        lon=(("nrows", "nobs"), lon_grid), lat=(("nrows", "nobs"), lat_grid)
    )

    # add attributes to lon and lat
    data.lon.attrs["epsg"] = int(target_crs.split(":")[-1])
    data.lat.attrs["epsg"] = int(target_crs.split(":")[-1])

    # drop nrows and nobs from data vars
    data = data.drop_vars(["nrows", "nobs"])

    # add x and y coordinates
    data["x"] = (
        ("nrows", "nobs"),
        domain.xx + domain.x_offset,
        {"epsg": int(microhh_crs.split(":")[-1])},
    )
    data["y"] = (
        ("nrows", "nobs"),
        domain.yy + domain.y_offset,
        {"epsg": int(microhh_crs.split(":")[-1])},
    )

    return data


def convert_microhh_timestep_to_date(
    case: init.Case,
    data: xr.Dataset,
    start_date: Optional[Union[str, pd.Timestamp]] = None,
    date_format: str = "%Y-%m-%d %H:%M:%S",
) -> xr.Dataset:
    """Convert time into a timestamp to make the dataset digestible by `ddeq`.

    Use 'datetime_utc' if available in the .ini file.

    Args:
        case:
            MicroHH case containing the time configuration.
        data:
            The dataset to convert.
        start_date:
            The start date as a string or a pandas Timestamp object.
        date_format:
            The format of the start_date string.

    Returns:
        Dataset with the time converted to a timestamp.
    """
    # use 'datetime_utc' if available in the .ini file
    if "datetime_utc" in case.config["time"] and start_date is None:
        start_date = case.config["time"]["datetime_utc"]
    elif "datetime_utc" in case.config["time"] and start_date is not None:
        logger.info(
            "Using %s to overwrite %s in .ini file",
            start_date,
            case.config["time"]["datetime_utc"],
        )

    if start_date is None:
        raise ValueError(
            "Please provide 'start_date' or set 'datetime_utc' in the ini file."
        )

    tz = "UTC"

    if isinstance(start_date, str):
        converted = pd.to_datetime(start_date, format=date_format)
        if pd.isna(converted):
            raise ValueError(f"Could not parse start_date: {start_date}")
        start_date = cast(pd.Timestamp, converted)

    if isinstance(start_date, pd.Timestamp) and start_date.tz is not None:
        start_date = start_date.tz_convert("UTC").tz_localize(None)

    # Force ns precision for the reference timestamp as well.
    ts_start_date = pd.Timestamp(start_date).as_unit("ns")

    if np.issubdtype(data["time"].dtype, np.datetime64):
        time_values = pd.to_datetime(data["time"].values).astype("datetime64[ns]")
        first_time = cast(pd.Timestamp, pd.to_datetime(time_values[0]))
        time_diff = first_time - ts_start_date
        new_time = time_values - time_diff

    elif np.issubdtype(data["time"].dtype, np.integer) or np.issubdtype(
        data["time"].dtype, np.floating
    ):
        # convert MicroHH seconds to timedelta, then add to start date
        time_delta = pd.to_timedelta(data["time"].to_numpy(), unit="s")
        new_time = pd.DatetimeIndex(start_date + time_delta).astype("datetime64[ns]")

    else:
        raise ValueError("Data type of 'time' was not recognized.")

    data = data.assign_coords(time=new_time)
    data["time"].attrs.update({"timezone": tz})

    return data


class PostProcessing:
    """Class used for post-processing MicroHH cases and handling the output.

    Args:
        case:
            Case to post-process.
    """

    def __init__(self, case: init.Case) -> None:
        self.case = case

    def process_case_on_alps(
        self,
        alps_setup: cscs.AlpsSetup,
        py_file_path: str,
        job_name: str,
        python_path: str,
        additional_args: Optional[dict] = None,
        overwrite: bool = False,
        start_jobs: bool = True,
    ) -> cscs.SlurmJob:
        """Submit a bash script to process a case on alps.

        Make sure that the python venv used for processing was created with the
        same uenv as the one in alps_setup.

        Args:
            alps_setup:
                Alps setup which should be used to process the case.
            py_file_path:
                File to the python script which should be executed to process the case.
            job_name:
                Name of the job. Determines the name of the resulting slurm.out script
                and the job name in the queue.
            python_path:
                Path of the python environment to use.
            additional_args:
                Additional arguments for the function called in the python script.
            overwrite:
                Specify if an existing sbatch script should be overwritten.
            start_jobs:
                Specify if the job should be started.

        Returns:
            Union[str, Path]:
                Path to the created sbatch script.
        """
        if additional_args is None:
            additional_args = {}

        job_name_formatted = job_name.replace(" ", "_").replace("-", "_")

        # create command for bash script
        cmd_parts = [
            # "srun",
            str(python_path),
            str(py_file_path),
            "--directory",
            str(self.case.directory),
        ]

        for key, value in additional_args.items():
            if isinstance(value, list):  # convert list to string for command
                cmd_parts.extend([f"--{key}", *[str(item) for item in value]])
            else:
                cmd_parts.extend([f"--{key}", str(value)])

        cmd = shlex.join(cmd_parts)

        # submit sbatch script to run post processing in parallel on alps
        job_file = Path(self.case.scratch_path, f"{job_name_formatted}.sbatch")

        if job_file.exists() and overwrite:
            job_file.unlink()

        if not job_file.exists() or overwrite:
            slurm_job = alps_setup.create_sbatch_script(
                commands=cmd,
                job_name=job_name_formatted,
                file_path=job_file,
                scratch_path=self.case.scratch_path,
            )
        else:
            logger.info("Reading existing sbatch script %s", job_file)

            slurm_job = cscs.SlurmJob(job_file, ssh_connect=alps_setup.ssh_connect)

        if start_jobs:
            slurm_job.start(wait=False)

        return slurm_job

    def create_3d_output(
        self,
        variables: Union[str, list],
        precision: Union[Precision, str] = Precision.SINGLE,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        sample_time: Optional[int] = None,
        nprocs: Optional[int] = None,
        overwrite: bool = False,
    ) -> None:
        """Create 3D netCDF files for the variables specified.

        Args:
            variables:
                Modelled variables which should be processed.
                E.g. ["u", "v", "w", "ch4", "qt", "th"]
            precision:
                Precision used for the created .nc file.
            start_time:
                Start time in s of the data to process.
            end_time:
                End time in s of the data to process.
            sample_time:
                Sample time in s of the output .nc file.
            nprocs:
                Number of worker processes for parallel processing.
            overwrite:
                Specify if existing .nc files should be overwritten.
        """
        variables = np.atleast_1d(variables).tolist()

        if variables:
            convert_3d_to_nc.run_conversion(
                filename=f"{self.case.name}.ini",
                directory=self.case.directory,
                variables=variables,
                precision=(
                    precision.value if isinstance(precision, Precision) else precision
                ),
                starttime=start_time,
                endtime=end_time,
                sampletime=sample_time,
                nprocs=1 if nprocs is None else nprocs,
                overwrite=overwrite,
            )

    def create_2d_output(
        self,
        variables: Union[str, list],
        modes: Optional[list] = None,
        indexes: Optional[list] = None,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        sample_time: Optional[int] = None,
        precision: Union[Precision, str] = Precision.SINGLE,
        nprocs: Optional[int] = None,
        compression: bool = True,
        order: Optional[int] = None,
        overwrite: bool = False,
    ) -> None:
        """Create cross-section netCDF files for the variables specified.

        Args:
            variables:
                Modelled variables which should be processed.
                E.g. ["u", "v", "w", "ch4", "qt", "th"]
            modes:
                Cross-section modes to process. Available options are "xy", "xz", "yz".
                If not provided, modes are determined from the .ini file.
            indexes:
                Indices of the cross-sections to process.
            start_time:
                Start time in s of the data to process.
            end_time:
                End time in s of the data to process.
            sample_time:
                Sample time in s of the cross-section output.
            precision:
                Precision used for the created .nc file.
            nprocs:
                Number of worker processes for parallel processing.
            compression:
                Enable compression in the netCDF file.
            order:
                Spatial order used to read the MicroHH grid.
            overwrite:
                Specify if existing files should be overwritten.
        """
        variables_list = np.atleast_1d(variables).tolist()
        if variables_list:
            cross_to_nc.run_conversion(
                filename=f"{self.case.name}.ini",
                directory=self.case.directory,
                modes=modes,
                variables=variables_list,
                indexes=indexes,
                starttime=start_time,
                endtime=end_time,
                sampletime=sample_time,
                precision=(
                    precision.value if isinstance(precision, Precision) else precision
                ),
                nprocs=1 if nprocs is None else nprocs,
                compression=compression,
                order=order,
                overwrite=overwrite,
            )

    def archive_case_on_store(
        self,
        store_path: Union[str, Path],
        overwrite: bool = False,
        variables: Optional[Union[str, list]] = None,
    ) -> None:
        """Move files used for running MicroHH to store for reproducibility.

        Args:
            store_path:
                Path to the folder where the files should be moved to.
            overwrite:
                Specify if existing files on 'store' should be overwritten.
            variables:
                Names of the processed {var}.nc files to move.
        """
        store_path = Path(store_path) / self.case.full_name

        # Handle target path
        if not store_path.exists():
            store_path.mkdir(parents=True, exist_ok=True)
        elif overwrite:
            shutil.rmtree(store_path)
            logger.warning("Overwriting %s.", store_path)
            store_path.mkdir(parents=True)

        # Copy files for reproducibility
        file_patterns = [
            "*" + self.case.name + ".ini",
            "*" + self.case.name + "_input.nc",
            "*" + self.case.name + ".out",
            "*" + self.case.name + ".default.*.nc",
            "init_microhh_*.out",
            "run_microhh_*.out",
            "run_microhh.sbatch",
            "microhh",
            "jobreport_*",
            "*_benchmark.json",
        ]
        for pattern in file_patterns:
            self.move_pattern_to_store(
                pattern, store_path=store_path, overwrite=overwrite
            )

        if variables is not None:
            variables_list = np.atleast_1d(variables).tolist()
            for var in variables_list:
                pattern = f"{var}.nc"
                self.move_pattern_to_store(
                    pattern, store_path=store_path, overwrite=overwrite
                )

    def move_pattern_to_store(
        self, pattern: str, store_path: Union[str, Path], overwrite: bool = False
    ) -> None:
        """Move files or folders with a certain pattern to store.

        Args:
            pattern:
                Pattern of files or folders which should be moved.
            store_path:
                Path to the folder where the files or folders should be moved to.
            overwrite:
                Specify if existing files or folders should be overwritten.
        """
        store_path = Path(store_path)

        for item in self.case.directory.glob(pattern):
            dest_item = store_path / item.name

            if dest_item.exists() and not overwrite:
                logger.info(
                    "'%s' already exists in '%s'. Skipping.", item.name, store_path
                )
                continue

            if dest_item.exists() and overwrite:
                if dest_item.is_dir():
                    shutil.rmtree(dest_item)
                else:
                    dest_item.unlink()

            if item.is_dir():
                shutil.copytree(item, dest_item)
            else:
                shutil.copy(item, store_path)

            logger.info("Successfully copied '%s' to '%s'.", item.name, store_path)

    def zip_binary_files(
        self, file_pattern: str = "^.*", remove_files: bool = False
    ) -> None:
        """Zip all binary files matching the given pattern together.

        Args:
            file_pattern:
                Pattern of files to zip.
            remove_files:
                Specify if the grid files should be removed after zipping them together.
        """
        zip_filename = (
            f"{file_pattern}.zip" if file_pattern != "^.*" else "binary_files.zip"
        )
        zip_file_path = Path(self.case.directory, zip_filename)

        if zip_file_path.exists():
            logger.info("%s already exists. Skipping...", zip_file_path)
            return

        pattern = re.compile(rf"{file_pattern}\.\d{{7}}$")
        matching_files = [
            file for file in self.case.directory.iterdir() if pattern.match(file.name)
        ]

        if not matching_files:
            logger.warning(
                "No files matching the pattern '%s' were found. Skipping...",
                file_pattern,
            )
            return

        with zipfile.ZipFile(zip_file_path, "w") as zip_file:
            for file in tqdm(matching_files, desc="Zipping files", unit="file"):
                zip_file.write(file, file.name)

        logger.info("%d files were zipped to '%s'.", len(matching_files), zip_file_path)

        if remove_files:
            for file in tqdm(matching_files, desc="Removing files", unit="file"):
                file.unlink()

    def unzip_binary_files(self, remove_zip_file: bool = False) -> None:
        """Unzip the binary grid files to scratch."""
        zip_filename = Path(self.case.directory, "binary_grid_files.zip")

        if not zip_filename.exists():
            logger.warning("%s does not exist. Skipping...", zip_filename)
            return

        with zipfile.ZipFile(zip_filename, "r") as zip_ref:
            zip_ref.extractall(self.case.directory)

        logger.info(
            "Files from '%s' successfully extracted to '%s'",
            zip_filename,
            self.case.directory,
        )

        if remove_zip_file:
            zip_filename.unlink()

    def open_3d_prognostic_variable(
        self, variable: str, chunks: Optional[Union[str, dict]] = None
    ) -> xr.Dataset:
        """Open 3D output netCDF file of 'variable' from model output.

        Args:
            variable:
                Variable name of dataset to open.
            chunks:
                Chunks used when processing the netCDF files with xarray.

        Returns:
            3D NetCDF file from model output.
        """
        file_path = Path(self.case.directory, f"{variable.lower()}.nc")

        if not file_path.exists():
            self.create_3d_output(variables=variable.lower())

        data = io.open_microhh_nc(case=self.case, variable=variable, chunks=chunks)
        data = self._nc_assign_names_and_attributes(data)

        return data

    def _nc_assign_names_and_attributes(self, data: xr.Dataset) -> xr.Dataset:
        """Assign long_name and unit attributes to variables in a .nc file.

        FIXME: Check if swvmr is in ini file

        Args:
            data:
                MicroHH dataset to which the attributes and units should be assigned.
        """
        swvmr = np.all(self.case.config["source"]["swvmr"])
        if swvmr:
            unit = "mol mol-1"
            prefix = "volume"
        else:
            unit = "kg kg-1"
            prefix = "mass"

        variables = {
            var: (f"{var} {prefix} mixing ratio", unit) for var in GAS_MOLAR_MASS
        }
        variables.update(
            {
                "th": ("potential temperature", "K"),
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
            var_name = str(var).lower()
            if var_name in variables:
                long_name, unit = variables[var_name]
                data[var].attrs.update({"long_name": long_name, "units": unit})

        return data

    def handle_prognostic_variables(
        self,
        variables: Union[str, list],
        filename: str,
        overwrite: bool = False,
        save: bool = True,
        chunks: Optional[dict] = None,
        interpolate: bool = True,
    ) -> xr.Dataset:
        """Combine .nc files of different variables to one file.

        Args:
            variables:
                Modelled variables which should be processed.
                E.g. ["u", "v", "w", "ch4", "qt", "th"]
            filename:
                Name of the file without path or file extension.
            overwrite:
                Indicate if an existing file should be overwritten.
            save:
                Indicate if the created file should be saved.
            chunks:
                Chunks used when processing the netCDF files with xarray.
            interpolate:
                Whether to interpolate the variables to the grid midpoint.

        Returns:
            Dataset with combined variables.
        """
        file_path = Path(self.case.directory, f"{filename}.nc")

        # Check if the output file already exists
        if file_path.exists() and not overwrite:
            # If the file exists, open and return it
            return io.open_microhh_nc(case=self.case, variable=filename, chunks=chunks)

        # combine the prognostic variables into one dataset
        combined_dataset = self.combine_prognostic_variables(
            variables=variables, chunks=chunks, interpolate=interpolate
        )

        # Save the combined dataset to a new NetCDF file
        if save:
            logger.info("Saving %s", file_path)
            combined_dataset.to_netcdf(file_path)

        return combined_dataset

    def combine_prognostic_variables(
        self,
        variables: Union[str, list],
        data: Optional[xr.Dataset] = None,
        chunks: Optional[Union[str, dict]] = None,
        interpolate: bool = True,
    ) -> xr.Dataset:
        """Combine multiple prognostic variables of the MicroHH output.

        Combine the specified variables into one dataset.
        Specify either the variables to combine or give a dataset to
        which the new variables should be added.

        Args:
            variables:
                Variable(s) to add together or to assign to the given dataset.
            data:
                If given, the variable(s) are added to this dataset
            chunks:
                Chunks used when processing the netCDF files with xarray.
            interpolate:
                Whether to interpolate the variables to the grid midpoint.

        Returns:
            Dataset containing the combined variables.
        """
        variables_list = np.atleast_1d(variables).tolist()
        datasets = [] if data is None else [data]

        for var in variables_list:
            dataset = self.open_3d_prognostic_variable(var, chunks=chunks)

            if interpolate:
                # homogenise dimensions
                dataset = self.interpolate_to_grid_midpoint(dataset)
                # dataset = dataset.persist()
            datasets.append(dataset)

        # Merge all datasets into one
        logger.info("Combining datasets...")
        combined_dataset = xr.merge(
            datasets, combine_attrs="no_conflicts", join="outer"
        )

        return combined_dataset

    def interpolate_to_grid_midpoint(self, data: xr.Dataset) -> xr.Dataset:
        """Interpolate variables at grid edges 'h' to the middle of the grid.

        Args:
            data: Dataset with the variables to interpolate.

        Returns:
            Dataset with the interpolated variables.
        """
        dim = self._get_dims_at_grid_boundaries(data)

        if dim is None:
            return data

        new_dim_name = dim[:-1]
        data = data.rename({dim: new_dim_name})

        logger.info("Interpolating %s to %s", dim, new_dim_name)

        if dim == "zh":
            # Use initial conditions since vertical grid can be stretched
            midpoints = io.open_initial_cond(self.case).z
        else:
            spacing = data[new_dim_name].diff(new_dim_name)
            half = spacing.median(skipna=True) / 2
            midpoints = data[new_dim_name] + half

        data_interpolated = data.interp(
            {new_dim_name: midpoints}, kwargs={"fill_value": "extrapolate"}
        )

        return data_interpolated

    def _get_dims_at_grid_boundaries(self, data: xr.Dataset) -> Optional[str]:
        """Get the name of the dimension at the grid boundaries.

        Args:
            data:
                Dataset which contains a dimension which is sampled at the boundaries
                of the grid.

        Returns:
            str: Name of dimension which is sampled at the grid boundaries.
        """
        # iterate over the dimensions of the Dataset
        for dim in list(data.dims):
            # check if the dimension name ends with 'h'
            dim_name = str(dim)
            if dim_name.endswith("h"):
                return dim_name

        return None

    def create_sources_dataset(
        self,
        lat: Union[list, np.ndarray],
        lon: Union[list, np.ndarray],
        source_names: Optional[Union[list, np.ndarray]] = None,
        source_labels: Optional[Union[list, np.ndarray]] = None,
    ) -> xr.Dataset:
        """Create a sources dataset used for the plume detection and in 'ddeq'.

        Args:
            lat:
                List of latitudes for the locations of a source.
            lon:
                List of longitudes for the locations of a source.
            source_names:
                List of names of the sources. Should not contain any special
                characters. If not specified, the sources will be numbered
                starting with source1 etc.
            source_labels:
                List of names of the sources used for plotting.
                If not specified, the sources will be numbered
                starting with source1 etc.

        Returns:
            Dataset with source information.
        """
        source_list = np.atleast_1d(self.case.config["source"]["sourcelist"])
        nr_sources = len(source_list)

        source_names = (
            source_names
            or source_labels
            or [f"source{i + 1}" for i in range(nr_sources)]
        )
        source_labels = source_labels or source_names

        sigma_x = np.array(self.case.config["source"]["sigma_x"])
        sigma_y = np.array(self.case.config["source"]["sigma_y"])
        source_diameter = np.maximum(sigma_x, sigma_y) * 4  # 4 sigma rule

        data = {
            "lon": (["source"], np.atleast_1d(np.array(lon, dtype=float))),
            "lat": (["source"], np.atleast_1d(np.array(lat, dtype=float))),
            "source_x0": (
                ["source"],
                np.atleast_1d(
                    np.array(self.case.config["source"]["source_x0"], dtype=float)
                ),
            ),
            "source_y0": (
                ["source"],
                np.atleast_1d(
                    np.array(self.case.config["source"]["source_y0"], dtype=float)
                ),
            ),
            "source_z0": (
                ["source"],
                np.atleast_1d(
                    np.array(self.case.config["source"]["source_z0"], dtype=float)
                ),
            ),
            "label": (
                ["source"],
                np.atleast_1d(np.array(source_labels, dtype=object)),
            ),
            "diameter": (
                ["source"],
                np.atleast_1d(np.array(source_diameter, dtype=float)),
            ),
            "strength": (
                ["source"],
                np.atleast_1d(
                    np.array(
                        src.get_source_strength(self.case, to_unit="kg/h"), dtype=float
                    )
                ),
            ),
        }

        coords = {"source": np.array(source_names, dtype=object)}

        sources = xr.Dataset(data, coords=coords)
        sources.lon.attrs["units"] = "longitude of source"
        sources.lat.attrs["units"] = "latitude of source"
        sources.diameter.attrs["units"] = "m"
        sources.strength.attrs["units"] = "kg h-1"

        for dim in ["x", "y", "z"]:
            sources[f"source_{dim}0"].attrs["units"] = "m"
            sources[f"source_{dim}0"].attrs["description"] = (
                f"{dim} position in MicroHH grid"
            )

        return sources

    def _broadcast_coords_to_2d_grid(
        self, data: xr.Dataset, coord_names: tuple[str, str]
    ) -> xr.Dataset:
        """Broadcast 1D x and y coordinates to a 2D grid.

        Args:
            data:
                Dataset with 1D x and y coordinates.
            coord_names:
                Names of the x and y coordinates in the dataset.

        Returns:
            Dataset with 2D x and y coordinates.
        """
        # Create 2D grids for x and y
        x_2d, y_2d = np.meshgrid(data[coord_names[0]], data[coord_names[1]])

        # Assign the new 2D coordinates
        data = data.assign_coords(x=(coord_names, x_2d), y=(coord_names, y_2d))

        return data

    def convert_to_concentration(
        self,
        data: xr.Dataset,
        gas: str = "ch4",
        to_unit: str = "kg m-3",
        chunks: Optional[Union[str, dict]] = None,
    ) -> xr.Dataset:
        """Convert the gas given in data to concentrations.

        Convert mixing ratio to concentraion using modelled pressure
        and potential temperature.

        Args:
            data:
                Dataset containing the variable which should be converted to
                concentrations.
            gas:
                Gas for which the columns should be computed.
            to_unit:
                Concentration units to which the gas should be converted.
            chunks:
                Chunks used when processing the netCDF files with xarray.
        """
        required_prognostic_vars = [gas, "th"]

        # check if all required prognostic variables are in the dataset
        for var in required_prognostic_vars:
            if var not in data.data_vars:
                logger.info("Adding '%s' to the dataset.", var)
                data = self.combine_prognostic_variables(
                    variables=var, data=data, chunks=chunks
                )

        # check if pressure is in the dataset
        if "phydro" not in data.data_vars:
            logger.info("Adding 'phydro' to the dataset.")
            data = self.add_statistics_variable(data, "phydro")

        # convert potential temperature to absolute temperature
        if "ta" not in data.data_vars:
            logger.info("Calculating absolute temperature.")
            data = self.add_absolute_temperature(data)

        from_unit = data[gas].attrs.get("units", None)
        if from_unit is None:
            raise ValueError(f"No units defined for variable '{gas}'.")

        logger.info(
            "Converting %s mixing ratios [%s] to concentrations [%s]. "
            "This might take a while.",
            gas,
            from_unit,
            to_unit,
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
        """Add a variable from the statistics file to the given dataset.

        Args:
            data:
                Dataset to which the variable should be added.
            variable:
                Name of the variable in the statistics file which should be added
                to the dataset.

        Returns:
            Dataset with the added variable.
        """
        # read in statistics file
        stats = io.open_statistics(case=self.case)

        # check if variable is in statistics file
        if variable not in stats.data_vars:
            raise KeyError(f"No variable named '{variable}'.")

        # check if target file has the necessary dimensions:
        for dim in stats.sizes:
            if dim not in data.dims or dim not in data.coords:
                data = data.assign_coords({dim: stats[dim]})

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

        return data

    def add_absolute_temperature(self, data: xr.Dataset) -> xr.Dataset:
        """Add absolute temperature derived from potential temperature and pressure.

        Args:
            data: Dataset containing potential temperature and pressure fields,
                or enough metadata to add pressure fields from statistics.

        Returns:
            Dataset with absolute temperature added as ``ta``.
        """
        required_vars = ["phydro", "phydroh"]

        for var in required_vars:
            if var not in data.data_vars:
                logger.info("Adding '%s' to the dataset.", var)
                data = self.add_statistics_variable(data, variable=var)

        # Calculate absolute temperature
        data["ta"] = PostProcessing.calculate_absolute_temperature(
            theta=data.th, p=data.phydro, p0=data.phydroh.sel(zh=0)
        )

        # Add attributes to 'ta' variable
        data["ta"].attrs["long_name"] = "Absolute temperature"
        data["ta"].attrs["units"] = "K"

        return data

    @staticmethod
    def calculate_absolute_temperature(
        theta: Union[xr.DataArray, np.ndarray, int, float],
        p: Union[xr.DataArray, np.ndarray, int, float],
        p0: Union[xr.DataArray, np.ndarray, int, float],
    ) -> Union[xr.DataArray, np.ndarray, int, float]:
        """Calculate absolute temperature from potential temperature and pressure.

        Args:
            theta: Potential temperature.
            p: Pressure at the target level.
            p0: Reference pressure.

        Returns:
            Absolute temperature using Poisson's equation.
        """
        # Constants
        r = scipy.constants.R  # Specific gas constant for dry air in J/(kg·K)
        cp = 1004  # Specific heat capacity at constant pressure for dry air in J/(kg·K)

        # Calculate absolute temperature
        absolute_temperature = theta * (p / p0) ** (r / cp)

        return absolute_temperature

    def calculate_3d_tke(
        self, data: xr.Dataset, chunks: Optional[Union[str, dict]] = None
    ) -> xr.Dataset:
        """Calculate the TKE based on the 3D prognostic variables of MicroHH.

        Args:
            data:
                Dataset containing the required variables. If it does not contain
                all variables, they are added.
            chunks:
                Chunks used when processing the netCDF files with xarray.

        Returns:
            Dataset containing TKE as a variable.
        """
        wind_components = ["u", "v", "w"]

        for var in wind_components:
            if var not in data.data_vars:
                data = self.combine_prognostic_variables(
                    variables=var, data=data, chunks=chunks
                )

        # compute temporal deviation for each wind component
        for wind_component in wind_components:
            data[f"{wind_component}_std"] = data[wind_component] - data[
                wind_component
            ].mean(dim="time")
            data[f"{wind_component}_std"].attrs["units"] = data[wind_component].attrs[
                "units"
            ]
            data[f"{wind_component}_std"].attrs["description"] = (
                f"deviation of {wind_component} wind speed from mean"
            )

        # compute TKE
        data["TKE"] = 0.5 * (
            data["u_std"] ** 2 + data["v_std"] ** 2 + data["w_std"] ** 2
        ).mean(dim="time")
        data["TKE"].attrs["units"] = "m2 s-2"
        data["TKE"].attrs["long_name"] = "turbulent kinetic energy"

        return data

    def calculate_mass_budget(
        self, read_method: Literal["binary", "netcdf"] = "binary"
    ) -> dict:
        """Calculate the mass budget of the modelled scalars.

        Args:
            read_method:
                Method to read the modelled scalar fields. Choose 'binary' to read
                the binary output files or 'netcdf' to read the netCDF output files.
        """
        stats = io.open_statistics(case=self.case)
        rhoref = stats.rhoref.values
        zh = stats.zh.values
        dz = zh[1:] - zh[:-1]

        itot = self.case.config["grid"]["itot"]
        jtot = self.case.config["grid"]["jtot"]
        ktot = self.case.config["grid"]["ktot"]

        xsize = self.case.config["grid"]["xsize"]
        ysize = self.case.config["grid"]["ysize"]

        dx = xsize / itot
        dy = ysize / jtot

        scalars = np.atleast_1d(self.case.config["source"]["sourcelist"])
        strength = np.atleast_1d(self.case.config["source"]["strength"])

        time_range = np.arange(
            0,
            int(self.case.config["time"]["endtime"] + 1),
            int(self.case.config["time"]["savetime"]),
        )

        def read_mass_from_binary(scalar: str, time_range: np.ndarray) -> list[float]:
            """Read scalar mass from MicroHH binary files for each output time."""
            mass_array = []
            dtype = get_dtype()
            for time in time_range:
                fld = np.fromfile(f"{self.case.directory}/{scalar}.{time:07d}", dtype)
                fld = fld.reshape((ktot, jtot, itot))
                mass = np.sum(rhoref[:, None, None] * fld * dx * dy * dz[:, None, None])
                mass_array.append(float(mass))
            return mass_array

        def read_mass_from_netcdf(scalar: str, time_range: np.ndarray) -> list[float]:
            """Read scalar mass from NetCDF output for each output time."""
            ds = io.open_microhh_nc(self.case, variable=scalar)
            # Assume variable name matches scalar, dimensions (time, z, y, x)
            arr = ds[scalar].values
            mass_array = []
            for i, _ in enumerate(time_range):
                fld = arr[i]
                mass = np.sum(rhoref[:, None, None] * fld * dx * dy * dz[:, None, None])
                mass_array.append(float(mass))
            ds.close()
            return mass_array

        def get_dtype() -> Union[type[np.float64], type[np.float32]]:
            """Return the floating-point dtype used by MicroHH binary output."""
            grid = mht.Read_grid(itot, jtot, ktot)

            if grid.prec == "d":
                return np.float64
            else:
                return np.float32

        mass_dict: dict[str, object] = {"time": time_range}
        for i, scalar in enumerate(scalars):
            scalar = str(scalar)
            if read_method == "binary":
                simulated_scalar = read_mass_from_binary(scalar, time_range)
            elif read_method == "netcdf":
                simulated_scalar = read_mass_from_netcdf(scalar, time_range)
            else:
                raise ValueError("Unknown read_method: choose 'binary' or 'netcdf'")

            mass_dict[f"prescribed_{scalar}_mass"] = np.asarray(
                strength[i] * time_range
            )
            mass_dict[f"simulated_{scalar}_mass"] = simulated_scalar

        return mass_dict


def merge_restart_statistics(
    stat_file1: Union[str, Path],
    stat_file2: Union[str, Path],
    combined_stat_file: Union[str, Path],
    split_time: int,
) -> None:
    """Merge two MicroHH statistics files from restarted cases into one file.

    Args:
        stat_file1:
            Path to the first statistics file (before restart).
        stat_file2:
            Path to the second statistics file (after restart).
        combined_stat_file:
            Path to the output file where the combined statistics will be saved.
        split_time:
            Time in seconds at which the two files should be merged. Data from
            stat_file1 will be used for time < split_time and data from
            stat_file2 will be used for time >= split_time.
    """
    if Path(combined_stat_file).exists():
        logger.info("%s already exists. Skipping...", combined_stat_file)
        return

    def clean_time(ds: xr.Dataset, times: np.ndarray) -> xr.Dataset:
        """Assign integer-valued time coordinates without float noise."""
        times_clean = np.rint(times).astype(np.int64)

        # Drop any existing time coordinate/variable, then assign the clean one.
        if "time" in ds.coords:
            ds = ds.drop_vars("time")

        return ds.assign_coords(time=("time", times_clean))

    with xr.open_dataset(stat_file1, decode_times=False) as ds1_root:
        times1 = np.rint(ds1_root.time.values).astype(np.int64)

    with xr.open_dataset(stat_file2, decode_times=False) as ds2_root:
        times2 = np.rint(ds2_root.time.values).astype(np.int64)

    with netCDF4.Dataset(stat_file1, "r") as nc:
        group_names = list(nc.groups.keys())

    # Write root group first.
    all_groups = [None, *group_names]

    for i, group in enumerate(all_groups):
        ds1 = xr.open_dataset(stat_file1, group=group, decode_times=False)
        ds2 = xr.open_dataset(stat_file2, group=group, decode_times=False)

        ds1 = clean_time(ds1, times1)
        ds2 = clean_time(ds2, times2)

        ds1_sliced = ds1.where(ds1.time < split_time, drop=True)
        ds2_sliced = ds2.where(ds2.time >= split_time, drop=True)

        ds_out = xr.concat(
            [ds1_sliced, ds2_sliced],
            dim="time",
            data_vars="all",
        )

        # Force the concatenated output coordinate to stay clean.
        ds_out = ds_out.assign_coords(
            time=("time", np.rint(ds_out.time.values).astype(np.int64))
        )

        ds_out = ds_out.transpose(*ds1.dims)

        write_mode = "w" if i == 0 else "a"

        ds_out.to_netcdf(
            combined_stat_file,
            group=group,
            mode=write_mode,
            engine="netcdf4",
        )

        ds1.close()
        ds2.close()

    logger.info("Finished merging into %s", combined_stat_file)


def add_gaussian_noise(
    data: xr.Dataset, var_name: str, gas: str, std: float, noise_units: str
) -> xr.Dataset:
    """Add Gaussian noise to a specified variable in the dataset.

    Args:
        data:
            The dataset containing the variable to which noise will be added.
        var_name:
            The name of the variable to which Gaussian noise will be added.
        gas:
            The gas for which noise is being added.
        std:
            The standard deviation of the Gaussian noise distribution.
        noise_units:
            Unit of the noise added.

    Returns:
        Dataset with the specified variable modified to include Gaussian noise.
    """
    # Create a deep copy of the dataset to avoid modifying the original in-place
    data_copy = data.copy(deep=True)

    # convert noise to same unit as data
    data_units = data_copy[var_name].attrs.get("units", None)

    if data_units is None:
        raise ValueError(
            f"Units for variable '{var_name}' are not defined in the dataset."
        )

    std_conv = ucat.convert_columns(
        std, from_unit=noise_units, to_unit=data_units, molar_mass=gas
    )

    # assume no bias (center = 0) but noise (= std_conv)
    noise = np.random.normal(0, std_conv, data_copy[var_name].size)
    noise = noise.reshape(data_copy[var_name].shape)

    data_copy[var_name] = data_copy[var_name] + noise
    data_copy[var_name].attrs.update(data_copy[var_name].attrs)
    data_copy[var_name].attrs["noise_level"] = std_conv

    # add std_conv for variable based on artificial noise
    data_copy[f"{var_name}_precision"] = (
        data_copy[var_name].dims,
        np.full_like(data_copy[var_name], std_conv),
    )
    data_copy[f"{var_name}_precision"].attrs["units"] = data_units

    return data_copy


def degrade_resolution(
    data: xr.Dataset, zonal_res: int, meridional_res: int
) -> xr.Dataset:
    """Degrade resolution of 3D or 2D MicroHH output by aggregating pixels.

    Args:
        data: 3D or 2D data
        zonal_res:
            Number of pixels in zonal direction which should be aggregated.
        meridional_res:
            Number of pixels in meridional direction which should be aggregated.

    Returns:
        Data with coarser resolution.
    """
    spatial_dims = {
        ("x", "y"): ("x", "y"),
        ("lon", "lat"): ("lon", "lat"),
        ("nrows", "nobs"): ("nrows", "nobs"),
        ("longitude", "latitude"): ("longitude", "latitude"),
    }

    for dims, (zonal_dim, meridional_dim) in spatial_dims.items():
        if all(dim in data.dims for dim in dims):
            degraded_data = (
                data.coarsen(
                    {zonal_dim: zonal_res, meridional_dim: meridional_res},
                    boundary="trim",
                )
            ).mean()  # pyright: ignore[reportAttributeAccessIssue]
            return degraded_data

    raise ValueError(f"Unknown spatial dimensions: {data.dims}")


def format_data_for_cdo(
    data_coord: xr.Dataset, out_path: Union[str, Path], save: bool = True
) -> xr.Dataset:
    """Format the data for use with the CDO library.

    Args:
        data_coord: Dataset with coordinates to format for CDO
            compatibility.
        out_path: Path where the formatted dataset should be saved.
        save: Whether to save the formatted dataset to out_path.

    Returns:
        Formatted dataset.
    """
    # Ensure out_path is a Path object
    out_path = Path(out_path)

    ds = data_coord.copy(deep=True)

    # Clean up dimensions and old vars
    ds = ds.drop_vars(["x", "y"], errors="ignore")
    if "vertices" not in ds.dims and "corners" in ds.dims:
        ds = ds.rename({"corners": "vertices"})

    data_vars = [v for v in ds.data_vars if v not in ["lonc", "latc", "time"]]

    # Force these variables to associate with lat/lon
    for var in data_vars:
        ds[var].encoding = {}
        ds[var].attrs["coordinates"] = "lat lon"

    if "time" in ds.coords:
        ds["time"].encoding["units"] = "seconds since 1970-01-01 00:00:00"
        ds["time"].encoding["calendar"] = "proleptic_gregorian"

    # Define metadata
    targets = {
        "lon": {
            "units": "degrees_east",
            "standard_name": "longitude",
            "bounds": "lonc",
        },
        "lat": {
            "units": "degrees_north",
            "standard_name": "latitude",
            "bounds": "latc",
        },
        "lonc": {"units": "degrees_east"},
        "latc": {"units": "degrees_north"},
    }

    # reset attributes for the main variables and bounds, to ensure CDO compatibility
    for var_name, new_attrs in targets.items():
        if var_name in ds.variables:
            ds[var_name].encoding = {}

            # Wipe and replace attributes
            ds[var_name].attrs = new_attrs

    # Remove automatically added 'coordinates' attributes that caused CDO warnings
    for bnd in ["lonc", "latc"]:
        if bnd in ds.variables:
            ds[bnd].encoding["coordinates"] = None
            ds[bnd].attrs.pop("coordinates", None)

    if save:
        ds.to_netcdf(out_path)

    return ds


def create_2d_3d_output_script(
    cases: list[init.Case],
    alps_setup: cscs.AlpsSetup,
    scratch_path: Union[str, Path],
    py_file_path: Union[str, Path],
    python_path: Union[str, Path],
    vars_3d: Union[str, list[str]],
    vars_2d: Union[str, list[str]],
    sample_time: Optional[int] = None,
    job_name: str = "create_2d_3d_output",
    overwrite: bool = False,
) -> cscs.SlurmJob:
    """Create a SLURM script to create 2D and 3D output for cases in parallel.

    The generated script launches one unbuffered Python process per case in the
    background and waits for all cases to finish. Running Python with ``-u``
    ensures that logging and print output from the 2D/3D conversion functions is
    written to the SLURM ``.out`` file while the job is still running.

    Args:
        cases:
            Cases to post-process. All cases are launched in parallel.
        alps_setup:
            Alps setup used to create the SLURM script header.
        scratch_path:
            Directory where the SLURM script and output file are written.
        py_file_path:
            Path to the Python script that creates the 2D and 3D output.
        python_path:
            Python executable to use.
        vars_3d:
            Variables to create 3D output for.
        vars_2d:
            Variables to create 2D output for.
        sample_time:
            Sampling interval passed to the processing script.
        job_name:
            Name of the SLURM job and script.
        overwrite:
            Overwrite an existing script.

    Returns:
        Path to the created SLURM script.
    """
    if not cases:
        raise ValueError("At least one case is required.")

    # check if cpus_per_task in alps_setup has been set and is sufficient for
    # the number of cases and variables
    ncases = len(utils.object_to_list(cases))
    nvars = len(utils.object_to_list(vars_3d))
    ncpus_per_task = alps_setup.cpus_per_task

    if ncpus_per_task is not None and ncpus_per_task < ncases * nvars:
        logger.warning(
            "cpus_per_task in alps_setup is set to %d, but %d CPUs are required to "
            "process all variables in parallel. cpus_per_task will be set to %d.",
            ncpus_per_task,
            ncases * nvars,
            ncases * nvars,
        )
        alps_setup.cpus_per_task = ncases * nvars
    if ncpus_per_task is None:
        alps_setup.cpus_per_task = ncases * nvars
        logger.info(
            "cpus_per_task in alps_setup is not set. Setting it to %d.",
            ncases * nvars,
        )

    job_name_formatted = job_name.replace(" ", "_").replace("-", "_")
    job_file = Path(scratch_path, f"{job_name_formatted}.sbatch")

    if job_file.exists() and not overwrite:
        logger.info("Reading existing bash script %s", job_file)
        return cscs.SlurmJob(job_file, ssh_connect=alps_setup.ssh_connect)

    commands = _get_2d_3d_output_commands(
        cases=cases,
        py_file_path=py_file_path,
        python_path=python_path,
        vars_3d=[str(var) for var in utils.object_to_list(vars_3d)],
        vars_2d=[str(var) for var in utils.object_to_list(vars_2d)],
        sample_time=sample_time,
    )

    job = alps_setup.create_sbatch_script(
        commands=commands,
        job_name=job_name_formatted,
        file_path=job_file,
        scratch_path=scratch_path,
    )

    return job


def _get_2d_3d_output_commands(
    cases: list[init.Case],
    py_file_path: Union[str, Path],
    python_path: Union[str, Path],
    vars_3d: list[str],
    vars_2d: list[str],
    sample_time: Optional[int] = None,
) -> list[str]:
    commands = [
        "# Define post-processing script and variables",
        f"SCRIPT={shlex.quote(str(py_file_path))}",
        f"PY_PATH={shlex.quote(str(python_path))}",
        "VARS_3D=(",
    ]

    for var in vars_3d:
        commands.append(f"    {shlex.quote(var)}")

    commands.extend(
        [
            ")",
            "VARS_2D=(",
        ]
    )

    for var in vars_2d:
        commands.append(f"    {shlex.quote(var)}")

    commands.extend(
        [
            ")",
            "",
            "CASE_DIRS=(",
        ]
    )

    for case in cases:
        commands.append(f"    {shlex.quote(str(case.directory))}")

    commands.extend(
        [
            ")",
            "",
            f"# Launch all {len(cases)} cases in the background",
            'for CASE_DIR in "${CASE_DIRS[@]}"; do',
            '    "$PY_PATH" -u "$SCRIPT" \\',
            '        --directory "$CASE_DIR" \\',
            '        --vars_3d "${VARS_3D[@]}" \\',
            '        --vars_2d "${VARS_2D[@]}" \\',
        ]
    )

    if sample_time is not None:
        commands.append(f"        --sample_time {sample_time} &")
    else:
        commands[-1] = '        --vars_2d "${VARS_2D[@]}" &'

    commands.extend(
        [
            "done",
            "",
            "# Wait for all background jobs to finish",
            "wait",
            'echo "All cases completed."',
        ]
    )

    return commands


__all__ = [
    "PostProcessing",
    "add_gaussian_noise",
    "create_2d_3d_output_script",
    "degrade_resolution",
    "format_data_for_cdo",
    "merge_restart_statistics",
]
