import glob
import importlib
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

sys.path.append(os.path.join('/users', os.environ["USER"], 'microhh', 'python'))
sys.path.append(os.path.join('/users', os.environ["USER"], 'microhh', 'processing_chain', 'src'))
import microhh_tools as mht
import initialisation as init

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
        """Check if the setup specified in the .ini file is valid
        based on the selected number of nodes and the model segreation
        and number of grid cells.

        Raises:
            ValueError: Raise error if the specified grid and model segreation
                do not match in x direction.
            ValueError: Raise error if the specified grid and model segreation
                do not match in y direction.
            ValueError: Raise error if the segregation and
                number of nodes do not match.
        """
        if not self.mpi and (self.ini_file['master']['npx'] != 1 or self.ini_file['master']['npy'] != 1):
            for dim in ["x", "y"]:
                self.ini_file.set_value(group="master", variable=f"np{dim}", value=1)
            self.ini_file.save(self.ini_path, allow_overwrite=True)
            print("Overwriting 'npx' and 'npy' because MPI is disabled.")

        if self.ini_file['grid']['itot'] % self.ini_file['master']['npx']:
            raise ValueError(
                f"itot = {self.ini_file['grid']['itot']} is not a multiple "
                f"of npx = {self.ini_file['master']['npx']}"
                )
        elif self.ini_file['grid']['jtot'] % self.ini_file['master']['npy']:
            raise ValueError(
                f"itot = {self.ini_file['grid']['jtot']} is not a multiple "
                f"of npy = {self.ini_file['master']['npy']}"
                )
        elif self.ini_file['master']['npx'] * self.ini_file['master']['npy'] != self.ntasks_per_node:
            raise ValueError(
                f"Task per node {self.ntasks_per_node} does not equal "
                f"npx * npy = {self.ini_file['master']['npx']} * "
                f"{self.ini_file['master']['npy']}"
                )

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

        if not isinstance(vars, list):
            self.vars = [vars]
        else:
            self.vars = vars

    def run_postprocessing(self, overwrite: bool=False,
                           precision: str="double"):
        """Run all post-processing steps.

        Args:
            overwrite (bool, optional. Default: False):
                Specify if existing files on 'scratch' and 
                'store' should be overwritten.
            precision (str, optional. Default "double"):
                Precision used for the created .nc file.
        """

        self.create_3d_output(
            overwrite=overwrite,
            precision=precision
            )
        self.move_to_store(
            overwrite=overwrite
            )


    def create_3d_output(self, overwrite: bool=False,
                         precision: str="double"):
        """Create 3D netCDF files for the variables specified

        Args:
            overwrite (bool, option. Default: False):
                Specify if existing files on 'scratch' and 
                'store' should be overwritten.
            precision (str, optional. Default: "double"):
                Precision used for the created .nc file.
        """
        start_time = self.ini_file['time']['starttime']
        end_time = self.ini_file['time']['endtime']
        sample_time = self.ini_file['time']['savetime']

        for var in self.vars:
            nc_file_path = os.path.join(self.scratch_path, f"{var}.nc")

            if os.path.exists(nc_file_path):
                if not overwrite:
                    raise FileExistsError(
                        f"File {nc_file_path} already exists. "
                        "Set 'overwrite=True' to allow overwriting."
                    )
                os.remove(nc_file_path)
                print(f"Overwriting {nc_file_path}")

        post_command = (
            f"python 3d_to_nc.py -d {self.scratch_path} "
            f"-f {self.case_name}.ini -v {' '.join(self.vars)} "
            f"-p {precision} -t0 {start_time} -t1 {end_time} "
            f"-tstep {sample_time}"
        )
        subprocess.run(
            post_command,
            shell=True,
            cwd=os.path.join('/users', self.user, 'microhh', 'python')
            )
        
        # assign names and units to the resulting netCDF file
        for var in self.vars:
            self._nc_assign_names_and_attributes(os.path.join(self.scratch_path, f"{var}.nc"))
        
    def _nc_assign_names_and_attributes(self, nc_file_path):
        """Assign long_name and unit attributes to variables in a .nc file.
    
        Args:
            nc_file_path (str): Path to the .nc file.
        """
        # Define the dictionary with variable attributes using a dictionary comprehension
        variables = {var: (f"{var} mixing ratio", "-") for var in gas_mass.keys()}
        variables.update({
            'thl': ('liquid water potential temperature', 'K'),
            'qt': ('total specific humidity', 'kg kg-1'),
            'u': ('zonal wind component', 'm s-1'),
            'v': ('meridional wind component', 'm s-1'),
            'w': ('vertical wind component', 'm s-1'),
            'p': ('air pressure', 'Pa'),
        })

        try:
            ds = xr.open_dataset(nc_file_path)
        except ValueError:
            ds = xr.open_dataset(nc_file_path, decode_times=False)

        # Assign attributes to variables
        for var in ds.variables:
            if var in variables:
                long_name, unit = variables[var]
                ds[var].attrs.update({'long_name': long_name, 'unit': unit})

        # remove existing file because overwriting is not allowed
        os.remove(nc_file_path)

        ds.to_netcdf(nc_file_path)


    def interpolate_to_grid_midpoint(self, nc_file: xr.Dataset):
        dim = self._get_dims_at_grid_boundaries(nc_file)

        if dim is None:
            return nc_file

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
    
    def combine_nc_files(self):
        # Create a list to store individual xarray.Dataset objects
        datasets = []

        # Iterate through each variable
        for var in self.vars:
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




    def move_to_store(self, overwrite: bool):
        """Move files used for running MicroHH to store for reproducibility.

        Args:
            overwrite (bool, optional):
                Specify if existing files on 'store' should be overwritten.

        Raises:
            FileExistsError: Raise error if the folder
            already exists and 'overwrite=False'.
        """

        # Handle target path
        if not os.path.exists(self.store_path):
            os.makedirs(self.store_path)
        elif overwrite:
            shutil.rmtree(self.store_path)
            print(f"Overwriting {self.store_path}.")
            os.makedirs(self.store_path)
        else:
            raise FileExistsError(
                f"Case {self.store_path} already exists. Set 'overwrite=True'."
                )

        # Copy files for reproducibility
        file_patterns = ['*'+self.case_name+'.ini', '*'+self.case_name+'_input.nc', 'slurm-*.out']
        for pattern in file_patterns:
            for file in glob.glob(os.path.join(self.scratch_path, pattern)):
                shutil.copy(file, self.store_path)
                print(f"Successfully copied '{file}' to '{self.store_path}'.")

        # Copy 3D .nc files of variables
        for var in self.vars:
            var_file = os.path.join(self.scratch_path, f"{var}.nc")
            if os.path.exists(var_file):
                shutil.copy(var_file, self.store_path)
                print(f"Successfully copied '{var_file}' to '{self.store_path}'.")
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

        try:
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