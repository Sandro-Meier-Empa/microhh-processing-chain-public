import shutil
import subprocess
from pathlib import Path
import os
import sys
import importlib




# TODO
# - Modify .ini file
# 



def create_batch_script(case_name, job_name,
                       run_time, nodes=1, ntasks_per_core=1,
                       ntasks_per_node=36, cpus_per_task=1, partition="debug",
                       path="/scratch/snx3000/smeier", start_jobs=False):
    
    job_file = os.path.join(path, "bash_script.sbash")

    if os.path.exists(path):

        with open(job_file, "w") as bash_script:
            header = "#!/bin/bash -l \n\n"
            
            header += f"#SBATCH --job-name={job_name} \n"
            header += "#SBATCH --account=em05\n"
            header += "#SBATCH --mail-type=ALL\n"
            header += "#SBATCH --mail-user=sandro.meier@empa.ch\n"
            header += f"#SBATCH --time={run_time}\n"
            header += f"#SBATCH --nodes={nodes}\n"
            header += f"#SBATCH --ntasks-per-core={ntasks_per_core}\n"
            header += f"#SBATCH --ntasks-per-node={ntasks_per_node}\n"
            header += f"#SBATCH --cpus-per-task={cpus_per_task}\n"
            header += f"#SBATCH --partition={partition} \n"
            header += "#SBATCH --constraint=mc\n"
            header += "#SBATCH --hint=nomultithread\n"
            header += f"#SBATCH --chdir={str(path)}\n\n"

            header += "export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK\n\n"

            header += f"srun {path}/microhh init {case_name}\n"
            header += f"srun {path}/microhh run {case_name}"

            bash_script.write(header)

    else:
        raise FileNotFoundError(f"Directory {path} does not exist yet. Copy files to scratch first")

        
    if start_jobs:
        # Launch the job on daint
        run_sim(job_file, False)


def run_sim(job_file: Path, wait: bool = True):
    """Run a simulation using sbatch"""

    print(f"Starting {job_file=}")

    os.system(f"sbatch {'--wait' if wait else ''} {job_file}")


def run_chain(case_name, job_name, mpi=True,
              suffix=None, run_time="00:10:00",
              nodes=1, ntasks_per_core=1,
              ntasks_per_node=36, cpus_per_task=1, partition="debug",
              path="/scratch/snx3000/smeier", start_jobs=False, overwrite=True):
    
    sys.path.append(f'/users/smeier/microhh/cases/{case_name}')
    module = importlib.import_module(f"{case_name}_input")

    version = "mpi" if mpi else "nompi"
    
    # Create nc file for initial conditions
    module.create_input_nc(path=f"/users/smeier/microhh/cases/{case_name}")

    # Copy microhh executable
    copy_command = f"cp /users/smeier/microhh/build.{version}/microhh /users/smeier/microhh/cases/{case_name}"
    subprocess.run(copy_command, shell=True)

    # Copy the folder to scratch directory
    new_name = f"{path}/{case_name}.{suffix}" if suffix else f"{path}/{case_name}"

    if os.path.exists(new_name) and overwrite:
        shutil.rmtree(new_name)
        print(f"Overwriting {new_name}...")

    copy_to_scratch_command = f"cp -r /users/smeier/microhh/cases/{case_name}/ {new_name}"
    subprocess.run(copy_to_scratch_command, shell=True)

    # Clean up the directories
    os.remove(f"/users/smeier/microhh/cases/{case_name}/{case_name}_input.nc")
    os.remove(f"/users/smeier/microhh/cases/{case_name}/microhh")

    # Create bash script
    create_batch_script(case_name, job_name,
                        run_time, nodes, ntasks_per_core,
                        ntasks_per_node, cpus_per_task, partition,
                        new_name, start_jobs)