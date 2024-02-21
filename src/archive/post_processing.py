import shutil
import subprocess
import os
import sys

sys.path.append(f'/users/smeier/microhh/python')
import microhh_tools as mht

def postprocessing(case_name, vars,
                   precision="single",
                   suffix=None,
                   overwrite=True,
                   path="/scratch/snx3000/smeier",
                   target_path="/store/empa/em05/smeier/microhh_output"):

    create_3d_output(case_name, vars,
                     precision,
                     suffix,
                     path)
    
    move_to_store(case_name, vars, suffix,
                  overwrite,
                  path, target_path)


def create_3d_output(case_name, vars,
                     precision="single",
                     suffix=None,
                     path="/scratch/snx3000/smeier"):
    
    name = f"{path}/{case_name}.{suffix}" if suffix else f"{path}/{case_name}"
    nl = mht.Read_namelist(f'{name}/{case_name}.ini')

    start_time = nl['time']['starttime']
    end_time = nl['time']['endtime']
    sample_time = nl['time']['savetime']

    post_command = f"python 3d_to_nc.py -d {name} -f {case_name}.ini -v {' '.join(vars)} -p {precision} -t0 {start_time} -t1 {end_time} -tstep {sample_time}"
    subprocess.run(post_command, shell=True, cwd="/users/smeier/microhh/python")




def move_to_store(case_name, vars,
                  suffix=None,
                  overwrite=True,
                  source_path="/scratch/snx3000/smeier",
                  target_path="/store/empa/em05/smeier/microhh_output"):
    
    source = f"{source_path}/{case_name}.{suffix}" if suffix else f"{source_path}/{case_name}"
    target = f"{target_path}/{case_name}.{suffix}" if suffix else f"{target_path}/{case_name}"

    if not os.path.exists(target):
        os.makedirs(target)

    elif os.path.exists(target) and overwrite:
        shutil.rmtree(target)
        print(f"Overwriting {target}...")

    # copy .ini file for reproducibility
    copy_ini_to_store_command = f"cp -r {source}/{case_name}.ini {target}"
    subprocess.run(copy_ini_to_store_command, shell=True)

    # copy initial conditions file for reproducibility
    copy_ic_to_store_command = f"cp -r {source}/{case_name}_input.nc {target}"
    subprocess.run(copy_ic_to_store_command, shell=True)
    
    # copy 3D .nc files of variables
    for var in vars:
        if os.path.exists(f"{source}/{var}.nc"):
            copy_to_store_command = f"cp -r {source}/{var}.nc {target}"
            subprocess.run(copy_to_store_command, shell=True)
            print(f"Successfully copied '{source}/{var}.nc' to '{target_path}'.")
        else:
            print(f"'{source}' does not exist.")

