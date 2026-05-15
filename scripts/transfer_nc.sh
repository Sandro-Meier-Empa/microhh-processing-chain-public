#!/bin/bash
#SBATCH --job-name=rsync_microhh
#SBATCH --account=em05
#SBATCH --output=rsync_microhh_%j.out
#SBATCH --time=05:00:00
#SBATCH --partition=xfer
#SBATCH --constraint=mc

set -euo pipefail

ROOT_DIR="/capstor/scratch/cscs/smeier/microhh_tests"
DEST_ROOT="/capstor/store/cscs/userlab/lp171/smeier/microhh_output"

CASES=(
    "microhh_tests.test_ls2d_1d_coarse_met_wind_512x512"
    "microhh_tests.test_ls2d_1d_coarse_met_wind_1024x1024_combined"
)

for CASE in "${CASES[@]}"; do
    mkdir -p "${DEST_ROOT}/${CASE}"

    rsync -avh --progress \
        "${ROOT_DIR}/${CASE}/"{u.nc,v.nc,w.nc} \
        "${DEST_ROOT}/${CASE}/"
done
