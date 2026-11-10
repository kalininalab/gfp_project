#!/bin/bash
set -euo pipefail
cd /nethome/akolchina/gfp_project
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
raw=/data/users/akolchina/gfp_project/artificial_peak_holdout_seed42_46_runs
exec /nethome/akolchina/miniconda3/envs/combi/bin/python -m scripts.peak_holdout.run \
  --directory "$raw" --target "$1" --mode "$2" --seed "$3" --model "$4" --device "$5"
