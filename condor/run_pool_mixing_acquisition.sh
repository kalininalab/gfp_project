#!/bin/bash
set -euo pipefail
cd /nethome/akolchina/gfp_project
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
exec /nethome/akolchina/miniconda3/envs/combi/bin/python -m scripts.pool_mixing_acquisition.run_round \
  --target "$1" --plate-size "$2" --round "$3" --seed "$4" --model "$5" \
  --device "$6" --reference-raw "$7" --raw "$8"
