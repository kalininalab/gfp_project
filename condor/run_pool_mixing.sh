#!/bin/bash
set -euo pipefail
cd /nethome/akolchina/gfp_project
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
if [[ $# -ge 6 ]]; then
  exec /nethome/akolchina/miniconda3/envs/combi/bin/python -m scripts.pool_mixing.run \
    --target "$1" --n-target "$2" --seed "$3" --model "$4" --device "$5" --raw "$6"
else
  exec /nethome/akolchina/miniconda3/envs/combi/bin/python -m scripts.pool_mixing.run \
    --target "$1" --n-target "$2" --seed "$3" --model "$4" --device "$5"
fi
