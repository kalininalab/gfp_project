#!/bin/bash
set -euo pipefail
cd /nethome/akolchina/gfp_project
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
exec /nethome/akolchina/miniconda3/envs/combi/bin/python -m scripts.ortholog_target_included.run \
  --directory "$1" --target "$2" --condition "$3" --seed "$4" --model "$5" --device "$6"
