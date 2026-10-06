#!/bin/bash
set -euo pipefail
cd /nethome/akolchina/gfp_project
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
exec /nethome/akolchina/miniconda3/envs/combi/bin/python -m scripts.paper_reproduction.run \
  --model "$1" --device "$2" --seed "${3:-42}"
