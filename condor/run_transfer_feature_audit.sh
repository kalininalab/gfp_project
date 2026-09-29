#!/bin/bash
set -euo pipefail
cd /nethome/akolchina/gfp_project
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec /nethome/akolchina/miniconda3/envs/combi/bin/python -m scripts.transfer_benchmark.audit_features
