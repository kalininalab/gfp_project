#!/bin/bash
set -euo pipefail
cd /nethome/akolchina/gfp_project
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/gfp-matplotlib"
exec /nethome/akolchina/miniconda3/envs/combi/bin/python -m scripts.finalize_esm
