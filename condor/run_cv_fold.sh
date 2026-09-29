#!/usr/bin/env bash
set -euo pipefail
cd /nethome/akolchina/gfp_project
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/gfp-matplotlib"
exec /nethome/akolchina/miniconda3/envs/combi/bin/python scripts/run_cv_fold.py \
  --config results/cv10_cgreGFP_seed42/config.json --fold "$1"
