#!/bin/bash
set -euo pipefail
cd /nethome/akolchina/gfp_project
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export MPLCONFIGDIR="${TMPDIR:-/tmp}/gfp-matplotlib"
model="$1"
fold="$2"
device="$3"
if [[ "$fold" == holdout ]]; then
    split=holdout
else
    split="results/cv10_cgreGFP_seed42/splits/fold_${fold}.csv"
fi
exec /nethome/akolchina/miniconda3/envs/combi/bin/python -m scripts.esm_benchmark.run --model "$model" --split "$split" --split-name "$fold" --device "$device"
