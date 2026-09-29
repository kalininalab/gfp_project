#!/bin/bash
set -euo pipefail
cd /nethome/akolchina/gfp_project
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
fold="$1"
if [[ "$fold" == holdout ]]; then
    split=holdout
else
    split="results/cv10_cgreGFP_seed42/splits/fold_${fold}.csv"
fi
exec /nethome/akolchina/miniconda3/envs/combi/bin/python -m scripts.train_jannis_ohe --split "$split" --split-name "$fold" --device cuda
