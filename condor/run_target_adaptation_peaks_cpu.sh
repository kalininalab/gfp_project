#!/bin/bash
set -euo pipefail
cd /nethome/akolchina/gfp_project
pairs=(cgreGFP_to_artificial artificial_to_cgreGFP)
models=(aubin_1_10_1 aubin_linear mlp_small)
index=$1
model=${models[$((index % 3))]}
seed=$((42 + (index / 3) % 5))
pair=${pairs[$((index / 15))]}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR="$(mktemp -d /tmp/gfp-target-peaks.XXXXXX)"
trap 'rm -rf "$MPLCONFIGDIR"' EXIT
exec /nethome/akolchina/miniconda3/envs/combi/bin/python -m scripts.target_adaptation.run --directory results/target_adaptation_peaks_seed42_46 --pair "$pair" --seed "$seed" --model "$model"
