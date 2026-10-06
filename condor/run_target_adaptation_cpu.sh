#!/bin/bash
set -euo pipefail
cd /nethome/akolchina/gfp_project
pairs=(cgreGFP_to_amacGFP cgreGFP_to_ppluGFP amacGFP_to_cgreGFP amacGFP_to_ppluGFP ppluGFP_to_cgreGFP ppluGFP_to_amacGFP)
models=(aubin_1_10_1 aubin_linear mlp_small)
index=$1
model=${models[$((index % 3))]}
seed=$((42 + (index / 3) % 5))
pair=${pairs[$((index / 15))]}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR="$(mktemp -d /tmp/gfp-target-adapt.XXXXXX)"
trap 'rm -rf "$MPLCONFIGDIR"' EXIT
exec /nethome/akolchina/miniconda3/envs/combi/bin/python -m scripts.target_adaptation.run --pair "$pair" --seed "$seed" --model "$model"
