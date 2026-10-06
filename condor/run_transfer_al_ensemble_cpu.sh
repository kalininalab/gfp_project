#!/bin/bash
set -euo pipefail
cd /nethome/akolchina/gfp_project
mixes=(cgre amac pplu cgre_amac cgre_pplu amac_pplu cgre_amac_pplu artificial)
models=(aubin_1_10_1 aubin_linear mlp_small)
index=$1
model=${models[$((index % 3))]}
seed=$((42 + (index / 3) % 5))
mix=${mixes[$((index / 15))]}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR="$(mktemp -d /tmp/gfp-transfer-ensemble.XXXXXX)"
trap 'rm -rf "$MPLCONFIGDIR"' EXIT
exec /nethome/akolchina/miniconda3/envs/combi/bin/python -m scripts.transfer_active_learning.run --directory results/transfer_al_ensemble_cpu_seed42_46 --mix "$mix" --seed "$seed" --model "$model" --method fancy --ensemble-members 5 --device cpu
