#!/bin/bash
set -euo pipefail
cd /nethome/akolchina/gfp_project
mixes=(cgre amac pplu cgre_amac cgre_pplu amac_pplu cgre_amac_pplu artificial)
models=(aubin_1_10_1 aubin_linear mlp_small)
index=$1
model=${models[$((index % 3))]}
seed=$((42 + (index / 3) % 5))
mix=${mixes[$((index / 15))]}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR="$(mktemp -d /tmp/gfp-transfer-al.XXXXXX)"
trap 'rm -rf "$MPLCONFIGDIR"' EXIT
exec /nethome/akolchina/miniconda3/envs/combi/bin/python -m scripts.transfer_active_learning.run --mix "$mix" --seed "$seed" --model "$model" --device cpu
