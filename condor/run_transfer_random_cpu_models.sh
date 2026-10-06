#!/bin/bash
set -euo pipefail
cd /nethome/akolchina/gfp_project
mixes=(cgre amac pplu cgre_amac cgre_pplu amac_pplu cgre_amac_pplu artificial)
models=(aubin_1_10_1 mlp_small)
directories=(transfer_random_aubin_cgreGFP_seed42_46 transfer_random_mlp_cgreGFP_seed42_46)
index=$1
model_index=$((index % 2))
seed=$((42 + (index / 2) % 5))
mix=${mixes[$((index / 10))]}
model=${models[$model_index]}
directory=${directories[$model_index]}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR="$(mktemp -d /tmp/gfp-random-cpu-models.XXXXXX)"
trap 'rm -rf "$MPLCONFIGDIR"' EXIT
exec /nethome/akolchina/miniconda3/envs/combi/bin/python -m scripts.transfer_active_learning.run --directory "results/$directory" --mix "$mix" --seed "$seed" --model "$model" --method random --device cpu
