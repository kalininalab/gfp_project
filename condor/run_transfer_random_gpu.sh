#!/bin/bash
set -euo pipefail
cd /nethome/akolchina/gfp_project
mixes=(cgre amac pplu cgre_amac cgre_pplu amac_pplu cgre_amac_pplu artificial)
index=$1
seed=$((42 + index % 5))
mix=${mixes[$((index / 5))]}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR="$(mktemp -d /tmp/gfp-transfer-random.XXXXXX)" CUBLAS_WORKSPACE_CONFIG=:4096:8
trap 'rm -rf "$MPLCONFIGDIR"' EXIT
exec /nethome/akolchina/miniconda3/envs/combi/bin/python -m scripts.transfer_active_learning.run --directory results/transfer_random_cgreGFP_seed42_46 --mix "$mix" --seed "$seed" --model CNN_Jannis_OHE --method random --device cuda
