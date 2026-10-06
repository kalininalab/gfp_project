#!/bin/bash
set -euo pipefail
mixes=(cgre amac pplu cgre_amac cgre_pplu amac_pplu cgre_amac_pplu artificial)
models=(aubin_1_10_1 aubin_linear mlp_small)
index=$1
model=${models[$((index % 3))]}
seed=$((42 + (index / 3) % 5))
mix=${mixes[$((index / 15))]}
exec /bin/bash /nethome/akolchina/gfp_project/condor/run_transfer_model.sh "$mix" "$seed" "$model" cpu
