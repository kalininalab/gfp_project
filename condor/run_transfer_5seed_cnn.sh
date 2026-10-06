#!/bin/bash
set -euo pipefail
mixes=(cgre amac pplu cgre_amac cgre_pplu amac_pplu cgre_amac_pplu artificial)
index=$1
seed=$((42 + index % 5))
mix=${mixes[$((index / 5))]}
exec /bin/bash /nethome/akolchina/gfp_project/condor/run_transfer_model.sh "$mix" "$seed" CNN_Jannis_OHE cuda
