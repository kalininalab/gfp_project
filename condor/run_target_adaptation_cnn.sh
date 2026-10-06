#!/bin/bash
set -euo pipefail
cd /nethome/akolchina/gfp_project
kind=$1
index=$2
if [[ "$kind" == "proteins" ]]; then
  directory=results/target_adaptation_cnn_seed42_46
  pairs=(cgreGFP_to_amacGFP cgreGFP_to_ppluGFP amacGFP_to_cgreGFP amacGFP_to_ppluGFP ppluGFP_to_cgreGFP ppluGFP_to_amacGFP)
else
  directory=results/target_adaptation_peaks_cnn_seed42_46
  pairs=(cgreGFP_to_artificial artificial_to_cgreGFP)
fi
seed=$((42 + index % 5))
pair=${pairs[$((index / 5))]}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR="$(mktemp -d /tmp/gfp-target-cnn.XXXXXX)"
trap 'rm -rf "$MPLCONFIGDIR"' EXIT
exec /nethome/akolchina/miniconda3/envs/combi/bin/python -m scripts.target_adaptation.run \
  --directory "$directory" --pair "$pair" --seed "$seed" --model CNN_Jannis_OHE --device cuda
