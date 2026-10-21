#!/bin/bash
set -euo pipefail
cd /nethome/akolchina/gfp_project
model="$1"
if [[ "$model" == "xgboost" ]]; then
  export PYTHONPATH=/data/users/akolchina/gfp_project_xgboost/python_packages
  device=cpu
else
  device=cuda
fi
exec /nethome/akolchina/miniconda3/envs/combi/bin/python -m scripts.hurdle_predictor.proteinbert_run \
  --model "$model" --gene "$2" --seed "$3" --device "$device"
