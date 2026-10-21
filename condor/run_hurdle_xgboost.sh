#!/bin/bash
set -euo pipefail
cd /nethome/akolchina/gfp_project
export PYTHONPATH=/data/users/akolchina/gfp_project_xgboost/python_packages
exec /nethome/akolchina/miniconda3/envs/combi/bin/python -m scripts.hurdle_predictor.xgboost_run \
  --representation "$1" --gene "$2" --seed "$3"
