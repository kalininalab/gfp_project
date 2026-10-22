#!/bin/bash
set -euo pipefail
cd /nethome/akolchina/gfp_project
export MPLCONFIGDIR=/tmp/matplotlib-gmm-gate
exec /nethome/akolchina/miniconda3/envs/combi/bin/python -m scripts.hurdle_predictor.gmm_gate_run \
  --gene "$1" --seed "$2" --device cuda
