#!/bin/bash
set -euo pipefail
cd /nethome/akolchina/gfp_project
export MPLCONFIGDIR=/tmp/matplotlib-hurdle
exec /nethome/akolchina/miniconda3/envs/combi/bin/python -m scripts.hurdle_predictor.neural_run \
  --architecture "$1" --gene "$2" --seed "$3" --device cuda
