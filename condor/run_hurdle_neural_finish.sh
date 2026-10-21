#!/bin/bash
set -euo pipefail
cd /nethome/akolchina/gfp_project
export MPLCONFIGDIR=/tmp/matplotlib-hurdle-final
/nethome/akolchina/miniconda3/envs/combi/bin/python -m scripts.hurdle_predictor.neural_finalize
