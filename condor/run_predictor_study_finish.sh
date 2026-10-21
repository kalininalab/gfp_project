#!/bin/bash
set -euo pipefail
cd /nethome/akolchina/gfp_project
export MPLCONFIGDIR=/tmp/matplotlib-predictor-study
exec /nethome/akolchina/miniconda3/envs/combi/bin/python -m scripts.hurdle_predictor.predictor_study_finalize
