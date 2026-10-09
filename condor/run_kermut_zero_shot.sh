#!/bin/bash
set -euo pipefail

cd /nethome/akolchina/gfp_project
export HF_HOME=/data/users/akolchina/hf
export TRANSFORMERS_OFFLINE=1
exec /nethome/akolchina/miniconda3/envs/combi/bin/python -m scripts.kermut.preprocess_zero_shot
