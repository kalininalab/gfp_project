#!/bin/bash
set -euo pipefail
cd /nethome/akolchina/gfp_project
export MPLCONFIGDIR=/tmp/matplotlib-pool-mixing
exec /nethome/akolchina/miniconda3/envs/combi/bin/python -m scripts.pool_mixing.finalize
