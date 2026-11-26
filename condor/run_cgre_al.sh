#!/bin/bash
set -euo pipefail
exec /nethome/akolchina/miniconda3/envs/combi/bin/python -m scripts.cgre_active_learning.run --model "$1" --seed "$2" --strategy "$3" --device "$4"
