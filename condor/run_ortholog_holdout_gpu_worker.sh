#!/bin/bash
set -euo pipefail
cd /nethome/akolchina/gfp_project
raw=/data/users/akolchina/gfp_project/ortholog_leave_one_out_v2_seed42_46_runs
worker=$1
for index in "$worker" "$((worker + 47))" "$((worker + 94))"; do
  (( index < 140 )) || continue
  read -r target condition seed < <(sed -n "$((index + 1))p" "$raw/gpu_jobs.txt")
  /bin/bash /nethome/akolchina/gfp_project/condor/run_ortholog_holdout.sh \
    "$target" "$condition" "$seed" CNN_Jannis_OHE cuda
done
