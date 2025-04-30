#!/usr/bin/env bash
# fetch data from gpfs 

remote_user_host="de7281@della-gpu.princeton.edu"
remote_base="/scratch/gpfs/de7281/final_qa/hang_results"
local_base="q_a"
exps=(real_only low_aug moderate_aug high_aug full_aug synthetic_only)

for i in {1..12}; do
  variant="variant_${i}_final"
  for exp in "${exps[@]}"; do
    remote_dir="${remote_base}/${variant}/${exp}"
    local_dir="${local_base}/${i}/${exp}"
    mkdir -p "${local_dir}"
    if scp -q "${remote_user_host}:${remote_dir}/epoch_logs.json" "${local_dir}/"; then
      echo "${variant}/${exp}/epoch_logs.json"
    else
      echo "missing: ${variant}/${exp}/epoch_logs.json"
    fi
    if scp -q "${remote_user_host}:${remote_dir}/classification_report.json" "${local_dir}/"; then
      echo "${variant}/${exp}/classification_report.json"
    else
      echo "missing: ${variant}/${exp}/classification_report.json"
    fi
  done
done
