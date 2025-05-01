#!/usr/bin/env bash
# fetch data from gpfs 

remote_user_host="de7281@della-gpu.princeton.edu"
remote_base="/scratch/gpfs/de7281/final_sentiment/hang_results/ace"
local_base="sentiment"
exps=(real_only low_aug moderate_aug high_aug full_aug synthetic_only)

for i in {1..12}; do
  variant="variant_${i}_final"
  for exp in "${exps[@]}"; do
    remote_dir="${remote_base}/${variant}/${exp}"
    local_dir="${local_base}/${i}/${exp}"
    mkdir -p "${local_dir}"

    if scp -q "${remote_user_host}:${remote_dir}/evaluation_results.json" "${local_dir}/"; then
      echo "${variant}/${exp}/evaluation_results.json"
    else
      echo "missing: ${variant}/${exp}/evaluation_results.json"
    fi

    if scp -q "${remote_user_host}:${remote_dir}/classification_report.txt" "${local_dir}/"; then
      echo "${variant}/${exp}/classification_report.txt"
    else
      echo "missing: ${variant}/${exp}/classification_report.txt"
    fi
  done
done
