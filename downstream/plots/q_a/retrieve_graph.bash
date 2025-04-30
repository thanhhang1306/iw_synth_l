#!/usr/bin/env bash

remote_user_host="de7281@della-gpu.princeton.edu"
remote_base="/scratch/gpfs/de7281/final_qa/hang_plots"
local_base="all_qa_plots"

mkdir -p "${local_base}"

# include only .png, exclude everything else
rsync -avz -e ssh \
  --include='*/' \
  --include='*.png' \
  --exclude='*' \
  "${remote_user_host}:${remote_base}/" \
  "${local_base}/"

