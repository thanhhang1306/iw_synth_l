ssh de7281@della-gpu.princeton.edu \
  "cd /scratch/gpfs/de7281/final_qa/hang_results && \
   find . -type f \( -name epoch_logs.json -o -name classification_report.json \) | \
   tar czf - -T -" \
> logs.tar.gz

mkdir -p q_a
tar xzf logs.tar.gz -C q_a
