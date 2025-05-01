ssh de7281@della-gpu.princeton.edu \
  "cd /scratch/gpfs/de7281/final_sentiment/hang_results/ace && \
   find . -type f \( -name evaluation_results.json -o -name classification_report.txt \) | \
   tar czf - -T -" \
> sentiment_logs.tar.gz

# then unpack:
mkdir -p sentiment
tar xzf sentiment_logs.tar.gz -C sentiment