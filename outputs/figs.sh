python3.11 aggregate_instability_figs.py \
  --benign-root mistral/outputs/baseline \
  --attacks-root mistral/outputs/tests \
  --outdir mistral/figs \
  --model-label "Mistral-7B-Instruct-v0.3" \
  --start-step 11 --end-step 40 \
  --mid-high-frac 0.25 --tail-cut-frac 0.15 \
  --scatter-k 29 --k-min 11 --k-max 40 \
  --summary-csv mistral/summary.csv

python3.11 aggregate_instability_figs.py \
  --benign-root nous/outputs/baseline \
  --attacks-root nous/outputs/tests \
  --outdir nous/figs \
  --model-label "Nous-Capybara-7B-V1.9" \
  --start-step 1 --end-step 10 \
  --mid-high-frac 0.25 --tail-cut-frac 0.15 \
  --scatter-k 10 --k-min 1 --k-max 10 \
  --summary-csv nous/summary.csv
