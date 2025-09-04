python3.11 analyze_thresholds.py --attacks_root mistral/outputs/tests --benign_root mistral/outputs/baseline --target_fpr 0.01 --also_target_fpr 0.05 --cv --out_csv summary.csv --start_step 11 --end_step 40 --mid_high_fracs 0.25 0.33 0.40 --tail_cut_fracs 0.10 0.15

python3.11 analyze_thresholds.py --attacks_root nous/outputs/tests --benign_root nous/outputs/baseline --target_fpr 0.01 --also_target_fpr 0.05 --cv --out_csv summary.csv --start_step 1 --end_step 10 --mid_high_fracs 0.25 0.33 0.40 --tail_cut_fracs 0.10 0.15
