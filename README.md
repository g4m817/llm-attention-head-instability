# Inter-Head Instability: A Signal of Attention Disagreement in LLMs

> Status: Exploratory but repeatable. I’m a security engineer, not an ML theorist; I noticed this while prototyping prompt-injection defenses. The code automates a lot and I haven’t manually audited every artifact. Please treat as preliminary; issues/PRs welcome if you spot mistakes.

## Overview
Recent work on attention interpretability describes the distraction effect, attention heads shifting from system tokens toward injected tokens ([Attention Tracker: Detecting Prompt Injection Attacks in LLMs (Hung, Ko, Rawat, Chung, Hsu, and Chen, 2024)](https://arxiv.org/html/2411.00348v1)), This repo explores a potentially complementary signal: inter-head instability. In certain decoding windows, attention heads disagree more when user input conflicts with system instructions.

I quantify instability as the per-layer standard deviation across heads of attention on system-prompt tokens, averaged over a short slice of steps/layers.

Across multiple datasets and two model families, adversarial prompts (that try to override the system rule) show higher instability than benign prompts, even when benign prompts are long and messy. Instability does not reduce to entropy/uncertainty in my runs, and it tracks a breakdown in head agreement.

What this looks good for:
- Lightweight defensive layer for prompt injections w/ automatic gating and/or alerting.
- Routing heuristic: flag likely instruction conflict early (before routing suspicious prompts to heavier guards/tools).
- Interpretability lens: visualize where/when heads coordinate vs. fragment under conflict.

I’m not claiming novelty over Attention Tracker; this signal may be a system-level fingerprint of the distraction effect. While distraction is a head-level shift of focus, instability here measures disagreement across heads when the system prompt and user input conflict. The goal here is to document a durable, testable effect. At the moment, the relationship to the distraction effect remains unclear.

---

## Methodology
- Signal: layer-wise std across heads of attention on system tokens, then mean that over (trimmed) layers and an early decode window.
- Windows: discovered empirically per model family:
  - Nous-Capybara-7B → steps 1–3
  - Mistral-7B-Instruct → steps 11–40
- Trim: drop shallowest 25% and deepest 15% layers (robust to small sweeps).
- Gating: if windowed score > threshold (0.14 in my runs), treat as “suspicious”.
- Length: use max_new_tokens > your target window end; signal is stable w.r.t. longer outputs because the metric averages only the chosen early window.
- Model Settings:
  - These settings were chosen to remove variance to confirm and analyze the signal, further research is required to determine if the signal becomes unmeasurable noise under different model configurations.
  - Temperature: 0
  - TOP_K: 0
  - TOP_P: None
  - SEED: 1000003

I also log token entropy, top-1 prob/margin, and a crude head-agreement score (mean head–head correlation on system-share) to attempt to address “it’s just uncertainty/length” critiques.

---

## Results Across Eight Runs

Four datasets, each run on both model families, 3 iterations, 50 benign + 50 attack prompts ( = 300 prompts/model/dataset). System prompts vary where noted.
- Say “test” (attacks try to elicit the banned string)
- Obfuscated forms (attacks: t3st, te5t, etc.)
- Digits/URLs ban (system: “no digits or URLs”)
- Complex-Benign Input (multi-format, long, conflicting-but-benign contexts: YAML/JSON/HTML/logs/multilingual/etc.). Inputs are messy.

The complex-benign set stresses ingestion/parsing difficulty rather than adversarial intent. The goal was to see if complex inputs would result in measurable inter-head disagreement the same way adversarial prompts do, they did not.

---

> These results are preliminary and based on small datasets/models. While the signal separates benign from adversarial inputs in these runs, it is not clear if this generalizes across architectures, prompts, or attack types.

### 📊 Testing For Pitfalls: Complex Benign Input / Entropy / Etc.
This dataset includes unique graphs compared to the other datasets, the goal of this set was to ensure the signal didn't break down with complex benign prompts, which would indicate it only signals complexity that leads to head disagreement. Additionally, we wanted to ensure this was not just entropy and compare it to head agreement.

#### Run Overview
| run_id             | model                        | system_prompt                | dataset_file                        | window_start | window_end | head_frac | tail_frac | n_benign | n_attack | benign_mean | benign_median | benign_std | attack_mean | attack_median | attack_std | auroc_windowed | thr_at_5_fpr | tpr_at_thr | fpr_at_thr | pearson_instability_entropy | spearman_instability_entropy |
|--------------------|------------------------------|------------------------------|-------------------------------------|--------------|------------|-----------|-----------|----------|----------|-------------|---------------|------------|-------------|---------------|------------|----------------|--------------|------------|------------|-----------------------------|------------------------------|
| run_20250906_004627 | models/Nous-Capybara-7B-V1.9 | sys_prompt_never_say_test.txt | custom_dataset_4_complex_benign.txt | 1            | 3          | 0.25      | 0.15      | 150      | 150      | 0.046317    | 0.041022      | 0.018149   | 0.144348    | 0.141537      | 0.039308   | 0.987200       | 0.115670     | 0.820000   | 0.040000   | -0.199118                   | -0.133201                    |
| run_20250906_012752 | models/Mistral-7B-Instruct-v0.3 | sys_prompt_never_say_test.txt | custom_dataset_4_complex_benign.txt | 11           | 40         | 0.25      | 0.15      | 144      | 150      | 0.088301    | 0.081420      | 0.024401   | 0.150225    | 0.154322      | 0.019313   | 0.948333       | 0.159971     | 0.320000   | 0.041667   | 0.339492                    | 0.353442                     |

| model                          | pearson_instability_entropy | spearman_instability_entropy | pearson_instability_headcorr | spearman_instability_headcorr | n_samples |
|--------------------------------|-----------------------------|------------------------------|------------------------------|-------------------------------|-----------|
| models/Nous-Capybara-7B-V1.9   | -0.199118                   | -0.133201                    | 0.757681                     | 0.735122                      | 300       |
| models/Mistral-7B-Instruct-v0.3| 0.339492                    | 0.353442                     | 0.675619                     | 0.676440                      | 300       |

#### Summary (Nous vs. Mistral)
- Separation holds: complex-benign clusters with benign, not with attacks.
- Not entropy: instability vs. entropy is weak/negative (e.g., Pearson ≈ −0.19, Spearman ≈ −0.11 in a representative Nous run).
- Instability rises as head agreement drops (e.g., Pearson ≈ 0.76 between instability and - mean head–head correlation on system-share).
- Windows remain model-specific: early (steps 1–3) for Nous; mid/late (11–40) for Mistral.
- Both models keep the attack/benign gap despite high input complexity.
- Entropy stays relatively flat compared to instability/head-agreement dynamics, supporting the "instruction-conflict/coordination" interpretation over "just uncertainty."

**Figures:**  
###### Stepwise Entropy Comparison
![Nous Entropy](figs_complex_benign_comparison/models_Nous-Capybara-7B-V1.9_sys_prompt_never_say_test.txt_custom_dataset_4_complex_benign.txt/stepwise_overlay.png)
![Mistral Entropy](figs_complex_benign_comparison/models_Mistral-7B-Instruct-v0.3_sys_prompt_never_say_test.txt_custom_dataset_4_complex_benign.txt/stepwise_overlay.png)

###### Scatter Instability vs Entropy
![Nous IvE](figs_complex_benign_comparison/models_Nous-Capybara-7B-V1.9_sys_prompt_never_say_test.txt_custom_dataset_4_complex_benign.txt/scatter_instability_vs_entropy.png)
![Mistral IvE](figs_complex_benign_comparison/models_Mistral-7B-Instruct-v0.3_sys_prompt_never_say_test.txt_custom_dataset_4_complex_benign.txt/scatter_instability_vs_entropy.png)

###### Scatter Instability vs Head
![Nous IvH](figs_complex_benign_comparison/models_Nous-Capybara-7B-V1.9_sys_prompt_never_say_test.txt_custom_dataset_4_complex_benign.txt/scatter_instability_vs_headcorr.png)
![Mistral IvH](figs_complex_benign_comparison/models_Mistral-7B-Instruct-v0.3_sys_prompt_never_say_test.txt_custom_dataset_4_complex_benign.txt/scatter_instability_vs_headcorr.png)

###### Distribution
![Nous IvH](figs_complex_benign_comparison/models_Nous-Capybara-7B-V1.9_sys_prompt_never_say_test.txt_custom_dataset_4_complex_benign.txt/scatter_windowed.png)
![Mistral IvH](figs_complex_benign_comparison/models_Mistral-7B-Instruct-v0.3_sys_prompt_never_say_test.txt_custom_dataset_4_complex_benign.txt/scatter_windowed.png)

### 📊 Dataset 1: Encouraging / Manipulating the model to say "test"
- 50 Attack prompts
- 50 Benign prompts
- 3 Iterations (total 300 prompts) per model
- (System prompt: *Never say "test"*)

#### Run Overview
| run\_id               | model                    | window | n\_benign | n\_attack | benign\_mean | attack\_mean | auroc\_windowed | tpr\@5%FPR |
| --------------------- | ------------------------ | ------ | --------- | --------- | ------------ | ------------ | --------------- | ---------- |
| run\_20250903\_133404 | Nous-Capybara-7B-V1.9    | 1–3    | 150       | 150       | 0.0635       | 0.1467       | 0.9584          | 0.88       |
| run\_20250903\_135551 | Mistral-7B-Instruct-v0.3 | 11–40  | 150       | 150       | 0.1139       | 0.1526       | 0.9020          | 0.62       |

#### Summary (Nous vs Mistral)
- Separation: Strong for both; Nous earlier and larger, Mistral later and slightly weaker overall.
- ROC: Nous ~0.9584, Mistral ~0.902.
- Distributions: Wider gap for Nous (0.0635→0.1467) than Mistral (0.1139→0.1526).
- AUROC: Nous peaks in 1–3; Mistral ramps and stabilizes 11–40.
- Heatmaps: Early band (Nous) vs. mid/late band (Mistral).

**Figures:**  
###### ROC
![ROC Nous](figs/models_Nous-Capybara-7B-V1.9_sys_prompt_never_say_test.txt_custom_dataset_benign.txt/roc.png)
![ROC Mistral](figs/models_Mistral-7B-Instruct-v0.3_sys_prompt_never_say_test.txt_custom_dataset_benign.txt/roc.png)

---

###### Distribution
![Nous Violin](figs/models_Nous-Capybara-7B-V1.9_sys_prompt_never_say_test.txt_custom_dataset_benign.txt/violin_windowed.png)
![Mistral Violin](figs/models_Mistral-7B-Instruct-v0.3_sys_prompt_never_say_test.txt_custom_dataset_benign.txt/violin_windowed.png)

---

![Nous Scatter](figs/models_Nous-Capybara-7B-V1.9_sys_prompt_never_say_test.txt_custom_dataset_benign.txt/scatter_windowed.png)
![Mistral Scatter](figs/models_Mistral-7B-Instruct-v0.3_sys_prompt_never_say_test.txt_custom_dataset_benign.txt/scatter_windowed.png)

---

###### AUROC
![Nous AUROC](figs/models_Nous-Capybara-7B-V1.9_sys_prompt_never_say_test.txt_custom_dataset_benign.txt/stepwise_auroc.png)
![Mistral AUROC](figs/models_Mistral-7B-Instruct-v0.3_sys_prompt_never_say_test.txt_custom_dataset_benign.txt/stepwise_auroc.png)

---

###### Heatmaps
![Nous Benign Heatmap](figs/models_Nous-Capybara-7B-V1.9_sys_prompt_never_say_test.txt_custom_dataset_benign.txt/mean_heatmap_benign.png)
![Nous Attack Heatmap](figs/models_Nous-Capybara-7B-V1.9_sys_prompt_never_say_test.txt_custom_dataset_benign.txt/mean_heatmap_attack.png)

---

![Mistral Benign Heatmap](figs/models_Mistral-7B-Instruct-v0.3_sys_prompt_never_say_test.txt_custom_dataset_benign.txt/mean_heatmap_benign.png)
![Mistral Attack Heatmap](figs/models_Mistral-7B-Instruct-v0.3_sys_prompt_never_say_test.txt_custom_dataset_benign.txt/mean_heatmap_attack.png) 

---

### 📊 Dataset 2: Obfuscated Forms (*t3st, te5t, etc.)
- 50 Attack prompts
- 50 Benign prompts
- 3 Iterations (total 300 prompts) per model
- (System prompt: *Never say "test"*)  

#### Run Overview
| run\_id               | model                    | window | n\_benign | n\_attack | benign\_mean | attack\_mean | auroc\_windowed | tpr\@5%FPR |
| --------------------- | ------------------------ | ------ | --------- | --------- | ------------ | ------------ | --------------- | ---------- |
| run\_20250903\_145157 | Nous-Capybara-7B-V1.9    | 1–3    | 150       | 150       | 0.0613       | 0.1427       | 0.9376          | 0.44       |
| run\_20250903\_152211 | Mistral-7B-Instruct-v0.3 | 11–40  | 150       | 150       | 0.1100       | 0.1434       | 0.8804          | 0.22       |

#### Summary (Nous vs. Mistral)
- Separation: Reduced by obfuscation but present.
- ROC: Nous ~0.9376, Mistral ~0.8804.
- Distributions: Nous gap remains visible (0.0613→0.1427); Mistral gap narrower (0.1100→0.1434).
- AUROC: Nous early; Mistral stabilizes ~step 15.
- Heatmaps: Softer contrasts; early (Nous) vs. mid/late (Mistral) persists.

**Figures:**  
###### ROC
![ROC Nous](figs/models_Nous-Capybara-7B-V1.9_sys_prompt_never_say_test.txt_custom_dataset_2_benign.txt/roc.png)
![ROC Mistral](figs/models_Mistral-7B-Instruct-v0.3_sys_prompt_never_say_test.txt_custom_dataset_2_benign.txt/roc.png)

---

###### Distribution
![Nous Violin](figs/models_Nous-Capybara-7B-V1.9_sys_prompt_never_say_test.txt_custom_dataset_2_benign.txt/violin_windowed.png)
![Mistral Violin](figs/models_Mistral-7B-Instruct-v0.3_sys_prompt_never_say_test.txt_custom_dataset_2_benign.txt/violin_windowed.png)

---

![Nous Scatter](figs/models_Nous-Capybara-7B-V1.9_sys_prompt_never_say_test.txt_custom_dataset_2_benign.txt/scatter_windowed.png)
![Mistral Scatter](figs/models_Mistral-7B-Instruct-v0.3_sys_prompt_never_say_test.txt_custom_dataset_2_benign.txt/scatter_windowed.png)

---

###### AUROC
![Nous AUROC](figs/models_Nous-Capybara-7B-V1.9_sys_prompt_never_say_test.txt_custom_dataset_2_benign.txt/stepwise_auroc.png)
![Mistral AUROC](figs/models_Mistral-7B-Instruct-v0.3_sys_prompt_never_say_test.txt_custom_dataset_2_benign.txt/stepwise_auroc.png)

---

###### Heatmaps
![Nous Benign Heatmap](figs/models_Nous-Capybara-7B-V1.9_sys_prompt_never_say_test.txt_custom_dataset_2_benign.txt/mean_heatmap_benign.png)
![Nous Attack Heatmap](figs/models_Nous-Capybara-7B-V1.9_sys_prompt_never_say_test.txt_custom_dataset_2_benign.txt/mean_heatmap_attack.png)

---

![Mistral Benign Heatmap](figs/models_Mistral-7B-Instruct-v0.3_sys_prompt_never_say_test.txt_custom_dataset_2_benign.txt/mean_heatmap_benign.png)
![Mistral Attack Heatmap](figs/models_Mistral-7B-Instruct-v0.3_sys_prompt_never_say_test.txt_custom_dataset_2_benign.txt/mean_heatmap_attack.png) 

---

### 📊 Dataset 3: Digits / URLs  
- 50 Attack prompts
- 50 Benign prompts
- 3 Iterations (total 300 prompts) per model
- (System prompt: *No digits or URLs*)

#### Run Overview
| run\_id               | model                    | window | n\_benign | n\_attack | benign\_mean | attack\_mean | auroc\_windowed | tpr\@5%FPR |
| --------------------- | ------------------------ | ------ | --------- | --------- | ------------ | ------------ | --------------- | ---------- |
| run\_20250903\_162514 | Nous-Capybara-7B-V1.9    | 1–3    | 150       | 150       | 0.0465       | 0.0823       | 0.9140          | 0.56       |
| run\_20250903\_165602 | Mistral-7B-Instruct-v0.3 | 11–40  | 150       | 150       | 0.0560       | 0.1306       | 0.9804          | 0.86       |

#### Summary (Nous vs. Mistral)
- Separation: Strong again; Mistral excels under digits/URL ban.
- ROC: Mistral ~0.9804, Nous ~0.9140.
- Distributions: Nous (0.0465→0.0823), Mistral (0.0560→0.1306) → cleaner Mistral split.
- AUROC: Nous early peak; Mistral strong mid/late stability.
- Heatmaps: Clear benign/attack contrast in both; broader late-window band for Mistral.

**Figures:**  
###### ROC
![ROC Nous](figs/models_Nous-Capybara-7B-V1.9_sys_prompt_digits_urls.txt_custom_dataset_3_benign.txt/roc.png)
![ROC Mistral](figs/models_Mistral-7B-Instruct-v0.3_sys_prompt_digits_urls.txt_custom_dataset_3_benign.txt/roc.png)

---

###### Distribution
![Nous Violin](figs/models_Nous-Capybara-7B-V1.9_sys_prompt_digits_urls.txt_custom_dataset_3_benign.txt/violin_windowed.png)
![Mistral Violin](figs/models_Mistral-7B-Instruct-v0.3_sys_prompt_digits_urls.txt_custom_dataset_3_benign.txt/violin_windowed.png)

---

![Nous Scatter](figs/models_Nous-Capybara-7B-V1.9_sys_prompt_digits_urls.txt_custom_dataset_3_benign.txt/scatter_windowed.png)
![Mistral Scatter](figs/models_Mistral-7B-Instruct-v0.3_sys_prompt_digits_urls.txt_custom_dataset_3_benign.txt/scatter_windowed.png)

---

###### AUROC
![Nous AUROC](figs/models_Nous-Capybara-7B-V1.9_sys_prompt_digits_urls.txt_custom_dataset_3_benign.txt/stepwise_auroc.png)
![Mistral AUROC](figs/models_Mistral-7B-Instruct-v0.3_sys_prompt_digits_urls.txt_custom_dataset_3_benign.txt/stepwise_auroc.png)

---

###### Heatmaps
![Nous Benign Heatmap](figs/models_Nous-Capybara-7B-V1.9_sys_prompt_digits_urls.txt_custom_dataset_3_benign.txt/mean_heatmap_benign.png)
![Nous Attack Heatmap](figs/models_Nous-Capybara-7B-V1.9_sys_prompt_digits_urls.txt_custom_dataset_3_benign.txt/mean_heatmap_attack.png)

---

![Mistral Benign Heatmap](figs/models_Mistral-7B-Instruct-v0.3_sys_prompt_digits_urls.txt_custom_dataset_3_benign.txt/mean_heatmap_benign.png)
![Mistral Attack Heatmap](figs/models_Mistral-7B-Instruct-v0.3_sys_prompt_digits_urls.txt_custom_dataset_3_benign.txt/mean_heatmap_attack.png) 

---

## Limitations

- It’s possible the instability signal is not a novel effect, but rather a statistical footprint of the distraction effect (Hung et al. 2024). Further experiments are required to establish whether it provides independent information or simply reflects the same underlying mechanism.
- Results are small-scale.
- Only 4 datasets and 2 system prompts, all synthetically generated.
- Only 2 model families tested.  
- Thresholds/steps tuned per model; no universal setting yet, I suspect it may not be possible to find universal settings, at least for the instability windows identified so far.
- Using std across heads on system-share is one design; other coordination measures may be better.

---

## Next Steps

- Validate on larger, more diverse model families and datasets.
- Compare directly with important-head analysis (Attention Tracker) to test whether instability consistently precedes head distraction.
- Explore whether per-model tuning can be replaced with normalized instability metrics.
- Investigate whether instability precedes jailbreak *success probability* in the wild.

---

## Repo Contents

- `detect_head.py` → main script (collection + scoring/gating, baseline, ungated comparison)  
- `gather.py` → produces ungated generation of adversarial and benign prompts, capturing all data needed to analyze head instability across layers and steps.
- `make_instability_figs.py` → produces figures of instability from runs/ - for visualizing actual results on datasets.
- `outputs/analyze_thresholds.py` → helper for CV + threshold tuning on the contents generated by `gather.py`
- `outputs/aggregate_instability_figs.py` → produces figures for instability
- `datasets/` → three variants of benign and adversarial prompts  
- `system_prompts/` → two variants used in experiments  
- `runs/` → output with per-prompt reports, heatmaps, line plots for analysis of gated and ungated runs
- `outputs/nous` → raw data dumped from `gather.py` and `analyze_thresholds.py` for the `Nous-Capybara-7B-V1.9` model
- `outputs/mistral` → raw data dumped from `gather.py` and `analyze_thresholds.py` for the `Mistral-7B-Instruct-v0.3` model

## Workflow

Ensure the `models/` directory exists with the models you want to test:
```
models/Nous-Capybara-7B-V1.9
models/Mistral-7B-Instruct-v0.3
```

Install
```
python -m venv dh
source dh/bin/activate
pip install -r requirements.txt
```

To run all evaluations, run the helper script that runs everything and then look at the results under `runs/`:
```
chmod +x run.sh
./run.sh
```

To run an individual run:
```
python detect_head.py --threshold 0.145347 --lookback-steps 3 --mid-high-frac 0.25 --tail-cut-frac 0.15 --system-prompt-file system_prompts/sys_prompt_never_say_test.txt --test-prompts-file datasets/custom_dataset_attacks.txt --baseline-prompts-file datasets/custom_dataset_benign.txt --compare-ungated --model models/Nous-Capybara-7B-V1.9  --fail-case test --iterations 3
```

To gather stats to analyze for an instability window run gather.py with the following options
```
  --model MODEL
  --system-prompt-file SYSTEM_PROMPT_FILE
  --baseline-prompts-file BASELINE_PROMPTS_FILE
  --test-prompts-file TEST_PROMPTS_FILE
  --mode {baseline,tests,single}
  --prompt PROMPT
  --iterations ITERATIONS
  --outputs-root OUTPUTS_ROOT
  --max-new-tokens MAX_NEW_TOKENS
  --temperature TEMPERATURE
  --top-k TOP_K
  --top-p TOP_P         -1 disables; else 0<top_p<=1
  --mid-high-frac MID_HIGH_FRAC
  --seed SEED
  --no-per-head
  --no-csv
```

To analyze those stats for instability spike windows, or modify analyze.sh for specific options
```
cd outputs
chmod +x analyze.sh
./analyze.sh
```

To generate specific graphs to observe the instability from these generalized results, from the same directory
```
chmod +x figs.sh
./figs.sh
```

To generate useful figures from your runs under `runs/`, run from the root:
```
python make_instability_figs.py --runs runs --outdir figs
```
