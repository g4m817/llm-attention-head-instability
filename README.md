# Inter-Head Instability: A Signal of Attention Disagreement in LLMs

> DISCLAIMER: This is exploratory work. I am not an ML engineer, I’m a security engineer who noticed this signal while tinkering with prompt injection defenses. These experiments may be wrong or incomplete, but I wanted to document them so others, especially researchers, can validate, refine, or discard. If I've misstated something, please tell me via an issue so I can correct it!

This repo contains exploratory experiments showing that **attention heads often disagree when faced with adversarial input**, and that this may provide a useful signal of prompt injection attempts. Although, more research needs to be conducted to determine if its useful across datasets / instructions and model families.

Even if this instability signal proves too noisy or imprecise to serve as a practical security measure, it may still be useful for interpretability. In particular, the instability windows may highlight when during decoding a model is actively processing system instructions, when its heads disagree on that focus, and when it stabilizes toward either following those instructions or drifting toward adversarial input.

---

## 1. Overview
Recent work on attention interpretability has described the distraction effect ([Attention Tracker: Detecting Prompt Injection Attacks in LLMs (Hung, Ko, Rawat, Chung, Hsu, and Chen, 2024)](https://arxiv.org/html/2411.00348v1)), where certain attention heads shift from system tokens toward injected tokens. This repo explores what may be a complementary signal: inter-head instability. Instead of drifting together, attention heads often disagree, some cling to the system prompt, others wander. This disagreement shows up as increased variance across heads within specific decoding windows. It may represent an early or system-level footprint of the distraction effect, though the relationship is not yet clear.

This disagreement might ultimately be just an interpretability curiosity, but so far it shows up consistently in particular decoding windows across system prompts and datasets. Measuring variance across heads in their focus on system tokens provides a simple statistical lens on when the model’s internal attention patterns become unstable. This suggests the possibility of flagging early signs of the model being pushed off-course, but it remains unclear whether the signal generalizes broadly or is robust enough for practical detection.

There are, of course, outliers. Some benign prompts confuse the model’s heads, and some adversarial prompts slip through without measurable disagreement. This is expected, the underlying mechanism of why heads agree or disagree remains a black box. At this stage, the safest claim is that inter-head divergence correlates with adversarial prompting in certain settings, but it is not a reliable or universal discriminator.

---

## 2. Method
- **Why the chosen step windows:** Different models *appear* to reach instability at different windows before ultimately stabilizing, these windows appear to be different per model.
- **Windows:**  
  - Nous-Capybara-7B → steps **1–3**.  
  - Mistral-7B → steps **11–40**.  
- **Layer trimming:** Drop shallowest 25% and deepest 15%. 
- **Metric:** Take the std of heads each layer of a step, then average them out to arrive at a suspicion score, gated by a threshold informed by data (generally, 0.14). This number has appeared across two model families producing the best mitigation, with the least false positives (1-5% FPR).
- **Model Settings:** 
  - These settings were chosen to remove variance to confirm and analyze the signal, further research is required to determine if the signal becomes unmeasurable noise under different model configurations.
  - Temperature: 0
  - TOP_K: 0
  - TOP_P: None
  - SEED: 1000003

---

## 3. Results Across Six Runs

We evaluate across **3 datasets × 2 models**. Each dataset has two runs: one with Nous-Capybara-7B, one with Mistral-7B. the first two datasets use the same system prompt "never say test", the last dataset uses a new system prompt "never use digits or URLs". These were chosen to demonstrate the signal appeared across datasets.

---
> These results are preliminary and based on small datasets/models. While the signal separates benign from adversarial inputs in these runs, it is not clear if this generalizes across architectures, prompts, or attack types.

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
- Only 3 datasets and 2 system prompts, both synthetically generated with fragile pass/fail scoring.
- Only 2 model families tested.  
- Thresholds/steps tuned per model; no universal setting yet, I suspect it may not be possible to find universal settings, at least for the instability windows identified so far.
-  It’s also possible that inter-head instability is not a security-relevant signal at all. Complex or ambiguous prompting may naturally cause heads to diverge, making instability just a side-effect of normal generative complexity. In that framing, its real utility may lie in interpretability, helping highlight when and where models internally "disagree", rather than reliably catching attacks.
---

## Next Steps

- Validate on larger, more diverse model families and datasets.
- Compare directly with important-head analysis (Attention Tracker) to test whether instability consistently precedes head distraction.
- Explore whether per-model tuning can be replaced with normalized instability metrics.
- Investigate whether instability precedes jailbreak *success probability* in the wild.
- Because the detection of adversarial prompts via instability is less precise and prone to false positives on complex benign prompts, one possible use case is as a routing heuristic. Instead of sending every input through a costly guard LLM, inter-head instability could potentially flag only the suspicious ones, potentially saving significant resources for companies deploying such defenses.

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

To run an invidiual run:
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
