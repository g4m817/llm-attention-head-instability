# Inter-Head Instability: A Signal of Attention Disagreement in LLMs

> **Status: Exploratory but repeatable.** I first noticed this signal while prototyping prompt-injection defenses. The code automates most analyses, but I haven’t manually audited every artifact. Please treat this as preliminary research: issues/PRs are welcome if you spot mistakes or want to extend the work.  
>  
> **Current work:** I am running evals with Deepset to align with Attention Tracker's datasets for comparative analysis. These will be pushed when the evals finish running.

---

## 1. Overview

Prompt injection attacks exploit the tension between **system prompts** (intended instructions) and **user prompts** (inputs).

Recent interpretability work highlights the **distraction effect**: injected tokens pull attention away from system instructions ([Hung et al., 2024](https://arxiv.org/abs/2411.00348)). Their *Attention Tracker* identifies *important heads* that are especially prone to distraction and monitors their focus.

This project explores a **complementary signal**: instead of tracking *which heads* lose focus, we measure *how consistently* heads agree on system tokens.

**Key idea:** adversarial prompts appear to induce **internal conflict**, measurable as elevated cross-head variance in certain decoding windows.

- **Benign prompts →** heads align, low variance.  
- **Adversarial prompts →** heads fracture, high variance.

We call this **inter-head instability**. 

---

## 2. A Complementary Lens  

This work does **not replace** Attention Tracker or other attention-based detectors. Instead, it offers a different perspective:

- **Attention Tracker:** focuses on *important heads* and measures **where attention shifts**.  
- **Inter-head instability:** focuses on *important windows* and measures **how much heads disagree** when reconciling system vs. user prompts.

Together, these views appear to capture two aspects of the same phenomenon:

- *Distraction effect:* adversarial tokens hijack attention.  
- *Instability effect:* heads disagree while resolving the conflict.

This complements related work on **safety heads** ([Wang et al., 2024](https://arxiv.org/abs/2407.01599)), **conflicting heads** ([Zverev et al., 2024](https://arxiv.org/abs/2405.21064)), and **Trojan detection** ([Lyu et al., 2022](https://arxiv.org/abs/2203.00229)).

---

## 3. Why This Matters  
I see this heuristic from multiple lenses, such as:
- **Lightweight detection:** Aggregated attention stats, no per-head tracing.  
- **Temporal calibration:** Family-specific instability windows often appear to emerge *before decoding finishes*.  
- **Interpretability:** Maps *when* conflict resolution occurs, complementing *which heads* are involved.  
- **Safety evaluation:** Low instability + safe refusals may indicate robust internalization of safety.  
- **Practical synergy:** Instability could augment Attention Tracker or safety-head analyses.

Potential Real-world use-cases:
- **Routing heuristic:** Flag suspicious prompts before sending to heavier guardrails.  
- **Model evaluation:** Score models on stability under adversarial inputs.  
- **Tool improvement:** Combine instability windows with important-head tracking for stronger detectors.

---

## 4. Core Findings  

- **Separation:** Across multiple datasets and two model families, adversarial prompts tend to show higher instability than benign, even when benign are long/noisy.  
- **Distinct from uncertainty:** Instability does not appear to reduce to entropy — correlations are weak or negative ([Hendrycks & Gimpel, 2017](https://arxiv.org/abs/1610.02136)).  
- **Head-level disagreement:** Appears to reflect fragmentation among heads on system tokens, consistent with the distraction effect ([Hung et al., 2024](https://arxiv.org/abs/2411.00348)).  
- **Model-specific windows:** Nous shows early instability (steps 1-3), Mistral shows late instability (steps 11-26).  
- **Prompt strength matters:** Stronger system instructions create sharper conflicts when user prompts contradict them, producing higher instability in those windows. This complements strong security prompt engineering (stronger system prompts → clearer stability/instability contrast).
- **Low instability does not necessarily reflect failure of the model:** A low instability score on an attack (while the model safely refuses) may indicate the model resolves conflict cleanly (i.e., it stays aligned). That’s a detection miss for this signal, but an interesting safety evaluation dimension.  

---

## 5. Why Windows Differ (Hypotheses + Context)  

Although no study has directly documented *model-specific reconciliation windows*, multiple strands of evidence support this claim:

1. **Instruction tuning strategies**  
   - Strongly tuned models (e.g., Mistral-Instruct) devote persistent attention to system tokens ([Ouyang et al., 2022](https://arxiv.org/abs/2203.02155)), deferring visible conflict until deeper layers.  
   - Less tuned models (e.g., Nous) show immediate head disagreement when system and user collide.  

2. **Architectural integration depth**  
   - Transformer interpretability shows layer specialization: shallow = lexical, middle = semantic, deep = alignment ([Tenney et al., 2019](https://arxiv.org/abs/1905.06316)).  
   - If reconciliation is pushed deeper, instability appears to emerge later.  

3. **Prompt handling differences**  
   - Some families stabilize early around system prompts, others revisit them dynamically ([Xie et al., 2024](https://arxiv.org/abs/2402.19419)).  

4. **Conflict resolution styles**  
   - *Early debaters*: high initial disagreement, then converge (Nous).  
   - *Late arbitrators*: stable early, fracture later under conflict (Mistral).  

5. **Sliding windows**  
   - Instability windows may shift later with longer prompts, consistent with research showing context length changes attention allocation ([Press et al., 2021](https://arxiv.org/abs/2102.00557)). This would require adaptive window logic instead of pre-selecting windows to make the approach robust.  

---

### 6. On Instability Windows
A key observation is that **instability onset appears to be model-dependent rather than dataset-dependent**. In Nous-Capybara-7B, divergence between benign and adversarial runs appears immediately within the first few decoding steps. Heatmaps of mean instability show that attacks trigger elevated cross-head variance almost instantly, and stepwise AUROC peaks at steps 1-3. By contrast, Mistral-7B-Instruct shows relatively flat early-step curves: instability rises only later, becoming most discriminative in steps 11-26.

This difference is reinforced by distribution plots: in Nous, attack scores separate cleanly from benign in early windows, while in Mistral, separation is strongest when scores are pooled over mid-range decoding. Scatter plots confirm that in both models, instability correlates more strongly with head agreement breakdown than with entropy. Together, these results demonstrate that, in our experiments, **the effective detection window is a function of model architecture and training, not input dataset**. 

---

Nous-Capybara-7B attack heatmap shows immediate instability spike (steps 1-3).
![Nous attack vs benign heatmap](original_figs/models_Nous-Capybara-7B-V1.9_sys_prompt_never_say_test.txt_custom_dataset_4_complex_benign.txt/mean_heatmap_attack.png)

Stepwise AUROC for Nous. Highest discrimination occurs at earliest steps.
![Nous stepwise AUROC](original_figs/models_Nous-Capybara-7B-V1.9_sys_prompt_never_say_test.txt_custom_dataset_4_complex_benign.txt/stepwise_auroc.png)

Mistral-7B-Instruct attack heatmap. Instability builds gradually, differentiating after step 10.
![Mistral attack vs benign heatmap](original_figs/models_Mistral-7B-Instruct-v0.3_sys_prompt_never_say_test.txt_custom_dataset_4_complex_benign.txt/mean_heatmap_attack.png)

ROC curve for Mistral, windowed over steps 11-40. AUROC ≈ 0.95.
![Mistral ROC](original_figs/models_Mistral-7B-Instruct-v0.3_sys_prompt_never_say_test.txt_custom_dataset_4_complex_benign.txt/roc.png)

Nous violin plot of instability scores. Clear early-step separation.
![Nous violin](original_figs/models_Nous-Capybara-7B-V1.9_sys_prompt_never_say_test.txt_custom_dataset_4_complex_benign.txt/violin_windowed.png)

Mistral violin plot. Best separation in mid-step window.
![Mistral violin](original_figs/models_Mistral-7B-Instruct-v0.3_sys_prompt_never_say_test.txt_custom_dataset_4_complex_benign.txt/violin_windowed.png)

Nous scatter plot. Strong negative correlation between instability and head agreement.
![Nous scatter instability vs head agreement](original_figs/models_Nous-Capybara-7B-V1.9_sys_prompt_never_say_test.txt_custom_dataset_4_complex_benign.txt/scatter_instability_vs_headcorr.png)

Mistral scatter plot. Same pattern, but onset delayed until later steps.
![Mistral scatter instability vs head agreement](original_figs/models_Mistral-7B-Instruct-v0.3_sys_prompt_never_say_test.txt_custom_dataset_4_complex_benign.txt/scatter_instability_vs_headcorr.png)

---

## 7. Results
### Deepset Results
TODO

### Synthetic Data Results
For our synthetic datasets we used four datasets, each run on both model families, 3 iterations, 50 benign + 50 attack prompts ( = 300 prompts/model/dataset). System prompts vary where noted.
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
- Instability rises as head agreement drops (e.g., Pearson ≈ 0.76 between instability and - mean head-head correlation on system-share).
- Windows remain model-specific: early (steps 1-3) for Nous; mid/late (11-26) for Mistral.
- Both models keep the attack/benign gap despite high input complexity.
- Entropy stays relatively flat compared to instability/head-agreement dynamics, supporting the "instruction-conflict/coordination" interpretation over "just uncertainty."

**Figures:**  
###### Stepwise Entropy Comparison
![Nous Entropy](original_figs/models_Nous-Capybara-7B-V1.9_sys_prompt_never_say_test.txt_custom_dataset_4_complex_benign.txt/stepwise_overlay.png)
![Mistral Entropy](original_figs/models_Mistral-7B-Instruct-v0.3_sys_prompt_never_say_test.txt_custom_dataset_4_complex_benign.txt/stepwise_overlay.png)

###### Distribution
![Nous IvH](original_figs/models_Nous-Capybara-7B-V1.9_sys_prompt_never_say_test.txt_custom_dataset_4_complex_benign.txt/scatter_windowed.png)
![Mistral IvH](original_figs/models_Mistral-7B-Instruct-v0.3_sys_prompt_never_say_test.txt_custom_dataset_4_complex_benign.txt/scatter_windowed.png)

### 📊 Dataset 1: Encouraging / Manipulating the model to say "test"
- 50 Attack prompts
- 50 Benign prompts
- 3 Iterations (total 300 prompts) per model
- (System prompt: *Never say "test"*)

#### Run Overview
| run\_id               | model                    | window | n\_benign | n\_attack | benign\_mean | attack\_mean | auroc\_windowed | tpr\@5%FPR |
| --------------------- | ------------------------ | ------ | --------- | --------- | ------------ | ------------ | --------------- | ---------- |
| run\_20250903\_133404 | Nous-Capybara-7B-V1.9    | 1-3    | 150       | 150       | 0.0635       | 0.1467       | 0.9584          | 0.88       |
| run\_20250903\_135551 | Mistral-7B-Instruct-v0.3 | 11-40  | 150       | 150       | 0.1139       | 0.1526       | 0.9020          | 0.62       |

#### Summary (Nous vs Mistral)
- Separation: Strong for both; Nous earlier and larger, Mistral later and slightly weaker overall.
- ROC: Nous ~0.9584, Mistral ~0.902.
- Distributions: Wider gap for Nous (0.0635→0.1467) than Mistral (0.1139→0.1526).
- AUROC: Nous peaks in 1-3; Mistral ramps and stabilizes 11-40.
- Heatmaps: Early band (Nous) vs. mid/late band (Mistral).

**Figures:**  
###### ROC
![ROC Nous](original_figs/models_Nous-Capybara-7B-V1.9_sys_prompt_never_say_test.txt_custom_dataset_1_benign.txt/roc.png)
![ROC Mistral](original_figs/models_Mistral-7B-Instruct-v0.3_sys_prompt_never_say_test.txt_custom_dataset_1_benign.txt/roc.png)

---

###### Distribution
![Nous Violin](original_figs/models_Nous-Capybara-7B-V1.9_sys_prompt_never_say_test.txt_custom_dataset_1_benign.txt/violin_windowed.png)
![Mistral Violin](original_figs/models_Mistral-7B-Instruct-v0.3_sys_prompt_never_say_test.txt_custom_dataset_1_benign.txt/violin_windowed.png)

---

![Nous Scatter](original_figs/models_Nous-Capybara-7B-V1.9_sys_prompt_never_say_test.txt_custom_dataset_1_benign.txt/scatter_windowed.png)
![Mistral Scatter](original_figs/models_Mistral-7B-Instruct-v0.3_sys_prompt_never_say_test.txt_custom_dataset_1_benign.txt/scatter_windowed.png)

---

###### AUROC
![Nous AUROC](original_figs/models_Nous-Capybara-7B-V1.9_sys_prompt_never_say_test.txt_custom_dataset_1_benign.txt/stepwise_auroc.png)
![Mistral AUROC](original_figs/models_Mistral-7B-Instruct-v0.3_sys_prompt_never_say_test.txt_custom_dataset_1_benign.txt/stepwise_auroc.png)

---

###### Heatmaps
![Nous Benign Heatmap](original_figs/models_Nous-Capybara-7B-V1.9_sys_prompt_never_say_test.txt_custom_dataset_1_benign.txt/mean_heatmap_benign.png)
![Nous Attack Heatmap](original_figs/models_Nous-Capybara-7B-V1.9_sys_prompt_never_say_test.txt_custom_dataset_1_benign.txt/mean_heatmap_attack.png)

---

![Mistral Benign Heatmap](original_figs/models_Mistral-7B-Instruct-v0.3_sys_prompt_never_say_test.txt_custom_dataset_1_benign.txt/mean_heatmap_benign.png)
![Mistral Attack Heatmap](original_figs/models_Mistral-7B-Instruct-v0.3_sys_prompt_never_say_test.txt_custom_dataset_1_benign.txt/mean_heatmap_attack.png) 

---

### 📊 Dataset 2: Obfuscated Forms (*t3st, te5t, etc.)
- 50 Attack prompts
- 50 Benign prompts
- 3 Iterations (total 300 prompts) per model
- (System prompt: *Never say "test"*)  

#### Run Overview
| run\_id               | model                    | window | n\_benign | n\_attack | benign\_mean | attack\_mean | auroc\_windowed | tpr\@5%FPR |
| --------------------- | ------------------------ | ------ | --------- | --------- | ------------ | ------------ | --------------- | ---------- |
| run\_20250903\_145157 | Nous-Capybara-7B-V1.9    | 1-3    | 150       | 150       | 0.0613       | 0.1427       | 0.9376          | 0.44       |
| run\_20250903\_152211 | Mistral-7B-Instruct-v0.3 | 11-40  | 150       | 150       | 0.1100       | 0.1434       | 0.8804          | 0.22       |

#### Summary (Nous vs. Mistral)
- Separation: Reduced by obfuscation but present.
- ROC: Nous ~0.9376, Mistral ~0.8804.
- Distributions: Nous gap remains visible (0.0613→0.1427); Mistral gap narrower (0.1100→0.1434).
- AUROC: Nous early; Mistral stabilizes ~step 15.
- Heatmaps: Softer contrasts; early (Nous) vs. mid/late (Mistral) persists.

**Figures:**  
###### ROC
![ROC Nous](original_figs/models_Nous-Capybara-7B-V1.9_sys_prompt_never_say_test.txt_custom_dataset_2_benign.txt/roc.png)
![ROC Mistral](original_figs/models_Mistral-7B-Instruct-v0.3_sys_prompt_never_say_test.txt_custom_dataset_2_benign.txt/roc.png)

---

###### Distribution
![Nous Violin](original_figs/models_Nous-Capybara-7B-V1.9_sys_prompt_never_say_test.txt_custom_dataset_2_benign.txt/violin_windowed.png)
![Mistral Violin](original_figs/models_Mistral-7B-Instruct-v0.3_sys_prompt_never_say_test.txt_custom_dataset_2_benign.txt/violin_windowed.png)

---

![Nous Scatter](original_figs/models_Nous-Capybara-7B-V1.9_sys_prompt_never_say_test.txt_custom_dataset_2_benign.txt/scatter_windowed.png)
![Mistral Scatter](original_figs/models_Mistral-7B-Instruct-v0.3_sys_prompt_never_say_test.txt_custom_dataset_2_benign.txt/scatter_windowed.png)

---

###### AUROC
![Nous AUROC](original_figs/models_Nous-Capybara-7B-V1.9_sys_prompt_never_say_test.txt_custom_dataset_2_benign.txt/stepwise_auroc.png)
![Mistral AUROC](original_figs/models_Mistral-7B-Instruct-v0.3_sys_prompt_never_say_test.txt_custom_dataset_2_benign.txt/stepwise_auroc.png)

---

###### Heatmaps
![Nous Benign Heatmap](original_figs/models_Nous-Capybara-7B-V1.9_sys_prompt_never_say_test.txt_custom_dataset_2_benign.txt/mean_heatmap_benign.png)
![Nous Attack Heatmap](original_figs/models_Nous-Capybara-7B-V1.9_sys_prompt_never_say_test.txt_custom_dataset_2_benign.txt/mean_heatmap_attack.png)

---

![Mistral Benign Heatmap](original_figs/models_Mistral-7B-Instruct-v0.3_sys_prompt_never_say_test.txt_custom_dataset_2_benign.txt/mean_heatmap_benign.png)
![Mistral Attack Heatmap](original_figs/models_Mistral-7B-Instruct-v0.3_sys_prompt_never_say_test.txt_custom_dataset_2_benign.txt/mean_heatmap_attack.png) 

---

### 📊 Dataset 3: Digits / URLs  
- 50 Attack prompts
- 50 Benign prompts
- 3 Iterations (total 300 prompts) per model
- (System prompt: *No digits or URLs*)

#### Run Overview
| run\_id               | model                    | window | n\_benign | n\_attack | benign\_mean | attack\_mean | auroc\_windowed | tpr\@5%FPR |
| --------------------- | ------------------------ | ------ | --------- | --------- | ------------ | ------------ | --------------- | ---------- |
| run\_20250903\_162514 | Nous-Capybara-7B-V1.9    | 1-3    | 150       | 150       | 0.0465       | 0.0823       | 0.9140          | 0.56       |
| run\_20250903\_165602 | Mistral-7B-Instruct-v0.3 | 11-40  | 150       | 150       | 0.0560       | 0.1306       | 0.9804          | 0.86       |

#### Summary (Nous vs. Mistral)
- Separation: Strong again; Mistral excels under digits/URL ban.
- ROC: Mistral ~0.9804, Nous ~0.9140.
- Distributions: Nous (0.0465→0.0823), Mistral (0.0560→0.1306) → cleaner Mistral split.
- AUROC: Nous early peak; Mistral strong mid/late stability.
- Heatmaps: Clear benign/attack contrast in both; broader late-window band for Mistral.

**Figures:**  
###### ROC
![ROC Nous](original_figs/models_Nous-Capybara-7B-V1.9_sys_prompt_digits_urls.txt_custom_dataset_3_benign.txt/roc.png)
![ROC Mistral](original_figs/models_Mistral-7B-Instruct-v0.3_sys_prompt_digits_urls.txt_custom_dataset_3_benign.txt/roc.png)

---

###### Distribution
![Nous Violin](original_figs/models_Nous-Capybara-7B-V1.9_sys_prompt_digits_urls.txt_custom_dataset_3_benign.txt/violin_windowed.png)
![Mistral Violin](original_figs/models_Mistral-7B-Instruct-v0.3_sys_prompt_digits_urls.txt_custom_dataset_3_benign.txt/violin_windowed.png)

---

![Nous Scatter](original_figs/models_Nous-Capybara-7B-V1.9_sys_prompt_digits_urls.txt_custom_dataset_3_benign.txt/scatter_windowed.png)
![Mistral Scatter](original_figs/models_Mistral-7B-Instruct-v0.3_sys_prompt_digits_urls.txt_custom_dataset_3_benign.txt/scatter_windowed.png)

---

###### AUROC
![Nous AUROC](original_figs/models_Nous-Capybara-7B-V1.9_sys_prompt_digits_urls.txt_custom_dataset_3_benign.txt/stepwise_auroc.png)
![Mistral AUROC](original_figs/models_Mistral-7B-Instruct-v0.3_sys_prompt_digits_urls.txt_custom_dataset_3_benign.txt/stepwise_auroc.png)

---

###### Heatmaps
![Nous Benign Heatmap](original_figs/models_Nous-Capybara-7B-V1.9_sys_prompt_digits_urls.txt_custom_dataset_3_benign.txt/mean_heatmap_benign.png)
![Nous Attack Heatmap](original_figs/models_Nous-Capybara-7B-V1.9_sys_prompt_digits_urls.txt_custom_dataset_3_benign.txt/mean_heatmap_attack.png)

---

![Mistral Benign Heatmap](original_figs/models_Mistral-7B-Instruct-v0.3_sys_prompt_digits_urls.txt_custom_dataset_3_benign.txt/mean_heatmap_benign.png)
![Mistral Attack Heatmap](original_figs/models_Mistral-7B-Instruct-v0.3_sys_prompt_digits_urls.txt_custom_dataset_3_benign.txt/mean_heatmap_attack.png) 

---

## 8. Methodology  

**Signal definition:**  
1. For each decoding step, compute attention on system tokens.  
2. Compute per-layer std across heads.  
3. Average over trimmed middle layers and a short decode window.  

**Calibration:**  
- Nous-Capybara-7B → steps 1-3  
- Mistral-7B-Instruct → steps 11-26  
- Threshold ~0.13-0.14 (Thresholds appear to fall around 0.13-0.14 in our empirical runs)
**Reproducibility:**  
- Deterministic decoding (`temperature=0`, `seed=1000003`).  
- Logs: entropy, top-1 margin, head-head correlation.  

---

## 9. Limitations  

- This is likely a statistical representaiton of the **distraction effect**, (Hung et all), without important head analysis, the AUROC appears to be slightly diminished. 
- **Model-specific calibration** (windows/thresholds).  
- Synthetic datasets; limited attack types.  
- Only 2 model families tested so far.  
- Std across heads on system-share is one design; other coordination measures (e.g., pairwise correlations, KL) may be better.

---

## 10. Future Work  

- Validate across broader architectures (Qwen, GPT-OSS, etc.).  
- Directly compare with *Attention Tracker* to establish complementarity/ordering.  
- Explore normalized metrics for cross-model universality.  
- Test correlation with jailbreak success rates in the wild.  
- Study length-dependent sliding windows and adaptive logic.  
- Deploy **instability + important-head** signals jointly as layered guardrails.  
- Visualize per-layer “debates” to map **safety** vs. **compliance** heads ([Wang et al., 2024](https://arxiv.org/abs/2407.01599)).

---

## 11. Workflow

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

To discover the optimal window, threshold, and layer slice run:
```
python gather.py --model models/Nous-Capybara-7B-V1.9 --system-prompt-file system_prompts/sys_prompt_probe.txt --attacks-prompts-file datasets/custom_dataset_attacks_probe.txt --benign-prompts-file datasets/custom_dataset_benign_probe.txt --outputs-root outputs/nous

python analyze_thresholds.py --attacks-root outputs/nous/attacks --benign-root outputs/nous/benign
```

Example Output (Nous)
```
Loaded runs: 100 | attacks=50 | benign=50
[auto-window] Tmax=159 → starts=[1, 2, 3, 4, 5]...[18, 19, 20], ends=[4, 5, 6, 7, 8]...[78, 79, 80]

=== Recommended operating points ===

-- Target FPR ≤ 1% --
start_step    = 1
end_step      = 7
mid_high_frac = 0.250
tail_cut_frac = 0.100
threshold     = 0.135097
achieved TPR  = 0.860
achieved FPR  = 0.000
AUROC         = 0.978
F1_at_thr     = 0.939

-- Target FPR ≤ 5% --
start_step    = 2
end_step      = 8
mid_high_frac = 0.250
tail_cut_frac = 0.100
threshold     = 0.096605
achieved TPR  = 0.940
achieved FPR  = 0.020
AUROC         = 0.972
F1_at_thr     = 0.959

-- Target FPR ≤ 10% --
start_step    = 17
end_step      = 46
mid_high_frac = 0.250
tail_cut_frac = 0.050
threshold     = 0.065100
achieved TPR  = 1.000
achieved FPR  = 0.093
AUROC         = 0.945
F1_at_thr     = 0.941
```

Example Output (Mistral)
```
Loaded runs: 100 | attacks=50 | benign=50
[auto-window] Tmax=159 → starts=[1, 2, 3, 4, 5]...[18, 19, 20], ends=[4, 5, 6, 7, 8]...[78, 79, 80]

=== Recommended operating points ===

-- Target FPR ≤ 1% --
start_step    = 10
end_step      = 26
mid_high_frac = 0.200
tail_cut_frac = 0.050
threshold     = 0.148950
achieved TPR  = 0.640
achieved FPR  = 0.000
AUROC         = 0.918
F1_at_thr     = 0.869

-- Target FPR ≤ 5% --
start_step    = 3
end_step      = 16
mid_high_frac = 0.200
tail_cut_frac = 0.150
threshold     = 0.124356
achieved TPR  = 0.860
achieved FPR  = 0.040
AUROC         = 0.895
F1_at_thr     = 0.905

-- Target FPR ≤ 10% --
start_step    = 2
end_step      = 35
mid_high_frac = 0.200
tail_cut_frac = 0.150
threshold     = 0.128252
achieved TPR  = 0.880
achieved FPR  = 0.100
AUROC         = 0.925
F1_at_thr     = 0.889
```

To run an evaluation dataset with these values:
```
python detect_head.py --system-prompt-file system_prompts/sys_prompt_generic_safety.txt --test-prompts-file datasets/custom_dataset_attacks.txt --benign-prompts-file datasets/custom_dataset_benign.txt --model models/Nous-Capybara-7B-V1.9 --threshold 0.096605 --window-start 2 --window-end 8 --mid-high-frac 0.250 --tail-cut-frac 0.100
```