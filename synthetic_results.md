These are historic results form toy instructions and synthetic datasets. They are being kept for historic purposes.

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