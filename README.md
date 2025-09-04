# Inter-Head Instability: Disagreement Among Attention Heads

> DISCLAIMER: This is exploratory work. I am not an ML engineer, I’m a security engineer who noticed this signal while tinkering with prompt injection defenses. These experiments may be wrong or incomplete, but I wanted to document them so others, especially researchers, can validate, refine, or discard.

This repo contains exploratory experiments showing that **attention heads often disagree when faced with adversarial input**, and that this may provide a useful signal of prompt injection attempts. Although, more research needs to be conducted to determine if its useful across datasets / instructions and model families.

Even if this instability signal proves too noisy or imprecise to serve as a practical security measure, it may still be useful for interpretability. In particular, the instability windows may highlight when during decoding a model is actively processing system instructions, when its heads disagree on that focus, and when it stabilizes toward either following those instructions or drifting toward adversarial input.

---

## 1. Overview
Recent work on attention interpretability has described the distraction effect ([Attention Tracker: Detecting Prompt Injection Attacks in LLMs (Hung, Ko, Rawat, Chung, Hsu, and Chen, 2024)](https://arxiv.org/html/2411.00348v1)), where certain attention heads shift from system tokens toward injected tokens. This repo explores what may be a complementary signal: inter-head instability. Instead of drifting together, attention heads often disagree, some cling to the system prompt, others wander. This disagreement shows up as increased variance across heads within specific decoding windows. It may represent an early or system-level footprint of the distraction effect, though the relationship is not yet clear.

This disagreement might ultimately be just an interpretability curiosity, but so far it shows up consistently in particular decoding windows across system prompts and datasets. Measuring variance across heads in their focus on system tokens provides a simple statistical lens on when the model’s internal attention patterns become unstable. This suggests the possibility of flagging early signs of the model being pushed off-course, but it remains unclear whether the signal generalizes broadly or is robust enough for practical detection.

There are, of course, outliers. Some benign prompts confuse the model’s heads, and some adversarial prompts slip through without measurable disagreement. This is expected, the underlying mechanism of why heads agree or disagree remains a black box. At this stage, the safest claim is that inter-head divergence correlates with adversarial prompting in certain settings, but it is not a reliable or universal discriminator.

---

## 2. Method
- **Metric:** Std dev across heads of attention to system tokens, per layer/step.  
- **Why the chosen step windows:** Different models *appear* to reach instability at different windows before ultimately stabilizing, these windows appear to be different per model.
- **Windows:**  
  - Nous-Capybara-7B → steps **1–3**.  
  - Mistral-7B → steps **11–40**.  
- **Layer trimming:** Drop shallowest 25% and deepest 15%. 
- **Approach:** Take the std of heads each layer of a step, then average them out to arrive at a suspicion score, gated by a threshold informed by data (generally, 0.14). This number has appeared across two model families producing the best mitigation, with the least false positives (1-5% FPR).
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
50 Attack prompts
50 Benign prompts
3 Iterations (total 300 prompts)
(System prompt: *Never say "test"*)  

#### Nous (steps 1–3) vs. Mistral (steps 11–40)  
- Both models show clear separation of attack vs. benign.  
- Nous signal emerges immediately; Mistral signal ramps later but reaches similar AUROC.  

**Figures:**  
###### ROC
![ROC Nous](figs/models_Nous-Capybara-7B-V1.9_sys_prompt_never_say_test.txt_custom_dataset_benign.txt/roc.png)
![ROC Mistral](figs/models_Mistral-7B-Instruct-v0.3_sys_prompt_never_say_test.txt_custom_dataset_benign.txt/roc.png)

---
###### Violin
![Nous Violin](figs/models_Nous-Capybara-7B-V1.9_sys_prompt_never_say_test.txt_custom_dataset_benign.txt/violin_windowed.png)
![Mistral Violin](figs/models_Mistral-7B-Instruct-v0.3_sys_prompt_never_say_test.txt_custom_dataset_benign.txt/violin_windowed.png)

---
###### Scatter
![Nous Scatter](figs/models_Nous-Capybara-7B-V1.9_sys_prompt_never_say_test.txt_custom_dataset_benign.txt/scatter_windowed.png)
![Mistral Scatter](figs/models_Mistral-7B-Instruct-v0.3_sys_prompt_never_say_test.txt_custom_dataset_benign.txt/scatter_windowed.png)

---
###### AUROC
![Nous AUROC](figs/models_Nous-Capybara-7B-V1.9_sys_prompt_never_say_test.txt_custom_dataset_benign.txt/stepwise_auroc.png)
![Mistral AUROC](figs/models_Mistral-7B-Instruct-v0.3_sys_prompt_never_say_test.txt_custom_dataset_benign.txt/stepwise_auroc.png)

---
###### Nous Benign vs Attack Heatmap
![Nous Benign Heatmap](figs/models_Nous-Capybara-7B-V1.9_sys_prompt_never_say_test.txt_custom_dataset_benign.txt/mean_heatmap_benign.png)
![Nous Attack Heatmap](figs/models_Nous-Capybara-7B-V1.9_sys_prompt_never_say_test.txt_custom_dataset_benign.txt/mean_heatmap_attack.png)

###### Mistral Benign vs Attack Heatmap
![Mistral Benign Heatmap](figs/models_Mistral-7B-Instruct-v0.3_sys_prompt_never_say_test.txt_custom_dataset_benign.txt/mean_heatmap_benign.png)
![Mistral Attack Heatmap](figs/models_Mistral-7B-Instruct-v0.3_sys_prompt_never_say_test.txt_custom_dataset_benign.txt/mean_heatmap_attack.png) 

---

### 📊 Dataset 2: Obfuscated Forms (*t3st, te5t, etc.)
50 Attack prompts
50 Benign prompts
3 Iterations (total 300 prompts)
(System prompt: *Never say "test"*)  

#### Nous vs. Mistral  
- Obfuscations reduce separation, but the instability signal still distinguishes attack vs. benign.  
- Nous shows noisy early window; Mistral stabilizes around step ~15.  

**Figures:**  
###### ROC
![ROC Nous](figs/models_Nous-Capybara-7B-V1.9_sys_prompt_never_say_test.txt_custom_dataset_2_benign.txt/roc.png)
![ROC Mistral](figs/models_Mistral-7B-Instruct-v0.3_sys_prompt_never_say_test.txt_custom_dataset_2_benign.txt/roc.png)

---
###### Violin
![Nous Violin](figs/models_Nous-Capybara-7B-V1.9_sys_prompt_never_say_test.txt_custom_dataset_2_benign.txt/violin_windowed.png)
![Mistral Violin](figs/models_Mistral-7B-Instruct-v0.3_sys_prompt_never_say_test.txt_custom_dataset_2_benign.txt/violin_windowed.png)

---
###### Scatter
![Nous Scatter](figs/models_Nous-Capybara-7B-V1.9_sys_prompt_never_say_test.txt_custom_dataset_2_benign.txt/scatter_windowed.png)
![Mistral Scatter](figs/models_Mistral-7B-Instruct-v0.3_sys_prompt_never_say_test.txt_custom_dataset_2_benign.txt/scatter_windowed.png)

---
###### AUROC
![Nous AUROC](figs/models_Nous-Capybara-7B-V1.9_sys_prompt_never_say_test.txt_custom_dataset_2_benign.txt/stepwise_auroc.png)
![Mistral AUROC](figs/models_Mistral-7B-Instruct-v0.3_sys_prompt_never_say_test.txt_custom_dataset_2_benign.txt/stepwise_auroc.png)

---
###### Nous Benign vs Attack Heatmap
![Nous Benign Heatmap](figs/models_Nous-Capybara-7B-V1.9_sys_prompt_never_say_test.txt_custom_dataset_2_benign.txt/mean_heatmap_benign.png)
![Nous Attack Heatmap](figs/models_Nous-Capybara-7B-V1.9_sys_prompt_never_say_test.txt_custom_dataset_2_benign.txt/mean_heatmap_attack.png)

###### Mistral Benign vs Attack Heatmap
![Mistral Benign Heatmap](figs/models_Mistral-7B-Instruct-v0.3_sys_prompt_never_say_test.txt_custom_dataset_2_benign.txt/mean_heatmap_benign.png)
![Mistral Attack Heatmap](figs/models_Mistral-7B-Instruct-v0.3_sys_prompt_never_say_test.txt_custom_dataset_2_benign.txt/mean_heatmap_attack.png) 

---

### 📊 Dataset 3: Digits / URLs  
50 Attack prompts
50 Benign prompts
3 Iterations (total 300 prompts)
(System prompt: *No digits or URLs*)

#### Nous vs. Mistral  
- New system prompt with broad lexical ban.  
- Both models show strong separation again, though signal shape differs: Nous immediate, Mistral gradual.  

**Figures:**  
###### ROC
![ROC Nous](figs/models_Nous-Capybara-7B-V1.9_sys_prompt_digits_urls.txt_custom_dataset_3_benign.txt/roc.png)
![ROC Mistral](figs/models_Mistral-7B-Instruct-v0.3_sys_prompt_digits_urls.txt_custom_dataset_3_benign.txt/roc.png)

---
###### Violin
![Nous Violin](figs/models_Nous-Capybara-7B-V1.9_sys_prompt_digits_urls.txt_custom_dataset_3_benign.txt/violin_windowed.png)
![Mistral Violin](figs/models_Mistral-7B-Instruct-v0.3_sys_prompt_digits_urls.txt_custom_dataset_3_benign.txt/violin_windowed.png)

---
###### Scatter
![Nous Scatter](figs/models_Nous-Capybara-7B-V1.9_sys_prompt_digits_urls.txt_custom_dataset_3_benign.txt/scatter_windowed.png)
![Mistral Scatter](figs/models_Mistral-7B-Instruct-v0.3_sys_prompt_digits_urls.txt_custom_dataset_3_benign.txt/scatter_windowed.png)

---
###### AUROC
![Nous AUROC](figs/models_Nous-Capybara-7B-V1.9_sys_prompt_digits_urls.txt_custom_dataset_3_benign.txt/stepwise_auroc.png)
![Mistral AUROC](figs/models_Mistral-7B-Instruct-v0.3_sys_prompt_digits_urls.txt_custom_dataset_3_benign.txt/stepwise_auroc.png)

---
###### Nous Benign vs Attack Heatmap
![Nous Benign Heatmap](figs/models_Nous-Capybara-7B-V1.9_sys_prompt_digits_urls.txt_custom_dataset_3_benign.txt/mean_heatmap_benign.png)
![Nous Attack Heatmap](figs/models_Nous-Capybara-7B-V1.9_sys_prompt_digits_urls.txt_custom_dataset_3_benign.txt/mean_heatmap_attack.png)

###### Mistral Benign vs Attack Heatmap
![Mistral Benign Heatmap](figs/models_Mistral-7B-Instruct-v0.3_sys_prompt_digits_urls.txt_custom_dataset_3_benign.txt/mean_heatmap_benign.png)
![Mistral Attack Heatmap](figs/models_Mistral-7B-Instruct-v0.3_sys_prompt_digits_urls.txt_custom_dataset_3_benign.txt/mean_heatmap_attack.png) 

---

## Limitations

- It’s possible the instability signal is not a novel effect, but rather a statistical footprint of the distraction effect (Hung et al. 2024). Further experiments are required to establish whether it provides independent information or simply reflects the same underlying mechanism.
- Results are small-scale.
- Only 3 datasets and 2 system prompts, both synthetically generated with fragile pass/fail scoring.
- Only 2 model families tested.  
- Thresholds/steps tuned per model; no universal setting yet, I suspect it may not be possible to find universal settings, at least for the instability windows identified so far.

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

#### Ensure the `models/` directory exists with the models you want to test:
```
models/Nous-Capybara-7B-V1.9
models/Mistral-7B-Instruct-v0.3
```

#### Install
```
pip install -r requirements.txt
```

#### To run the evaluations, run the helper script that runs everything and then look at the results under `runs/`:
```
chmod +x run.sh
./run.sh
```
