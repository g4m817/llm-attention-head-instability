# Inter-Head Instability: Disagreement Among Attention Heads

> DISCLAIMER: I am not an ML engineer, I have no academic or formal background in ML. I am a security engineer that was only tinkering with this as a side-project when I observed the signal. I could be completely wrong, but I'm attempting to formalize it into a digestable form to hopefully allow academics to take it forward to validate/improve/discard as meaningless. The more work I do the more I think it may just be the distraction effect re-discovered through std dev of inter-heads attention on system share. The repo is still under-construction as I run more experiments (slowly, consumer hardware, lol).

This repo contains experiments showing that **attention heads often disagree when faced with adversarial input**, and that this is a useful signal of prompt injection attempts. Although, more research needs to be conducted to determine if its useful across datasets / instructions and model families. Even if it turns out to not be a useful detection mechanism for prompt injection, it may at least provide some insight into why models attention drifts towards adversarial input during generation.

---

## 1. Overview
Recent work on attention interpretability has described a distraction effect, see: [Attention Tracker: Detecting Prompt Injection Attacks in LLMs (Hung, Ko, Rawat, Chung, Hsu, and Chen, 2024)](https://arxiv.org/html/2411.00348v1), which showed attention drifting from system tokens to adversarial ones. We are attempting to extend that view by highlighting a different signal — inter-head instability. Instead of drifting together, attention heads often disagree: some cling to the system prompt, others ignore it, fracturing the model’s internal consensus. This _may_ lead to the distraction effect, but more research is required.

This disagreement may just be an interpretability curiosity, but it shows up consistently in the right decoding windows across system prompts and datasets, so far at least. By measuring variance across heads in their focus on system tokens, we may be able reliably flag when the model is being pushed off-course, early.

Of course, there are outliers. Some benign prompts confuse the model’s heads, and some adversarial prompts slip through without disagreement. That’s expected — we’re not explaining why heads agree or disagree (that remains a black box), only observing that divergence correlates with adversarial prompting.

---

## 2. Method
- **Metric:** Std dev across heads of attention to system tokens, per layer/step.  
- **Why the chosen step windows:** Different models *appear* to reach instability at different windows before ultimately converging, these windows appear to be different per model.
- **Windows:**  
  - Nous-Capybara-7B → steps **1–3**.  
  - Mistral-7B → steps **11–40**.  
- **Layer trimming:** Drop shallowest 25% and deepest 15%. 
- **Approach:** Take the std of heads each layer of a step, then average them out to arrive at a suspcion score, gated by a threshold informed by data (generally, 0.14). This number has appeared across two model families producing the best mitigation, with the least false positives (1-5% FPR).
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

- Results are small-scale.
- Only 3 datasets and 2 system prompts, both synthetically generated with fragile pass/fail scoring.
- Only 2 model families tested.  
- Thresholds/steps tuned per model; no universal setting yet, I suspect it may not be possible to find universal settings, at least for the start/end windows.

---

## Next Steps

- Validate on larger, more diverse model families and datasets.
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

#### To run the evaluaitons, run the helper script that runs everything and then look at the results under `runs/`:
```
chmod +x run.sh
./run.sh
```
