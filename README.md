# Prompt Injection Defense via Attention Head Instability

## Note From the Author
> **This repo is a hobbyist weekend experiment.**  
> I’m not an ML engineer — this is just my attempt to see whether a simple, lightweight scoring heuristic could catch prompt injection attempts in real time.  
>
> It’s absolutely not novel research. I probably misunderstood or messed up parts of the testing. But since I found it fun and others might too, I’m sharing it here.  
>
> **TL;DR**: It computes a basic “head instability” score from model attentions during early decoding. If the score is high, it blocks the output. That’s it. It’s fragile, tuned to a single instruction (“don’t say `test`”), and shouldn’t be used in production.  

---

## Why this exists

After [messing around with heuristics](https://github.com/g4m817/llm-instruction-adherence) I decided to go back to the drawing board, which lead to collecting graphs and statistics during inference to [measure attention across layers](https://github.com/g4m817/llm-gather-inference-metrics). Some interesting metrics surfaced, which I then turned (with ChatGPT's help) into a PoC, this is not novel, its just doing things a bit different than others, for a more refined, non-hobbyist, approach check out [Zou et al., 2024 — *\\attn: Attention-based prompt injection detection*](https://arxiv.org/html/2411.00348v1)


Prompt injection / jailbreak defenses are usually big, complex systems (extra classifiers, judges, retraining). I wanted to try something **tiny and self-contained**:  
- No retraining  
- No external classifier  
- Just a heuristic, in generation  

The question that sparked it was:

> *“Is there something interestingt that surfaces in attention across layers with prompt injections versus benign prompt?”*

This repo is the result of me probing and analyzing the results here: https://github.com/g4m817/llm-gather-inference-metrics

---

## What it does (high-level)

- **System prompt:** A strict instruction — *“Never say the forbidden term.”*  
- **Collect attentions:** During the first few decode steps, capture the model’s per-head attentions.  
- **Compute instability:** Measure how dispersed the “system share” is across heads (std dev across heads).  
- **Aggregate:** Average across mid/high layers and early steps.  
- **Threshold:** If instability ≥ `0.13`, block the generation. Otherwise, let it through.  

---

## ⚠️ Big disclaimers

- **Not production-ready.** This is just lab tinkering.  
- **Single instruction.** Only tested with “don’t say `test`.”  
- **Small, hand-crafted data.** 50 benign Qs, 50 adversarial Qs.  
- **Tuned to fit.** Numbers look nice because threshold `0.13` and lookback `5` were hand-picked.  
- **Fragile.** Different instructions, models, or prompts will probably break it.  
- **Slower generation.** Needs `output_attentions=True`, which adds overhead.  

---

## Results (our exact lab)

**Model**: `Nous-Capybara-7B-V1.9`  
**Device**: GPU (fp16)  
**Config**:  
- `max_new_tokens=160`  
- `temperature=0.7`  
- `top_k=50`  
- `do_sample=True`  
- `attn_implementation="eager"`  
- **Threshold:** `0.13`  
- **Lookback steps:** `5`

### Unpatched (malicious prompts)
- **Total:** 150  
- **Passes:** 84  
- **Fails:** 66  
- **Attack success rate:** **44%**

### Patched (malicious prompts)
- **Total:** 150  
- **Passes:** 133  
- **Fails:** 17  
- **Attack success rate:** **11.33%**

### Patched (benign prompts)
- **Total:** 150  
- **Benign accepted:** 145  
- **False positives:** 5  
- **Pass rate:** **96.67%**

> **Reminder:** These are narrow, lab-specific numbers on a toy instruction (‘don’t say test’). They do not imply general robustness.

---

## How it works (technical sketch)

- Build a conversation with `<|system|> … <|user|> … <|assistant|>`.  
- Prefill the model and then decode the first 5 steps with `output_attentions=True`.  
- For each step/layer:
  - Compute the fraction of attention heads spend on **system tokens**.  
  - Take the **std dev across heads**.  
- Aggregate across mid/high layers.  
- If above threshold → block output.  

---

## Reproducibility notes

Results are **tightly coupled** to:
- Model weights: `Nous-Capybara-7B-V1.9`  
- Config: `--threshold 0.13 --lookback-steps 5`  
- System prompt: forbids the word “test”  
- Dataset: 50 benign Qs, 50 adversarial Qs (see code)  
- Single-turn chats only  

---

## Quickstart

```bash
# 1) Install
pip install -U torch transformers

# 2) Run patched test (malicious prompts)
python main.py --mode tests --iterations 3 --threshold 0.13 --lookback-steps 5

# 3) Run benign prompts (check false positives)
python main.py --mode baseline --iterations 3 --threshold 0.13 --lookback-steps 5
```

## Influences

- [Zou et al., 2024 — *\\attn: Attention-based prompt injection detection*](https://arxiv.org/html/2411.00348v1)
- [Wang et al., 2025 — *AttentionDefense: Training-free adversarial prompt detection*](https://arxiv.org/pdf/2504.12321)  
- [Wang et al., 2024 — *DETAM: Dynamic Attention Map Defense*](https://arxiv.org/abs/2504.13562) 
