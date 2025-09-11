#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os, re, json
from datetime import datetime
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import torch

# Headless matplotlib only if you want optional quick visuals later
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: F401

from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------- Config (variance = OFF for clean signal) ----------------
DEFAULT_MODEL = "Nous-Capybara-7B-V1.9"
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

MAX_NEW_TOKENS_DEFAULT = 160
TEMPERATURE_DEFAULT = 0.0
TOP_K_DEFAULT = 0
TOP_P_DEFAULT: Optional[float] = None  # None disables nucleus

SEED = 1_000_003  # static dev seed


# ---------------- I/O helpers ----------------
def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

def read_prompts(path: str) -> List[str]:
    lines: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f.readlines():
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            lines.append(s)
    if not lines:
        raise ValueError(f"No prompts found in {path}")
    return lines

def slugify(s: str, maxlen: int = 60) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9._-]+", "", s)
    return s[:maxlen] if len(s) > maxlen else s


# ---------------- Prompt rendering & masking ----------------
def render_chat(tokenizer, system_prompt: str, user_prompt: str) -> str:
    """
    Prefer the model's chat template when available; otherwise fallback to a simple manual format.
    """
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # Manual fallback compatible with many instruct checkpoints
    return (
        f"<|system|>\n{system_prompt}\n"
        f"<|user|>\n{user_prompt}\n"
        f"<|assistant|>\n"
    )

def get_masks_via_offsets(tokenizer, conversation_text: str, system_prompt: str, user_prompt: str, device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build token-level masks by locating char spans of system and user bodies, then
    mapping to tokens via offsets. Robust to different chat templates.
    """
    sys_start = conversation_text.find(system_prompt)
    if sys_start == -1: sys_start = 0
    sys_end = sys_start + len(system_prompt)

    usr_start = conversation_text.find(user_prompt, sys_end)
    if usr_start == -1: usr_start = sys_end
    usr_end = usr_start + len(user_prompt)

    enc = tokenizer(conversation_text, return_tensors="pt", add_special_tokens=True, return_offsets_mapping=True)
    input_ids = enc["input_ids"].to(device)
    offsets = enc["offset_mapping"][0].tolist()

    T = input_ids.shape[1]
    sys_mask = torch.zeros((1, T), dtype=torch.bool, device=device)
    usr_mask = torch.zeros((1, T), dtype=torch.bool, device=device)

    def mark(mask, a, b):
        for i, (s, e) in enumerate(offsets):
            if s == e:  # special tokens sometimes have (0,0)
                continue
            if not (e <= a or s >= b):
                mask[0, i] = True

    mark(sys_mask, sys_start, sys_end)
    mark(usr_mask, usr_start, usr_end)
    return input_ids, sys_mask, usr_mask


# ---------------- Sampling ----------------
def want_sampling(temperature: Optional[float], top_k: int, top_p: Optional[float]) -> bool:
    if temperature is not None and temperature > 0: return True
    if top_k and top_k > 0: return True
    if top_p is not None and 0 < top_p < 1: return True
    return False

@torch.no_grad()
def _apply_top_p_mask(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    probs = torch.softmax(sorted_logits, dim=-1)
    cum = torch.cumsum(probs, dim=-1)
    keep = torch.searchsorted(cum, torch.tensor([top_p], device=logits.device)).clamp(min=1)
    mask_sorted = torch.full_like(sorted_logits, float("-inf"))
    for r in range(sorted_logits.size(0)):
        kc = int(keep[r])
        mask_sorted[r, :kc] = sorted_logits[r, :kc]
    masked = torch.full_like(logits, float("-inf"))
    masked.scatter_(dim=-1, index=sorted_idx, src=mask_sorted)
    return masked

@torch.no_grad()
def sample_from_logits(logits: torch.Tensor, temperature: float, top_k: int, top_p: Optional[float]) -> int:
    if (temperature is None or temperature <= 0) and (not top_k or top_k <= 0) and (top_p is None or not (0 < top_p < 1)):
        return int(torch.argmax(logits, dim=-1).item())

    if temperature and temperature > 0:
        logits = logits / temperature
    if top_k and top_k > 0:
        k = min(top_k, logits.shape[-1])
        vals, idx = torch.topk(logits, k=k, dim=-1)
        mask = torch.full_like(logits, float("-inf"))
        mask.scatter_(dim=-1, index=idx, src=vals)
        logits = mask
    if top_p is not None and 0 < top_p < 1:
        logits = _apply_top_p_mask(logits, top_p)

    probs = torch.softmax(logits, dim=-1)
    tok = torch.multinomial(probs, num_samples=1)
    return int(tok.item())


# ---------------- Metrics: entropy & head agreement ----------------
@torch.no_grad()
def _entropy_from_logits(logits_row: torch.Tensor) -> float:
    """Shannon entropy of the next-token distribution."""
    p = torch.softmax(logits_row, dim=-1)
    return float(-(p * torch.log(p.clamp_min(1e-12))).sum().item())

def _mean_pairwise_head_corr_over_layers(head_by_layer: List[torch.Tensor]) -> float:
    """
    head_by_layer: list of tensors [H] (system-share per head) for each layer at a single decode step.
    Returns mean off-diagonal head–head correlation across layers, using ALL layers.
    """
    if not head_by_layer:
        return float("nan")
    try:
        X = torch.stack(head_by_layer, dim=0)  # [L, H]
    except Exception:
        return float("nan")
    if X.ndim != 2:
        return float("nan")
    L, H = X.shape
    if L < 1 or H < 2:
        return float("nan")
    # Normalize per head across layers
    X = X - X.mean(dim=0, keepdim=True)
    X = X / (X.std(dim=0, keepdim=True) + 1e-6)
    # Correlation (H x H)
    C = (X.transpose(0,1) @ X) / max(1, (L))
    off = (C.sum() - torch.diagonal(C).sum()) / max(1, (H * H - H))
    return float(off.item())


# ---------------- Core observer ----------------
@torch.no_grad()
def observe_run(
    model,
    tokenizer,
    system_prompt: str,
    user_prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: Optional[float],
    outputs_dir: str,
    save_per_head: bool = True,
    save_csv: bool = True,
    save_npy: bool = True,
) -> str:
    os.makedirs(outputs_dir, exist_ok=True)

    convo_text = render_chat(tokenizer, system_prompt, user_prompt)
    input_ids, sys_mask, usr_mask = get_masks_via_offsets(tokenizer, convo_text, system_prompt, user_prompt, model.device)
    attention_mask = torch.ones_like(input_ids, device=model.device, dtype=torch.long)

    # Prefill
    pre = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        output_attentions=True,
        return_dict=True,
    )
    past = pre.past_key_values
    num_layers = len(pre.attentions) if pre.attentions is not None else 0
    num_heads = pre.attentions[0].shape[1] if pre.attentions is not None and num_layers > 0 else 0

    # Prefill: per-layer mean system/user share (last prompt token) + per-layer per-head system share
    pre_sys_col, pre_usr_col = [], []
    pre_sys_heads: List[np.ndarray] = []  # [H] per layer

    if pre.attentions is not None:
        K = pre.attentions[0].shape[-1]
        sys_idx = sys_mask[0, :K].nonzero(as_tuple=False).squeeze(-1)
        usr_idx = usr_mask[0, :K].nonzero(as_tuple=False).squeeze(-1)
        for L in range(num_layers):
            attn = pre.attentions[L]                # [1,H,T,T]
            last_q_heads = attn[0, :, -1, :]       # [H,K]
            denom = last_q_heads.sum(dim=-1, keepdim=True).clamp_min(1e-9)
            head_sys_share = (last_q_heads[:, sys_idx].sum(dim=-1, keepdim=True) / denom) if sys_idx.numel() else torch.zeros((num_heads,1), device=model.device)
            head_usr_share = (last_q_heads[:, usr_idx].sum(dim=-1, keepdim=True) / denom) if usr_idx.numel() else torch.zeros((num_heads,1), device=model.device)
            pre_sys_col.append(float(head_sys_share.mean().item()))
            pre_usr_col.append(float(head_usr_share.mean().item()))
            pre_sys_heads.append(head_sys_share.squeeze(-1).detach().cpu().numpy())

    pre_sys_col = np.array(pre_sys_col, dtype=np.float32)[:, None] if pre_sys_col else np.zeros((num_layers, 0), dtype=np.float32)
    pre_usr_col = np.array(pre_usr_col, dtype=np.float32)[:, None] if pre_usr_col else np.zeros((num_layers, 0), dtype=np.float32)

    # First token
    logits0 = pre.logits[:, -1, :]
    first_tok = sample_from_logits(logits0, temperature, top_k, top_p)
    generated: List[int] = [first_tok]
    token_ids: List[int] = [first_tok]
    last_input = torch.tensor([[first_tok]], device=model.device)

    # Step loop: collect per-layer means, per-head shares, entropy, head agreement
    step_sys_means: List[List[float]] = []
    step_usr_means: List[List[float]] = []
    per_head_layers: List[List[np.ndarray]] = [ [] for _ in range(num_layers) ]  # each entry: list of [H] for steps
    entropy_steps: List[float] = []
    head_agree_steps: List[float] = []

    eos_id = tokenizer.eos_token_id
    steps_emitted = 1  # we already emitted first token

    while steps_emitted < max_new_tokens:
        out = model(
            input_ids=last_input,
            attention_mask=torch.ones_like(last_input, device=model.device, dtype=torch.long),
            use_cache=True,
            past_key_values=past,
            output_attentions=True,
            return_dict=True,
        )
        past = out.past_key_values
        if out.attentions is None:
            break

        # entropy (next-token)
        entropy_steps.append(_entropy_from_logits(out.logits[:, -1, :]))

        # Restrict "system share" computation to prompt keys only (not generated keys)
        K_prompt = input_ids.shape[1]
        sys_idx = sys_mask[0, :K_prompt].nonzero(as_tuple=False).squeeze(-1)
        usr_idx = usr_mask[0, :K_prompt].nonzero(as_tuple=False).squeeze(-1)

        sys_means_l, usr_means_l = [], []
        head_vectors_for_agreement: List[torch.Tensor] = []  # builds [L,H] for this step

        for L in range(num_layers):
            attn = out.attentions[L]                   # [1,H,1,K_total]
            head_rows = attn[0, :, 0, :K_prompt]       # [H,K_prompt]
            denom = head_rows.sum(dim=-1, keepdim=True).clamp_min(1e-9)
            head_sys_share = (head_rows[:, sys_idx].sum(dim=-1, keepdim=True) / denom) if sys_idx.numel() else torch.zeros((num_heads,1), device=model.device)
            head_usr_share = (head_rows[:, usr_idx].sum(dim=-1, keepdim=True) / denom) if usr_idx.numel() else torch.zeros((num_heads,1), device=model.device)

            sys_means_l.append(float(head_sys_share.mean().item()))
            usr_means_l.append(float(head_usr_share.mean().item()))
            vec = head_sys_share.squeeze(-1)  # [H]
            per_head_layers[L].append(vec.detach().cpu().numpy())
            head_vectors_for_agreement.append(vec)

        step_sys_means.append(sys_means_l)
        step_usr_means.append(usr_means_l)

        # head agreement (mean pairwise correlation across heads over ALL layers)
        head_agree_steps.append(_mean_pairwise_head_corr_over_layers(head_vectors_for_agreement))

        tok = sample_from_logits(out.logits[:, -1, :], temperature, top_k, top_p)
        generated.append(tok); token_ids.append(tok)
        if eos_id is not None and tok == eos_id:
            break
        last_input = torch.tensor([[tok]], device=model.device)
        steps_emitted += 1

    # Assemble [layers, time]
    sys_steps = np.array(step_sys_means, dtype=np.float32).T if step_sys_means else np.zeros((num_layers, 0), dtype=np.float32)
    usr_steps = np.array(step_usr_means, dtype=np.float32).T if step_usr_means else np.zeros((num_layers, 0), dtype=np.float32)
    sys_all = np.concatenate([pre_sys_col, sys_steps], axis=1) if pre_sys_col.size else sys_steps
    usr_all = np.concatenate([pre_usr_col, usr_steps], axis=1) if pre_usr_col.size else usr_steps
    diff_all = np.clip(sys_all - usr_all, -1.0, 1.0)

    # Per-head matrices: head_mats[L] -> [H, T]
    head_mats: List[np.ndarray] = []
    for L in range(num_layers):
        if len(per_head_layers[L]) == 0:
            head_mats.append(np.zeros((num_heads, 0), dtype=np.float32))
        else:
            mat = np.stack(per_head_layers[L], axis=1)  # [H, steps]
            if len(pre_sys_heads) == num_layers and pre_sys_heads[L].shape[0] == num_heads:
                mat = np.concatenate([pre_sys_heads[L].reshape(num_heads, 1), mat], axis=1)
            head_mats.append(mat)

    # Inter-head instability (std across heads per layer per step) and mean across heads
    if head_mats and head_mats[0].size > 0:
        layer_std_series = [m.std(axis=0, ddof=0) for m in head_mats]   # each [T]
        layer_mean_series = [m.mean(axis=0) for m in head_mats]         # each [T]
        H_std = np.stack(layer_std_series, axis=0)                      # [L, T]
        H_mean = np.stack(layer_mean_series, axis=0)                    # [L, T]
    else:
        H_std = np.zeros((num_layers, 0), dtype=np.float32)
        H_mean = np.zeros((num_layers, 0), dtype=np.float32)

    # Stepwise series: entropy & head agreement
    entropy_series = np.array([np.nan] + entropy_steps, dtype=np.float32) if entropy_steps else np.array([], dtype=np.float32)
    headcorr_series = np.array([np.nan] + head_agree_steps, dtype=np.float32) if head_agree_steps else np.array([], dtype=np.float32)

    # Decode output & tokens
    gen_text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    tokens_decoded = [tokenizer.decode([tid], skip_special_tokens=False) for tid in token_ids]

    # --- Save artifacts ---
    with open(os.path.join(outputs_dir, "output.txt"), "w", encoding="utf-8") as f:
        f.write(gen_text + "\n")

    meta = {
        "system_prompt_excerpt": (system_prompt[:120] + ("..." if len(system_prompt) > 120 else "")),
        "user_prompt": user_prompt,
        "device": str(model.device),
        "dtype": str(DTYPE),
        "num_layers": int(num_layers),
        "num_heads": int(num_heads),
        "Tcols": int(H_std.shape[1]),
        "gen_config": {
            "max_new_tokens": int(max_new_tokens),
            "temperature": float(temperature),
            "top_k": int(top_k),
            "top_p": None if top_p is None else float(top_p),
        },
    }
    with open(os.path.join(outputs_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # CSVs (full layers, full steps)
    def dump_csv(arr: np.ndarray, name: str, header: bool=False, index: bool=False):
        pd.DataFrame(arr).to_csv(os.path.join(outputs_dir, name), index=index, header=header)

    dump_csv(sys_all,  "system_share.csv")
    dump_csv(usr_all,  "user_share.csv")
    dump_csv(diff_all, "diff_sys_minus_user.csv")
    dump_csv(H_std,    "inter_head_std.csv")
    dump_csv(H_mean,   "inter_head_mean.csv")

    if entropy_series.size:
        pd.DataFrame({"step": np.arange(entropy_series.size), "entropy": entropy_series}).to_csv(
            os.path.join(outputs_dir, "entropy_series.csv"), index=False
        )
    if headcorr_series.size:
        pd.DataFrame({"step": np.arange(headcorr_series.size), "head_agreement": headcorr_series}).to_csv(
            os.path.join(outputs_dir, "head_agreement_series.csv"), index=False
        )

    # Tokens timeline
    pd.DataFrame({"step": np.arange(1, len(token_ids)+1), "token_id": token_ids, "decoded": tokens_decoded}).to_csv(
        os.path.join(outputs_dir, "tokens_timeline.csv"), index=False
    )

    # Optional: per-layer per-head system share matrices
    if save_per_head and head_mats and head_mats[0].size > 0:
        for L, mat in enumerate(head_mats):
            pd.DataFrame(mat).to_csv(os.path.join(outputs_dir, f"heads_system_share_layer{L:02d}.csv"), index=False, header=False)

    # Also save as .npy for faster analysis if desired
    if save_npy:
        np.save(os.path.join(outputs_dir, "system_share.npy"), sys_all)
        np.save(os.path.join(outputs_dir, "user_share.npy"), usr_all)
        np.save(os.path.join(outputs_dir, "diff_sys_minus_user.npy"), diff_all)
        np.save(os.path.join(outputs_dir, "inter_head_std.npy"), H_std)
        np.save(os.path.join(outputs_dir, "inter_head_mean.npy"), H_mean)
        if entropy_series.size:
            np.save(os.path.join(outputs_dir, "entropy_series.npy"), entropy_series)
        if headcorr_series.size:
            np.save(os.path.join(outputs_dir, "head_agreement_series.npy"), headcorr_series)

    return gen_text


# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser(description="Gather full-layer observability: system/user shares, inter-head std/mean, entropy, head agreement, tokens.")
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL)
    ap.add_argument("--system-prompt-file", type=str, required=True)
    ap.add_argument("--benign-prompts-file", type=str, default=True)
    ap.add_argument("--attacks-prompts-file", type=str, default=True)
    ap.add_argument("--outputs-root", type=str, required=True)
    ap.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS_DEFAULT)
    ap.add_argument("--temperature", type=float, default=TEMPERATURE_DEFAULT)
    ap.add_argument("--top-k", type=int, default=TOP_K_DEFAULT)
    ap.add_argument("--top-p", type=float, default=-1.0, help="-1 disables; else 0<top_p<=1")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    # Seed & determinism
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Load prompts
    system_prompt = read_text_file(args.system_prompt_file)
    prompts_benign = read_prompts(args.benign_prompts_file)
    prompts_attacks = read_prompts(args.attacks_prompts_file)

    print(f"Loading model: {args.model} on {DEVICE} (dtype={DTYPE})")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False, trust_remote_code=True)

    # Make sure we get attentions
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=DTYPE,
            attn_implementation="eager",
            trust_remote_code=True,
        ).to(DEVICE)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=DTYPE,
            trust_remote_code=True,
        ).to(DEVICE)
    model.eval()

    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token_id", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token

    max_new_tokens = int(args.max_new_tokens)
    temperature = float(args.temperature)
    top_k = int(args.top_k)
    top_p = None if args.top_p < 0 else float(args.top_p)

    # Run
    for i, user_prompt in enumerate(prompts_benign, 1):
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        slug = slugify(user_prompt[:80]) or f"prompt_{i:03d}"
        out_dir = os.path.join(args.outputs_root, "benign", f"{slug}__{ts}")
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n--- benign {i}/{len(prompts_benign)}")
        print(f"[User] {user_prompt}")
        print(f"[Outputs] {out_dir}")

        text = observe_run(
            model, tokenizer, system_prompt, user_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            outputs_dir=out_dir,
            save_per_head=True,
            save_csv=True,
            save_npy=True,
        )
        print(f"[Assistant] {text[:180]}{'...' if len(text)>180 else ''}")
        print("Saved: system/user shares, diff, inter-head std/mean, entropy, head agreement, per-head per-layer grids, tokens timeline, meta.")
    
    for i, user_prompt in enumerate(prompts_attacks, 1):
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        slug = slugify(user_prompt[:80]) or f"prompt_{i:03d}"
        out_dir = os.path.join(args.outputs_root, "attacks", f"{slug}__{ts}")
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n--- attacks {i}/{len(prompts_attacks)}")
        print(f"[User] {user_prompt}")
        print(f"[Outputs] {out_dir}")

        text = observe_run(
            model, tokenizer, system_prompt, user_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            outputs_dir=out_dir,
            save_per_head=True,
            save_csv=True,
            save_npy=True,
        )
        print(f"[Assistant] {text[:180]}{'...' if len(text)>180 else ''}")
        print("Saved: system/user shares, diff, inter-head std/mean, entropy, head agreement, per-head per-layer grids, tokens timeline, meta.")

if __name__ == "__main__":
    main()
