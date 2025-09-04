#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os, re, json
from datetime import datetime
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------- Config (variance = OFF for clean signal) ----------------
DEFAULT_MODEL = "Nous-Capybara-7B-V1.9"
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

MAX_NEW_TOKENS_DEFAULT = 160
TEMPERATURE_DEFAULT = 0.0
TOP_K_DEFAULT = 0
TOP_P_DEFAULT: Optional[float] = None  # None disables nucleus

# Static dev seed
SEED = 1_000_003

# Layer slice default (you can override on CLI post-hoc in analysis)
MID_HIGH_FRAC_DEFAULT = 0.33

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
    Build token-level masks by:
      1) Finding the character spans where the exact system_prompt and user_prompt bodies appear
      2) Tokenizing the full conversation with offsets
      3) Marking tokens whose offsets overlap those char spans
    This is robust to different chat templates.
    """
    # Char spans
    sys_start = conversation_text.find(system_prompt)
    if sys_start == -1: sys_start = 0
    sys_end = sys_start + len(system_prompt)

    usr_start = conversation_text.find(user_prompt, sys_end)  # user should come after system
    if usr_start == -1: usr_start = sys_end
    usr_end = usr_start + len(user_prompt)

    enc = tokenizer(conversation_text, return_tensors="pt", add_special_tokens=True, return_offsets_mapping=True)
    input_ids = enc["input_ids"].to(device)
    offsets = enc["offset_mapping"][0].tolist()  # List[(start,end)] per token

    T = input_ids.shape[1]
    sys_mask = torch.zeros((1, T), dtype=torch.bool, device=device)
    usr_mask = torch.zeros((1, T), dtype=torch.bool, device=device)

    def mark(mask, a, b):
        for i, (s, e) in enumerate(offsets):
            if s == e:  # special tokens sometimes have (0,0)
                continue
            # overlap of [s,e) with [a,b)
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
def sample_from_logits(logits: torch.Tensor, temperature: float, top_k: int, top_p: Optional[float]) -> int:
    # With our defaults (0,0,None) this falls back to argmax via multinomial on a sharp distro
    if temperature and temperature > 0:
        logits = logits / temperature
    if top_k and top_k > 0:
        k = min(top_k, logits.shape[-1])
        topk_vals, topk_idx = torch.topk(logits, k=k, dim=-1)
        mask = torch.full_like(logits, float("-inf"))
        mask.scatter_(dim=-1, index=topk_idx, src=topk_vals)
        logits = mask
    if top_p is not None and 0 < top_p < 1:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumprobs = torch.cumsum(probs, dim=-1)
        cutoff = torch.sum(cumprobs <= top_p, dim=-1, keepdim=True).clamp_min(1)
        mask = torch.full_like(sorted_logits, float("-inf"))
        rows = torch.arange(sorted_logits.size(0), device=logits.device).unsqueeze(-1)
        mask[rows, :cutoff] = sorted_logits[rows, :cutoff]
        unsorted = torch.full_like(logits, float("-inf"))
        unsorted.scatter_(dim=-1, index=sorted_indices, src=mask)
        logits = unsorted
    probs = torch.softmax(logits, dim=-1)
    tok = torch.multinomial(probs, num_samples=1)
    return int(tok.item())

# ---------------- Plot helpers ----------------
def save_heatmap(matrix: np.ndarray, title: str, fname: str, vmin: float, vmax: float, cmap: str, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    plt.figure(figsize=(max(6, matrix.shape[1] * 0.35), max(4, matrix.shape[0] * 0.35)))
    plt.imshow(matrix, aspect='auto', interpolation='nearest', vmin=vmin, vmax=vmax, cmap=cmap)
    plt.colorbar()
    plt.xlabel("time (0 = prefill; then each generated token)")
    plt.ylabel("layer")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, fname), dpi=180)
    plt.close()

def save_tokens_timeline(tokens: List[str], outdir: str, fname: str = "tokens_timeline.png"):
    steps = list(range(len(tokens)))
    plt.figure(figsize=(max(8, len(tokens) * 0.35), 2.2))
    plt.scatter(steps, [1]*len(tokens), s=0)
    N = max(1, len(tokens)//24)
    for i, tok in enumerate(tokens):
        if i % N == 0 or i in (0, len(tokens)-1):
            t = tok.replace("\n","⏎").replace("\t","⇥")
            if len(t) > 12: t = t[:12] + "…"
            plt.text(i, 1, t, rotation=90, va="center", ha="center", fontsize=8)
    plt.yticks([])
    xt = steps[::N]
    if steps[-1] not in xt: xt = xt + [steps[-1]]
    plt.xticks(xt)
    plt.xlabel("time (0 = prefill; then each generated token)")
    plt.title("Prefill + Generated tokens")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, fname), dpi=180)
    plt.close()

def decode_tokens(tokenizer, ids: List[int]) -> List[str]:
    toks = []
    for tid in ids:
        s = tokenizer.decode([tid], skip_special_tokens=False)
        s = s if s.strip() != "" else repr(s)
        toks.append(s)
    return toks

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
    mid_high_frac: float = MID_HIGH_FRAC_DEFAULT,
    save_per_head: bool = True,
    save_csv: bool = True,
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

    # Prefill per-layer average system/user shares at last prompt token
    pre_sys_col, pre_usr_col = [], []
    pre_sys_heads: List[np.ndarray] = []  # [H] per layer, system share at prefill

    if pre.attentions is not None:
        K = pre.attentions[0].shape[-1]
        sys_idx = sys_mask[0, :K].nonzero(as_tuple=False).squeeze(-1)
        usr_idx = usr_mask[0, :K].nonzero(as_tuple=False).squeeze(-1)
        for L in range(num_layers):
            attn = pre.attentions[L]                # [1,H,T,T]
            last_q_heads = attn[0, :, -1, :]       # [H,K]
            denom = last_q_heads.sum(dim=-1, keepdim=True).clamp_min(1e-9)
            if sys_idx.numel():
                head_sys_share = last_q_heads[:, sys_idx].sum(dim=-1, keepdim=True) / denom
            else:
                head_sys_share = torch.zeros((num_heads, 1), device=model.device)
            if usr_idx.numel():
                head_usr_share = last_q_heads[:, usr_idx].sum(dim=-1, keepdim=True) / denom
            else:
                head_usr_share = torch.zeros((num_heads, 1), device=model.device)

            pre_sys_col.append(float(head_sys_share.mean().item()))
            pre_usr_col.append(float(head_usr_share.mean().item()))
            pre_sys_heads.append(head_sys_share.squeeze(-1).detach().cpu().numpy())

    pre_sys_col = np.array(pre_sys_col, dtype=np.float32)[:, None] if pre_sys_col else np.zeros((num_layers, 0), dtype=np.float32)
    pre_usr_col = np.array(pre_usr_col, dtype=np.float32)[:, None] if pre_usr_col else np.zeros((num_layers, 0), dtype=np.float32)

    # First token
    logits0 = pre.logits[:, -1, :]
    first_tok = sample_from_logits(logits0, temperature, top_k, top_p)
    generated = [first_tok]
    last_input = torch.tensor([[first_tok]], device=model.device)

    # Step loop
    step_sys, step_usr = [], []
    per_head_layers: List[List[np.ndarray]] = [ [] for _ in range(num_layers) ]  # each: list of [H] per step

    eos_id = tokenizer.eos_token_id
    for _ in range(max_new_tokens - 1):
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

        K_prompt = input_ids.shape[1]
        sys_idx = sys_mask[0, :K_prompt].nonzero(as_tuple=False).squeeze(-1)
        usr_idx = usr_mask[0, :K_prompt].nonzero(as_tuple=False).squeeze(-1)

        sys_sh_l, usr_sh_l = [], []
        for L in range(num_layers):
            attn = out.attentions[L]                   # [1,H,1,K_total]
            head_rows = attn[0, :, 0, :K_prompt]       # [H,K_prompt]
            denom = head_rows.sum(dim=-1, keepdim=True).clamp_min(1e-9)
            if sys_idx.numel():
                head_sys_share = head_rows[:, sys_idx].sum(dim=-1, keepdim=True) / denom
            else:
                head_sys_share = torch.zeros((num_heads, 1), device=model.device)
            if usr_idx.numel():
                head_usr_share = head_rows[:, usr_idx].sum(dim=-1, keepdim=True) / denom
            else:
                head_usr_share = torch.zeros((num_heads, 1), device=model.device)

            sys_sh_l.append(float(head_sys_share.mean().item()))
            usr_sh_l.append(float(head_usr_share.mean().item()))
            per_head_layers[L].append(head_sys_share.squeeze(-1).detach().cpu().numpy())  # [H]

        step_sys.append(sys_sh_l)
        step_usr.append(usr_sh_l)

        tok = sample_from_logits(out.logits[:, -1, :], temperature, top_k, top_p)
        generated.append(tok)
        if eos_id is not None and tok == eos_id:
            break
        last_input = torch.tensor([[tok]], device=model.device)

    # Assemble [layers, time]
    sys_steps = np.array(step_sys, dtype=np.float32).T if step_sys else np.zeros((num_layers, 0), dtype=np.float32)
    usr_steps = np.array(step_usr, dtype=np.float32).T if step_usr else np.zeros((num_layers, 0), dtype=np.float32)
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

    # Inter-head instability H_std (std across heads) and mean across heads
    if head_mats and head_mats[0].size > 0:
        layer_std_series = [m.std(axis=0) for m in head_mats]   # each [T]
        layer_mean_series = [m.mean(axis=0) for m in head_mats] # each [T]
        H_std = np.stack(layer_std_series, axis=0)              # [L, T]
        H_mean = np.stack(layer_mean_series, axis=0)            # [L, T]
    else:
        H_std = np.zeros((num_layers, 0), dtype=np.float32)
        H_mean = np.zeros((num_layers, 0), dtype=np.float32)

    # Mid/high slice & cumulative rollups (skip prefill column 0 for cumulative)
    Tcols = H_std.shape[1]
    Llayers = H_std.shape[0]
    mid_start = max(2, int(Llayers * mid_high_frac))
    H_mid = H_std[mid_start:, :] if Llayers > mid_start else H_std

    per_step_mean_std_all = H_std.mean(axis=0) if Tcols else np.zeros((0,), dtype=np.float32)
    per_step_mean_std_mid = H_mid.mean(axis=0) if Tcols else np.zeros((0,), dtype=np.float32)

    cum_all, cum_mid = [], []
    for k in range(1, max(1, Tcols)):  # steps 1..T-1 (exclude prefill)
        cum_all.append(per_step_mean_std_all[1:k+1].mean())
        cum_mid.append(per_step_mean_std_mid[1:k+1].mean())
    cum_all = np.array(cum_all, dtype=np.float32)
    cum_mid = np.array(cum_mid, dtype=np.float32)

    # Decode output & tokens
    gen_text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    token_strings = ["<PF>"] + decode_tokens(tokenizer, generated)

    # ---- Save artifacts
    save_heatmap(sys_all,  "Attention share to SYSTEM tokens", "attn_share_system.png", vmin=0.0, vmax=1.0, cmap="Blues",   outdir=outputs_dir)
    save_heatmap(usr_all,  "Attention share to USER tokens",   "attn_share_user.png",   vmin=0.0, vmax=1.0, cmap="Oranges", outdir=outputs_dir)
    save_heatmap(diff_all, "SYSTEM minus USER attention share","attn_share_diff_sys_minus_user.png", vmin=-1.0, vmax=1.0, cmap="coolwarm", outdir=outputs_dir)
    save_heatmap(H_std,    "Inter-head instability (std across heads)", "inter_head_std_heatmap.png", vmin=0.0, vmax=0.5, cmap="viridis", outdir=outputs_dir)

    # Per-head grids
    if save_per_head and head_mats and head_mats[0].size > 0:
        for L, mat in enumerate(head_mats):
            plt.figure(figsize=(max(6, mat.shape[1]*0.35), max(3, mat.shape[0]*0.3)))
            plt.imshow(mat, aspect='auto', interpolation='nearest', vmin=0.0, vmax=1.0, cmap="Blues")
            plt.colorbar()
            plt.xlabel("time (0 = prefill; then each generated token)")
            plt.ylabel("head")
            plt.title(f"System share per-head — layer {L}")
            plt.tight_layout()
            plt.savefig(os.path.join(outputs_dir, f"heads_system_share_layer{L:02d}.png"), dpi=180)
            plt.close()

    save_tokens_timeline(token_strings, outputs_dir)

    with open(os.path.join(outputs_dir, "output.txt"), "w", encoding="utf-8") as f:
        f.write(gen_text + "\n")
    meta = {
        "system_prompt_excerpt": (system_prompt[:120] + ("..." if len(system_prompt) > 120 else "")),
        "user_prompt": user_prompt,
        "device": str(model.device),
        "dtype": str(DTYPE),
        "num_layers": int(num_layers),
        "num_heads": int(num_heads),
        "Tcols": int(Tcols),
        "mid_high_frac": float(mid_high_frac),
        "mid_start_layer_index": int(mid_start),
        "gen_config": {
            "max_new_tokens": int(max_new_tokens),
            "temperature": float(temperature),
            "top_k": int(top_k),
            "top_p": None if top_p is None else float(top_p),
        },
    }
    with open(os.path.join(outputs_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    if save_csv:
        def dump(arr: np.ndarray, name: str):
            pd.DataFrame(arr).to_csv(os.path.join(outputs_dir, name), index=False, header=False)
        dump(sys_all,  "system_share.csv")
        dump(usr_all,  "user_share.csv")
        dump(diff_all, "diff_sys_minus_user.csv")
        dump(H_std,    "inter_head_std.csv")
        dump(H_mean,   "inter_head_mean.csv")
        # Per-step and cumulative rollups
        pd.DataFrame({"step": np.arange(Tcols), "per_step_all": per_step_mean_std_all, "per_step_mid": per_step_mean_std_mid}).to_csv(
            os.path.join(outputs_dir, "instability_per_step.csv"), index=False
        )
        if cum_all.size:
            pd.DataFrame({"k": np.arange(1, cum_all.size+1), "cum_mean_std_all": cum_all, "cum_mean_std_mid": cum_mid}).to_csv(
                os.path.join(outputs_dir, "instability_cumulative_by_k.csv"), index=False
            )
        # Optional: per-head CSVs
        if save_per_head and head_mats and head_mats[0].size > 0:
            for L, mat in enumerate(head_mats):
                pd.DataFrame(mat).to_csv(os.path.join(outputs_dir, f"heads_system_share_layer{L:02d}.csv"), index=False, header=False)

    return gen_text

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser(description="Head-disagreement observability: inter-head std, shares, and rollups.")
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL)

    ap.add_argument("--system-prompt-file", type=str, required=True)
    ap.add_argument("--baseline-prompts-file", type=str, default=None)
    ap.add_argument("--test-prompts-file", type=str, default=None)
    ap.add_argument("--mode", type=str, choices=["baseline","tests","single"], default="tests")
    ap.add_argument("--prompt", type=str, default="")  # used in single mode

    ap.add_argument("--iterations", type=int, default=1)
    ap.add_argument("--outputs-root", type=str, default="outputs")

    # generation (defaults = variance off)
    ap.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS_DEFAULT)
    ap.add_argument("--temperature", type=float, default=TEMPERATURE_DEFAULT)
    ap.add_argument("--top-k", type=int, default=TOP_K_DEFAULT)
    ap.add_argument("--top-p", type=float, default=-1.0, help="-1 disables; else 0<top_p<=1")

    # analysis knobs (just recorded; no blocking decisions here)
    ap.add_argument("--mid-high-frac", type=float, default=MID_HIGH_FRAC_DEFAULT)

    # determinism
    ap.add_argument("--seed", type=int, default=SEED)

    # dumps
    ap.add_argument("--no-per-head", action="store_true")
    ap.add_argument("--no-csv", action="store_true")

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

    if args.mode == "single":
        prompts = [args.prompt] if args.prompt else ["Say hello."]
    elif args.mode == "baseline":
        if not args.baseline_prompts_file:
            raise ValueError("--baseline-prompts-file is required for mode=baseline")
        prompts = read_prompts(args.baseline_prompts_file)
    else:
        if not args.test_prompts_file:
            raise ValueError("--test-prompts-file is required for mode=tests")
        prompts = read_prompts(args.test_prompts_file)

    # Load model/tokenizer
    print(f"Loading model: {args.model} on {DEVICE} (dtype={DTYPE})")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=DTYPE,
        attn_implementation="eager",      # ensure attentions are produced
        trust_remote_code=True,
    ).to(DEVICE)
    model.eval()

    # Optional pad token for batching later (not strictly needed here)
    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token_id", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token

    # Resolve generation defaults
    max_new_tokens = int(args.max_new_tokens)
    temperature = float(args.temperature)
    top_k = int(args.top_k)
    top_p = None if args.top_p < 0 else float(args.top_p)

    # Run
    for it in range(args.iterations):
        for i, user_prompt in enumerate(prompts, 1):
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            slug = slugify(user_prompt[:80]) or f"prompt_{i}"
            out_dir = os.path.join(args.outputs_root, args.mode, f"{slug}__{ts}")
            os.makedirs(out_dir, exist_ok=True)
            print(f"\n--- iter {it+1}/{args.iterations} | {args.mode} {i}/{len(prompts)}")
            print(f"[User] {user_prompt}")
            print(f"[Outputs] {out_dir}")

            text = observe_run(
                model, tokenizer, system_prompt, user_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                outputs_dir=out_dir,
                mid_high_frac=float(args.mid_high_frac),
                save_per_head=not args.no_per_head,
                save_csv=not args.no_csv,
            )
            print(f"[Assistant] {text[:180]}{'...' if len(text)>180 else ''}")
            print("Saved: system/user shares, inter-head std, per-head grids (optional), per-step/cumulative CSVs, tokens timeline, meta.")

if __name__ == "__main__":
    main()

