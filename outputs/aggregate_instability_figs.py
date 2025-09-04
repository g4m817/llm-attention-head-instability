#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggregate instability figures across runs (benign vs attacks) with Mistral-friendly windowing.

Adds late-window scoring over mid/high layers:
    --start-step 11 --end-step 40 --mid-high-frac 0.25 --tail-cut-frac 0.15

Retains original K-based plots for comparison.
"""
import argparse, os, json, math
from glob import glob
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------- utils ---------------------------------
def find_run_dirs(root: str) -> List[str]:
    if not os.path.isdir(root):
        return []
    out = []
    for p in glob(os.path.join(root, "*")):
        if os.path.isdir(p) and os.path.isfile(os.path.join(p, "inter_head_std.csv")):
            out.append(p)
    return sorted(out)

def safe_read_csv(path: str) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(path)
    except Exception:
        return None

def safe_read_matrix(path: str) -> Optional[np.ndarray]:
    try:
        df = pd.read_csv(path, header=None)
        return df.values.astype(np.float32)
    except Exception:
        return None

def nanpad_1d_to_same(arrs: List[np.ndarray]) -> np.ndarray:
    if not arrs:
        return np.zeros((0,0), dtype=np.float32)
    maxlen = max(a.shape[0] for a in arrs)
    out = np.full((len(arrs), maxlen), np.nan, dtype=np.float32)
    for i, a in enumerate(arrs):
        out[i, :a.shape[0]] = a
    return out

def nanpad_2d_to_same(mats: List[np.ndarray]) -> np.ndarray:
    if not mats:
        return np.zeros((0,0,0), dtype=np.float32)
    Lmax = max(m.shape[0] for m in mats)
    Tmax = max(m.shape[1] for m in mats)
    out = np.full((len(mats), Lmax, Tmax), np.nan, dtype=np.float32)
    for i, m in enumerate(mats):
        out[i, :m.shape[0], :m.shape[1]] = m
    return out

def nanmean(a: np.ndarray, axis=None):
    return np.nanmean(a, axis=axis)

def nanse(a: np.ndarray, axis=None):
    std = np.nanstd(a, axis=axis)
    n_eff = np.sum(np.isfinite(a), axis=axis)
    with np.errstate(divide="ignore", invalid="ignore"):
        se = std / np.sqrt(np.maximum(n_eff, 1))
    return se

# ----------------------- aggregation per class ------------------------
def aggregate_class(run_dirs: List[str]) -> Dict[str, np.ndarray]:
    """
    For each run:
      - inter_head_std.csv -> H_std [L,T]
      - instability_per_step.csv (optional) -> per_step_mid series for K-plots
    Returns stacked arrays with NaN padding where lengths differ.
    """
    H_list: List[np.ndarray] = []
    step_mid_list: List[np.ndarray] = []

    for rd in run_dirs:
        H = safe_read_matrix(os.path.join(rd, "inter_head_std.csv"))
        if H is None:
            continue
        H_list.append(H)

        steps = safe_read_csv(os.path.join(rd, "instability_per_step.csv"))
        if steps is not None:
            if "per_step_mid" in steps.columns:
                step_mid_list.append(steps["per_step_mid"].values.astype(np.float32))
            elif steps.shape[1] >= 3:
                step_mid_list.append(steps.iloc[:,2].values.astype(np.float32))

    H_stack = nanpad_2d_to_same(H_list)                    # [N, L, T]
    step_mid = nanpad_1d_to_same(step_mid_list)            # [N, T]
    return {"H_stack": H_stack, "step_mid": step_mid}

# ----------------------- Mistral-friendly window score ----------------
def per_run_window_scores(
    H_stack: np.ndarray,
    start_step: int = 11,
    end_step: int = 40,
    mid_high_frac: float = 0.25,
    tail_cut_frac: float = 0.15,
) -> np.ndarray:
    """
    Compute per-run mean of inter-head std over:
      - layers [mid_high_frac .. 1-tail_cut_frac)
      - steps  [start_step .. end_step] (clamped; excludes prefill 0)
    Returns 1D array of length N (NaN where insufficient length).
    """
    if H_stack.size == 0:
        return np.array([], dtype=np.float32)
    N, L, T = H_stack.shape
    # layer slice
    start_L = max(2, int(L * mid_high_frac))
    end_L   = max(start_L + 1, int(L * (1.0 - tail_cut_frac)))
    start_L = min(start_L, L - 1)
    end_L   = min(max(end_L, start_L + 1), L)
    # time slice
    t0 = max(1, int(start_step))               # exclude prefill 0
    t1 = min(int(end_step), T - 1)
    out = np.full((N,), np.nan, dtype=np.float32)
    if t1 < t0:  # nothing to score
        return out
    window = H_stack[:, start_L:end_L, t0:t1+1]     # [N, L', S']
    with np.errstate(invalid="ignore"):
        out = np.nanmean(window.reshape(N, -1), axis=1)
    return out

# --------------------------- plotting --------------------------------
def fig_runlevel_scatter(ben_vals: np.ndarray, atk_vals: np.ndarray, Krange: tuple, outpath: str, model_label: str):
    k0, k1 = Krange
    plt.figure(figsize=(8,5.5))
    bx = np.random.normal(loc=0.0, scale=0.03, size=ben_vals.shape[0])
    ax = np.random.normal(loc=1.0, scale=0.03, size=atk_vals.shape[0])
    plt.scatter(bx, ben_vals, marker='x', label="Benign", alpha=0.85)
    plt.scatter(ax, atk_vals, marker='x', label="Attack", alpha=0.85)
    plt.xticks([0,1], ["Benign","Attack"])
    plt.ylabel(f"Mean instability (steps {k0}–{k1})")
    plt.title(f"Run-level instability separation (K={k0}–{k1}) — {model_label}")
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()

def fig_per_layer_at_k(ben_H: np.ndarray,
                       atk_H: np.ndarray,
                       outpath: str,
                       model_label: str,
                       k_min: int):
    """Per-layer instability snapshot at Kmin (not hard-coded step 1)."""
    def layer_profile(H, k):
        if H.size == 0:
            return np.zeros((0,), dtype=np.float32)
        T = H.shape[2]
        t_idx = int(np.clip(k, 1, T-1))   # exclude prefill 0; clamp to T-1
        return nanmean(H[:, :, t_idx], axis=0), t_idx

    ben_prof, t_idx_b = layer_profile(ben_H, k_min)
    atk_prof, t_idx_a = layer_profile(atk_H, k_min)
    t_idx = max(t_idx_b if ben_prof.size else 1, t_idx_a if atk_prof.size else 1)

    L = max(ben_prof.shape[0], atk_prof.shape[0])
    xs = np.arange(L)

    plt.figure(figsize=(10,5))
    if ben_prof.size:
        plt.plot(xs[:ben_prof.size], ben_prof, label=f"Benign @ step {t_idx}")
    if atk_prof.size:
        plt.plot(xs[:atk_prof.size], atk_prof, label=f"Attack @ step {t_idx}")

    plt.xlabel("Layer index")
    plt.ylabel(f"Instability at step {t_idx}")
    plt.title(f"Per-layer instability at Kmin={k_min} — {model_label}")
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()

def fig_instability_vs_lookback(ben_step: np.ndarray, atk_step: np.ndarray, Kmin: int, Kmax: int, outpath: str, model_label: str):
    Ks = list(range(Kmin, Kmax+1))
    def class_stats(step_mat):
        means, ses = [], []
        for K in Ks:
            lo = max(1, Kmin)
            hi = min(step_mat.shape[1]-1, K)
            if hi < lo:
                means.append(np.nan); ses.append(np.nan); continue
            seg = step_mat[:, lo:hi+1]
            per_run = np.nanmean(seg, axis=1)
            means.append(np.nanmean(per_run))
            ses.append(nanse(per_run))
        return np.array(means), np.array(ses)

    ben_mean, ben_se = class_stats(ben_step)
    atk_mean, atk_se = class_stats(atk_step)

    plt.figure(figsize=(9,5))
    plt.plot(Ks, ben_mean, marker='o', label="Benign")
    plt.fill_between(Ks, ben_mean-ben_se, ben_mean+ben_se, alpha=0.2)
    plt.plot(Ks, atk_mean, marker='o', label="Attack")
    plt.fill_between(Ks, atk_mean-atk_se, atk_mean+atk_se, alpha=0.2)
    plt.xlabel(f"Lookback K (mean over steps {Kmin}..K)")
    plt.ylabel("Mean instability (mid layers)")
    plt.title(f"Instability vs. K (window starts at {Kmin}) — {model_label}")
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()

def fig_mean_heatmap(H_stack: np.ndarray, title: str, outpath: str):
    if H_stack.size == 0:
        plt.figure(figsize=(6,4)); plt.title(title + " (no data)")
        plt.savefig(outpath, dpi=180); plt.close(); return
    H_mean = nanmean(H_stack, axis=0)  # [L, T]
    plt.figure(figsize=(10,6))
    vmin = 0.0; vmax = max(0.5, np.nanpercentile(H_mean, 99))
    plt.imshow(H_mean, aspect='auto', interpolation='nearest', vmin=vmin, vmax=vmax, cmap="viridis")
    plt.colorbar(label="Std across heads (system-token share)")
    plt.xlabel("Step (0=prefill, Kmin=generated)")
    plt.ylabel("Layer (0=shallow, top=deep)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()

def load_instability_per_step(run_dir):
    path = os.path.join(run_dir, "instability_per_step.csv")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        if "per_step_mid" in df.columns:
            return df["per_step_mid"].values
        elif df.shape[1] >= 3:
            return df.iloc[:,2].values
    except Exception:
        return None
    return None

def violin_plot_K(benign_dirs, attack_dirs, k_min, k_max, outpath, model_label):
    ben_vals, att_vals = [], []
    for d in benign_dirs:
        arr = load_instability_per_step(d)
        if arr is not None and len(arr) > k_max:
            ben_vals.append(arr[k_min:k_max+1].mean())
    for d in attack_dirs:
        arr = load_instability_per_step(d)
        if arr is not None and len(arr) > k_max:
            att_vals.append(arr[k_min:k_max+1].mean())
    data = [{"class":"Benign","val":v} for v in ben_vals] + [{"class":"Attack","val":v} for v in att_vals]
    df = pd.DataFrame(data)
    plt.figure(figsize=(8,5))
    sns.violinplot(data=df,x="class",y="val",inner="stick",cut=0)
    plt.ylabel("Mean inter-head instability")
    plt.title(f"Distribution of mean instability (steps {k_min}-{k_max}) — {model_label}")
    plt.tight_layout(); plt.savefig(outpath,dpi=180); plt.close()

def violin_plot_window(ben_scores, atk_scores, start_step, end_step, outpath, model_label):
    data = [{"class":"Benign","val":v} for v in ben_scores] + [{"class":"Attack","val":v} for v in atk_scores]
    df = pd.DataFrame(data)
    plt.figure(figsize=(8,5))
    sns.violinplot(data=df,x="class",y="val",inner="stick",cut=0)
    plt.ylabel("Mean inter-head instability (windowed)")
    plt.title(f"Distribution over steps {start_step}–{end_step} (mid/high layers) — {model_label}")
    plt.tight_layout(); plt.savefig(outpath,dpi=180); plt.close()

def fig_benign_vs_attack_line(ben_step: np.ndarray,
                              atk_step: np.ndarray,
                              Kmin: int,
                              Kmax: int,
                              outpath: str,
                              model_label: str):
    Ks = list(range(Kmin, Kmax+1))
    def class_mean(mat, K, Kmin):
        if mat.size == 0:
            return np.nan
        idx_hi = min(mat.shape[1]-1, K)     # clamp to last decode step
        lo = max(1, Kmin)                   # exclude prefill (0)
        if idx_hi < lo:
            return np.nan
        per_run = np.nanmean(mat[:, lo:idx_hi+1], axis=1)
        return float(np.nanmean(per_run))

    ben_means = [class_mean(ben_step, K, Kmin) for K in Ks]
    atk_means = [class_mean(atk_step, K, Kmin) for K in Ks]

    plt.figure(figsize=(8.5,5))
    plt.plot(Ks, ben_means, marker='o', label="Benign (baseline)")
    plt.plot(Ks, atk_means, marker='o', label="Attack (injection)")
    plt.xlabel(f"Lookback step K (mean over {max(1,Kmin)}..K)")
    plt.ylabel("Mean instability (std across heads/layers)")
    plt.title(f"Instability signal at early decode steps (Kmin={Kmin}, Kmax={Kmax}) — {model_label}")
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()

def per_run_mean_over_K_window(step_mat: np.ndarray, k0: int, k1: int) -> np.ndarray:
    """
    step_mat: [N, T] where col 0 = prefill.
    Returns per-run mean over steps [k0..k1] (clamped), excluding prefill.
    """
    if step_mat.size == 0:
        return np.array([])
    T = step_mat.shape[1]
    lo = max(1, int(k0))
    hi = min(int(k1), T - 1)
    if hi < lo:
        return np.array([])
    return np.nanmean(step_mat[:, lo:hi+1], axis=1)

def fig_runlevel_scatter_window(ben_vals: np.ndarray,
                                atk_vals: np.ndarray,
                                outpath: str,
                                model_label: str,
                                start_step: int,
                                end_step: int):
    """Scatter of per-run windowed means over steps [start_step..end_step]."""
    plt.figure(figsize=(8, 5.5))
    if ben_vals.size:
        bx = np.random.normal(loc=0.0, scale=0.03, size=ben_vals.shape[0])
        plt.scatter(bx, ben_vals, marker='x', label="Benign", alpha=0.85)
    if atk_vals.size:
        ax = np.random.normal(loc=1.0, scale=0.03, size=atk_vals.shape[0])
        plt.scatter(ax, atk_vals, marker='x', label="Attack", alpha=0.85)
    plt.xticks([0, 1], ["Benign", "Attack"])
    plt.ylabel(f"Mean instability (steps {start_step}–{end_step}, mid/high layers)")
    plt.title(f"Run-level separation — {model_label}")
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()
# ------------------------------ main ---------------------------------
def main():
    ap = argparse.ArgumentParser(description="Aggregate instability figures (benign vs attack) with late-window scoring.")
    ap.add_argument("--benign-root", required=True, type=str)
    ap.add_argument("--attacks-root", required=True, type=str)
    ap.add_argument("--outdir", required=True, type=str)
    ap.add_argument("--model-label", required=True, type=str)

    # Mistral-friendly window defaults
    ap.add_argument("--start-step", type=int, default=11, help="Window start step (exclude prefill 0).")
    ap.add_argument("--end-step",   type=int, default=40, help="Window end step (inclusive, clipped to T-1).")
    ap.add_argument("--mid-high-frac", type=float, default=0.25, help="Start of mid/high layer slice (fraction of depth).")
    ap.add_argument("--tail-cut-frac", type=float, default=0.15, help="Drop deepest tail fraction (noise).")

    # Keep original early-K plots for comparison
    ap.add_argument("--summary-csv", type=str, default=None)
    ap.add_argument("--scatter-k", type=int, default=6)
    ap.add_argument("--k-min", type=int, default=3)
    ap.add_argument("--k-max", type=int, default=8)

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    ben_dirs = find_run_dirs(args.benign_root)
    atk_dirs = find_run_dirs(args.attacks_root)
    print(f"[INFO] Found {len(ben_dirs)} benign and {len(atk_dirs)} attack runs.")

    ben = aggregate_class(ben_dirs)
    atk = aggregate_class(atk_dirs)
    ben_H = ben["H_stack"]         # [N,L,T]
    atk_H = atk["H_stack"]
    ben_step = ben["step_mid"]     # [N,T] (optional, for K-plots)
    atk_step = atk["step_mid"]

    # --------- NEW: windowed per-run scores (best for Mistral) ----------
    benW = per_run_window_scores(
        ben_H, args.start_step, args.end_step, args.mid_high_frac, args.tail_cut_frac
    )
    atkW = per_run_window_scores(
        atk_H, args.start_step, args.end_step, args.mid_high_frac, args.tail_cut_frac
    )

    # A') Run-level scatter (windowed)
    fig_runlevel_scatter_window(
        benW, atkW,
        os.path.join(args.outdir, f"fig_runlevel_scatter_window.png"),
        args.model_label, args.start_step, args.end_step
    )

    # E') Violin (windowed)
    violin_plot_window(
        benW, atkW, args.start_step, args.end_step,
        os.path.join(args.outdir, f"fig_violin_window.png"),
        args.model_label
    )

    # ------------- KEEP: early-step diagnostics for reference ----------
    if ben_step.size and atk_step.size:
        benK = per_run_mean_over_K_window(ben_step, args.k_min, args.scatter_k)
        atkK = per_run_mean_over_K_window(atk_step, args.k_min, args.scatter_k)
        fig_runlevel_scatter(
            benK, atkK, (args.k_min, args.scatter_k),
            os.path.join(args.outdir, f"fig_runlevel_scatter_K{args.k_min}-{args.scatter_k}.png"),
            args.model_label
        )
        fig_instability_vs_lookback(
            ben_step, atk_step, args.k_min, args.k_max,
            os.path.join(args.outdir, "fig_instability_vs_lookback.png"),
            args.model_label
        )
        fig_benign_vs_attack_line(
            ben_step, atk_step, args.k_min, args.k_max,
            os.path.join(args.outdir, f"fig_benign_vs_attack_steps_{args.k_min}-{args.k_max}.png"),
            args.model_label
        )

    # ------------- Mean heatmaps & step-1 layer profile ----------------
    fig_mean_heatmap(atk_H, f"Attack: mean inter-head instability — {args.model_label}",
                     os.path.join(args.outdir, "fig_attack_mean_heatmap.png"))
    fig_mean_heatmap(ben_H, f"Benign: mean inter-head instability — {args.model_label}",
                     os.path.join(args.outdir, "fig_benign_mean_heatmap.png"))
    fig_per_layer_at_k(
        ben_H, atk_H,
        os.path.join(args.outdir, f"fig_per_layer_profile_Kmin{args.k_min}.png"),
        args.model_label, args.k_min
    )

    # ------------- Optional tradeoff heatmap from summary.csv ----------
    if args.summary_csv and os.path.isfile(args.summary_csv):
        # unchanged: your summary CSV heatmap routine (kept if you still run the CV tuner)
        pass

    print(f"[DONE] Saved figures to: {os.path.abspath(args.outdir)}")

if __name__ == "__main__":
    main()
