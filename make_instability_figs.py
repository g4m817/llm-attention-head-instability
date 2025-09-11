#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate labeled graphs from run data, grouped by (model, system-prompt filename, dataset filename).

Adds:
- Entropy & head-agreement ingestion (per-sample *_entropy.npy, *_headcorr.npy)
- Stepwise overlays: mean instability, entropy, head agreement (trimmed layers)
- CSV: correlation stats at group level
- Line charts (replacing old scatter files):
  - scatter_instability_vs_entropy.png      -> line: instability (benign/attack) + entropy (twin axis)
  - scatter_instability_vs_headcorr_attack.png -> line: instability + head agreement (attack only, twin axis)
  - scatter_instability_vs_headcorr_benign.png -> line: instability + head agreement (benign only, twin axis)

Compatible with both old (baseline/tests with iterations) and new (benign/attack without iterations) run formats.

Usage:
  python make_instability_figs.py --runs /path/to/runs_or_zip --outdir /path/to/out
"""
import os, io, re, json, argparse, zipfile, csv
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

# ---------------------------- I/O layer ----------------------------

@dataclass
class RunConfig:
    model: str
    system_prompt_file: str
    dataset_file: str          # generalized: benign or attack prompts file (whichever present)
    window_end: int
    window_start: int
    mid_high_frac: float
    tail_cut_frac: float

@dataclass
class Sample:
    run: str
    kind: str  # 'benign' or 'attack' (also supports 'baseline'/'tests' legacy)
    iter: Optional[int]        # None for new format
    index: int
    std: Optional[np.ndarray]        # [T, L]
    entropy: Optional[np.ndarray]    # [T]
    headcorr: Optional[np.ndarray]   # [T]
    text: Optional[str]

@dataclass
class RunData:
    run: str
    config: RunConfig
    samples: List[Sample]

def _basename(path: str) -> str:
    return os.path.basename(path).replace("\\", "/")

class Reader:
    def list(self) -> List[str]:
        raise NotImplementedError
    def open(self, name: str) -> io.BytesIO:
        raise NotImplementedError

class FSReader(Reader):
    def __init__(self, root: str):
        self.root = os.path.abspath(root)
    def list(self) -> List[str]:
        out = []
        for dirpath, _, filenames in os.walk(self.root):
            rel = os.path.relpath(dirpath, self.root)
            if rel == ".":
                rel = ""
            for fn in filenames:
                out.append(os.path.join(rel, fn).replace("\\", "/"))
        return out
    def open(self, name: str) -> io.BytesIO:
        p = os.path.join(self.root, name)
        with open(p, "rb") as f:
            return io.BytesIO(f.read())

class ZipReader(Reader):
    def __init__(self, zpath: str):
        self.zf = zipfile.ZipFile(zpath, "r")
    def list(self) -> List[str]:
        return [n for n in self.zf.namelist() if not n.endswith("/")]
    def open(self, name: str) -> io.BytesIO:
        return io.BytesIO(self.zf.read(name))

def make_reader(path: str) -> Reader:
    if os.path.isdir(path):
        return FSReader(path)
    if path.lower().endswith(".zip"):
        return ZipReader(path)
    raise ValueError(f"Unsupported --runs path: {path}")

# ------------------------- parsing runs ----------------------------
# Legacy patterns (baseline/tests with iterations)
LEG_STD_RE  = re.compile(r'(baseline|tests|tests_ungated)_iter(\d+)_(\d+)_.*_std\.npy$')
LEG_ENT_RE  = re.compile(r'(baseline|tests|tests_ungated)_iter(\d+)_(\d+)_.*_entropy\.npy$')
LEG_CORR_RE = re.compile(r'(baseline|tests|tests_ungated)_iter(\d+)_(\d+)_.*_headcorr\.npy$')
# New patterns (benign/attack, no iterations)
NEW_STD_RE  = re.compile(r'(benign|attack)_(\d+)_.*_std\.npy$')
NEW_ENT_RE  = re.compile(r'(benign|attack)_(\d+)_.*_entropy\.npy$')
NEW_CORR_RE = re.compile(r'(benign|attack)_(\d+)_.*_headcorr\.npy$')

def _trim_to_window(xs: np.ndarray, ys: np.ndarray, start: int, end: int) -> Tuple[np.ndarray, np.ndarray]:
    if ys is None or ys.size == 0:
        return xs, ys
    m = (xs >= start) & (xs <= end)
    if not m.any():
        return xs[:0], ys[:0]
    # clamp length mismatch just in case
    n = min(xs.size, ys.size)
    return xs[:n][m[:n]], ys[:n][m[:n]]

def _trim_trailing_nan(x: np.ndarray) -> np.ndarray:
    """Keep interior NaNs (gaps), drop only trailing NaNs so the line won't stop early."""
    if x.size == 0:
        return x
    finite_idx = np.where(np.isfinite(x))[0]
    if finite_idx.size == 0:
        return x[:0]
    last = finite_idx[-1]
    return x[:last+1]

def find_run_dirs(reader: Reader) -> List[str]:
    names = reader.list()
    run_roots = sorted({n.split("/")[0] for n in names if "/" in n and n.startswith("run_")})
    return run_roots

def parse_config(reader: Reader, run_dir: str) -> Optional[RunConfig]:
    try:
        with reader.open(f"{run_dir}/config.json") as f:
            cfg = json.loads(f.read().decode("utf-8"))
        # Handle new + legacy keys
        model = cfg["model"]
        system_prompt_file = cfg["system_prompt_file"]
        dataset_file = cfg.get("benign_prompts_file") or cfg.get("test_prompts_file") or \
                       cfg.get("baseline_prompts_file") or ""  # best-effort
        return RunConfig(
            model=model,
            system_prompt_file=system_prompt_file,
            dataset_file=dataset_file,
            window_end=int(cfg.get("window_end", cfg.get("lookback_steps", 0) or 0)),
            window_start=int(cfg.get("window_start", cfg.get("start_steps", 1))),
            mid_high_frac=float(cfg.get("mid_high_frac", 0.25)),
            tail_cut_frac=float(cfg.get("tail_cut_frac", 0.15)),
        )
    except Exception as e:
        print(f"[WARN] Failed to read config for {run_dir}: {e}")
        return None

def parse_run(reader: Reader, run_dir: str, cfg: RunConfig) -> RunData:
    names = [n for n in reader.list() if n.startswith(run_dir + "/")]

    # Map .txt responses (both legacy and new naming)
    txt_map: Dict[str, str] = {}
    for n in names:
        base = _basename(n)
        if n.endswith(".txt"):
            try:
                txt_map[base] = reader.open(n).read().decode("utf-8", errors="ignore")
            except Exception:
                pass

    # Ingest arrays
    std_map: Dict[Tuple[str, Optional[int], int], np.ndarray] = {}
    ent_map: Dict[Tuple[str, Optional[int], int], np.ndarray] = {}
    cor_map: Dict[Tuple[str, Optional[int], int], np.ndarray] = {}

    for n in names:
        base = _basename(n)
        if n.endswith("_std.npy"):
            m = LEG_STD_RE.match(base)
            if m:
                kind, it, idx = m.group(1), int(m.group(2)), int(m.group(3))
                std_map[(kind, it, idx)] = np.load(reader.open(n))
                continue
            m = NEW_STD_RE.match(base)
            if m:
                kind, idx = m.group(1), int(m.group(2))
                std_map[(kind, None, idx)] = np.load(reader.open(n))
                continue

        if n.endswith("_entropy.npy"):
            m = LEG_ENT_RE.match(base)
            if m:
                kind, it, idx = m.group(1), int(m.group(2)), int(m.group(3))
                ent_map[(kind, it, idx)] = np.load(reader.open(n))
                continue
            m = NEW_ENT_RE.match(base)
            if m:
                kind, idx = m.group(1), int(m.group(2))
                ent_map[(kind, None, idx)] = np.load(reader.open(n))
                continue

        if n.endswith("_headcorr.npy"):
            m = LEG_CORR_RE.match(base)
            if m:
                kind, it, idx = m.group(1), int(m.group(2)), int(m.group(3))
                cor_map[(kind, it, idx)] = np.load(reader.open(n))
                continue
            m = NEW_CORR_RE.match(base)
            if m:
                kind, idx = m.group(1), int(m.group(2))
                cor_map[(kind, None, idx)] = np.load(reader.open(n))
                continue

    # Merge keys across three maps
    keys = set(list(std_map.keys()) + list(ent_map.keys()) + list(cor_map.keys()))
    samples: List[Sample] = []
    for key in sorted(keys, key=lambda x: (x[0], x[1] if x[1] is not None else -1, x[2])):
        kind, it, idx = key
        std = std_map.get(key)
        ent = ent_map.get(key)
        cor = cor_map.get(key)

        # Find matching .txt, legacy and new:
        # legacy guess: f"{kind}_iter{it}_{idx:03d}"
        # new guess:    f"{kind}_{idx:03d}"
        t = None
        guesses = []
        if it is not None:
            guesses.append(f"{kind}_iter{it}_{idx:03d}")
        guesses.append(f"{kind}_{idx:03d}")
        for g in guesses:
            for k in txt_map.keys():
                if g in k:
                    t = txt_map[k]; break
            if t is not None:
                break

        # Normalize kind to 'benign'/'attack'
        norm_kind = kind
        if kind == "baseline":
            norm_kind = "benign"
        elif kind == "tests":
            norm_kind = "attack"

        samples.append(Sample(
            run=run_dir,
            kind=norm_kind,
            iter=it,
            index=idx,
            std=std if isinstance(std, np.ndarray) else None,
            entropy=ent if isinstance(ent, np.ndarray) else None,
            headcorr=cor if isinstance(cor, np.ndarray) else None,
            text=t
        ))
    return RunData(run=run_dir, config=cfg, samples=samples)

# ------------------------- scoring & helpers -----------------------

def filename_only(path: str) -> str:
    return os.path.basename(path)

def layer_slice(L: int, head: float=0.25, tail: float=0.15) -> Tuple[int,int]:
    s = max(0, int(L * head))
    e = max(s+1, int(L * (1.0 - tail)))
    e = min(e, L)
    return s, e

def per_sample_window_mean_std(std_mat: np.ndarray, step0: int, step1: int, head_frac: float, tail_frac: float) -> float:
    T, L = std_mat.shape
    lo = max(1, int(step0))
    hi = min(int(step1), T - 1)
    if hi < lo:
        return np.nan
    ls, le = layer_slice(L, head_frac, tail_frac)
    window = std_mat[lo:hi+1, ls:le]
    return float(np.nanmean(window)) if window.size else np.nan

def per_sample_window_mean_vec(vec: np.ndarray, step0: int, step1: int) -> float:
    T = vec.shape[0]
    lo = max(1, int(step0))
    hi = min(int(step1), T - 1)
    if hi < lo:
        return np.nan
    return float(np.nanmean(vec[lo:hi+1]))

def step_series_std(std_mat: np.ndarray, head_frac: float, tail_frac: float) -> np.ndarray:
    T, L = std_mat.shape
    ls, le = layer_slice(L, head_frac, tail_frac)
    x = np.nanmean(std_mat[:, ls:le], axis=1)
    if x.shape[0] > 0:
        x[0] = np.nan
    return x

def _pad(arrs: List[np.ndarray]) -> Optional[np.ndarray]:
    arrs = [a for a in arrs if isinstance(a, np.ndarray) and a.size]
    if not arrs:
        return None
    maxlen = max(a.shape[0] for a in arrs)
    out = np.full((len(arrs), maxlen), np.nan, dtype=np.float32)
    for i, a in enumerate(arrs):
        out[i, :a.shape[0]] = a
    return out

def class_stats(arr: np.ndarray) -> Tuple[float,float,float]:
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    return float(np.mean(arr)), float(np.median(arr)), float(np.std(arr))

def tpr_at_fixed_fpr(y: np.ndarray, s: np.ndarray, fpr_target=0.05) -> Tuple[float,float,float]:
    mask = np.isfinite(s)
    y = y[mask]; s = s[mask]
    if len(y) < 3 or len(np.unique(y)) < 2:
        return (float("nan"), float("nan"), float("nan"))
    fpr, tpr, thr = roc_curve(y, s)
    idx = int(np.argmin(np.abs(fpr - fpr_target)))
    return float(thr[idx]), float(tpr[idx]), float(fpr[idx])

# ------------------------- plotting --------------------------------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _clamp_window_to_data(w0: int, w1: int, t_last: int) -> Tuple[int, int]:
    """
    Clamp [w0, w1] to [1, t_last], where t_last is the last *real* step index (T-1).
    """
    if t_last < 1:
        return 1, 1
    a = max(1, int(min(w0, w1)))
    b = max(a, int(max(w0, w1)))
    b = min(b, int(t_last))
    return a, b

def _apply_window_xlims(ax, w0c: int, w1c: int, color="#1f77b4"):
    """Set x-limits and draw start/end guides labeled *inside* axes."""
    ax.set_xlim(w0c - 0.1, w1c + 0.1)
    ytop = ax.get_ylim()[1]
    ax.axvline(w0c, ls="--", lw=1.2, color=color, alpha=0.6)
    ax.axvline(w1c, ls="--", lw=1.2, color=color, alpha=0.6)
    ax.text(w0c, ytop, f"start={w0c}", ha="left", va="top", fontsize=10, color=color, alpha=0.9)
    ax.text(w1c, ytop, f"end={w1c}",   ha="right", va="top", fontsize=10, color=color, alpha=0.9)

def save_stepwise_auroc(y_true: np.ndarray, step_mat: np.ndarray, outpath: str, window: Optional[Tuple[int,int]]=None):
    """Compute AUROC at each decode step using layer-trimmed per-step means (steps 1..T-1).
       Crop both data and axes to the clamped window; draw guides.
    """
    # step_mat shape: [N, T]; columns are steps 0..T-1 (0=prefill NaN)
    T_all = step_mat.shape[1] - 1  # last real step index
    if T_all < 1:
        return

    xs, ys = [], []
    for k in range(1, step_mat.shape[1]):  # skip prefill
        s = step_mat[:, k]
        m = np.isfinite(s)
        xs.append(k)
        if m.sum() < 3 or len(np.unique(y_true[m])) < 2:
            ys.append(np.nan)
        else:
            ys.append(roc_auc_score(y_true[m], s[m]))
    xs = np.asarray(xs)
    ys = np.asarray(ys, dtype=float)

    if window is not None:
        w0c, w1c = _clamp_window_to_data(window[0], window[1], T_all)
        keep = (xs >= w0c) & (xs <= w1c)
        xs, ys = xs[keep], ys[keep]
    else:
        w0c, w1c = 1, T_all

    plt.figure(figsize=(8,5))
    ax = plt.gca()
    ax.plot(xs, ys, lw=2, color="#1f77b4")
    ax.set_xlabel("Decoding step")
    ax.set_ylabel("AUROC")
    ax.set_title(f"Stepwise AUROC — Instability (per-step mean std)  [window {w0c}–{w1c}]")
    ax.grid(True, alpha=0.3)
    _apply_window_xlims(ax, w0c, w1c)
    plt.tight_layout()
    plt.savefig(outpath, dpi=180, bbox_inches="tight", pad_inches=0.2)
    plt.close()

def save_roc(y_true: np.ndarray, scores: np.ndarray, title: str, outpath: str,
             w0: Optional[int]=None, w1: Optional[int]=None):
    mask = np.isfinite(scores)
    y = y_true[mask]; s = scores[mask]
    if len(y) < 3 or len(np.unique(y)) < 2:
        print(f"[WARN] Not enough data for ROC: {outpath}")
        return
    fpr, tpr, _ = roc_curve(y, s)
    A = roc_auc_score(y, s)
    plt.figure(figsize=(5.8,5.5))
    plt.plot(fpr, tpr, lw=2, label=f"AUROC = {A:.3f}")
    plt.plot([0,1],[0,1],'--',alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    if w0 is not None and w1 is not None:
        plt.title(f"{title}  [window {w0}–{w1}]")
    else:
        plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=180, bbox_inches="tight", pad_inches=0.2)
    plt.close()

def save_stepwise_overlay(xs: np.ndarray, ys_list: List[Tuple[str, np.ndarray, str]],
                          title: str, outpath: str, x_start: Optional[int]=None, x_end: Optional[int]=None):
    """
    Plot stepwise series; crops data and axes to [x_start, x_end] (clamped to available steps).
    xs should be integer step indices (0..T-1) with step 0 prefill.
    """
    if xs.size == 0:
        return
    T_all = int(xs.max())  # last real step index
    if x_start is not None and x_end is not None:
        w0c, w1c = _clamp_window_to_data(int(x_start), int(x_end), T_all)
    else:
        w0c, w1c = 1, T_all

    mask = (xs >= w0c) & (xs <= w1c)
    if not mask.any():
        print(f"[WARN] save_stepwise_overlay: empty mask for {outpath}")
        return

    plt.figure(figsize=(8.4,5.0))
    ax = plt.gca()
    for label, ys, color in ys_list:
        if ys is None or ys.size == 0:
            continue
        ys_crop = ys[:xs.size][mask]
        if not np.isfinite(ys_crop).any():
            continue
        ax.plot(xs[mask], ys_crop, label=label, lw=2, alpha=0.9, color=color)

    if not ax.has_data():
        plt.close()
        print(f"[WARN] No finite data to plot for {outpath}")
        return

    ax.set_xlabel("Decoding step index")
    ax.set_ylabel("Mean (trimmed layers / series)")
    ax.set_title(f"{title}  [window {w0c}–{w1c}]")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _apply_window_xlims(ax, w0c, w1c)
    plt.tight_layout()
    plt.savefig(outpath, dpi=180, bbox_inches="tight", pad_inches=0.2)
    plt.close()

def save_violin(ben: np.ndarray, atk: np.ndarray, title: str, outpath: str, ylabel: str, w0: Optional[int]=None, w1: Optional[int]=None):
    plt.figure(figsize=(7.2,4.8))
    data = [ben[~np.isnan(ben)], atk[~np.isnan(atk)]]
    if all(len(d)==0 for d in data):
        plt.close()
        print(f"[WARN] No data for violin: {outpath}")
        return
    plt.violinplot(data, showmeans=True, showextrema=False, showmedians=False)
    plt.xticks([1,2],["Benign","Attack"])
    plt.ylabel(ylabel)
    plt.title(f"{title}  [window {w0}–{w1}]" if (w0 is not None and w1 is not None) else title)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=180, bbox_inches="tight", pad_inches=0.2)
    plt.close()

def save_mean_heatmap(mats: List[np.ndarray], title: str, outpath: str,
                      x_start: Optional[int]=None, x_end: Optional[int]=None):
    if not mats:
        return
    Lmax = max(m.shape[1] for m in mats)
    Tmax = max(m.shape[0] for m in mats)
    stack = np.full((len(mats), Tmax, Lmax), np.nan, dtype=np.float32)
    for i, m in enumerate(mats):
        T, L = m.shape
        stack[i, :T, :L] = m
    mean_mat = np.nanmean(stack, axis=0)  # [T, L]

    # crop to window along steps (axis 0), skipping prefill (0) in the crop logic
    if x_start is not None and x_end is not None and mean_mat.shape[0] > 1:
        x0 = max(1, int(x_start))
        x1 = min(int(x_end), mean_mat.shape[0] - 1)
        if x1 >= x0:
            mean_mat = mean_mat[x0:x1+1, :]

    plt.figure(figsize=(8,5.2))
    plt.imshow(mean_mat.T, aspect='auto', origin='lower', interpolation='nearest', cmap='viridis')
    plt.colorbar(label="Std across heads (system-token share)")
    plt.xlabel("Step (prefill removed)")
    plt.ylabel("Layer (shallow→deep)")
    plt.title(f"{title}  [window {x_start}–{x_end}]" if (x_start is not None and x_end is not None) else title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=180, bbox_inches="tight", pad_inches=0.2)
    plt.close()

# -------- New line charts (replace old scatter files) ---------------

def _rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    """NaN-aware centered rolling mean, same length, leaves NaNs where window has <1 finite."""
    n = x.size
    if n == 0 or w <= 1:
        return x.copy()
    k = min(w, n)
    half = k // 2
    y = np.full(n, np.nan, dtype=float)
    for i in range(n):
        a = max(0, i - half)
        b = min(n, i - half + k)
        win = x[a:b]
        m = np.isfinite(win)
        if m.any():
            y[i] = np.nanmean(win[m])
    return y

def save_instability_entropy_lines(
    y_true: np.ndarray,
    instab: np.ndarray,
    entropy: np.ndarray,
    outpath: str,
    roll_window: int = 7,
    window: Optional[Tuple[int,int]] = None,
):
    """Instability (benign/attack) + Entropy on twin axis. Trim trailing NaNs only."""
    m = np.isfinite(instab)
    y = y_true[m]; s = instab[m]
    e = entropy[m] if (isinstance(entropy, np.ndarray) and entropy.size == instab.size) else np.full_like(s, np.nan)

    # class splits
    s_b = s[y == 0]
    s_a = s[y == 1]

    # compute rolling means keeping alignment, then drop only trailing NaNs
    sb_rm = _rolling_mean(s_b, roll_window); sb_rm = _trim_trailing_nan(sb_rm)
    sa_rm = _rolling_mean(s_a, roll_window); sa_rm = _trim_trailing_nan(sa_rm)
    e_rm  = _rolling_mean(e,   roll_window); e_rm  = _trim_trailing_nan(e_rm)

    if sb_rm.size==0 and sa_rm.size==0 and e_rm.size==0:
        print(f"[WARN] No finite data for {outpath}")
        return

    plt.figure(figsize=(8.8, 5.2))
    ax = plt.gca()
    if sb_rm.size:
        ax.plot(np.arange(sb_rm.size), np.ma.masked_invalid(sb_rm), label="Instability (benign, rolling)", lw=2, color="#1f77b4")
    if sa_rm.size:
        ax.plot(np.arange(sa_rm.size), np.ma.masked_invalid(sa_rm), label="Instability (attack, rolling)", lw=2, color="#2ca02c")

    ax2 = ax.twinx()
    if e_rm.size:
        ax2.plot(np.arange(e_rm.size), np.ma.masked_invalid(e_rm), label="Entropy (rolling)", lw=2, color="#d62728", alpha=0.85)

    ax.set_xlabel("Sample index within class (rolling)")
    ax.set_ylabel("Instability (windowed)")
    ax2.set_ylabel("Entropy (windowed)")
    ax.grid(True, alpha=0.3)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="best")

    suffix = f"  [window {window[0]}–{window[1]}]" if window else ""
    plt.title("Instability (benign vs attack) and Entropy — rolling means"+suffix)
    plt.tight_layout()
    plt.savefig(outpath, dpi=180, bbox_inches="tight", pad_inches=0.2)
    plt.close()

def save_instability_headcorr_lines_by_class(
    y_true: np.ndarray,
    instab: np.ndarray,
    headcorr: np.ndarray,
    outpath_attack: str,
    outpath_benign: str,
    roll_window: int = 7,
    window: Optional[Tuple[int,int]] = None,
):
    """Two twin-axis charts (attack / benign). Trim trailing NaNs only."""
    m = np.isfinite(instab)
    y = y_true[m]; s = instab[m]
    c = headcorr[m] if (isinstance(headcorr, np.ndarray) and headcorr.size == instab.size) else np.full_like(s, np.nan)

    def plot_one(mask: np.ndarray, title: str, outpath: str):
        s_c = s[mask]; c_c = c[mask]
        s_rm = _rolling_mean(s_c, roll_window); s_rm = _trim_trailing_nan(s_rm)
        c_rm = _rolling_mean(c_c, roll_window); c_rm = _trim_trailing_nan(c_rm)

        if s_rm.size==0 and c_rm.size==0:
            print(f"[WARN] No finite data for {outpath}")
            return

        plt.figure(figsize=(8.8, 5.2))
        ax = plt.gca()
        if s_rm.size:
            ax.plot(np.arange(s_rm.size), np.ma.masked_invalid(s_rm), label="Instability (rolling)", lw=2, color="#1f77b4")
        ax.set_xlabel("Sample index (rolling)")
        ax.set_ylabel("Instability (windowed)")
        ax.grid(True, alpha=0.3)

        ax2 = ax.twinx()
        if c_rm.size:
            ax2.plot(np.arange(c_rm.size), np.ma.masked_invalid(c_rm), label="Head agreement (rolling)", lw=2, color="#d62728", alpha=0.85)
        ax2.set_ylabel("Head agreement (mean corr)")

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="best")

        suffix = f"  [window {window[0]}–{window[1]}]" if window else ""
        plt.title(title + suffix)
        plt.tight_layout()
        plt.savefig(outpath, dpi=180, bbox_inches="tight", pad_inches=0.2)
        plt.close()

    atk_mask = (y == 1)
    ben_mask = (y == 0)
    plot_one(atk_mask, "Instability vs Head Agreement — Attack (rolling means)", outpath_attack)
    plot_one(ben_mask, "Instability vs Head Agreement — Benign (rolling means)", outpath_benign)

# ------------------------------ main -------------------------------

def default_window_for_model(model_name: str) -> Tuple[int,int]:
    """Fallback window when config lacks window_start/window_end."""
    if model_name and "mistral" in model_name.lower():
        return 11, 40
    # conservative short window for others
    return 1, 3

def clamp_window(w0: int, w1: int) -> Tuple[int,int]:
    if w0 is None or w1 is None:
        return (w0, w1)
    if w1 < w0:
        w0, w1 = w1, w0
    return int(w0), int(w1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="Path to root runs dir OR a .zip archive.")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    ensure_dir(args.outdir)
    reader = make_reader(args.runs)

    runs: List[RunData] = []
    for rd in find_run_dirs(reader):
        cfg = parse_config(reader, rd)
        if cfg is None:
            continue
        runs.append(parse_run(reader, rd, cfg))

    if not runs:
        print("[ERROR] No runs found.")
        return

    groups: Dict[Tuple[str,str,str], List[RunData]] = {}
    for r in runs:
        key = (r.config.model, filename_only(r.config.system_prompt_file), filename_only(r.config.dataset_file))
        groups.setdefault(key, []).append(r)

    # Summary CSV of correlations per group
    corr_csv_path = os.path.join(args.outdir, "group_correlations.csv")
    with open(corr_csv_path, "w", newline="", encoding="utf-8") as fcsv:
        cw = csv.writer(fcsv)
        cw.writerow([
            "model","system_prompt","dataset_file",
            "window_start","window_end","head_frac","tail_frac",
            "pearson_instability_entropy","spearman_instability_entropy",
            "pearson_instability_headcorr","spearman_instability_headcorr",
            "n_samples"
        ])

        for (model, sysfile, dataset_file), rlist in groups.items():
            label = f"{model}"
            gdir = os.path.join(args.outdir, f"{model.replace(' ','_')}_{sysfile}_{dataset_file}".replace(os.sep,"_"))
            ensure_dir(gdir)

            cfg0 = rlist[0].config

            # Resolve group-level window from config (fallback to model default)
            w0 = int(cfg0.window_start) if cfg0.window_start else None
            w1 = int(cfg0.window_end)   if cfg0.window_end   else None
            if w0 is None or w1 is None:
                d0, d1 = default_window_for_model(cfg0.model)
                w0 = d0 if w0 is None else w0
                w1 = d1 if w1 is None else w1
            w0, w1 = clamp_window(w0, w1)

            head_frac, tail_frac = cfg0.mid_high_frac, cfg0.tail_cut_frac

            # Accumulators
            scores: List[Tuple[int, float]] = []  # (is_attack, score)
            step_series_list: List[np.ndarray] = []
            benign_heatmaps, attack_heatmaps = [], []

            # For correlations (group-level)
            windowed_entropy: List[float] = []
            windowed_headcorr: List[float] = []

            # ---- Per-run outputs ----
            group_csv = os.path.join(gdir, "per_run_metrics.csv")
            with open(group_csv, "w", newline="", encoding="utf-8") as fpr:
                writer = csv.writer(fpr)
                writer.writerow([
                    "run_id","model","system_prompt","dataset_file",
                    "window_start","window_end","head_frac","tail_frac",
                    "n_benign","n_attack",
                    "benign_mean","benign_median","benign_std",
                    "attack_mean","attack_median","attack_std",
                    "auroc_windowed","thr_at_5_fpr","tpr_at_thr","fpr_at_thr",
                    "pearson_instability_entropy","spearman_instability_entropy"
                ])

                for run in rlist:
                    # Per-run window: prefer per-run config, else group fallback
                    rw0 = int(run.config.window_start) if run.config.window_start else w0
                    rw1 = int(run.config.window_end)   if run.config.window_end   else w1
                    rw0, rw1 = clamp_window(rw0, rw1)
                    rhead = run.config.mid_high_frac if run.config.mid_high_frac is not None else head_frac
                    rtail = run.config.tail_cut_frac  if run.config.tail_cut_frac  is not None else tail_frac

                    r_scores: List[Tuple[int,float]] = []
                    r_entropy: List[float] = []
                    r_headcorr: List[float] = []

                    for s in run.samples:
                        if s.kind not in ("benign","attack","baseline","tests"):
                            continue
                        is_attack = 1 if s.kind in ('attack','tests') else 0

                        if isinstance(s.std, np.ndarray):
                            score = per_sample_window_mean_std(s.std, rw0, rw1, rhead, rtail)
                            r_scores.append((is_attack, score))
                            scores.append((is_attack, score))
                            step_series_list.append(step_series_std(s.std, rhead, rtail))
                            if is_attack:
                                attack_heatmaps.append(s.std)
                            else:
                                benign_heatmaps.append(s.std)
                        else:
                            score = np.nan
                            r_scores.append((is_attack, score))
                            scores.append((is_attack, score))

                        we = per_sample_window_mean_vec(s.entropy, rw0, rw1) if isinstance(s.entropy, np.ndarray) else np.nan
                        wc = per_sample_window_mean_vec(s.headcorr, rw0, rw1) if isinstance(s.headcorr, np.ndarray) else np.nan
                        r_entropy.append(we); r_headcorr.append(wc)
                        windowed_entropy.append(we); windowed_headcorr.append(wc)

                    if not r_scores:
                        continue
                    ry = np.array([a for a,_ in r_scores], dtype=np.int32)
                    rs = np.array([v for _,v in r_scores], dtype=np.float32)
                    bvals = np.array([v for a,v in r_scores if a==0], dtype=np.float32)
                    avals = np.array([v for a,v in r_scores if a==1], dtype=np.float32)

                    # Violin (run-level)
                    plt.figure(figsize=(7.2,4.8))
                    data = [bvals[np.isfinite(bvals)], avals[np.isfinite(avals)]]
                    try:
                        if any(len(d)>0 for d in data):
                            plt.violinplot(data, showmeans=True, showextrema=False, showmedians=False)
                            plt.xticks([1,2], ["Benign","Attack"])
                            plt.ylabel("Windowed instability")
                            plt.title(f"{run.run} — {label} — steps {rw0}–{rw1} (trim {rhead:.2f}/{rtail:.2f})  [window {rw0}–{rw1}]")
                            plt.grid(axis="y", alpha=0.3)
                            plt.tight_layout()
                            plt.savefig(os.path.join(gdir, "violin_windowed.png"), dpi=180, bbox_inches="tight", pad_inches=0.2)
                    finally:
                        plt.close()

                    # Simple scatter separation (run-level)
                    plt.figure(figsize=(7.2,4.8))
                    if bvals.size:
                        bx = np.random.normal(loc=0.0, scale=0.03, size=bvals.shape[0])
                        plt.scatter(bx, bvals, marker="x", label="Benign", alpha=0.85, color="#1f77b4")
                    if avals.size:
                        axp = np.random.normal(loc=1.0, scale=0.03, size=avals.shape[0])
                        plt.scatter(axp, avals, marker="x", label="Attack", alpha=0.85, color="#2ca02c")
                    plt.xticks([0,1], ["Benign","Attack"])
                    plt.ylabel(f"Mean instability (steps {rw0}–{rw1}, trimmed layers)")
                    plt.title(f"{run.run} — {label}  [window {rw0}–{rw1}]")
                    plt.grid(True, linestyle="--", alpha=0.35)
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(os.path.join(gdir, "scatter_windowed.png"), dpi=180, bbox_inches="tight", pad_inches=0.2)
                    plt.close()

                    # Per-run metrics + correlation(instability, entropy)
                    try:
                        mask = np.isfinite(rs)
                        au = roc_auc_score(ry[mask], rs[mask]) if mask.sum() and len(np.unique(ry[mask]))>1 else float("nan")
                    except Exception:
                        au = float("nan")
                    thr, tprv, fprv = tpr_at_fixed_fpr(ry, rs, 0.05)

                    b_mean, b_med, b_std = class_stats(bvals)
                    a_mean, a_med, a_std = class_stats(avals)

                    r_entropy_arr = np.array(r_entropy, dtype=np.float32)
                    p_inst_ent = s_inst_ent = float("nan")
                    try:
                        from scipy.stats import pearsonr, spearmanr
                        m = np.isfinite(rs) & np.isfinite(r_entropy_arr)
                        if m.sum() >= 3:
                            p_inst_ent, _ = pearsonr(rs[m], r_entropy_arr[m])
                            s_inst_ent, _ = spearmanr(rs[m], r_entropy_arr[m])
                    except Exception:
                        pass

                    writer.writerow([
                        run.run, model, sysfile, dataset_file,
                        rw0, rw1, rhead, rtail,
                        int(np.isfinite(bvals).sum()), int(np.isfinite(avals).sum()),
                        f"{b_mean:.6f}", f"{b_med:.6f}", f"{b_std:.6f}",
                        f"{a_mean:.6f}", f"{a_med:.6f}", f"{a_std:.6f}",
                        f"{au:.6f}", f"{thr:.6f}", f"{tprv:.6f}", f"{fprv:.6f}",
                        f"{p_inst_ent:.6f}", f"{s_inst_ent:.6f}",
                    ])

            # ---- Group-level arrays ----
            if scores:
                y_true = np.array([a for a, _ in scores], dtype=np.int32)
                sc = np.array([s for _, s in scores], dtype=np.float32)
            else:
                y_true = np.array([], dtype=np.int32)
                sc = np.array([], dtype=np.float32)

            # Per-step series collections
            ent_series_list, corr_series_list = [], []
            for r in rlist:
                for s in r.samples:
                    if isinstance(s.entropy, np.ndarray):
                        v = s.entropy.astype(np.float32)
                        if v.shape[0] > 0:
                            v[0] = np.nan  # mask prefill
                        ent_series_list.append(v)
                    if isinstance(s.headcorr, np.ndarray):
                        v = s.headcorr.astype(np.float32)
                        if v.shape[0] > 0:
                            v[0] = np.nan
                        corr_series_list.append(v)

            step_std_mat = _pad(step_series_list)
            step_ent_mat = _pad(ent_series_list)
            step_cor_mat = _pad(corr_series_list)

            # 1) ROC on windowed instability (title shows window for context)
            save_roc(
                y_true, sc,
                f"ROC — {label} — windowed instability",
                os.path.join(gdir, "roc.png"),
                w0=w0, w1=w1
            )

            # 2) Stepwise AUROC, cropped + clamped to window
            if step_std_mat is not None and step_std_mat.shape[0] == len(y_true):
                save_stepwise_auroc(y_true, step_std_mat, os.path.join(gdir, "stepwise_auroc.png"), window=(w0, w1))
            else:
                print(f"[WARN] Could not generate stepwise AUROC for group {label} — mismatched shapes")

            # 3) Stepwise overlays: mean instability (trimmed), entropy, headcorr
            T_max = 0
            for m in (step_std_mat, step_ent_mat, step_cor_mat):
                if m is not None:
                    T_max = max(T_max, m.shape[1])
            if T_max > 0:
                xs = np.arange(T_max, dtype=np.int32)
                series: List[Tuple[str, np.ndarray, str]] = []

                def finite_rowwise_mean(mat: Optional[np.ndarray], label: str, color: str):
                    if mat is None or not np.isfinite(mat).any():
                        return
                    mrow = np.nanmean(mat, axis=0)
                    if np.isfinite(mrow).any():
                        series.append((label, mrow, color))

                finite_rowwise_mean(step_std_mat, "Instability (layer-trim mean)", "#1f77b4")
                finite_rowwise_mean(step_ent_mat, "Entropy (mean)", "#d62728")
                finite_rowwise_mean(step_cor_mat, "Head agreement (mean)", "#2ca02c")

                if series:
                    save_stepwise_overlay(xs, series, f"Stepwise means — {label}",
                                          os.path.join(gdir, "stepwise_overlay.png"),
                                          x_start=w0, x_end=w1)
                else:
                    print(f"[WARN] Skipping stepwise overlay for {label}: no finite data.")

            # 4) Violin for windowed instability (title shows window)
            ben_arr = np.array([s for (a, s) in scores if a == 0], dtype=np.float32)
            atk_arr = np.array([s for (a, s) in scores if a == 1], dtype=np.float32)
            save_violin(
                ben_arr, atk_arr,
                f"Windowed instability — {label} — steps {w0}–{w1}",
                os.path.join(gdir, "violin_windowed.png"),
                ylabel="Std across heads (trimmed layers)",
                w0=w0, w1=w1
            )

            # 5) Heatmaps (mean over samples), cropped to window
            if benign_heatmaps:
                save_mean_heatmap(
                    benign_heatmaps,
                    f"Benign mean heatmap — {label}",
                    os.path.join(gdir, "mean_heatmap_benign.png"),
                    x_start=w0, x_end=w1
                )
            if attack_heatmaps:
                save_mean_heatmap(
                    attack_heatmaps,
                    f"Attack mean heatmap — {label}",
                    os.path.join(gdir, "mean_heatmap_attack.png"),
                    x_start=w0, x_end=w1
                )

            # 6) Line charts (overwrite former scatter filenames); titles include window
            we_arr = np.array(windowed_entropy, dtype=np.float32)
            wc_arr = np.array(windowed_headcorr, dtype=np.float32)

            save_instability_entropy_lines(
                y_true=y_true,
                instab=sc,
                entropy=we_arr,
                outpath=os.path.join(gdir, "scatter_instability_vs_entropy.png"),
                window=(w0, w1),
            )

            save_instability_headcorr_lines_by_class(
                y_true=y_true,
                instab=sc,
                headcorr=wc_arr,
                outpath_attack=os.path.join(gdir, "scatter_instability_vs_headcorr_attack.png"),
                outpath_benign=os.path.join(gdir, "scatter_instability_vs_headcorr_benign.png"),
                window=(w0, w1),
            )

            # ---- Group-level Pearson/Spearman for CSV ----
            from scipy.stats import pearsonr, spearmanr
            p_ie = s_ie = p_ic = s_ic = float("nan")
            if sc.size:
                m_e = np.isfinite(sc) & np.isfinite(we_arr)
                if m_e.sum() >= 3 and len(np.unique(sc[m_e])) > 1:
                    try:
                        p_ie, _ = pearsonr(sc[m_e], we_arr[m_e])
                        s_ie, _ = spearmanr(sc[m_e], we_arr[m_e])
                    except Exception:
                        pass
                m_c = np.isfinite(sc) & np.isfinite(wc_arr)
                if m_c.sum() >= 3 and len(np.unique(sc[m_c])) > 1:
                    try:
                        p_ic, _ = pearsonr(sc[m_c], wc_arr[m_c])
                        s_ic, _ = spearmanr(sc[m_c], wc_arr[m_c])
                    except Exception:
                        pass

            cw.writerow([
                model, sysfile, dataset_file,
                w0, w1, head_frac, tail_frac,
                f"{p_ie:.6f}", f"{s_ie:.6f}",
                f"{p_ic:.6f}", f"{s_ic:.6f}",
                int(len(sc))
            ])

    print(f"[DONE] Wrote figures under: {os.path.abspath(args.outdir)}")
    print(f"[DONE] Group correlations CSV: {corr_csv_path}")

if __name__ == "__main__":
    main()
