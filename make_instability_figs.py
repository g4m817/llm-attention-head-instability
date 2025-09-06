#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate labeled graphs from run data, grouped by (model, system-prompt filename, dataset filename).

Adds:
- Entropy & head-agreement ingestion (per-sample *_entropy.npy, *_headcorr.npy)
- Scatter: windowed instability vs windowed entropy (+ Pearson/Spearman)
- Stepwise overlays: mean instability, entropy, head agreement (trimmed layers)
- CSV: correlation stats at group level

Usage:
  python make_instability_figs.py --runs /path/to/runs_or_zip --outdir /path/to/out
"""
import os, io, re, json, argparse, zipfile, math, csv
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
    baseline_prompts_file: str
    fail_terms: List[str]
    lookback_steps: int
    start_steps: int
    mid_high_frac: float
    tail_cut_frac: float

@dataclass
class Sample:
    run: str
    kind: str  # 'baseline' or 'tests' or 'tests_ungated'
    iter: int
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

RUN_STD_RE = re.compile(r'(baseline|tests|tests_ungated)_iter(\d+)_(\d+)_.*_std\.npy$')
RUN_ENT_RE = re.compile(r'(baseline|tests|tests_ungated)_iter(\d+)_(\d+)_.*_entropy\.npy$')
RUN_CORR_RE= re.compile(r'(baseline|tests|tests_ungated)_iter(\d+)_(\d+)_.*_headcorr\.npy$')

def find_run_dirs(reader: Reader) -> List[str]:
    names = reader.list()
    run_roots = sorted({n.split("/")[0] for n in names if "/" in n and n.startswith("run_")})
    return run_roots

def parse_config(reader: Reader, run_dir: str) -> Optional[RunConfig]:
    try:
        with reader.open(f"{run_dir}/config.json") as f:
            cfg = json.loads(f.read().decode("utf-8"))
        return RunConfig(
            model=cfg["model"],
            system_prompt_file=cfg["system_prompt_file"],
            baseline_prompts_file=cfg["baseline_prompts_file"],
            fail_terms=list(cfg.get("fail_terms", [])),
            lookback_steps=int(cfg.get("lookback_steps", 0)),
            start_steps=int(cfg.get("start_steps", 1)),
            mid_high_frac=float(cfg.get("mid_high_frac", 0.25)),
            tail_cut_frac=float(cfg.get("tail_cut_frac", 0.15)),
        )
    except Exception as e:
        print(f"[WARN] Failed to read config for {run_dir}: {e}")
        return None

def parse_run(reader: Reader, run_dir: str, cfg: RunConfig) -> RunData:
    names = [n for n in reader.list() if n.startswith(run_dir + "/")]
    txt_map: Dict[str, str] = {}
    for n in names:
        if n.endswith(".txt") and ("baseline_iter" in n or "tests_iter" in n):
            try:
                txt_map[_basename(n)] = reader.open(n).read().decode("utf-8", errors="ignore")
            except Exception:
                pass

    std_map, ent_map, corr_map = {}, {}, {}
    for n in names:
        base = _basename(n)
        if n.endswith("_std.npy"):
            m = RUN_STD_RE.match(base)
            if m:
                std_map[(m.group(1), int(m.group(2)), int(m.group(3)))] = np.load(reader.open(n))
        elif n.endswith("_entropy.npy"):
            m = RUN_ENT_RE.match(base)
            if m:
                ent_map[(m.group(1), int(m.group(2)), int(m.group(3)))] = np.load(reader.open(n))
        elif n.endswith("_headcorr.npy"):
            m = RUN_CORR_RE.match(base)
            if m:
                corr_map[(m.group(1), int(m.group(2)), int(m.group(3)))] = np.load(reader.open(n))

    samples: List[Sample] = []
    keys = set(list(std_map.keys()) + list(ent_map.keys()) + list(corr_map.keys()))
    for key in sorted(keys):
        kind, it, idx = key
        std = std_map.get(key)
        ent = ent_map.get(key)
        cor = corr_map.get(key)
        txt_name_guess = f"{kind}_iter{it}_{idx:03d}"
        # find matching .txt (best-effort)
        t = None
        for k in txt_map.keys():
            if txt_name_guess in k:
                t = txt_map[k]; break
        samples.append(Sample(
            run=run_dir,
            kind=kind,
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

def choose_window(model: str, default_nous=(1,3), default_mistral=(11,40)) -> Tuple[int,int]:
    m = model.lower()
    if "mistral" in m:
        return default_mistral
    return default_nous

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
    x[0] = np.nan
    return x

def pearson_spearman(x: np.ndarray, y: np.ndarray) -> Tuple[float,float]:
    from scipy.stats import pearsonr, spearmanr
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan"), float("nan")
    px, _ = pearsonr(x[m], y[m])
    sx, _ = spearmanr(x[m], y[m])
    return float(px), float(sx)

# ------------------------- plotting --------------------------------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_roc(y_true: np.ndarray, scores: np.ndarray, title: str, outpath: str):
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
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()

def save_stepwise_overlay(xs: np.ndarray, ys_list: List[Tuple[str, np.ndarray, str]], title: str, outpath: str):
    plt.figure(figsize=(8.4,5.0))
    for label, ys, color in ys_list:
        plt.plot(xs, ys, label=label, lw=2, alpha=0.9, color=color)
    plt.xlabel("Decoding step index")
    plt.ylabel("Mean (trimmed layers / series)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()

def save_scatter(x: np.ndarray, y: np.ndarray, title: str, outpath: str, xlabel: str, ylabel: str, subtitle: Optional[str]=None):
    m = np.isfinite(x) & np.isfinite(y)
    plt.figure(figsize=(6.0,5.0))
    plt.scatter(x[m], y[m], alpha=0.65, s=20)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if subtitle:
        plt.title(f"{title}\n{subtitle}")
    else:
        plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()

def save_violin(ben: np.ndarray, atk: np.ndarray, title: str, outpath: str, ylabel: str):
    plt.figure(figsize=(7.2,4.8))
    data = [ben[~np.isnan(ben)], atk[~np.isnan(atk)]]
    plt.violinplot(data, showmeans=True, showextrema=False, showmedians=False)
    plt.xticks([1,2],["Benign","Attack"])
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()

def save_mean_heatmap(mats: List[np.ndarray], title: str, outpath: str):
    if not mats:
        return
    Lmax = max(m.shape[1] for m in mats)
    Tmax = max(m.shape[0] for m in mats)
    stack = np.full((len(mats), Tmax, Lmax), np.nan, dtype=np.float32)
    for i, m in enumerate(mats):
        T, L = m.shape
        stack[i, :T, :L] = m
    mean_mat = np.nanmean(stack, axis=0)  # [T, L]
    plt.figure(figsize=(8,5.2))
    plt.imshow(mean_mat.T, aspect='auto', origin='lower', interpolation='nearest', cmap='viridis')
    plt.colorbar(label="Std across heads (system-token share)")
    plt.xlabel("Step (0=prefill)")
    plt.ylabel("Layer (shallow→deep)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()

# ------------------------- grouping logic --------------------------

def group_key(cfg: RunConfig) -> Tuple[str,str,str]:
    return (cfg.model, filename_only(cfg.system_prompt_file), filename_only(cfg.baseline_prompts_file))

def label_for_group(model: str, sysfile: str, dataset_file: str) -> str:
    return f"{model}"

# ------------------------------ main -------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="Path to root runs dir OR a .zip archive.")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--nous-window", type=str, default="1,3", help="start,end for Nous (default 1,3)")
    ap.add_argument("--mistral-window", type=str, default="11,40", help="start,end for Mistral (default 11,40)")
    ap.add_argument("--head-frac", type=float, default=0.25, help="fraction of shallow layers to drop")
    ap.add_argument("--tail-frac", type=float, default=0.15, help="fraction of deepest layers to drop")
    args = ap.parse_args()

    ensure_dir(args.outdir)
    reader = make_reader(args.runs)

    ns0, ns1 = [int(x) for x in args.nous_window.split(",")]
    ms0, ms1 = [int(x) for x in args.mistral_window.split(",")]

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
        key = group_key(r.config)
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
            label = label_for_group(model, sysfile, dataset_file)
            gdir = os.path.join(args.outdir, f"{model.replace(' ','_')}_{sysfile}_{dataset_file}".replace(os.sep,"_"))
            ensure_dir(gdir)

            cfg0 = rlist[0].config
            if cfg0.start_steps and cfg0.lookback_steps:
                w0, w1 = cfg0.start_steps, cfg0.lookback_steps
                head_frac, tail_frac = cfg0.mid_high_frac, cfg0.tail_cut_frac
            elif "mistral" in model.lower():
                w0, w1 = ms0, ms1
                head_frac, tail_frac = args.head_frac, args.tail_frac
            else:
                w0, w1 = ns0, ns1
                head_frac, tail_frac = args.head_frac, args.tail_frac

            # Accumulators
            scores: List[Tuple[int, float]] = []
            step_series_list: List[np.ndarray] = []
            benign_heatmaps, attack_heatmaps = [], []

            # For correlations
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

                def class_stats(arr: np.ndarray) -> Tuple[float,float,float]:
                    arr = arr[np.isfinite(arr)]
                    if arr.size == 0:
                        return (float("nan"), float("nan"), float("nan"))
                    return float(np.mean(arr)), float(np.median(arr)), float(np.std(arr))

                from sklearn.metrics import roc_curve, roc_auc_score
                def tpr_at_fixed_fpr(y: np.ndarray, s: np.ndarray, fpr_target=0.05) -> Tuple[float,float,float]:
                    mask = np.isfinite(s)
                    y = y[mask]; s = s[mask]
                    if len(y) < 3 or len(np.unique(y)) < 2:
                        return (float("nan"), float("nan"), float("nan"))
                    fpr, tpr, thr = roc_curve(y, s)
                    idx = int(np.argmin(np.abs(fpr - fpr_target)))
                    return float(thr[idx]), float(tpr[idx]), float(fpr[idx])

                for run in rlist:
                    rw0 = run.config.start_steps if run.config.start_steps is not None else (ms0 if 'mistral' in run.config.model.lower() else ns0)
                    rw1 = run.config.lookback_steps if run.config.lookback_steps else (ms1 if 'mistral' in run.config.model.lower() else ns1)
                    rhead = run.config.mid_high_frac if run.config.mid_high_frac is not None else head_frac
                    rtail = run.config.tail_cut_frac if run.config.tail_cut_frac is not None else tail_frac

                    r_scores: List[Tuple[int,float]] = []
                    r_entropy: List[float] = []
                    r_headcorr: List[float] = []

                    for s in run.samples:
                        is_attack = 1 if s.kind == 'tests' else 0 if s.kind == 'baseline' else None
                        if is_attack is None:
                            continue

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

                        # windowed entropy/headcorr (if present)
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

                    # Violin
                    plt.figure(figsize=(7.2,4.8))
                    data = [bvals[np.isfinite(bvals)], avals[np.isfinite(avals)]]
                    try:
                        plt.violinplot(data, showmeans=True, showextrema=False, showmedians=False)
                        plt.xticks([1,2], ["Benign","Attack"])
                        plt.ylabel("Windowed instability")
                        plt.title(f"{run.run} — {label} — steps {rw0}–{rw1} (trim {rhead:.2f}/{rtail:.2f})")
                        plt.grid(axis="y", alpha=0.3)
                        plt.tight_layout()
                        plt.savefig(os.path.join(gdir, "violin_windowed.png"), dpi=180)
                    finally:
                        plt.close()

                    # Scatter
                    plt.figure(figsize=(7.2,4.8))
                    if bvals.size:
                        bx = np.random.normal(loc=0.0, scale=0.03, size=bvals.shape[0])
                        plt.scatter(bx, bvals, marker="x", label="Benign", alpha=0.85)
                    if avals.size:
                        ax = np.random.normal(loc=1.0, scale=0.03, size=avals.shape[0])
                        plt.scatter(ax, avals, marker="x", label="Attack", alpha=0.85)
                    plt.xticks([0,1], ["Benign","Attack"])
                    plt.ylabel(f"Mean instability (steps {rw0}–{rw1}, trimmed layers)")
                    plt.title(f"{run.run} — {label}")
                    plt.grid(axis="y", linestyle="--", alpha=0.35)
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(os.path.join(gdir, "scatter_windowed.png"), dpi=180)
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
                    # align with rs by finite mask
                    p_inst_ent, s_inst_ent = np.nan, np.nan
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

            # Stepwise means — align different lengths
            def _pad(arrs):
                arrs = [a for a in arrs if isinstance(a, np.ndarray) and a.size]
                if not arrs:
                    return None
                maxlen = max(a.shape[0] for a in arrs)
                out = np.full((len(arrs), maxlen), np.nan, dtype=np.float32)
                for i, a in enumerate(arrs):
                    out[i, :a.shape[0]] = a
                return out

            # Build per-sample step series for std (we already have these), entropy, headcorr
            step_std_mat = _pad(step_series_list)

            # For entropy/headcorr, we don't have layer-trim. Use direct per-step means across samples.
            ent_series_list = []
            corr_series_list = []
            for r in rlist:
                for s in r.samples:
                    if isinstance(s.entropy, np.ndarray):
                        v = s.entropy.astype(np.float32)
                        v[0] = np.nan  # mask prefill
                        ent_series_list.append(v)
                    if isinstance(s.headcorr, np.ndarray):
                        v = s.headcorr.astype(np.float32)
                        v[0] = np.nan
                        corr_series_list.append(v)

            step_ent_mat = _pad(ent_series_list)
            step_cor_mat = _pad(corr_series_list)

            # 1) ROC on windowed instability
            save_roc(
                y_true, sc,
                f"ROC — {label} — steps {w0}–{w1}",
                os.path.join(gdir, "roc.png")
            )

            # 2) Stepwise overlays: mean instability (trimmed), entropy, headcorr
            # Build shared x-axis by max length across present mats
            T_max = max([m.shape[1] if m is not None else 0 for m in [step_std_mat]]) if step_std_mat is not None else 0
            T_max = max(T_max, step_ent_mat.shape[1] if step_ent_mat is not None else 0)
            T_max = max(T_max, step_cor_mat.shape[1] if step_cor_mat is not None else 0)
            if T_max > 0:
                xs = np.arange(T_max, dtype=np.int32)
                series = []
                if step_std_mat is not None:
                    series.append(("Instability (layer-trim mean)", np.nanmean(step_std_mat, axis=0), "#1f77b4"))
                if step_ent_mat is not None:
                    series.append(("Entropy (mean)", np.nanmean(step_ent_mat, axis=0), "#d62728"))
                if step_cor_mat is not None:
                    series.append(("Head agreement (mean)", np.nanmean(step_cor_mat, axis=0), "#2ca02c"))
                save_stepwise_overlay(xs, series, f"Stepwise means — {label}", os.path.join(gdir, "stepwise_overlay.png"))

            # 3) Violin for windowed instability
            ben_arr = np.array([s for (a, s) in scores if a == 0], dtype=np.float32)
            atk_arr = np.array([s for (a, s) in scores if a == 1], dtype=np.float32)
            save_violin(
                ben_arr, atk_arr,
                f"Windowed instability — {label} — steps {w0}–{w1}",
                os.path.join(gdir, "violin_windowed.png"),
                ylabel="Std across heads (trimmed layers)"
            )

            # 4) Heatmaps (mean over runs)
            if benign_heatmaps:
                save_mean_heatmap(
                    benign_heatmaps,
                    f"Benign mean heatmap — {label}",
                    os.path.join(gdir, "mean_heatmap_benign.png")
                )
            if attack_heatmaps:
                save_mean_heatmap(
                    attack_heatmaps,
                    f"Attack mean heatmap — {label}",
                    os.path.join(gdir, "mean_heatmap_attack.png")
                )

            # 5) Scatter: windowed instability vs windowed entropy (and vs headcorr)
            we_arr = np.array(windowed_entropy, dtype=np.float32)
            wc_arr = np.array(windowed_headcorr, dtype=np.float32)

            # compute correlations at group level
            from scipy.stats import pearsonr, spearmanr
            p_ie = s_ie = np.nan
            m = np.isfinite(sc) & np.isfinite(we_arr)
            if m.sum() >= 3:
                p_ie, _ = pearsonr(sc[m], we_arr[m])
                s_ie, _ = spearmanr(sc[m], we_arr[m])

            p_ic = s_ic = np.nan
            mc = np.isfinite(sc) & np.isfinite(wc_arr)
            if mc.sum() >= 3:
                p_ic, _ = pearsonr(sc[mc], wc_arr[mc])
                s_ic, _ = spearmanr(sc[mc], wc_arr[mc])

            subtitle = f"Pearson={p_ie:.3f}, Spearman={s_ie:.3f}"
            save_scatter(we_arr, sc,
                         f"Instability vs Entropy — {label}",
                         os.path.join(gdir, "scatter_instability_vs_entropy.png"),
                         xlabel="Windowed entropy", ylabel="Windowed instability",
                         subtitle=subtitle)
            subtitle2 = f"Pearson={p_ic:.3f}, Spearman={s_ic:.3f}"
            save_scatter(wc_arr, sc,
                         f"Instability vs Head agreement — {label}",
                         os.path.join(gdir, "scatter_instability_vs_headcorr.png"),
                         xlabel="Windowed head agreement (mean corr)",
                         ylabel="Windowed instability",
                         subtitle=subtitle2)

            # write correlations to summary csv
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
