#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate labeled graphs from run data, grouped by (model, system-prompt filename, dataset filename).

Supported inputs:
- A directory containing run_* subfolders, each with a config.json and *_std.npy / *.txt files
- A .zip archive with the same structure

Outputs (per group and per run):
- ROC curve (attack vs benign) using model-appropriate early windows
- Step-wise AUROC vs decoding step
- Mean heatmaps (benign vs attack) over layers x steps
- Distribution/box plots of windowed instability (benign vs attack)

The titles and filenames include: model name, system prompt filename, dataset filename.
Layer trimming: drop first 25% and last 15% of layers when computing windowed means.
Windows (decoding steps, inclusive; step 0 is prefill and ignored):
- Nous-* : steps 1..3
- Mistral-* : steps 11..40
You can override these with CLI flags.

Usage:
  python make_instability_figs.py --runs /path/to/runs_or_zip --outdir /path/to/out

Author: ChatGPT
"""
import os, io, re, json, argparse, zipfile, math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Iterable

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, auc

# ---------------------------- I/O layer ----------------------------

@dataclass
class RunConfig:
    model: str
    system_prompt_file: str
    baseline_prompts_file: str  # benign dataset filename
    fail_terms: List[str]
    lookback_steps: int
    start_steps: Optional[int]
    mid_high_frac: Optional[float]
    tail_cut_frac: Optional[float]
    start_steps: int
    mid_high_frac: float
    tail_cut_frac: float

@dataclass
class Sample:
    run: str
    kind: str  # 'baseline' or 'tests' or 'tests_ungated'
    iter: int
    index: int
    std: np.ndarray  # [steps, layers] or [T, L]
    text: Optional[str]

@dataclass
class RunData:
    run: str
    config: RunConfig
    samples: List[Sample]

def _basename(path: str) -> str:
    return os.path.basename(path).replace("\\", "/")

class Reader:
    """Abstract base reader."""
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

RUN_RE = re.compile(r'(baseline|tests|tests_ungated)_iter(\d+)_(\d+)_.*_std\.npy$')

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
    # Load texts for labeling success if needed
    txt_map: Dict[str, str] = {}
    for n in names:
        if n.endswith(".txt") and ("baseline_iter" in n or "tests_iter" in n):
            try:
                txt_map[_basename(n)] = reader.open(n).read().decode("utf-8", errors="ignore")
            except Exception:
                pass

    samples: List[Sample] = []
    for n in names:
        if not n.endswith("_std.npy"):
            continue
        base = _basename(n)
        m = RUN_RE.match(base)
        if not m:
            continue
        kind, it, idx = m.groups()
        try:
            arr = np.load(reader.open(n))
        except Exception as e:
            print(f"[WARN] Failed to load {n}: {e}")
            continue
        # Ensure shape [T, L]
        if arr.ndim == 2:
            steps, layers = arr.shape
        elif arr.ndim == 3:
            # Some exports may be [H,L,T] or [L,T,H]; try to reduce
            arr = np.nanmean(arr, axis=0)
            steps, layers = arr.shape
        else:
            print(f"[WARN] Unexpected array shape in {n}: {arr.shape}")
            continue
        txt_name = base.replace("_std.npy", ".txt")
        samples.append(Sample(
            run=run_dir,
            kind=kind,
            iter=int(it),
            index=int(idx),
            std=arr,   # [T, L]
            text=txt_map.get(txt_name)
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
    """Return [start, end) layer slice indices after trimming head and tail fractions."""
    s = max(0, int(L * head))
    e = max(s+1, int(L * (1.0 - tail)))
    e = min(e, L)
    return s, e

def per_sample_window_mean(std_mat: np.ndarray, step0: int, step1: int, head_frac: float, tail_frac: float) -> float:
    """Mean over layers [trimmed] and steps [step0..step1], ignoring prefill at 0 and clamping bounds."""
    T, L = std_mat.shape
    lo = max(1, int(step0))
    hi = min(int(step1), T - 1)
    if hi < lo:
        return np.nan
    ls, le = layer_slice(L, head_frac, tail_frac)
    window = std_mat[lo:hi+1, ls:le]  # [S, L']
    return float(np.nanmean(window)) if window.size else np.nan

def step_series(std_mat: np.ndarray, head_frac: float, tail_frac: float) -> np.ndarray:
    """Return per-step mean across trimmed layers (ignoring prefill step 0)."""
    T, L = std_mat.shape
    ls, le = layer_slice(L, head_frac, tail_frac)
    x = np.nanmean(std_mat[:, ls:le], axis=1)  # [T,]
    x[0] = np.nan  # mask prefill
    return x

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

def save_stepwise_auroc(y_true: np.ndarray, step_mat: np.ndarray, title: str, outpath: str):
    """Compute AUROC at each decode step using layer-trimmed per-step means."""
    # step_mat: [N, T] where T includes prefill 0 (NaN)
    T = step_mat.shape[1]
    xs, ys = [], []
    for k in range(1, T):  # skip prefill
        s = step_mat[:, k]
        m = np.isfinite(s)
        if m.sum() < 3 or len(np.unique(y_true[m])) < 2:
            xs.append(k); ys.append(np.nan); continue
        xs.append(k)
        ys.append(roc_auc_score(y_true[m], s[m]))
    plt.figure(figsize=(8,4.8))
    plt.plot(xs, ys, marker='o')
    plt.xlabel("Decoding step index")
    plt.ylabel("AUROC (attack vs benign)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()

def save_violin(ben: np.ndarray, atk: np.ndarray, title: str, outpath: str, ylabel: str):
    plt.figure(figsize=(7.2,4.8))
    data = [ben[~np.isnan(ben)], atk[~np.isnan(atk)]]
    parts = plt.violinplot(data, showmeans=True, showextrema=False, showmedians=False)
    plt.xticks([1,2],["Benign","Attack"])  # violinplot is 1-indexed on x
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()

def save_mean_heatmap(mats: List[np.ndarray], title: str, outpath: str):
    if not mats:
        return
    # Align by (L,T) via NaN padding
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
    return f"{model} | sys={sysfile} | data={dataset_file}"

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

    # Parse windows
    ns0, ns1 = [int(x) for x in args.nous_window.split(",")]
    ms0, ms1 = [int(x) for x in args.mistral_window.split(",")]

    # Collect runs
    runs = []
    for rd in find_run_dirs(reader):
        cfg = parse_config(reader, rd)
        if cfg is None:
            continue
        runs.append(parse_run(reader, rd, cfg))

    if not runs:
        print("[ERROR] No runs found.")
        return

    # Group by (model, sysfile, dataset)
    groups: Dict[Tuple[str,str,str], List[RunData]] = {}
    for r in runs:
        key = group_key(r.config)
        groups.setdefault(key, []).append(r)

    for (model, sysfile, dataset_file), rlist in groups.items():
        label = label_for_group(model, sysfile, dataset_file)
        gdir = os.path.join(args.outdir, f"{model.replace(' ','_')}_{sysfile}_{dataset_file}".replace(os.sep,"_"))
        ensure_dir(gdir)

        # Determine window for this group (prefer config.json if valid)
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

        # Gather per-sample measurements (group-level accumulators)
        scores: List[Tuple[int, float]] = []              # (is_attack, windowed score)
        step_series_list: List[np.ndarray] = []           # per-sample per-step means (trimmed layers)
        benign_heatmaps, attack_heatmaps = [], []         # for group mean heatmaps

        # ---- Per-run outputs (violin/scatter + CSV metrics) ----
        import csv
        group_csv = os.path.join(gdir, "per_run_metrics.csv")
        with open(group_csv, "w", newline="", encoding="utf-8") as fcsv:
            writer = csv.writer(fcsv)
            writer.writerow([
                "run_id","model","system_prompt","dataset_file",
                "window_start","window_end","head_frac","tail_frac",
                "n_benign","n_attack",
                "benign_mean","benign_median","benign_std",
                "attack_mean","attack_median","attack_std",
                "auroc_windowed","thr_at_5_fpr","tpr_at_thr","fpr_at_thr"
            ])

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

            for run in rlist:
                # Window & layer-trim for THIS RUN (fall back to CLI/model defaults if missing)
                rw0 = run.config.start_steps if run.config.start_steps is not None else (ms0 if 'mistral' in run.config.model.lower() else ns0)
                rw1 = run.config.lookback_steps if run.config.lookback_steps else (ms1 if 'mistral' in run.config.model.lower() else ns1)
                rhead = run.config.mid_high_frac if run.config.mid_high_frac is not None else args.head_frac
                rtail = run.config.tail_cut_frac if run.config.tail_cut_frac is not None else args.tail_frac

                # Collect per-run scores
                r_scores: List[Tuple[int,float]] = []
                for s in run.samples:
                    is_attack = 1 if s.kind == 'tests' else 0 if s.kind == 'baseline' else None
                    if is_attack is None:
                        continue
                    mat = np.array(s.std, dtype=np.float32)  # [T, L]
                    # per-sample windowed score (this run's window + trim)
                    score = per_sample_window_mean(mat, rw0, rw1, rhead, rtail)
                    r_scores.append((is_attack, score))

                    # accumulate at group level
                    scores.append((is_attack, score))
                    step_series_list.append(step_series(mat, rhead, rtail))
                    if is_attack:
                        attack_heatmaps.append(mat)
                    else:
                        benign_heatmaps.append(mat)

                if not r_scores:
                    continue

                # Per-run plots + metrics
                ry = np.array([a for a,_ in r_scores], dtype=np.int32)
                rs = np.array([v for _,v in r_scores], dtype=np.float32)

                rdir = os.path.join(gdir, run.run)

                # Violin (Benign vs Attack)
                bvals = np.array([v for a,v in r_scores if a==0], dtype=np.float32)
                avals = np.array([v for a,v in r_scores if a==1], dtype=np.float32)

                plt.figure(figsize=(7.2,4.8))
                data = [bvals[np.isfinite(bvals)], avals[np.isfinite(avals)]]
                plt.violinplot(data, showmeans=True, showextrema=False, showmedians=False)
                plt.xticks([1,2], ["Benign","Attack"])   # violinplot is 1-indexed
                plt.ylabel("Windowed instability")
                plt.title(f"{run.run} — {label} — steps {rw0}–{rw1} (trim {rhead:.2f}/{rtail:.2f})")
                plt.grid(axis="y", alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(gdir, "violin_windowed.png"), dpi=180)
                plt.close()

                # Scatter (jittered)
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

                # Per-run metrics
                try:
                    mask = np.isfinite(rs)
                    au = roc_auc_score(ry[mask], rs[mask]) if mask.sum() and len(np.unique(ry[mask]))>1 else float("nan")
                except Exception:
                    au = float("nan")
                thr, tprv, fprv = tpr_at_fixed_fpr(ry, rs, 0.05)

                b_mean, b_med, b_std = class_stats(bvals)
                a_mean, a_med, a_std = class_stats(avals)
                n_benign = int(np.isfinite(bvals).sum())
                n_attack = int(np.isfinite(avals).sum())

                writer.writerow([
                    run.run, model, sysfile, dataset_file,
                    rw0, rw1, rhead, rtail,
                    n_benign, n_attack,
                    f"{b_mean:.6f}", f"{b_med:.6f}", f"{b_std:.6f}",
                    f"{a_mean:.6f}", f"{a_med:.6f}", f"{a_std:.6f}",
                    f"{au:.6f}", f"{thr:.6f}", f"{tprv:.6f}", f"{fprv:.6f}",
                ])

        # ---- Plots per group ----
        # Build group-level arrays now that we accumulated from all runs
        if scores:
            y_true = np.array([a for a, _ in scores], dtype=np.int32)
            sc = np.array([s for _, s in scores], dtype=np.float32)
        else:
            y_true = np.array([], dtype=np.int32)
            sc = np.array([], dtype=np.float32)

        # Pad step-wise series (runs may have different T)
        def _pad_to_same_len(arrs):
            if not arrs:
                return None
            maxlen = max(a.shape[0] for a in arrs)
            out = np.full((len(arrs), maxlen), np.nan, dtype=np.float32)
            for i, a in enumerate(arrs):
                out[i, :a.shape[0]] = a
            return out

        step_mat = _pad_to_same_len(step_series_list)

        # Label the group window for titles:
        # If all runs in the group share the same window, show it; else say "per-run config windows".
        same_window = True
        if rlist:
            r0 = rlist[0].config
            s0 = r0.start_steps if r0.start_steps is not None else None
            e0 = r0.lookback_steps if r0.lookback_steps else None
            for rr in rlist[1:]:
                s = rr.config.start_steps if rr.config.start_steps is not None else None
                e = rr.config.lookback_steps if rr.config.lookback_steps else None
                if s != s0 or e != e0:
                    same_window = False
                    break
        if same_window and rlist and s0 is not None and e0 is not None:
            group_window_label = f"steps {s0}–{e0}"
        else:
            group_window_label = "per-run config windows"

        # 1) ROC (attack vs benign) over windowed scores
        save_roc(
            y_true, sc,
            f"ROC — {label} — {group_window_label}",
            os.path.join(gdir, "roc.png")
        )

        # 2) Step-wise AUROC (layer-trimmed per-step means)
        if step_mat is not None and step_mat.size:
            save_stepwise_auroc(
                y_true, step_mat,
                f"Step-wise AUROC — {label}",
                os.path.join(gdir, "stepwise_auroc.png")
            )

        # 3) Group violin (windowed instability; benign vs attack)
        ben_arr = np.array([s for (a, s) in scores if a == 0], dtype=np.float32)
        atk_arr = np.array([s for (a, s) in scores if a == 1], dtype=np.float32)
        save_violin(
            ben_arr, atk_arr,
            f"Windowed instability — {label} — {group_window_label}",
            os.path.join(gdir, "violin_windowed.png"),
            ylabel="Std across heads (trimmed layers)"
        )

        # 4) Heatmaps (mean over runs) for benign vs attack
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


        # Also: save a small text summary
        try:
            mask = np.isfinite(sc)
            au = roc_auc_score(y_true[mask], sc[mask]) if mask.sum() and len(np.unique(y_true[mask]))>1 else float("nan")
        except Exception:
            au = float("nan")
        with open(os.path.join(gdir, "summary.txt"), "w", encoding="utf-8") as f:
            f.write(f"group: {label}\n")
            f.write(f"window: steps {w0}..{w1}, layer trim: head={args.head_frac}, tail={args.tail_frac}\n")
            f.write(f"n_samples: {len(scores)}\n")
            f.write(f"AUROC (windowed): {au:.4f}\n")

    print(f"[DONE] Wrote figures under: {os.path.abspath(args.outdir)}")

if __name__ == "__main__":
    main()
