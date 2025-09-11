#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterable
import numpy as np
import pandas as pd

# ---------- IO ----------
def load_run_dir(run_dir: Path) -> Optional[dict]:
    H_path = run_dir / "inter_head_std.csv"
    M_path = run_dir / "meta.json"
    if not H_path.exists() or not M_path.exists():
        return None
    try:
        H = pd.read_csv(H_path, header=None).to_numpy(dtype=float)  # [L, T] (T0=prefill)
        meta = json.loads(M_path.read_text())
        user_prompt = meta.get("user_prompt", "")
        sys_excerpt = meta.get("system_prompt_excerpt", "")
        run_id = str(run_dir)
        return {"H_std": H, "user_prompt": user_prompt, "system_excerpt": sys_excerpt, "run_id": run_id}
    except Exception as e:
        print(f"[WARN] Skipping {run_dir}: {e}")
        return None

def iter_leaf_dirs(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    for d in sorted(root.glob("*")):
        if d.is_dir():
            yield d

def load_all(attacks_root: Path, benign_root: Path) -> pd.DataFrame:
    rows = []
    for root, lab in [(attacks_root, 1), (benign_root, 0)]:
        if not root.exists():
            print(f"[WARN] Missing folder: {root}")
            continue
        for d in iter_leaf_dirs(root):
            rec = load_run_dir(d)
            if rec is None:
                continue
            rec["label"] = lab
            rec["suite"] = (Path(rec["run_id"]).parent.name + "::" + rec["system_excerpt"][:60]).strip()
            rows.append(rec)
    if not rows:
        raise RuntimeError("No runs found under attacks/benign roots.")
    return pd.DataFrame(rows)

# ---------- Metric ----------
def compute_score_window(
    H_std: np.ndarray,
    start_step: int,
    end_step: Optional[int],
    mid_high_frac: float,
    tail_cut_frac: float
) -> float:
    """
    H_std: [L, T], T includes prefill at col 0.
    Score = mean over layers [start_L:end_L) and steps [start_step..end_step], excluding prefill.
    """
    if not isinstance(H_std, np.ndarray) or H_std.ndim != 2:
        return float("nan")
    L, T = H_std.shape
    if T <= 1 or L < 2:
        return float("nan")

    # ---- layer slice (mid/high with tail cut)
    start_L = max(0, int(L * mid_high_frac))
    end_L = int(L * (1.0 - tail_cut_frac))
    end_L = max(start_L + 1, min(L, end_L))
    start_L = min(start_L, end_L - 1)

    # ---- time slice (skip prefill 0)
    t0 = max(1, int(start_step))
    t1 = int(end_step) if end_step is not None else (T - 1)
    t1 = min(t1, T - 1)
    if t1 < t0:
        return float("nan")

    window = H_std[start_L:end_L, t0:t1+1]
    return float(np.nanmean(window)) if window.size else float("nan")

def score_grid(
    df: pd.DataFrame,
    start_steps: Iterable[int],
    end_steps: Iterable[int],
    starts: Iterable[float],
    tails: Iterable[float],
) -> pd.DataFrame:
    """
    For each run and (start_step, end_step, mid_high_frac, tail_cut_frac), compute score.
    Returns long-form DataFrame with: run_id, label, suite, user_prompt, params..., score
    """
    out = []
    for _, row in df.iterrows():
        H = row["H_std"]
        for ss in start_steps:
            for ee in end_steps:
                if ee <= ss:  # end must be after start
                    continue
                for s in starts:
                    for t in tails:
                        sc = compute_score_window(H, start_step=ss, end_step=ee,
                                                  mid_high_frac=float(s), tail_cut_frac=float(t))
                        out.append({
                            "run_id": row["run_id"],
                            "label": int(row["label"]),
                            "suite": row["suite"],
                            "user_prompt": row["user_prompt"],
                            "start_step": int(ss),
                            "end_step": int(ee),
                            "mid_high_frac": float(s),
                            "tail_cut_frac": float(t),
                            "score": sc
                        })
    return pd.DataFrame(out)

# ---------- Thresholding ----------
def metrics_at_threshold(scores: np.ndarray, labels: np.ndarray, thr: float) -> Dict[str, float]:
    pred = (scores >= thr).astype(int)  # 1=attack if above threshold
    tp = int(np.sum((pred==1) & (labels==1)))
    fp = int(np.sum((pred==1) & (labels==0)))
    tn = int(np.sum((pred==0) & (labels==0)))
    fn = int(np.sum((pred==0) & (labels==1)))
    tpr = tp / (tp + fn + 1e-9)
    fpr = fp / (fp + tn + 1e-9)
    prec = tp / (tp + fp + 1e-9)
    rec  = tpr
    f1 = 2*prec*rec / (prec + rec + 1e-9)
    j = tpr - fpr
    return dict(tp=tp, fp=fp, tn=tn, fn=fn, tpr=tpr, fpr=fpr, precision=prec, recall=rec, f1=f1, youdenJ=j)

def thresholds_from_scores(scores: np.ndarray) -> np.ndarray:
    vals = scores[~np.isnan(scores)]
    if vals.size == 0:
        return np.array([])
    uniq = np.unique(vals)
    # include -inf and +inf guards so we can always find a bracket
    return np.concatenate(([-np.inf], uniq, [np.inf]))

def best_thr_at_fpr(scores: np.ndarray, labels: np.ndarray, fpr_target: float) -> Dict[str, float]:
    """
    Choose threshold that achieves FPR ≤ target with maximum TPR.
    If none achieve ≤ target, pick the closest (minimal FPR gap).
    """
    cands = thresholds_from_scores(scores)
    if cands.size == 0:
        return {"thr": np.nan, "tpr": np.nan, "fpr": np.nan}
    best_idx = None
    best_tpr = -1.0
    best_fpr = 1.0
    # First pass: valid region (fpr <= target), maximize TPR, tie-break by lower FPR then lower thr
    for i, thr in enumerate(cands):
        m = metrics_at_threshold(scores, labels, thr)
        if m["fpr"] <= fpr_target:
            if (m["tpr"] > best_tpr) or (np.isclose(m["tpr"], best_tpr) and (m["fpr"] < best_fpr)):
                best_idx, best_tpr, best_fpr = i, m["tpr"], m["fpr"]
    if best_idx is not None:
        thr = float(cands[best_idx])
        return {"thr": thr, "tpr": best_tpr, "fpr": best_fpr}
    # Fallback: pick threshold whose FPR is closest *below or above* target, tie-break by max TPR
    best_idx = None
    best_gap = 9e9
    best_tpr = -1.0
    for i, thr in enumerate(cands):
        m = metrics_at_threshold(scores, labels, thr)
        gap = abs(m["fpr"] - fpr_target)
        if (gap < best_gap) or (np.isclose(gap, best_gap) and (m["tpr"] > best_tpr)):
            best_idx, best_gap, best_tpr, best_fpr = i, gap, m["tpr"], m["fpr"]
    thr = float(cands[best_idx])
    return {"thr": thr, "tpr": best_tpr, "fpr": best_fpr}

def approx_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels==1]; neg = scores[labels==0]
    if len(pos)==0 or len(neg)==0: return float("nan")
    # rank-based AUROC approximation (Mann–Whitney U)
    s = pd.Series(scores)
    r = s.rank(method="average")
    n_pos, n_neg = len(pos), len(neg)
    u = r[labels==1].sum() - n_pos*(n_pos+1)/2
    return float(u / (n_pos * n_neg))

def summarize_by_params(scores_df: pd.DataFrame, fpr_targets=(0.01, 0.05, 0.10)) -> pd.DataFrame:
    rows = []
    for (ss, ee, s, t), g in scores_df.groupby(["start_step","end_step","mid_high_frac","tail_cut_frac"]):
        sc = g["score"].to_numpy()
        lb = g["label"].to_numpy()
        # drop NaNs jointly
        m = np.isfinite(sc)
        sc = sc[m]; lb = lb[m]
        if sc.size == 0 or lb.size == 0:
            continue
        auroc = approx_auroc(sc, lb)
        # global bests (F1 / Youden J) — not required for FPR targets, but useful diagnostics
        cands = thresholds_from_scores(sc)
        best_F1_thr, best_F1 = np.nan, -1.0
        best_J_thr,  best_J  = np.nan, -1.0
        for thr in cands:
            met = metrics_at_threshold(sc, lb, thr)
            if met["f1"] > best_F1:
                best_F1 = met["f1"]; best_F1_thr = float(thr)
            if met["youdenJ"] > best_J:
                best_J = met["youdenJ"]; best_J_thr = float(thr)

        row = {
            "start_step": ss,
            "end_step": ee,
            "mid_high_frac": s,
            "tail_cut_frac": t,
            "n": len(g),
            "AUROC": auroc,
            "thr_F1": best_F1_thr,
            "F1_at_thr": best_F1,
            "thr_J": best_J_thr,
            "J_at_thr": best_J,
        }
        # add FPR targets
        for ft in fpr_targets:
            pick = best_thr_at_fpr(sc, lb, ft)
            row[f"thr_FPR@{int(ft*100)}%"] = pick["thr"]
            row[f"TPR_at_FPR@{int(ft*100)}%"] = pick["tpr"]
            row[f"FPR_at_FPR@{int(ft*100)}%"] = pick["fpr"]
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    # Order: prefer higher TPR@1%, then AUROC, then F1
    order_cols = [f"TPR_at_FPR@1%","AUROC","F1_at_thr"]
    for oc in order_cols:
        if oc not in rows[0]:
            order_cols.remove(oc)
    df = pd.DataFrame(rows).sort_values(order_cols, ascending=[False]*len(order_cols)).reset_index(drop=True)
    return df

def recommend(summary: pd.DataFrame, target_fpr: float) -> Dict[str, float]:
    key_thr = f"thr_FPR@{int(target_fpr*100)}%"
    key_tpr = f"TPR_at_FPR@{int(target_fpr*100)}%"
    key_fpr = f"FPR_at_FPR@{int(target_fpr*100)}%"
    cand = summary[summary[key_thr].notna()].copy()
    if not len(cand):
        # fall back to overall best J (or F1) if FPR slice missing
        best = summary.iloc[0]
        thr = best["thr_J"] if not pd.isna(best["thr_J"]) else best["thr_F1"]
        return {
            "start_step": int(best["start_step"]),
            "end_step": int(best["end_step"]),
            "mid_high_frac": float(best["mid_high_frac"]),
            "tail_cut_frac": float(best["tail_cut_frac"]),
            "threshold": float(thr),
            "achieved_TPR": float("nan"),
            "achieved_FPR": float("nan"),
            "AUROC": float(best.get("AUROC", np.nan)),
            "F1_at_thr": float(best.get("F1_at_thr", np.nan)),
        }
    # pick maximum TPR at target, tie-break by lower FPR, then higher AUROC
    cand = cand.sort_values(by=[key_tpr, key_fpr, "AUROC"], ascending=[False, True, False])
    best = cand.iloc[0]
    return {
        "start_step": int(best["start_step"]),
        "end_step": int(best["end_step"]),
        "mid_high_frac": float(best["mid_high_frac"]),
        "tail_cut_frac": float(best["tail_cut_frac"]),
        "threshold": float(best[key_thr]),
        "achieved_TPR": float(best[key_tpr]),
        "achieved_FPR": float(best[key_fpr]),
        "AUROC": float(best.get("AUROC", np.nan)),
        "F1_at_thr": float(best.get("F1_at_thr", np.nan)),
    }

# ---------- Auto window sweep helpers ----------
def infer_Tmax(df: pd.DataFrame) -> int:
    Tmax = 0
    for _, r in df.iterrows():
        H = r["H_std"]
        if isinstance(H, np.ndarray) and H.ndim == 2 and H.shape[1] > Tmax:
            Tmax = H.shape[1]
    return max(1, Tmax - 1)  # exclude prefill

def build_auto_windows(Tmax: int, start_cap: int = 20, max_span: int = 60) -> Tuple[List[int], List[int]]:
    """
    Start in [1..min(start_cap, Tmax-1)], end in [start+1 .. min(start+max_span, Tmax)].
    Returns deduplicated sorted unique starts and ends.
    """
    starts = list(range(1, min(start_cap, Tmax - 1) + 1))
    ends = sorted(set(
        min(Tmax, s + d)
        for s in starts
        for d in (3,5,8,10,15,20,30,40,50, max_span)
        if s + d <= Tmax
    ))
    return starts, ends

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Tune head-instability detector: best window, layer trim, and threshold for 1/5/10% FPR.")
    ap.add_argument("--attacks-root", required=True)
    ap.add_argument("--benign-root",  required=True)
    ap.add_argument("--out-csv", default="")

    # Manual grid (optional)
    ap.add_argument("--start-steps", type=int, nargs="*", default=None, help="Explicit start steps (prefill=0; 1=first generated token).")
    ap.add_argument("--end-steps",   type=int, nargs="*", default=None, help="Explicit end steps (inclusive).")
    ap.add_argument("--mid-high-fracs", type=float, nargs="*", default=[0.20, 0.25, 0.33, 0.40])
    ap.add_argument("--tail-cut-fracs", type=float, nargs="*", default=[0.05, 0.10, 0.15])

    # Auto window sweep
    ap.add_argument("--auto-window", action="store_true", help="Sweep starts/ends derived from data (recommended).")
    ap.add_argument("--auto-start-cap", type=int, default=20)
    ap.add_argument("--auto-max-span", type=int, default=60)

    # FPR targets
    ap.add_argument("--fprs", type=float, nargs="*", default=[0.01, 0.05, 0.10])

    args = ap.parse_args()

    df = load_all(Path(args.attacks_root), Path(args.benign_root))
    print(f"Loaded runs: {len(df)} | attacks={int((df['label']==1).sum())} | benign={int((df['label']==0).sum())}")

    # Build start/end grids
    if args.auto_window or args.start_steps is None or args.end_steps is None:
        Tmax = infer_Tmax(df)
        starts, ends = build_auto_windows(Tmax, start_cap=args.auto_start_cap, max_span=args.auto_max_span)
        print(f"[auto-window] Tmax={Tmax} → starts={starts[:5]}...{starts[-3:]}, ends={ends[:5]}...{ends[-3:]}")
    else:
        starts = sorted(set([int(x) for x in args.start_steps if x >= 1]))
        ends   = sorted(set([int(x) for x in args.end_steps if x >= 2]))

    # Grid scoring
    scores_df = score_grid(
        df,
        start_steps=starts,
        end_steps=ends,
        starts=tuple(args.mid_high_fracs),
        tails=tuple(args.tail_cut_fracs),
    )
    if not len(scores_df):
        raise RuntimeError("No scores were computed (empty grid or bad inputs).")

    summary = summarize_by_params(scores_df, fpr_targets=tuple(args.fprs))
    if not len(summary):
        raise RuntimeError("Empty summary — no valid parameter combinations produced finite scores.")

    # pd.set_option("display.max_columns", 200)
    # print("\n=== Per-parameter summary (top 12) ===")
    # print(summary.head(12).to_string(index=False))

    # Recommendations for each target FPR
    print("\n=== Recommended operating points ===")
    for fpr_t in args.fprs:
        rec = recommend(summary, target_fpr=fpr_t)
        print(f"\n-- Target FPR ≤ {int(fpr_t*100)}% --")
        print(f"start_step    = {rec['start_step']}")
        print(f"end_step      = {rec['end_step']}")
        print(f"mid_high_frac = {rec['mid_high_frac']:.3f}")
        print(f"tail_cut_frac = {rec['tail_cut_frac']:.3f}")
        print(f"threshold     = {rec['threshold']:.6f}")
        print(f"achieved TPR  = {rec['achieved_TPR']:.3f}")
        print(f"achieved FPR  = {rec['achieved_FPR']:.3f}")
        print(f"AUROC         = {rec['AUROC']:.3f}")
        print(f"F1_at_thr     = {rec['F1_at_thr']:.3f}")

    if args.out_csv:
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.out_csv, index=False)
        print(f"\nSaved summary to {args.out_csv}")

if __name__ == "__main__":
    main()