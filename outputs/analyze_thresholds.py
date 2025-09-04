#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
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

def load_all(attacks_root: Path, benign_root: Path) -> pd.DataFrame:
    rows = []
    for root, lab in [(attacks_root, 1), (benign_root, 0)]:
        if not root.exists():
            print(f"[WARN] Missing folder: {root}")
            continue
        for d in sorted(root.glob("*")):
            if not d.is_dir(): 
                continue
            rec = load_run_dir(d)
            if rec is None:
                continue
            rec["label"] = lab
            rec["suite"] = (Path(rec["run_id"]).parent.name + "::" + rec["system_excerpt"][:60]).strip()
            rows.append(rec)
    if not rows:
        raise RuntimeError("No runs found under attacks/benign roots.")
    return pd.DataFrame(rows)

# ---------- Metric (Mistral-tuned) ----------
def compute_score_window(
    H_std: np.ndarray,
    start_step: int = 11,   # skip early phase; 0=prefill so 11 = 11th generated token
    end_step: Optional[int] = 40,   # include this step (clipped to T-1)
    mid_high_frac: float = 0.25,    # start mid/high layers ~¼ depth
    tail_cut_frac: float = 0.15     # drop deepest 15% (noisy tail)
) -> float:
    """
    H_std: [L, T], T includes prefill at col 0.
    Score = mean over layers [start:end) and steps [start_step..end_step], excluding prefill.
    """
    L, T = H_std.shape
    if T <= 1:
        return float("nan")

    # ---- layer slice (mid/high with tail cut)
    start_L = max(2, int(L * mid_high_frac))
    end_L = max(start_L + 1, int(L * (1.0 - tail_cut_frac)))
    start_L = min(start_L, L - 1)
    end_L = min(max(end_L, start_L + 1), L)

    # ---- time slice (skip prefill 0)
    t0 = max(1, int(start_step))             # at least 1 to exclude prefill
    t1 = int(end_step) if end_step is not None else (T - 1)
    t1 = min(t1, T - 1)                      # clamp to last generated step
    if t1 < t0:
        return float("nan")

    window = H_std[start_L:end_L, t0:t1+1]
    return float(np.mean(window)) if window.size else float("nan")

def score_grid(
    df: pd.DataFrame,
    start_steps=(11, 15),          # late-start sweep (Mistral)
    end_steps=(30, 40, 50),        # where signal tends to peak/sustain
    starts=(0.25, 0.33, 0.40),     # mid/high layer start sweep
    tails=(0.10, 0.15)             # tail cut sweep
) -> pd.DataFrame:
    """
    For each run and (start_step, end_step, start_L, tail_L), compute score.
    Returns long-form DataFrame with: run_id, label, suite, user_prompt, params..., score
    """
    out = []
    for _, row in df.iterrows():
        H = row["H_std"]
        for ss in start_steps:
            for ee in end_steps:
                for s in starts:
                    for t in tails:
                        sc = compute_score_window(H, start_step=ss, end_step=ee,
                                                  mid_high_frac=s, tail_cut_frac=t)
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

def pick_thresholds(scores: np.ndarray, labels: np.ndarray, fpr_targets=(0.01, 0.05)):
    uniq = np.unique(scores[~np.isnan(scores)])
    if len(uniq) == 0:
        return {"by_target": {}, "best_F1": {"thr": np.nan, "f1": np.nan}, "best_J": {"thr": np.nan, "J": np.nan}}
    cands = np.concatenate(([-np.inf], uniq, [np.inf]))
    best_F1 = {"thr": None, "f1": -1.0}
    best_J  = {"thr": None, "J": -1.0}
    rows = []
    for thr in cands:
        m = metrics_at_threshold(scores, labels, thr)
        rows.append((thr, m))
        if m["f1"] > best_F1["f1"]:
            best_F1 = {"thr": float(thr), "f1": float(m["f1"])}
        if m["youdenJ"] > best_J["J"]:
            best_J = {"thr": float(thr), "J": float(m["youdenJ"])}
    dfm = pd.DataFrame([{"thr": t, **m} for t, m in rows]).sort_values("thr")
    out = {}
    for t in fpr_targets:
        sub = dfm[dfm["fpr"] <= t]
        row = sub.iloc[0] if len(sub) else dfm.iloc[dfm["fpr"].argmin()]
        out[f"FPR@{int(t*100)}%"] = {"thr": float(row["thr"]), "fpr": float(row["fpr"]), "tpr": float(row["tpr"])}
    return {"by_target": out, "best_F1": best_F1, "best_J": best_J}

def approx_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels==1]; neg = scores[labels==0]
    if len(pos)==0 or len(neg)==0: return float("nan")
    ranks = pd.Series(scores).rank(method="average")
    return float((ranks[labels==1].mean() - (len(pos)+1)/2) / len(neg))

def summarize_by_params(scores_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (ss, ee, s, t), g in scores_df.groupby(["start_step","end_step","mid_high_frac","tail_cut_frac"]):
        sc = g["score"].to_numpy()
        lb = g["label"].to_numpy()
        picks = pick_thresholds(sc, lb)
        auroc = approx_auroc(sc, lb)
        row = {
            "start_step": ss,
            "end_step": ee,
            "mid_high_frac": s,
            "tail_cut_frac": t,
            "n": len(g),
            "n_attacks": int((lb==1).sum()),
            "n_benign": int((lb==0).sum()),
            "AUROC": auroc,
            "thr_F1": picks["best_F1"]["thr"],
            "F1_at_thr": picks["best_F1"]["f1"],
            "thr_J": picks["best_J"]["thr"],
            "J_at_thr": picks["best_J"]["J"],
        }
        for key, val in picks["by_target"].items():
            row[f"thr_{key}"] = val["thr"]
            row[f"{key}_TPR"] = val["tpr"]
        rows.append(row)
    df = pd.DataFrame(rows).sort_values(["AUROC","F1_at_thr"], ascending=[False,False]).reset_index(drop=True)
    return df

def recommend(summary: pd.DataFrame, target_fpr: float) -> Dict[str, float]:
    target = f"FPR@{int(target_fpr*100)}%"
    thr_col = f"thr_{target}"
    tpr_col = f"{target}_TPR"
    cand = summary[summary[thr_col].notna()].copy()
    if not len(cand):
        best = summary.iloc[0]
        thr = best["thr_J"] if not pd.isna(best["thr_J"]) else best["thr_F1"]
        return {
            "start_step": int(best["start_step"]),
            "end_step": int(best["end_step"]),
            "mid_high_frac": float(best["mid_high_frac"]),
            "tail_cut_frac": float(best["tail_cut_frac"]),
            "threshold": float(thr),
            "achieved_TPR": float("nan"),
            "AUROC": float(best["AUROC"]),
            "F1_at_thr": float(best["F1_at_thr"])
        }
    cand = cand.sort_values(by=[tpr_col, "AUROC", "F1_at_thr"], ascending=[False, False, False])
    best = cand.iloc[0]
    return {
        "start_step": int(best["start_step"]),
        "end_step": int(best["end_step"]),
        "mid_high_frac": float(best["mid_high_frac"]),
        "tail_cut_frac": float(best["tail_cut_frac"]),
        "threshold": float(best[thr_col]),
        "achieved_TPR": float(best[tpr_col]),
        "AUROC": float(best["AUROC"]),
        "F1_at_thr": float(best["F1_at_thr"])
    }

# ---------- CV (optional) ----------
def leave_one_prompt_out(scores_df: pd.DataFrame, target_fpr: float) -> pd.DataFrame:
    prompts = sorted(scores_df["user_prompt"].unique().tolist())
    rows = []
    for held in prompts:
        train = scores_df[scores_df["user_prompt"] != held]
        test  = scores_df[scores_df["user_prompt"] == held]
        summ = summarize_by_params(train)
        rec  = recommend(summ, target_fpr=target_fpr)
        sub = test[
            (test["start_step"] == rec["start_step"]) &
            (test["end_step"]   == rec["end_step"]) &
            (np.isclose(test["mid_high_frac"], rec["mid_high_frac"])) &
            (np.isclose(test["tail_cut_frac"], rec["tail_cut_frac"]))
        ]
        if not len(sub):
            continue
        m = metrics_at_threshold(sub["score"].to_numpy(), sub["label"].to_numpy(), rec["threshold"])
        rows.append({
            "held_out_prompt": held[:80],
            **rec,
            "val_TPR": m["tpr"],
            "val_FPR": m["fpr"],
            "val_F1":  m["f1"],
        })
    return pd.DataFrame(rows)

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Tune head-instability detector (Mistral-late-window).")
    ap.add_argument("--attacks_root", default="outputs/tests")
    ap.add_argument("--benign_root",  default="outputs/baseline")
    ap.add_argument("--target_fpr", type=float, default=0.01)      # 1%
    ap.add_argument("--also_target_fpr", type=float, default=0.05) # 5%
    ap.add_argument("--cv", action="store_true")
    ap.add_argument("--out_csv", default="")

    # Mistral defaults
    ap.add_argument("--start_step", type=int, default=11)
    ap.add_argument("--end_step",   type=int, default=40)
    ap.add_argument("--mid_high_fracs", type=float, nargs="+", default=[0.25, 0.33, 0.40])
    ap.add_argument("--tail_cut_fracs", type=float, nargs="+", default=[0.10, 0.15])

    args = ap.parse_args()

    if not (0.0 < args.target_fpr < 1.0):
        raise ValueError(f"--target_fpr must be in (0,1), got {args.target_fpr}")
    if not (0.0 < args.also_target_fpr < 1.0):
        raise ValueError(f"--also_target_fpr must be in (0,1), got {args.also_target_fpr}")
    df = load_all(Path(args.attacks_root), Path(args.benign_root))
    print(f"Loaded runs: {len(df)} | attacks={int((df['label']==1).sum())} | benign={int((df['label']==0).sum())}")

    # Build grids from CLI defaults (you can pass multiple values)
    start_steps = [args.start_step] if isinstance(args.start_step, int) else list(args.start_step)
    end_steps   = [args.end_step]   if isinstance(args.end_step,   int) else list(args.end_step)

    scores_df = score_grid(
        df,
        start_steps=start_steps,
        end_steps=end_steps,
        starts=tuple(args.mid_high_fracs),
        tails=tuple(args.tail_cut_fracs),
    )
    summary   = summarize_by_params(scores_df)

    pd.set_option("display.max_columns", 200)
    print("\n=== Per-parameter summary (top 12) ===")
    print(summary.head(12).to_string(index=False))

    # Primary recommendation (target FPR)
    rec = recommend(summary, target_fpr=args.target_fpr)
    print("\n=== Recommended operating point ===")
    print(f"start_step    = {rec['start_step']}")
    print(f"end_step      = {rec['end_step']}")
    print(f"mid_high_frac = {rec['mid_high_frac']:.3f}")
    print(f"tail_cut_frac = {rec['tail_cut_frac']:.3f}")
    print(f"threshold     = {rec['threshold']:.6f}  (TPR@FPR≤{int(args.target_fpr*100)}%: {rec['achieved_TPR']:.3f})")
    print(f"AUROC={rec['AUROC']:.3f} | F1_at_thr={rec['F1_at_thr']:.3f}")

    # Secondary FPR point
    rec2 = recommend(summary, target_fpr=args.also_target_fpr)
    print("\n=== Secondary operating point ===")
    print(f"start_step    = {rec2['start_step']}")
    print(f"end_step      = {rec2['end_step']}")
    print(f"mid_high_frac = {rec2['mid_high_frac']:.3f}")
    print(f"tail_cut_frac = {rec2['tail_cut_frac']:.3f}")
    print(f"threshold     = {rec2['threshold']:.6f}  (TPR@FPR≤{int(args.also_target_fpr*100)}%: {rec2['achieved_TPR']:.3f})")
    print(f"AUROC={rec2['AUROC']:.3f} | F1_at_thr={rec2['F1_at_thr']:.3f}")

    if args.out_csv:
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.out_csv, index=False)
        print(f"\nSaved summary to {args.out_csv}")

    if args.cv:
        cvdf = leave_one_prompt_out(scores_df, target_fpr=args.target_fpr)
        if len(cvdf):
            print("\n=== Leave-one-prompt-out validation ===")
            print(cvdf.to_string(index=False))
            print("\nCV means:",
                  f"TPR={cvdf['val_TPR'].mean():.3f},",
                  f"FPR={cvdf['val_FPR'].mean():.3f},",
                  f"F1={cvdf['val_F1'].mean():.3f}")
        else:
            print("\n[CV] Not enough overlap to validate.")

if __name__ == "__main__":
    main()