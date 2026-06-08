#!/usr/bin/env python3
"""
Valeur — train a Ridge probe on CR4-NarrEmote passages.

Input:  cr4_passages.csv (produced by prepare_cr4_passages.py)
Output: a .joblib artefact with EXACT same schema as train_vad.py's output,
        so encode_vad.py can load it unmodified.

Differences from train_vad.py:
  - Training input is CR4 single-sentence passages (not "The word {}." templates)
  - Training target is citizen-aggregated VAD (mean over 5+ labels per passage)
  - Train/val split is by book (file_id), not random word split
  - Sample weighting by n_labels (more annotators = more reliable target)

Example:
    python train_cr4_probe.py --passages data/cr4_passages.csv \\
        --out models/ridge_cr4_nrc.joblib --target nrc
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import config
from utils import encode_texts, load_sbert, set_global_seed


def bootstrap_r_ci(y_true: np.ndarray, y_pred: np.ndarray, n_boot: int, alpha: float = 0.05):
    n = len(y_true)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        try:
            stats[b] = pearsonr(y_true[idx], y_pred[idx])[0]
        except Exception:
            stats[b] = 0.0
    return float(np.mean(stats)), (
        float(np.percentile(stats, 100 * alpha / 2)),
        float(np.percentile(stats, 100 * (1 - alpha / 2))),
    )


def permutation_test_r(y_true: np.ndarray, y_pred: np.ndarray, n_perm: int) -> float:
    orig = pearsonr(y_true, y_pred)[0]
    count = 0
    for _ in range(n_perm):
        perm = np.random.permutation(y_pred)
        try:
            r_perm = pearsonr(y_true, perm)[0]
        except Exception:
            r_perm = 0.0
        if abs(r_perm) >= abs(orig):
            count += 1
    return (count + 1) / (n_perm + 1)


def train_cr4(
    passages_csv: Path,
    out_path: Path,
    target: str = "nrc",        # "nrc" or "nrcbert"
    n_boot: int = 500,
    n_perm: int = 500,
) -> dict:
    t0 = time.time()
    set_global_seed()

    df = pd.read_csv(passages_csv, low_memory=False)
    print(f"[load] {len(df):,} CR4 passages")

    train = df[df["split"] == "train"].reset_index(drop=True)
    val   = df[df["split"] == "val"].reset_index(drop=True)

    vcol, acol, dcol = f"V_{target}", f"A_{target}", f"D_{target}"
    target_cols = [vcol, acol, dcol]
    train = train.dropna(subset=target_cols).reset_index(drop=True)
    val   = val.dropna(subset=target_cols).reset_index(drop=True)
    print(f"[split] train={len(train):,}  val={len(val):,}  (target={target})")

    # ------------------------------------------------------------------ SBERT
    sbert = load_sbert()
    print(f"[encode] {len(train):,} train passages → SBERT")
    X_tr = encode_texts(sbert, train["passage"].astype(str).tolist())
    print(f"[encode] {len(val):,} val passages → SBERT")
    X_va = encode_texts(sbert, val["passage"].astype(str).tolist())

    y_tr = train[target_cols].values.astype(float)
    y_va = val[target_cols].values.astype(float)
    w_tr = train["n_labels"].values.astype(float)

    # ------------------------------------------------------------------ scale
    x_scaler = StandardScaler().fit(X_tr)
    y_scaler = StandardScaler().fit(y_tr)
    X_tr_s = x_scaler.transform(X_tr)
    X_va_s = x_scaler.transform(X_va)
    y_tr_s = y_scaler.transform(y_tr)

    # ------------------------------------------------------------------ Ridge
    print(f"[ridge] fitting 3 × RidgeCV (alphas={len(config.ALPHAS)}, cv=5, sample_weight=n_labels)")
    models = []
    for dim in range(3):
        rc = RidgeCV(alphas=list(config.ALPHAS), cv=5)
        rc.fit(X_tr_s, y_tr_s[:, dim], sample_weight=w_tr)
        models.append(rc)

    # ------------------------------------------------------------------ eval
    preds_s = np.column_stack([m.predict(X_va_s) for m in models])
    preds   = y_scaler.inverse_transform(preds_s)

    diagnostics = {}
    print(f"\n[eval] held-out performance (CR4 / target={target})")
    for i, label in enumerate(["Valence", "Arousal", "Dominance"]):
        yt = y_va[:, i]
        yp = preds[:, i]
        r    = float(pearsonr(yt, yp)[0])
        rho  = float(spearmanr(yt, yp).correlation)
        rmse = math.sqrt(mean_squared_error(yt, yp))
        mae  = mean_absolute_error(yt, yp)
        r_mean, (lo, hi) = bootstrap_r_ci(yt, yp, n_boot=n_boot)
        p = permutation_test_r(yt, yp, n_perm=n_perm)
        diagnostics[label] = {
            "pearson_r":        r,
            "spearman_rho":     rho,
            "rmse":             rmse,
            "mae":              mae,
            "bootstrap_r_mean": r_mean,
            "bootstrap_ci_95":  [lo, hi],
            "permutation_p":    p,
            "chosen_alpha":     float(models[i].alpha_),
        }
        print(
            f"  {label:9s} r={r:.3f}  rho={rho:.3f}  RMSE={rmse:.3f}  MAE={mae:.3f}"
            f"  | 95% CI [{lo:.3f}, {hi:.3f}]  p={p:.4f}  alpha={models[i].alpha_:.2g}"
        )

    # ------------------------------------------------------------------ save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    artefact = {
        "lexicon_name": f"cr4_{target}",
        "sbert_model":  config.SBERT_MODEL,
        "template":     "<passage>",     # passage text used as-is, no template
        "models":       models,
        "x_scaler":     x_scaler,
        "y_scaler":     y_scaler,
        "diagnostics":  diagnostics,
        "n_train":      int(len(train)),
        "n_test":       int(len(val)),
        "scale":        "0 to 1 (NRC v2.1 normalized, CR4 release)",
        "trained_on":   f"cr4_narremote_passages_{target}",
        "n_books_train": int(train["file_id"].nunique()),
        "n_books_val":   int(val["file_id"].nunique()),
    }
    joblib.dump(artefact, out_path)
    print(f"\n[save] {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")
    print(f"[done] elapsed {time.time() - t0:.1f}s")
    return diagnostics


def main():
    ap = argparse.ArgumentParser(description="Train CR4 Ridge VAD probe")
    ap.add_argument("--passages", required=True, type=Path)
    ap.add_argument("--out",      required=True, type=Path)
    ap.add_argument("--target",   default="nrc", choices=["nrc", "nrcbert"])
    ap.add_argument("--bootstrap", type=int, default=500)
    ap.add_argument("--perm",      type=int, default=500)
    args = ap.parse_args()

    diagnostics = train_cr4(
        passages_csv=args.passages,
        out_path=args.out,
        target=args.target,
        n_boot=args.bootstrap,
        n_perm=args.perm,
    )
    json_path = args.out.with_suffix(".diagnostics.json")
    with open(json_path, "w") as f:
        json.dump(diagnostics, f, indent=2)
    print(f"[save] {json_path}")


if __name__ == "__main__":
    main()
