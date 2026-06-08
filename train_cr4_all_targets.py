#!/usr/bin/env python3
"""
Valeur — train CR4 Ridge probes against ALL THREE target families at once:
NRC-direct, NRCBERT (Piper's context-aware), EMO (EmoBank baseline).

Single SBERT encoding pass over CR4 passages, then fit independent RidgeCV
probes for each target family. Produces three artefacts:
    models/ridge_cr4_nrc.joblib
    models/ridge_cr4_nrcbert.joblib
    models/ridge_cr4_emo.joblib

Each artefact matches train_vad.py's schema, so encode_vad.py loads them
unmodified.

Example:
    python train_cr4_all_targets.py --passages data/cr4_passages.csv \\
        --out-dir models
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


TARGETS = ["nrc", "nrcbert", "emo"]


def bootstrap_r_ci(y_true, y_pred, n_boot, alpha=0.05):
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


def permutation_test_r(y_true, y_pred, n_perm):
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


def fit_one_target(
    target: str,
    X_tr_s: np.ndarray,
    X_va_s: np.ndarray,
    y_tr: np.ndarray,
    y_va: np.ndarray,
    w_tr: np.ndarray,
    n_boot: int,
    n_perm: int,
):
    """Fit a 3-dim Ridge probe for one target family. Returns (artefact_partial, diagnostics)."""
    y_scaler = StandardScaler().fit(y_tr)
    y_tr_s = y_scaler.transform(y_tr)

    models = []
    for dim in range(3):
        rc = RidgeCV(alphas=list(config.ALPHAS), cv=5)
        rc.fit(X_tr_s, y_tr_s[:, dim], sample_weight=w_tr)
        models.append(rc)

    preds_s = np.column_stack([m.predict(X_va_s) for m in models])
    preds   = y_scaler.inverse_transform(preds_s)

    diagnostics = {}
    print(f"\n[eval] target={target}")
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
            f"  {label:9s} r={r:+.3f}  rho={rho:+.3f}  RMSE={rmse:.3f}  MAE={mae:.3f}"
            f"  | 95% CI [{lo:+.3f}, {hi:+.3f}]  p={p:.4f}  alpha={models[i].alpha_:.2g}"
        )
    return models, y_scaler, diagnostics


def train_all(
    passages_csv: Path,
    out_dir: Path,
    n_boot: int = 500,
    n_perm: int = 500,
    embed_cache: Path | None = None,
):
    t0 = time.time()
    set_global_seed()

    df = pd.read_csv(passages_csv, low_memory=False)
    print(f"[load] {len(df):,} CR4 passages")

    train = df[df["split"] == "train"].reset_index(drop=True)
    val   = df[df["split"] == "val"].reset_index(drop=True)
    print(f"[split] train={len(train):,}  val={len(val):,}")

    # Drop rows missing ANY of the target columns (so the same passages support
    # every probe — clean comparison).
    target_cols = []
    for t in TARGETS:
        target_cols += [f"V_{t}", f"A_{t}", f"D_{t}"]
    train = train.dropna(subset=target_cols).reset_index(drop=True)
    val   = val.dropna(subset=target_cols).reset_index(drop=True)
    print(f"[clean] after dropna across all 3 target families: train={len(train):,}  val={len(val):,}")

    # ----------------------------------------------------------------- SBERT
    if embed_cache and embed_cache.exists():
        print(f"[cache] loading {embed_cache}")
        cached = np.load(embed_cache)
        X_tr, X_va = cached["X_tr"], cached["X_va"]
        if len(X_tr) != len(train) or len(X_va) != len(val):
            print("[cache] size mismatch — recomputing")
            X_tr = X_va = None
        else:
            print(f"[cache] X_tr={X_tr.shape}  X_va={X_va.shape}")
    else:
        X_tr = X_va = None

    if X_tr is None:
        sbert = load_sbert()
        print(f"[encode] {len(train):,} train passages → SBERT")
        X_tr = encode_texts(sbert, train["passage"].astype(str).tolist())
        print(f"[encode] {len(val):,} val passages → SBERT")
        X_va = encode_texts(sbert, val["passage"].astype(str).tolist())
        if embed_cache:
            embed_cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez(embed_cache, X_tr=X_tr, X_va=X_va)
            print(f"[cache] saved {embed_cache}")

    # ----------------------------------------------------------------- scale X
    x_scaler = StandardScaler().fit(X_tr)
    X_tr_s = x_scaler.transform(X_tr)
    X_va_s = x_scaler.transform(X_va)

    w_tr = train["n_labels"].values.astype(float)

    # ----------------------------------------------------------------- per-target
    out_dir.mkdir(parents=True, exist_ok=True)
    all_diagnostics = {}
    for t in TARGETS:
        print(f"\n{'='*60}\n[ridge] fitting target={t}\n{'='*60}")
        y_tr = train[[f"V_{t}", f"A_{t}", f"D_{t}"]].values.astype(float)
        y_va = val  [[f"V_{t}", f"A_{t}", f"D_{t}"]].values.astype(float)

        models, y_scaler, diagnostics = fit_one_target(
            t, X_tr_s, X_va_s, y_tr, y_va, w_tr, n_boot, n_perm
        )

        artefact = {
            "lexicon_name": f"cr4_{t}",
            "sbert_model":  config.SBERT_MODEL,
            "template":     "<passage>",
            "models":       models,
            "x_scaler":     x_scaler,
            "y_scaler":     y_scaler,
            "diagnostics":  diagnostics,
            "n_train":      int(len(train)),
            "n_test":       int(len(val)),
            "scale":        f"0 to 1 (CR4 release, target={t})",
            "trained_on":   f"cr4_narremote_passages_{t}",
            "n_books_train": int(train["file_id"].nunique()),
            "n_books_val":   int(val["file_id"].nunique()),
        }
        out_path = out_dir / f"ridge_cr4_{t}.joblib"
        joblib.dump(artefact, out_path)
        json_path = out_path.with_suffix(".diagnostics.json")
        with open(json_path, "w") as f:
            json.dump(diagnostics, f, indent=2)
        print(f"[save] {out_path}")
        print(f"[save] {json_path}")
        all_diagnostics[t] = diagnostics

    print(f"\n[done] total elapsed {time.time() - t0:.1f}s")

    # Summary table
    print(f"\n{'='*60}\nSUMMARY — held-out Pearson r per target × dim\n{'='*60}")
    print(f"{'target':12s} {'V':>8s} {'A':>8s} {'D':>8s}")
    for t in TARGETS:
        d = all_diagnostics[t]
        print(f"{t:12s} {d['Valence']['pearson_r']:+8.3f} "
              f"{d['Arousal']['pearson_r']:+8.3f} {d['Dominance']['pearson_r']:+8.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--passages", required=True, type=Path)
    ap.add_argument("--out-dir",  required=True, type=Path)
    ap.add_argument("--bootstrap", type=int, default=500)
    ap.add_argument("--perm",      type=int, default=500)
    ap.add_argument("--embed-cache", type=Path, default=Path("models/cr4_embeddings.npz"),
                    help="cache file for SBERT embeddings (saves ~7 min on rerun)")
    args = ap.parse_args()
    train_all(args.passages, args.out_dir,
              n_boot=args.bootstrap, n_perm=args.perm,
              embed_cache=args.embed_cache)


if __name__ == "__main__":
    main()
