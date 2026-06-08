#!/usr/bin/env python3
"""
Valeur — apply all 3 CR4 probe variants (nrc/nrcbert/emo) to Kafka windows
in a single SBERT pass. Writes three windows CSVs:

    results/windows_cr4_nrc.csv
    results/windows_cr4_nrcbert.csv
    results/windows_cr4_emo.csv

Each has the same schema as windows_warriner.csv / windows_nrc.csv.

Example:
    python encode_cr4_all.py --text data/metamorphosis.txt --out-dir results
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import config
from utils import (
    choose_odd_window,
    encode_texts,
    load_sbert,
    make_sliding_windows,
    segment_sentences,
    set_global_seed,
)


TARGETS = ["nrc", "nrcbert", "emo"]


def encode_all(text_path: Path, models_dir: Path, out_dir: Path) -> None:
    t0 = time.time()
    set_global_seed()

    # Segment + window
    sentences = segment_sentences(text_path)
    win_texts, win_meta = make_sliding_windows(sentences)
    print(f"[windows] {len(win_texts):,} × {config.WINDOW_SIZE}-sentence windows (stride={config.STEP})")

    # SBERT encode ONCE
    sbert = load_sbert()
    emb = encode_texts(sbert, win_texts)

    # Apply each probe
    out_dir.mkdir(parents=True, exist_ok=True)
    for target in TARGETS:
        model_path = models_dir / f"ridge_cr4_{target}.joblib"
        out_path   = out_dir / f"windows_cr4_{target}.csv"
        print(f"\n[load] {model_path}")
        art = joblib.load(model_path)
        models   = art["models"]
        x_scaler = art["x_scaler"]
        y_scaler = art["y_scaler"]

        emb_s = x_scaler.transform(emb)
        preds_s = np.column_stack([m.predict(emb_s) for m in models])
        vad = y_scaler.inverse_transform(preds_s)

        df = pd.DataFrame(win_meta)
        df["Valence"]   = vad[:, 0]
        df["Arousal"]   = vad[:, 1]
        df["Dominance"] = vad[:, 2]
        df["Narrative_Progress"] = np.linspace(0, 1, len(df))

        wl = choose_odd_window(len(df))
        if wl is not None:
            for col in ("Valence", "Arousal", "Dominance"):
                df[f"{col}_Smooth"] = savgol_filter(
                    df[col].values, window_length=wl, polyorder=config.SAVGOL_POLYORDER,
                )
        else:
            for col in ("Valence", "Arousal", "Dominance"):
                df[f"{col}_Smooth"] = df[col].values

        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"[save] {out_path}  ({len(df):,} rows)")

    print(f"\n[done] elapsed {time.time() - t0:.1f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text",      required=True, type=Path)
    ap.add_argument("--models-dir", type=Path, default=Path("models"))
    ap.add_argument("--out-dir",   type=Path, default=Path("results"))
    args = ap.parse_args()
    encode_all(args.text, args.models_dir, args.out_dir)


if __name__ == "__main__":
    main()
