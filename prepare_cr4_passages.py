#!/usr/bin/env python3
"""
Valeur — prepare CR4-NarrEmote passage-level training data.

Reads CR4NarrEmote_t1Yes.csv (one row per labeled annotation), groups by
subject_ids (= passage UID), mean-aggregates per-label NRC + NRCBERT VAD
targets, and splits BY file_id (= book) into 70/15/15 train/val/test.

This is the data-prep step for the CR4 probe (transfer-learning option B
in cr4_integration_plan.md). Book-level split is mandatory: within-book
sentences share author voice and register, so splitting by passage would
leak signal into validation.

Example:
    python prepare_cr4_passages.py \\
        --raw "../cr4_review_bundle/doi-10.5683-sp3-xn4zyz (2)/CR4NarrEmote_t1Yes.csv" \\
        --out data/cr4_passages.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import config

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


def prepare(
    raw_csv: Path,
    out_csv: Path,
    seed: int = config.RANDOM_STATE,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> pd.DataFrame:
    print(f"[load] {raw_csv}")
    df = pd.read_csv(raw_csv, low_memory=False)
    print(f"       {len(df):,} annotation rows")

    # Keep only rows with a valid NRC lexicon mapping (drops labels with no NRC entry)
    df = df.dropna(subset=["NRC_valence", "NRC_arousal", "NRC_dominance"])
    print(f"[filter] {len(df):,} rows with NRC mapping")

    # Group by passage UID; first() for stable per-passage cols, mean() for VAD targets
    agg = (
        df.groupby("subject_ids", sort=False)
          .agg(
              passage=("passage", "first"),
              file_id=("file_id", "first"),
              pub_date=("PUBL_DATE", "first"),
              genre=("Genre", "first"),
              category=("Category", "first"),
              code=("Code", "first"),
              V_nrc=("NRC_valence", "mean"),
              A_nrc=("NRC_arousal", "mean"),
              D_nrc=("NRC_dominance", "mean"),
              V_nrcbert=("NRCBERT_valence", "mean"),
              A_nrcbert=("NRCBERT_arousal", "mean"),
              D_nrcbert=("NRCBERT_dominance", "mean"),
              V_emo=("EMO_valence", "mean"),
              A_emo=("EMO_arousal", "mean"),
              D_emo=("EMO_dominance", "mean"),
              n_labels=("classification_id", "count"),
          )
          .reset_index()
    )
    print(f"[group] {len(agg):,} unique passages")

    # Drop passages with no usable file_id (cannot split by book)
    n_no_book = agg["file_id"].isna().sum()
    if n_no_book:
        print(f"[filter] dropping {n_no_book} passages with missing file_id")
        agg = agg.dropna(subset=["file_id"])

    # Book-level random split
    rng = np.random.default_rng(seed)
    books = agg["file_id"].astype(str).unique()
    rng.shuffle(books)
    n = len(books)
    cut_train = int(train_frac * n)
    cut_val   = int((train_frac + val_frac) * n)
    train_b = set(books[:cut_train])
    val_b   = set(books[cut_train:cut_val])

    def split_of(fid: str) -> str:
        if fid in train_b:
            return "train"
        if fid in val_b:
            return "val"
        return "test"

    agg["split"] = agg["file_id"].astype(str).map(split_of)

    print(f"[split] books   : {cut_train} train | {cut_val - cut_train} val | {n - cut_val} test  (total {n})")
    print(f"[split] passages: {agg['split'].value_counts().to_dict()}")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[save] {out_csv}  ({len(agg):,} rows)")
    return agg


def main():
    ap = argparse.ArgumentParser(description="Prep CR4-NarrEmote passages for probe training")
    ap.add_argument("--raw", required=True, type=Path, help="path to CR4NarrEmote_t1Yes.csv")
    ap.add_argument("--out", required=True, type=Path, help="output passages CSV")
    ap.add_argument("--train-frac", type=float, default=0.70)
    ap.add_argument("--val-frac",   type=float, default=0.15)
    args = ap.parse_args()
    prepare(args.raw, args.out, train_frac=args.train_frac, val_frac=args.val_frac)


if __name__ == "__main__":
    main()
