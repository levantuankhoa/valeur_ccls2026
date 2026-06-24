#!/usr/bin/env python3
"""
Valeur — EDA Figure: Lexicon Characterization
==============================================

3-panel scatter plot comparing NRC VAD v2.1 and Warriner (2013) at the
WORD LEVEL on their ~13,915 overlapping terms, annotated with:
  - Pearson r per dimension
  - inset table showing word-level vs trajectory-level agreement

Output: results/eda_lexicons_scatter.png

Usage:
    python plot_eda_lexicons.py
    python plot_eda_lexicons.py --out results/custom_name.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

# Trajectory-level numbers from ground_truth/matrix_*.csv (verified 2026-06-05)
# These are reported in §4.2 of the paper
TRAJECTORY_R = {"V": 0.818, "A": 0.621, "D": 0.193}
TRAJECTORY_TRIAL_R = {"V": 0.811, "A": 0.701, "D": 0.098}

DIM_LABELS = {"V": "Valence", "A": "Arousal", "D": "Dominance"}
COLORS = {"V": "#1f77b4", "A": "#ff7f0e", "D": "#2ca02c"}


def load_warriner(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["word_lower"] = df["Word"].str.lower().str.strip()
    out = df[["word_lower", "V.Mean.Sum", "A.Mean.Sum", "D.Mean.Sum"]].rename(
        columns={"V.Mean.Sum": "W_V", "A.Mean.Sum": "W_A", "D.Mean.Sum": "W_D"}
    ).copy()
    for col in ["W_V", "W_A", "W_D"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.dropna()


def load_nrc(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path, sep="\t", header=None, names=["word", "V", "A", "D"],
        encoding="utf-8-sig"
    )
    df["word_lower"] = df["word"].str.lower().str.strip()
    out = df[["word_lower", "V", "A", "D"]].rename(
        columns={"V": "N_V", "A": "N_A", "D": "N_D"}
    ).copy()
    for col in ["N_V", "N_A", "N_D"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.dropna()


def normalize_warriner(series: pd.Series) -> pd.Series:
    """Map Warriner 1-9 scale to -1 to +1 for side-by-side comparison.

    Midpoint of 1-9 scale is 5.  Formula: (x - 5) / 4.
    """
    return (series - 5.0) / 4.0


def scatter_panel(ax, x, y, dim, color, word_r, n_words):
    """Render one scatter panel with hexbin density + Pearson annotation."""
    # Hexbin for density: opaque + log colour scale so the structure reads
    # crisply instead of washing out to a pale cloud (linear bins + alpha).
    hb = ax.hexbin(x, y, gridsize=50, cmap="Blues", mincnt=1, bins="log", linewidths=0.0)

    ax.set_xlabel(f"Warriner {DIM_LABELS[dim]} (mapped to −1…+1)", fontsize=9)
    ax.set_ylabel(f"NRC v2.1 {DIM_LABELS[dim]}", fontsize=9)
    ax.set_title(DIM_LABELS[dim], fontsize=11, fontweight="bold", color=color)
    ax.axhline(0, color="gray", linewidth=0.6, linestyle="--", alpha=0.5)
    ax.axvline(0, color="gray", linewidth=0.6, linestyle="--", alpha=0.5)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)

    # Reference diagonal
    ax.plot([-1, 1], [-1, 1], "k--", linewidth=0.8, alpha=0.4, label="perfect agreement")

    # Pearson annotation
    traj_r = TRAJECTORY_R[dim]
    traj_trial_r = TRAJECTORY_TRIAL_R[dim]
    label = (
        f"Word-level r = {word_r:.2f}  (n={n_words:,})\n"
        f"Traj. r = {traj_r:.3f}  (Metamorphosis)\n"
        f"Traj. r = {traj_trial_r:.3f}  (Trial)"
    )
    ax.text(
        0.03, 0.97, label,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor=color),
    )

    return hb


def main(out_path: Path) -> None:
    # --- Load lexicons ---
    war_path = DATA / "warriner_2013.csv"
    nrc_path = DATA / "unigrams-NRC-VAD-Lexicon-v2.1.txt"

    if not war_path.exists():
        sys.exit(f"[ERROR] Missing {war_path}")
    if not nrc_path.exists():
        sys.exit(f"[ERROR] Missing {nrc_path}")

    war = load_warriner(war_path)
    nrc = load_nrc(nrc_path)

    merged = pd.merge(war, nrc, on="word_lower", how="inner")
    n = len(merged)
    print(f"[eda] Overlapping words: {n:,}")
    print(f"      Warriner total: {len(war):,}  |  NRC total: {len(nrc):,}")

    # Normalize Warriner to -1…+1
    merged["W_V_n"] = normalize_warriner(merged["W_V"])
    merged["W_A_n"] = normalize_warriner(merged["W_A"])
    merged["W_D_n"] = normalize_warriner(merged["W_D"])

    # Word-level Pearson r
    r_V, _ = pearsonr(merged["W_V_n"], merged["N_V"])
    r_A, _ = pearsonr(merged["W_A_n"], merged["N_A"])
    r_D, _ = pearsonr(merged["W_D_n"], merged["N_D"])
    print(f"[eda] Word-level Pearson r  — V: {r_V:.3f}  A: {r_A:.3f}  D: {r_D:.3f}")
    print(f"[eda] Trajectory-level r    — V: {TRAJECTORY_R['V']:.3f}  A: {TRAJECTORY_R['A']:.3f}  D: {TRAJECTORY_R['D']:.3f}")

    # --- Figure layout ---
    fig = plt.figure(figsize=(15, 5.5))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    dims = [("V", merged["W_V_n"], merged["N_V"], r_V),
            ("A", merged["W_A_n"], merged["N_A"], r_A),
            ("D", merged["W_D_n"], merged["N_D"], r_D)]

    axes = []
    for i, (dim, x, y, word_r) in enumerate(dims):
        ax = fig.add_subplot(gs[0, i])
        hb = scatter_panel(ax, x.values, y.values, dim, COLORS[dim], word_r, n)
        axes.append(ax)

    # Suptitle
    fig.suptitle(
        f"NRC VAD v2.1 vs Warriner (2013): Word-Level Agreement on {n:,} Overlapping Terms\n"
        "Warriner scores mapped to −1…+1 scale for comparison. "
        "Diagonal = perfect agreement. Annotations show word-level (this figure) "
        "and trajectory-level (Metamorphosis / Trial) Pearson r.",
        fontsize=9, y=1.01
    )

    # Footer note about D finding
    fig.text(
        0.5, -0.04,
        "Dominance (D) shows the lowest word-level agreement (r ≈ 0.33) — consistent with Mohammad (2025) — "
        "and diverges further at trajectory level (r = 0.19 on The Metamorphosis, r = 0.10 on The Trial), "
        "revealing that contextual application via SBERT amplifies the structural D-disagreement inherent to the lexicons.",
        ha="center", fontsize=8, style="italic", wrap=True,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff8e7", alpha=0.9, edgecolor="#ccaa00")
    )

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[plot] Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EDA lexicon scatter figure")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "results" / "eda_lexicons_scatter.png")
    args = parser.parse_args()
    main(args.out)
