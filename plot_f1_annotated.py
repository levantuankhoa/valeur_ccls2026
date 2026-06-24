#!/usr/bin/env python3
"""
Valeur — F1 Annotated: Warriner VAD Trajectory with Discrimination Evidence
============================================================================

Re-renders the canonical Warriner trajectory (Figure 1 for the JCLS paper)
with two annotated regions that demonstrate within-text discrimination:

  POSITIVE: W165-175 — chief clerk visit, Gregor's defensive monologue
            Peak at W167: NEI = 6.73 (V↓ A↑ D↓ all fire simultaneously)

  NEGATIVE: W698-706 — Gregor's death, "empty and peaceful rumination"
            W699-701: NEI = 0.00 (zA = 0 → gate fails → NEI = 0)

This is the headline construct-specificity evidence: SAME probe, SAME formula,
SAME text — two affectively opposite scenes correctly discriminated.

Sources:
  windows:  results/windows_warriner.csv
  NEI data: results/warriner.nei.csv
  episodes: results/warriner.episodes.csv
  verified: paper1_writing/ground_truth/verify_gregor_death_output.txt

Output:
  results/F1_warriner_trajectory_annotated.png
  ../paper1_writing/figures/F1_warriner_trajectory_kafka.png  (canonical location)

Usage:
    python plot_f1_annotated.py
    python plot_f1_annotated.py --no-paper-copy
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
PAPER_FIGURES = ROOT.parent / "paper1_writing" / "figures"

# Verified window indices (1-based Window_ID from CSVs; array index = W-1)
CHIEF_CLERK_ZONE = (165, 175)   # W165–W175: chief clerk visit, Gregor's defence
CHIEF_CLERK_PEAK = (167, 168)   # W167-168: NEI=6.73, 6.45 (peak discrimination)
DEATH_ZONE = (698, 706)         # W698–W706: Gregor's death scene
DEATH_FLAT = (699, 701)         # W699-701: NEI=0.00 (zA=0 gate failure)


def load_data():
    windows = pd.read_csv(RESULTS / "windows_warriner.csv", encoding="utf-8-sig")
    nei_df  = pd.read_csv(RESULTS / "warriner.nei.csv",     encoding="utf-8-sig")
    eps_df  = pd.read_csv(RESULTS / "warriner.episodes.csv", encoding="utf-8-sig")
    return windows, nei_df, eps_df


def build_episode_ranges(eps_df):
    return [(int(r.start_window), int(r.end_window)) for _, r in eps_df.iterrows()]


def add_zone_annotation(ax, w_start, w_end, ymin, ymax, label, label_y_frac,
                         color, alpha=0.18, edge_lw=2.0):
    """Shade a window range and add a bracket + label above/below."""
    ax.add_patch(mpatches.Rectangle(
        (w_start - 1.5, ymin), w_end - w_start + 2, ymax - ymin,
        facecolor=color, edgecolor=color, linewidth=edge_lw,
        alpha=alpha, zorder=3,
    ))
    mid = (w_start + w_end) / 2
    label_y = ymin + label_y_frac * (ymax - ymin)
    ax.annotate(
        label,
        xy=(mid, label_y),
        xycoords="data",
        fontsize=7.5,
        ha="center",
        va="center",
        color=color,
        fontweight="bold",
        zorder=12,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=color,
                  alpha=0.92, linewidth=1.2),
    )


def render(windows: pd.DataFrame, nei_df: pd.DataFrame,
           episodes, out_path: Path) -> None:

    V = windows["Valence_Smooth"].values
    A = windows["Arousal_Smooth"].values
    D = windows["Dominance_Smooth"].values
    N = len(V)
    x = np.arange(1, N + 1)   # 1-based Window_ID to match the paper

    ymin = min(V.min(), A.min(), D.min()) - 0.55
    ymax = max(V.max(), A.max(), D.max()) + 0.85  # extra headroom for top labels

    fig, ax = plt.subplots(figsize=(16, 5.5))

    # --- Background regions: episodes (pink) ---
    for (s, e) in episodes:
        ax.add_patch(mpatches.Rectangle(
            (s - 0.5, ymin), e - s + 1, ymax - ymin,
            facecolor="#ffb3d9", edgecolor="#ff3d81", linewidth=1.6, alpha=0.22, zorder=2,
        ))

    # --- ANNOTATION ZONE 1: Chief clerk (positive, W165-175) ---
    add_zone_annotation(
        ax, CHIEF_CLERK_ZONE[0], CHIEF_CLERK_ZONE[1],
        ymin, ymax,
        label=(
            "Chief clerk visit: Gregor's defence\n"
            f"W{CHIEF_CLERK_PEAK[0]}–W{CHIEF_CLERK_PEAK[1]}: NEI = 6.73, 6.45\n"
            "V↓ A↑ D↓ — all gate components fire"
        ),
        label_y_frac=0.88,
        color="#c0392b",
        alpha=0.10,
        edge_lw=2.5,
    )
    # Arrow from annotation to W167 peak
    peak_idx = CHIEF_CLERK_PEAK[0]
    ax.annotate(
        "",
        xy=(peak_idx, ymin + 0.83 * (ymax - ymin) - 0.3),
        xycoords="data",
        xytext=(peak_idx, ymin + 0.83 * (ymax - ymin) - 0.05),
        arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.5),
    )

    # --- ANNOTATION ZONE 2: Gregor's death (negative, W698-706) ---
    add_zone_annotation(
        ax, DEATH_ZONE[0], DEATH_ZONE[1],
        ymin, ymax,
        label=(
            "Gregor’s death: “empty and peaceful”\n"
            f"W{DEATH_FLAT[0]}–W{DEATH_FLAT[1]}: NEI = 0.00\n"
            "zₐ = 0 → gate fails → NEI = 0"
        ),
        label_y_frac=0.88,
        color="#2471a3",
        alpha=0.10,
        edge_lw=2.5,
    )

    # --- VAD trajectories ---
    l1, = ax.plot(x, V, label="Valence",   linewidth=1.8, color="#1f77b4", zorder=10)
    l2, = ax.plot(x, A, label="Arousal",   linewidth=1.8, color="#ff7f0e", zorder=10)
    l3, = ax.plot(x, D, label="Dominance", linewidth=1.8, color="#2ca02c", zorder=10)

    # --- Legend ---
    entr_patch = mpatches.Patch(facecolor="#ffb3d9", edgecolor="#ff3d81", alpha=0.5,
                                label="Entrapment episode")
    pos_patch  = mpatches.Patch(facecolor="#c0392b", edgecolor="#c0392b", alpha=0.2,
                                label="Chief clerk zone (positive, NEI peak)")
    neg_patch  = mpatches.Patch(facecolor="#2471a3", edgecolor="#2471a3", alpha=0.2,
                                label="Death zone (negative control, NEI = 0)")

    ax.legend(
        handles=[l1, l2, l3, entr_patch, pos_patch, neg_patch],
        loc="upper left", framealpha=0.95, fontsize=8, ncol=2,
    )

    ax.set_xlabel("Window index (narrative progress →)", fontsize=10)
    ax.set_ylabel("V / A / D score (Warriner scale 1–9)", fontsize=10)
    ax.set_title(
        "Affective Trajectory — Warriner probe (gated_sum NEI)\n"
        "Kafka, “Die Verwandlung” • Same probe, same formula, same text: "
        "within-text discrimination of entrapment (chief clerk) vs. defeat (Gregor’s death)",
        fontsize=10
    )
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(0, N + 1)
    ax.grid(alpha=0.18)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[plot] Saved: {out_path}")


def main(no_paper_copy: bool = False) -> None:
    windows, nei_df, eps_df = load_data()

    # Merge NEI columns into windows if not already present
    if "NEI" not in windows.columns and "NEI" in nei_df.columns:
        windows = windows.merge(
            nei_df[["Window_ID", "NEI", "Entrapment"]],
            on="Window_ID", how="left"
        )

    episodes = build_episode_ranges(eps_df)

    out = RESULTS / "F1_warriner_trajectory_annotated.png"
    render(windows, nei_df, episodes, out)

    # Copy to paper1_writing/figures/
    if not no_paper_copy:
        PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
        dest = PAPER_FIGURES / "F1_warriner_trajectory_kafka.png"
        shutil.copy2(out, dest)
        print(f"[copy] {dest}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render F1 annotated Warriner trajectory")
    parser.add_argument("--no-paper-copy", action="store_true",
                        help="Skip copying to paper1_writing/figures/")
    args = parser.parse_args()
    main(no_paper_copy=args.no_paper_copy)
