#!/usr/bin/env python3
"""
plot_dualprobe_trajectory.py
============================
Two-panel annotated VAD trajectory: Warriner (top) + NRC v2.1 (bottom).

Both panels share the same annotation zones (chief clerk + death).
The chief clerk zone fires in BOTH probes (positive example).
The death scene returns NEI=0 in BOTH probes (negative control).

Also generates:
  F3_consensus_nei_overlay.png  — NRC vs Warriner NEI overlaid on one axis,
  consensus episodes (W149-150, W167-173, W237-241) shaded.

Output:
  results/F1b_dualprobe_trajectory.png
  results/F3_consensus_nei_overlay.png
  + copies to ../paper1_writing/figures/
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
PAPER_FIGURES = ROOT.parent / "paper1_writing" / "figures"
JCLS_FIGURES  = ROOT.parent / "paper1_writing" / "JCLS-Template-1.0.6" / "JCLS-Template-1.0.6" / "figures"

CHIEF_ZONE = (165, 175)
CHIEF_PEAK = (167, 168)
DEATH_ZONE = (698, 706)
DEATH_FLAT = (699, 701)

# Consensus episodes (2-lexicon Warriner + NRC, top-5% overlap)
CONSENSUS_EPISODES = [
    (149, 150,  "Chief clerk's speech\n\"Mr. Samsa, what is wrong?\""),
    (167, 173,  "Gregor's defence\n\"I was quite alright last night\""),
    (237, 241,  "Traveller's monologue\n\"I'm trapped in a difficult situation\""),
]

TAU = 0.10


def compute_nei(df: pd.DataFrame):
    V = df["Valence_Smooth"].values
    A = df["Arousal_Smooth"].values
    D = df["Dominance_Smooth"].values
    med_V, med_A, med_D = np.median(V), np.median(A), np.median(D)
    dV = med_V - V
    dA = A - med_A
    dD = med_D - D
    zV = np.maximum((dV - dV.mean()) / dV.std(), 0.0)
    zA = np.maximum((dA - dA.mean()) / dA.std(), 0.0)
    zD = np.maximum((dD - dD.mean()) / dD.std(), 0.0)
    gate = (zV > TAU) & (zA > TAU) & (zD > TAU)
    nei = np.where(gate, zV + zA + zD, 0.0)
    return nei, zV, zA, zD


def add_zone(ax, w_start, w_end, ymin, ymax, label, label_y_frac, color,
             alpha=0.12, edge_lw=2.0):
    ax.add_patch(mpatches.Rectangle(
        (w_start - 1.5, ymin), w_end - w_start + 2, ymax - ymin,
        facecolor=color, edgecolor=color, linewidth=edge_lw,
        alpha=alpha, zorder=3,
    ))
    mid = (w_start + w_end) / 2.0
    ax.annotate(
        label,
        xy=(mid, ymin + label_y_frac * (ymax - ymin)),
        fontsize=7.0, ha="center", va="center",
        color=color, fontweight="bold", zorder=12,
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                  edgecolor=color, alpha=0.93, linewidth=1.1),
    )


# ─────────────────────────────────────────────────────────────
# Figure 1b: dual-probe (Warriner top, NRC bottom)
# ─────────────────────────────────────────────────────────────
def plot_dualprobe(dfw, dfn, neiw, nein):
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    fig.subplots_adjust(hspace=0.32)

    datasets = [
        (axes[0], dfw, neiw, "Warriner (1–9 scale, normalised)",
         "#1f77b4", "#ff7f0e", "#2ca02c", "A"),
        (axes[1], dfn, nein, "NRC v2.1 (−1 to +1 scale)",
         "#1f77b4", "#ff7f0e", "#2ca02c", "B"),
    ]

    for ax, df, nei, ylabel, cv, ca, cd, panel in datasets:
        V = df["Valence_Smooth"].values
        A = df["Arousal_Smooth"].values
        D = df["Dominance_Smooth"].values
        N = len(V)
        x = np.arange(1, N + 1)

        ymin = min(V.min(), A.min(), D.min()) - 0.45
        ymax = max(V.max(), A.max(), D.max()) + 0.75

        # Chief clerk zone
        add_zone(ax, CHIEF_ZONE[0], CHIEF_ZONE[1], ymin, ymax,
                 f"W{CHIEF_PEAK[0]}–{CHIEF_PEAK[1]}: Chief clerk\nGregor's defence (NEI peak)",
                 0.87, "#c0392b", alpha=0.10)
        # Death zone
        add_zone(ax, DEATH_ZONE[0], DEATH_ZONE[1], ymin, ymax,
                 f"W{DEATH_FLAT[0]}–{DEATH_FLAT[1]}: Gregor's death\nNEI = 0.00 (gate fails)",
                 0.87, "#2471a3", alpha=0.10)

        ax.plot(x, V, label="Valence",   lw=1.8, color=cv, zorder=10)
        ax.plot(x, A, label="Arousal",   lw=1.8, color=ca, zorder=10)
        ax.plot(x, D, label="Dominance", lw=1.8, color=cd, zorder=10)

        # NEI on twin axis
        ax2 = ax.twinx()
        ax2.fill_between(x, nei, alpha=0.25, color="#9b59b6", zorder=5)
        ax2.plot(x, nei, color="#9b59b6", lw=0.9, zorder=6)
        ax2.set_ylabel("NEI", color="#9b59b6", fontsize=9)
        ax2.tick_params(axis="y", labelcolor="#9b59b6")

        ax.set_ylim(ymin, ymax)
        ax.set_xlim(0, N + 1)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(alpha=0.15)
        ax.text(0.005, 0.97, f"({panel})", transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="top")
        ax.legend(loc="upper left", framealpha=0.9, fontsize=8)

    axes[1].set_xlabel("Window index (narrative progress →)", fontsize=10)
    fig.suptitle(
        "Warriner and NRC v2.1 affective trajectories — Kafka, Die Verwandlung\n"
        "Both probes fire at the chief clerk scene (W167–168) and return NEI = 0 at Gregor's death",
        fontsize=10, y=1.005,
    )
    out = RESULTS / "F1b_dualprobe_trajectory.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[plot] Saved: {out}")
    return out


# ─────────────────────────────────────────────────────────────
# Figure 3: consensus NEI overlay + 3 consensus episodes
# ─────────────────────────────────────────────────────────────
def plot_consensus(dfw, dfn, neiw, nein):
    fig, ax = plt.subplots(figsize=(16, 5.5))

    N = len(neiw)
    x = np.arange(1, N + 1)

    # Shade consensus episodes first
    episode_colors = ["#e67e22", "#27ae60", "#8e44ad"]
    for (s, e, label), ec in zip(CONSENSUS_EPISODES, episode_colors):
        ax.axvspan(s - 0.5, e + 0.5, alpha=0.18, color=ec, zorder=1)
        mid = (s + e) / 2.0
        ax.annotate(
            label,
            xy=(mid, max(neiw.max(), nein.max()) * 0.92),
            fontsize=7.5, ha="center", va="top",
            color=ec, fontweight="bold", zorder=12,
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                      edgecolor=ec, alpha=0.93, linewidth=1.1),
        )

    # NEI lines
    lw, = ax.plot(x, neiw, label="Warriner NEI", lw=1.8, color="#c0392b", zorder=10)
    ln, = ax.plot(x, nein, label="NRC v2.1 NEI", lw=1.8, color="#2471a3", zorder=10)
    ax.fill_between(x, neiw, alpha=0.12, color="#c0392b")
    ax.fill_between(x, nein, alpha=0.12, color="#2471a3")

    # Top-5% thresholds
    t95w = np.percentile(neiw, 95)
    t95n = np.percentile(nein, 95)
    ax.axhline(t95w, color="#c0392b", lw=0.9, ls="--", alpha=0.7,
               label=f"Warriner 95th pct ({t95w:.2f})")
    ax.axhline(t95n, color="#2471a3", lw=0.9, ls="--", alpha=0.7,
               label=f"NRC 95th pct ({t95n:.2f})")

    ep_patches = [
        mpatches.Patch(facecolor=ec, alpha=0.4, label=f"Ep {i+1}: {lab.split(chr(10))[0]}")
        for i, ((s, e, lab), ec) in enumerate(zip(CONSENSUS_EPISODES, episode_colors))
    ]
    ax.legend(handles=[lw, ln] + ep_patches, loc="upper right",
              framealpha=0.93, fontsize=8, ncol=2)

    ax.set_xlabel("Window index (narrative progress →)", fontsize=10)
    ax.set_ylabel("NEI (Narrative Entrapment Index)", fontsize=10)
    ax.set_title(
        "Cross-probe NEI consensus — Warriner vs NRC v2.1 on Die Verwandlung\n"
        "Shaded: 3 episodes where both probes exceed their 95th-percentile threshold",
        fontsize=10,
    )
    ax.set_xlim(0, N + 1)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.15)

    out = RESULTS / "F3_consensus_nei_overlay.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[plot] Saved: {out}")
    return out


def main():
    dfw = pd.read_csv(RESULTS / "windows_warriner.csv", encoding="utf-8-sig")
    dfn = pd.read_csv(RESULTS / "windows_nrc.csv",      encoding="utf-8-sig")

    neiw, *_ = compute_nei(dfw)
    nein, *_ = compute_nei(dfn)

    out1 = plot_dualprobe(dfw, dfn, neiw, nein)
    out2 = plot_consensus(dfw, dfn, neiw, nein)

    for out, dest_name in [(out1, "F1b_dualprobe_trajectory.png"),
                           (out2, "F3_consensus_nei_overlay.png")]:
        for dest_dir in [PAPER_FIGURES, JCLS_FIGURES]:
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(out, dest_dir / dest_name)
            print(f"[copy] {dest_dir / dest_name}")


if __name__ == "__main__":
    main()
