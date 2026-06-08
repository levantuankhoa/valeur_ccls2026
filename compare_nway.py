#!/usr/bin/env python3
"""
Valeur — N-way trajectory + NEI comparison across arbitrary probes.

Loads N windows_*.csv files, computes per-probe NEI (gated_sum), reports
pairwise Pearson on V/A/D and on NEI, extracts top-5% episode masks,
computes pairwise Jaccard + strict N-way + lenient ≥k-of-N consensus,
and renders an (N+1)-row plot.

Default 5-way: warriner + nrc + cr4_nrc + cr4_nrcbert + cr4_emo.

Example:
    python compare_nway.py
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import config
from nei_plot import compute_nei


DEFAULT_PROBES = {
    "warriner":     "results/windows_warriner.csv",
    "nrc":          "results/windows_nrc.csv",
    "cr4_nrc":      "results/windows_cr4_nrc.csv",
    "cr4_nrcbert":  "results/windows_cr4_nrcbert.csv",
    "cr4_emo":      "results/windows_cr4_emo.csv",
}


def extract_episode_mask(nei: np.ndarray, percentile: float, min_duration: int) -> np.ndarray:
    pos = nei[nei > 0]
    if len(pos) == 0:
        return np.zeros_like(nei, dtype=bool)
    cutoff = np.percentile(pos, 100 * percentile)
    raw = nei >= cutoff
    out = np.zeros_like(raw)
    i = 0
    while i < len(raw):
        if raw[i]:
            j = i
            while j < len(raw) and raw[j]:
                j += 1
            if j - i >= min_duration:
                out[i:j] = True
            i = j
        else:
            i += 1
    return out


def count_episodes(mask: np.ndarray) -> int:
    runs = 0
    in_run = False
    for v in mask:
        if v and not in_run:
            runs += 1
            in_run = True
        elif not v:
            in_run = False
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("results"))
    ap.add_argument("--prefix",  type=str, default="compare_5way")
    ap.add_argument("--percentile",   type=float, default=config.NEI_PERCENTILE)
    ap.add_argument("--min-duration", type=int,   default=config.MIN_EPISODE_DURATION)
    args = ap.parse_args()

    paths = {k: Path(v) for k, v in DEFAULT_PROBES.items()}
    dfs = {}
    for k, p in paths.items():
        if not p.exists():
            print(f"[skip] {k}: {p} not found")
            continue
        dfs[k] = pd.read_csv(p)
        print(f"[load] {k:14s} {p}  ({len(dfs[k]):,} rows)")

    probes = list(dfs.keys())
    n = min(len(d) for d in dfs.values())
    for k in dfs:
        dfs[k] = dfs[k].iloc[:n].reset_index(drop=True)

    # ------------------------------------------------------------- NEI
    neis = {}
    for k in probes:
        nei, _, _, _ = compute_nei(
            dfs[k]["Valence_Smooth"].values,
            dfs[k]["Arousal_Smooth"].values,
            dfs[k]["Dominance_Smooth"].values,
        )
        neis[k] = nei

    # ------------------------------------------------------------- V/A/D pairwise
    print(f"\n{'='*70}\nPAIRWISE PEARSON r ON SMOOTHED V/A/D\n{'='*70}")
    traj_rows = []
    for dim_label, col in [("V", "Valence_Smooth"), ("A", "Arousal_Smooth"), ("D", "Dominance_Smooth")]:
        print(f"\n  --- {dim_label} ---")
        for k1, k2 in itertools.combinations(probes, 2):
            r   = float(pearsonr(dfs[k1][col], dfs[k2][col])[0])
            rho = float(spearmanr(dfs[k1][col], dfs[k2][col]).correlation)
            print(f"  {k1:14s} ~ {k2:14s}  r={r:+.3f}  rho={rho:+.3f}")
            traj_rows.append({"channel": dim_label, "probe_a": k1, "probe_b": k2,
                              "pearson_r": r, "spearman_rho": rho})
    pd.DataFrame(traj_rows).to_csv(args.out_dir / f"{args.prefix}_trajectory_correlations.csv",
                                    index=False, encoding="utf-8-sig")

    # Compact V/A/D agreement matrices
    print(f"\n{'='*70}\nV/A/D AGREEMENT MATRICES (Pearson r)\n{'='*70}")
    for dim_label, col in [("V", "Valence_Smooth"), ("A", "Arousal_Smooth"), ("D", "Dominance_Smooth")]:
        print(f"\n  --- {dim_label} ---")
        mat = pd.DataFrame(index=probes, columns=probes, dtype=float)
        for k1 in probes:
            for k2 in probes:
                mat.loc[k1, k2] = float(pearsonr(dfs[k1][col], dfs[k2][col])[0])
        print(mat.to_string(float_format=lambda x: f"{x:+.3f}"))
        mat.to_csv(args.out_dir / f"{args.prefix}_matrix_{dim_label}.csv", encoding="utf-8-sig")

    # ------------------------------------------------------------- NEI pairwise
    print(f"\n{'='*70}\nPAIRWISE PEARSON/SPEARMAN ON NEI\n{'='*70}")
    nei_rows = []
    nei_mat = pd.DataFrame(index=probes, columns=probes, dtype=float)
    for k1, k2 in itertools.combinations(probes, 2):
        r   = float(pearsonr(neis[k1], neis[k2])[0])
        rho = float(spearmanr(neis[k1], neis[k2]).correlation)
        print(f"  {k1:14s} ~ {k2:14s}  r={r:+.3f}  rho={rho:+.3f}")
        nei_rows.append({"probe_a": k1, "probe_b": k2,
                         "pearson_r": r, "spearman_rho": rho})
    for k1 in probes:
        for k2 in probes:
            nei_mat.loc[k1, k2] = float(pearsonr(neis[k1], neis[k2])[0])
    print("\nNEI Pearson matrix:")
    print(nei_mat.to_string(float_format=lambda x: f"{x:+.3f}"))
    nei_mat.to_csv(args.out_dir / f"{args.prefix}_matrix_NEI.csv", encoding="utf-8-sig")
    pd.DataFrame(nei_rows).to_csv(args.out_dir / f"{args.prefix}_nei_correlations.csv",
                                   index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------- episodes
    print(f"\n{'='*70}\nEPISODES (top {(1 - args.percentile)*100:.0f}%, min {args.min_duration} windows)\n{'='*70}")
    masks = {}
    for k in probes:
        m = extract_episode_mask(neis[k], args.percentile, args.min_duration)
        masks[k] = m
        print(f"  {k:14s}  {m.sum():3d} windows flagged  in {count_episodes(m)} episodes")

    # ------------------------------------------------------------- pairwise Jaccard
    print(f"\n{'='*70}\nEPISODE JACCARD\n{'='*70}")
    jacc_rows = []
    jacc_mat = pd.DataFrame(index=probes, columns=probes, dtype=float)
    for k1, k2 in itertools.combinations(probes, 2):
        union = (masks[k1] | masks[k2]).sum()
        inter = (masks[k1] & masks[k2]).sum()
        j = inter / max(union, 1)
        print(f"  {k1:14s} ~ {k2:14s}  Jaccard={j:.3f}  (∩={inter}  ∪={union})")
        jacc_rows.append({"probe_a": k1, "probe_b": k2,
                          "intersection": int(inter), "union": int(union),
                          "jaccard": float(j)})
    for k1 in probes:
        for k2 in probes:
            u = (masks[k1] | masks[k2]).sum()
            i = (masks[k1] & masks[k2]).sum()
            jacc_mat.loc[k1, k2] = float(i / max(u, 1))
    print("\nJaccard matrix:")
    print(jacc_mat.to_string(float_format=lambda x: f"{x:.3f}"))
    jacc_mat.to_csv(args.out_dir / f"{args.prefix}_matrix_jaccard.csv", encoding="utf-8-sig")
    pd.DataFrame(jacc_rows).to_csv(args.out_dir / f"{args.prefix}_episode_jaccard.csv",
                                    index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------- N-way consensus
    print(f"\n{'='*70}\nN-WAY CONSENSUS (top {(1 - args.percentile)*100:.0f}%)\n{'='*70}")
    stacked = np.stack([masks[k] for k in probes], axis=0)  # (N, W)
    counts  = stacked.sum(axis=0)
    consensus = {}
    print(f"  {'k of N':10s} {'windows':>10s} {'episodes':>10s}")
    for k in range(1, len(probes) + 1):
        mask_k = counts >= k
        consensus[k] = mask_k
        print(f"  ≥{k}/{len(probes):<8d} {int(mask_k.sum()):>10d} {count_episodes(mask_k):>10d}")

    summary = pd.DataFrame({
        "probe_or_consensus": list(probes) + [f">={k}_of_{len(probes)}" for k in range(1, len(probes) + 1)],
        "windows_flagged":    [int(masks[k].sum()) for k in probes] + [int(consensus[k].sum()) for k in range(1, len(probes) + 1)],
        "episodes":           [count_episodes(masks[k]) for k in probes] + [count_episodes(consensus[k]) for k in range(1, len(probes) + 1)],
    })
    summary.to_csv(args.out_dir / f"{args.prefix}_consensus_summary.csv", index=False)

    # ------------------------------------------------------------- plot
    progress = dfs[probes[0]]["Narrative_Progress"].values
    nrow = len(probes) + 1
    fig, axes = plt.subplots(nrow, 1, figsize=(15, 2 + 1.6 * nrow), sharex=True)
    vad_colors = {"Valence": "tab:blue", "Arousal": "tab:orange", "Dominance": "tab:green"}

    for i, k in enumerate(probes):
        ax = axes[i]
        for ch, color in vad_colors.items():
            s = dfs[k][f"{ch}_Smooth"].values
            s_z = (s - s.mean()) / (s.std() + 1e-9)
            ax.plot(progress, s_z, color=color, label=ch, linewidth=1.0)
        # Shade strict N-way consensus on this row
        ax.fill_between(progress, -3, 3, where=consensus[len(probes)],
                        color="red", alpha=0.15)
        ax.set_ylabel(k, fontsize=9)
        ax.set_ylim(-3, 3)
        if i == 0:
            ax.legend(loc="upper right", fontsize=7, ncol=3)

    ax = axes[-1]
    nei_palette = plt.cm.tab10.colors
    for j, k in enumerate(probes):
        ax.plot(progress, neis[k], label=f"NEI({k})",
                color=nei_palette[j % len(nei_palette)], linewidth=1.0)
    # Lenient (>=k/N for k = ceil(N/2) + 1) shading
    half = (len(probes) // 2) + 1
    ymax = max(max(neis[k]) for k in probes)
    ax.fill_between(progress, 0, ymax, where=consensus[half],
                    color="orange", alpha=0.15, label=f"≥{half}/{len(probes)} agree")
    ax.fill_between(progress, 0, ymax, where=consensus[len(probes)],
                    color="red", alpha=0.30, label=f"strict {len(probes)}/{len(probes)}")
    ax.set_ylabel("NEI")
    ax.set_xlabel("Narrative progress")
    ax.legend(loc="upper right", fontsize=7, ncol=min(len(probes) + 2, 7))

    plt.suptitle(f"Valeur — {len(probes)}-way probe comparison", fontsize=12)
    plt.tight_layout()
    plot_path = args.out_dir / f"{args.prefix}_plot.png"
    plt.savefig(plot_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\n[save] {plot_path}")

    # Strict consensus window detail
    strict = consensus[len(probes)]
    if strict.sum() > 0:
        ref = dfs[probes[0]]
        rows = []
        i = 0
        while i < n:
            if strict[i]:
                j = i
                while j < n and strict[j]:
                    j += 1
                if j - i >= args.min_duration:
                    rows.append({
                        "start_window": int(ref.iloc[i]["Window_ID"]),
                        "end_window":   int(ref.iloc[j - 1]["Window_ID"]),
                        "n_windows":    j - i,
                        "center_text":  str(ref.iloc[(i + j) // 2]["Center_Sentence"]),
                    })
                i = j
            else:
                i += 1
        pd.DataFrame(rows).to_csv(args.out_dir / f"{args.prefix}_strict_episodes.csv",
                                   index=False, encoding="utf-8-sig")

    # Lenient consensus window detail (≥half)
    lenient = consensus[half]
    if lenient.sum() > 0:
        ref = dfs[probes[0]]
        rows = []
        i = 0
        while i < n:
            if lenient[i]:
                j = i
                while j < n and lenient[j]:
                    j += 1
                if j - i >= args.min_duration:
                    rows.append({
                        "start_window": int(ref.iloc[i]["Window_ID"]),
                        "end_window":   int(ref.iloc[j - 1]["Window_ID"]),
                        "n_windows":    j - i,
                        "center_text":  str(ref.iloc[(i + j) // 2]["Center_Sentence"]),
                    })
                i = j
            else:
                i += 1
        pd.DataFrame(rows).to_csv(args.out_dir / f"{args.prefix}_lenient_episodes.csv",
                                   index=False, encoding="utf-8-sig")
        print(f"[save] lenient ≥{half}/{len(probes)} episodes: {len(rows)}")


if __name__ == "__main__":
    main()
