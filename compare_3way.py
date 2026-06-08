#!/usr/bin/env python3
"""
Valeur — 3-way trajectory + NEI comparison: Warriner vs NRC vs CR4.

Loads the three windows_*.csv outputs, computes per-probe NEI (gated_sum),
reports pairwise Pearson on V/A/D and on NEI, extracts top-5% episode masks,
computes pairwise Jaccard on episode masks + strict 3-way consensus, and
renders a 6-row comparison plot (3 trajectory rows + 3 NEI rows).

Example:
    python compare_3way.py
"""

from __future__ import annotations

import argparse
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


PROBES_DEFAULT = {
    "warriner": "results/windows_warriner.csv",
    "nrc":      "results/windows_nrc.csv",
    "cr4_nrc":  "results/windows_cr4_nrc.csv",
}


def extract_episodes(nei: np.ndarray, percentile: float, min_duration: int) -> np.ndarray:
    """Return boolean mask of windows in top-`percentile` NEI episodes of duration >= min_duration."""
    pos = nei[nei > 0]
    if len(pos) == 0:
        return np.zeros_like(nei, dtype=bool)
    cutoff = np.percentile(pos, 100 * percentile)
    raw = nei >= cutoff
    # require contiguous run of length >= min_duration
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


def load_windows(paths: dict[str, Path]) -> dict[str, pd.DataFrame]:
    dfs = {}
    for name, path in paths.items():
        if not Path(path).exists():
            raise FileNotFoundError(f"Missing windows CSV for '{name}': {path}")
        dfs[name] = pd.read_csv(path)
        print(f"[load] {name:10s} {path}  ({len(dfs[name]):,} rows)")
    n = min(len(df) for df in dfs.values())
    if not all(len(df) == n for df in dfs.values()):
        print(f"[align] truncating all to common length {n}")
        for k in dfs:
            dfs[k] = dfs[k].iloc[:n].reset_index(drop=True)
    return dfs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("results"))
    ap.add_argument("--prefix",  type=str, default="compare_3way")
    ap.add_argument("--percentile", type=float, default=config.NEI_PERCENTILE)
    ap.add_argument("--min-duration", type=int, default=config.MIN_EPISODE_DURATION)
    args = ap.parse_args()

    paths = {k: Path(v) for k, v in PROBES_DEFAULT.items()}
    dfs = load_windows(paths)
    probes = list(dfs.keys())

    # ------------------------------------------------------------------ NEI
    neis = {}
    for k, df in dfs.items():
        nei, _, _, _ = compute_nei(
            df["Valence_Smooth"].values,
            df["Arousal_Smooth"].values,
            df["Dominance_Smooth"].values,
        )
        neis[k] = nei

    # ------------------------------------------------------------------ pairwise V/A/D corr
    print("\n=== PAIRWISE PEARSON r ON SMOOTHED V/A/D ===")
    rows = []
    for dim_label, col in [("V", "Valence_Smooth"), ("A", "Arousal_Smooth"), ("D", "Dominance_Smooth")]:
        for i, k1 in enumerate(probes):
            for k2 in probes[i + 1:]:
                r = float(pearsonr(dfs[k1][col], dfs[k2][col])[0])
                rho = float(spearmanr(dfs[k1][col], dfs[k2][col]).correlation)
                print(f"  {dim_label}: {k1:10s} ~ {k2:10s}  r={r:+.3f}  rho={rho:+.3f}")
                rows.append({"channel": dim_label, "probe_a": k1, "probe_b": k2,
                             "pearson_r": r, "spearman_rho": rho})
    pd.DataFrame(rows).to_csv(args.out_dir / f"{args.prefix}_trajectory_correlations.csv",
                              index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------------ pairwise NEI corr
    print("\n=== PAIRWISE PEARSON/SPEARMAN ON NEI ===")
    nei_rows = []
    for i, k1 in enumerate(probes):
        for k2 in probes[i + 1:]:
            r = float(pearsonr(neis[k1], neis[k2])[0])
            rho = float(spearmanr(neis[k1], neis[k2]).correlation)
            print(f"  NEI: {k1:10s} ~ {k2:10s}  r={r:+.3f}  rho={rho:+.3f}")
            nei_rows.append({"probe_a": k1, "probe_b": k2,
                             "pearson_r": r, "spearman_rho": rho})
    pd.DataFrame(nei_rows).to_csv(args.out_dir / f"{args.prefix}_nei_correlations.csv",
                                   index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------------ episodes
    print(f"\n=== ENTRAPMENT EPISODES (top {(1 - args.percentile)*100:.0f}%, min {args.min_duration} windows) ===")
    masks = {}
    for k, nei in neis.items():
        m = extract_episodes(nei, args.percentile, args.min_duration)
        masks[k] = m
        # contiguous runs count
        runs = 0
        in_run = False
        for v in m:
            if v and not in_run:
                runs += 1
                in_run = True
            elif not v:
                in_run = False
        print(f"  {k:10s}  flagged {m.sum():3d} windows  in {runs} episodes")

    # ------------------------------------------------------------------ episode overlap
    print("\n=== EPISODE MASK OVERLAP (Jaccard) ===")
    jacc_rows = []
    for i, k1 in enumerate(probes):
        for k2 in probes[i + 1:]:
            union = (masks[k1] | masks[k2]).sum()
            inter = (masks[k1] & masks[k2]).sum()
            j = inter / max(union, 1)
            print(f"  {k1:10s} ~ {k2:10s}  Jaccard={j:.3f}  (intersection {inter} / union {union})")
            jacc_rows.append({"probe_a": k1, "probe_b": k2,
                              "intersection": int(inter), "union": int(union), "jaccard": float(j)})
    pd.DataFrame(jacc_rows).to_csv(args.out_dir / f"{args.prefix}_episode_jaccard.csv",
                                    index=False, encoding="utf-8-sig")

    strict = masks[probes[0]].copy()
    for k in probes[1:]:
        strict &= masks[k]
    lenient = np.zeros_like(strict)
    stacked = np.stack([masks[k] for k in probes], axis=0)
    lenient = stacked.sum(axis=0) >= 2
    print(f"\n  STRICT 3-way consensus (all 3 flag):     {strict.sum()} windows")
    print(f"  LENIENT 3-way consensus (≥2 of 3 flag):  {lenient.sum()} windows")

    # ------------------------------------------------------------------ summary table
    summary = pd.DataFrame({
        "probe": probes,
        "n_windows_flagged": [int(masks[k].sum()) for k in probes],
    })
    summary.loc[len(summary)] = ["STRICT_3way", int(strict.sum())]
    summary.loc[len(summary)] = ["LENIENT_2of3", int(lenient.sum())]
    summary.to_csv(args.out_dir / f"{args.prefix}_episode_counts.csv", index=False)

    # ------------------------------------------------------------------ plot
    n_windows = len(neis[probes[0]])
    progress = dfs[probes[0]]["Narrative_Progress"].values

    fig, axes = plt.subplots(len(probes) + 1, 1, figsize=(14, 2 + 2 * len(probes)), sharex=True)
    colors_vad = {"Valence": "tab:blue", "Arousal": "tab:orange", "Dominance": "tab:green"}

    for i, k in enumerate(probes):
        ax = axes[i]
        for ch, color in colors_vad.items():
            s = dfs[k][f"{ch}_Smooth"].values
            # z-score for cross-probe visual comparability
            s_z = (s - s.mean()) / (s.std() + 1e-9)
            ax.plot(progress, s_z, color=color, label=ch, linewidth=1.2)
        # Shade strict consensus
        ax.fill_between(progress, -3, 3, where=strict, color="red", alpha=0.10, label="strict 3-way")
        ax.set_ylabel(f"{k}\n(z-scored)", fontsize=9)
        ax.set_ylim(-3, 3)
        if i == 0:
            ax.legend(loc="upper right", fontsize=7, ncol=4)

    # NEI panel — 3 lines on one axis
    ax = axes[-1]
    nei_colors = {"warriner": "tab:purple", "nrc": "tab:brown", "cr4_nrc": "tab:red"}
    for k in probes:
        ax.plot(progress, neis[k], label=f"NEI({k})", color=nei_colors.get(k, "k"), linewidth=1.0)
    # Lenient consensus shading (more inclusive — shows where ≥2 probes agree)
    ax.fill_between(progress, 0, ax.get_ylim()[1] if False else 12,
                    where=lenient, color="orange", alpha=0.15, label="≥2 of 3 agree")
    ax.fill_between(progress, 0, 12, where=strict, color="red", alpha=0.20, label="strict 3-way")
    ax.set_ylabel("NEI")
    ax.set_xlabel("Narrative progress")
    ax.legend(loc="upper right", fontsize=7, ncol=5)

    plt.suptitle("Valeur — 3-way probe comparison (Warriner / NRC / CR4)", fontsize=11)
    plt.tight_layout()
    plot_path = args.out_dir / f"{args.prefix}_plot.png"
    plt.savefig(plot_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\n[save] {plot_path}")

    # ------------------------------------------------------------------ consensus episode CSV with window text
    if strict.sum() > 0:
        ref = dfs[probes[0]]
        rows = []
        i = 0
        while i < n_windows:
            if strict[i]:
                j = i
                while j < n_windows and strict[j]:
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
        pd.DataFrame(rows).to_csv(args.out_dir / f"{args.prefix}_strict_consensus_episodes.csv",
                                   index=False, encoding="utf-8-sig")
        print(f"[save] strict consensus episodes: {len(rows)}")


if __name__ == "__main__":
    main()
