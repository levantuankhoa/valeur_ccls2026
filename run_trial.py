#!/usr/bin/env python3
"""
Valeur — Trial robustness run.

Strips Gutenberg boilerplate, segments, builds 3-sentence sliding windows,
encodes once with SBERT, applies 3 main probes (warriner / nrc / cr4_nrc),
computes NEI per probe + 3-way consensus, and looks for literary turning
points (opening arrest, cathedral parable, final execution).

Output:
    results/trial_clean.txt
    results/trial_warriner.csv, trial_nrc.csv, trial_cr4_nrc.csv
    results/trial_3way_*.csv (correlations, jaccard, consensus)
    results/trial_3way_plot.png
    Console: literary alignment check on top-NEI windows per probe
"""

from __future__ import annotations

import itertools
import re
import sys
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.stats import pearsonr, spearmanr

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import config
from nei_plot import compute_nei
from utils import (
    choose_odd_window,
    encode_texts,
    load_sbert,
    make_sliding_windows,
    set_global_seed,
)


RAW_PATH   = Path("data/thetrial.txt")
CLEAN_PATH = Path("data/thetrial_clean.txt")
OUT_DIR    = Path("results")
PROBES = [
    ("warriner", "models/ridge_warriner.joblib"),
    ("nrc",      "models/ridge_nrc.joblib"),
    ("cr4_nrc",  "models/ridge_cr4_nrc.joblib"),
]

# Known literary turning points in The Trial (Muir 1937 / Gutenberg)
# Identified by chapter content, not line number — we find them by keyword
KNOWN_SCENES = {
    "opening_arrest":    ["Someone must have been telling lies about Josef K",
                          "he was arrested one fine morning",
                          "arrested without having done anything wrong"],
    "first_interrogation": ["interrogation", "examining magistrate"],
    "flogger":           ["whip", "flogger", "Franz and Willem"],
    "cathedral":         ["cathedral", "Before the Law", "doorkeeper", "country man"],
    "execution":         ["Like a dog", "shame", "knife"],
}


def strip_gutenberg(raw_text: str) -> str:
    """Remove Gutenberg header/footer and license."""
    start = re.search(r"\*\*\* START OF THE PROJECT GUTENBERG EBOOK[^\n]*\n", raw_text)
    end   = re.search(r"\*\*\* END OF THE PROJECT GUTENBERG EBOOK", raw_text)
    if start and end:
        return raw_text[start.end():end.start()]
    return raw_text


def segment_for_trial(text_path: Path):
    """Trial-specific segmentation — Gutenberg .txt has hard line wraps inside paragraphs."""
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm", disable=["ner", "tagger", "lemmatizer"])
    except OSError:
        from spacy.cli import download
        download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm", disable=["ner", "tagger", "lemmatizer"])
    nlp.max_length = 5_000_000

    raw = text_path.read_text(encoding="utf-8", errors="ignore")
    # Replace single newlines (line wraps) with spaces; preserve paragraph breaks (double newlines)
    raw = re.sub(r"(?<!\n)\n(?!\n)", " ", raw)
    raw = re.sub(r"\s+", " ", raw)
    sents = [s.text.strip() for s in nlp(raw).sents
             if len(s.text.strip()) > config.MIN_SENTENCE_LENGTH]
    return sents


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


def main():
    t0 = time.time()
    set_global_seed()

    # ---------------------------------------------------------- strip Gutenberg
    raw = RAW_PATH.read_text(encoding="utf-8", errors="ignore")
    clean = strip_gutenberg(raw)
    CLEAN_PATH.write_text(clean, encoding="utf-8")
    print(f"[clean] raw={len(raw):,} chars → clean={len(clean):,} chars  → {CLEAN_PATH}")

    # ---------------------------------------------------------- segment
    sents = segment_for_trial(CLEAN_PATH)
    print(f"[segment] {len(sents):,} sentences")

    win_texts, win_meta = make_sliding_windows(sents)
    print(f"[windows] {len(win_texts):,} × {config.WINDOW_SIZE}-sentence windows")

    # ---------------------------------------------------------- SBERT once
    sbert = load_sbert()
    emb = encode_texts(sbert, win_texts)

    # ---------------------------------------------------------- apply probes
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    trajectories = {}
    for name, model_path in PROBES:
        print(f"\n[probe] {name}  ← {model_path}")
        art = joblib.load(model_path)
        x_scaler = art["x_scaler"]
        y_scaler = art["y_scaler"]
        models   = art["models"]

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

        out_csv = OUT_DIR / f"trial_{name}.csv"
        df.to_csv(out_csv, index=False, encoding="utf-8-sig")
        trajectories[name] = df
        print(f"[save] {out_csv}")

    # ---------------------------------------------------------- NEI per probe
    neis = {}
    for name, df in trajectories.items():
        nei, _, _, _ = compute_nei(
            df["Valence_Smooth"].values,
            df["Arousal_Smooth"].values,
            df["Dominance_Smooth"].values,
        )
        neis[name] = nei

    probes = list(trajectories.keys())

    # ---------------------------------------------------------- V/A/D cross-probe Pearson
    print(f"\n{'='*70}\nTRIAL — V/A/D pairwise Pearson (smoothed)\n{'='*70}")
    rows = []
    for dim, col in [("V", "Valence_Smooth"), ("A", "Arousal_Smooth"), ("D", "Dominance_Smooth")]:
        for k1, k2 in itertools.combinations(probes, 2):
            r   = float(pearsonr(trajectories[k1][col], trajectories[k2][col])[0])
            rho = float(spearmanr(trajectories[k1][col], trajectories[k2][col]).correlation)
            print(f"  {dim}: {k1:10s} ~ {k2:10s}  r={r:+.3f}  rho={rho:+.3f}")
            rows.append({"channel": dim, "probe_a": k1, "probe_b": k2,
                         "pearson_r": r, "spearman_rho": rho})
    pd.DataFrame(rows).to_csv(OUT_DIR / "trial_3way_trajectory_correlations.csv",
                              index=False, encoding="utf-8-sig")

    # ---------------------------------------------------------- NEI Pearson
    print(f"\n{'='*70}\nTRIAL — NEI pairwise\n{'='*70}")
    nei_rows = []
    for k1, k2 in itertools.combinations(probes, 2):
        r   = float(pearsonr(neis[k1], neis[k2])[0])
        rho = float(spearmanr(neis[k1], neis[k2]).correlation)
        print(f"  NEI: {k1:10s} ~ {k2:10s}  r={r:+.3f}  rho={rho:+.3f}")
        nei_rows.append({"probe_a": k1, "probe_b": k2,
                         "pearson_r": r, "spearman_rho": rho})
    pd.DataFrame(nei_rows).to_csv(OUT_DIR / "trial_3way_nei_correlations.csv",
                                  index=False, encoding="utf-8-sig")

    # ---------------------------------------------------------- episodes + consensus
    print(f"\n{'='*70}\nTRIAL — episodes (top 5%, min 2 windows) + consensus\n{'='*70}")
    masks = {}
    for k in probes:
        m = extract_episode_mask(neis[k], config.NEI_PERCENTILE, config.MIN_EPISODE_DURATION)
        masks[k] = m
        runs = 0
        in_run = False
        for v in m:
            if v and not in_run:
                runs += 1
                in_run = True
            elif not v:
                in_run = False
        print(f"  {k:10s}  {m.sum():3d} windows flagged in {runs} episodes")

    print(f"\n  pairwise Jaccard:")
    for k1, k2 in itertools.combinations(probes, 2):
        union = (masks[k1] | masks[k2]).sum()
        inter = (masks[k1] & masks[k2]).sum()
        j = inter / max(union, 1)
        print(f"    {k1:10s} ~ {k2:10s}  J={j:.3f}  (∩={inter} ∪={union})")

    stacked = np.stack([masks[k] for k in probes], axis=0)
    counts  = stacked.sum(axis=0)
    print(f"\n  consensus:")
    for k in range(1, len(probes) + 1):
        m = counts >= k
        runs = 0
        in_run = False
        for v in m:
            if v and not in_run:
                runs += 1
                in_run = True
            elif not v:
                in_run = False
        print(f"    ≥{k}/{len(probes):d}  {int(m.sum()):3d} windows in {runs} episodes")

    # ---------------------------------------------------------- literary scan
    print(f"\n{'='*70}\nTRIAL — top-10 NEI windows per probe (literary identification)\n{'='*70}")
    for name in probes:
        df = trajectories[name]
        nei = neis[name]
        top_idx = np.argsort(nei)[::-1][:10]
        print(f"\n  [{name}] top-10 windows by NEI:")
        for rank, idx in enumerate(top_idx, 1):
            row = df.iloc[idx]
            text = str(row["Center_Sentence"])[:120]
            print(f"    {rank:2d}. W{int(row['Window_ID']):4d}  NEI={nei[idx]:.3f}  prog={row['Narrative_Progress']:.2f}  {text!r}")

    # ---------------------------------------------------------- known-scene alignment
    print(f"\n{'='*70}\nTRIAL — known scene alignment\n{'='*70}")
    centres = trajectories[probes[0]]["Center_Sentence"].astype(str).str.lower()
    for scene_name, keywords in KNOWN_SCENES.items():
        matches = []
        for kw in keywords:
            hits = centres.str.contains(kw.lower(), regex=False, na=False)
            if hits.any():
                matches.extend(centres[hits].index.tolist()[:3])
        matches = sorted(set(matches))
        if matches:
            print(f"\n  [{scene_name}] windows matching keywords:")
            for idx in matches[:5]:
                row = trajectories[probes[0]].iloc[idx]
                nei_vals = {p: float(neis[p][idx]) for p in probes}
                voting   = sum(masks[p][idx] for p in probes)
                text = str(row["Center_Sentence"])[:100]
                print(f"    W{int(row['Window_ID']):4d}  prog={row['Narrative_Progress']:.2f}  "
                      f"NEI {nei_vals}  vote={voting}/{len(probes)}  {text!r}")
        else:
            print(f"\n  [{scene_name}] NO keyword match — scene may not be in corpus or different translation")

    # ---------------------------------------------------------- plot
    progress = trajectories[probes[0]]["Narrative_Progress"].values
    fig, axes = plt.subplots(len(probes) + 1, 1, figsize=(15, 2 + 1.6 * len(probes)), sharex=True)
    colors = {"Valence": "tab:blue", "Arousal": "tab:orange", "Dominance": "tab:green"}

    strict = counts >= len(probes)
    for i, k in enumerate(probes):
        ax = axes[i]
        for ch, c in colors.items():
            s = trajectories[k][f"{ch}_Smooth"].values
            s_z = (s - s.mean()) / (s.std() + 1e-9)
            ax.plot(progress, s_z, color=c, label=ch, linewidth=1.0)
        ax.fill_between(progress, -3, 3, where=strict, color="red", alpha=0.15)
        ax.set_ylabel(k, fontsize=9)
        ax.set_ylim(-3, 3)
        if i == 0:
            ax.legend(loc="upper right", fontsize=7, ncol=3)

    ax = axes[-1]
    nei_palette = ["tab:purple", "tab:brown", "tab:red"]
    for j, k in enumerate(probes):
        ax.plot(progress, neis[k], label=f"NEI({k})", color=nei_palette[j], linewidth=1.0)
    half = (len(probes) // 2) + 1
    ymax = max(max(neis[k]) for k in probes)
    ax.fill_between(progress, 0, ymax, where=counts >= half, color="orange", alpha=0.15,
                    label=f"≥{half}/{len(probes)}")
    ax.fill_between(progress, 0, ymax, where=strict, color="red", alpha=0.30,
                    label=f"strict {len(probes)}/{len(probes)}")
    ax.set_ylabel("NEI")
    ax.set_xlabel("Narrative progress")
    ax.legend(loc="upper right", fontsize=7, ncol=5)

    plt.suptitle("Valeur — The Trial 3-probe trajectory + NEI", fontsize=11)
    plt.tight_layout()
    plot_path = OUT_DIR / "trial_3way_plot.png"
    plt.savefig(plot_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\n[save] {plot_path}")
    print(f"[done] elapsed {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
