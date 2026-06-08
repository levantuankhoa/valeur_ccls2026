#!/usr/bin/env python3
"""Extract ≥2/3 consensus episodes from Trial run with full context text."""
from pathlib import Path
import numpy as np
import pandas as pd
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import config
from nei_plot import compute_nei

PROBES = ["warriner", "nrc", "cr4_nrc"]

def extract_mask(nei, pct, dur):
    pos = nei[nei > 0]
    if not len(pos):
        return np.zeros_like(nei, dtype=bool)
    raw = nei >= np.percentile(pos, 100 * pct)
    out = np.zeros_like(raw)
    i = 0
    while i < len(raw):
        if raw[i]:
            j = i
            while j < len(raw) and raw[j]:
                j += 1
            if j - i >= dur:
                out[i:j] = True
            i = j
        else:
            i += 1
    return out

dfs = {p: pd.read_csv(f"results/trial_{p}.csv") for p in PROBES}
neis = {p: compute_nei(dfs[p]["Valence_Smooth"].values,
                        dfs[p]["Arousal_Smooth"].values,
                        dfs[p]["Dominance_Smooth"].values)[0]
        for p in PROBES}
masks = {p: extract_mask(neis[p], config.NEI_PERCENTILE, config.MIN_EPISODE_DURATION)
         for p in PROBES}
stacked = np.stack([masks[p] for p in PROBES])
counts = stacked.sum(axis=0)

ref = dfs[PROBES[0]]
print(f"\n{'='*80}\n≥2/3 CONSENSUS EPISODES (Trial)\n{'='*80}\n")
i = 0
episode_n = 1
while i < len(counts):
    if counts[i] >= 2:
        j = i
        while j < len(counts) and counts[j] >= 2:
            j += 1
        if j - i >= config.MIN_EPISODE_DURATION:
            print(f"--- Episode {episode_n}: W{int(ref.iloc[i]['Window_ID'])}-W{int(ref.iloc[j-1]['Window_ID'])}  ({j-i} windows, prog {ref.iloc[i]['Narrative_Progress']:.2f}-{ref.iloc[j-1]['Narrative_Progress']:.2f}) ---")
            # Which probes voted at each window
            for k in range(i, j):
                votes = [p for p in PROBES if masks[p][k]]
                nei_str = " ".join(f"{p[:3]}={neis[p][k]:.1f}" for p in PROBES)
                text = str(ref.iloc[k]["Center_Sentence"])[:160]
                print(f"  W{int(ref.iloc[k]['Window_ID']):4d}  [{','.join(votes):20s}]  {nei_str}  {text!r}")
            episode_n += 1
            print()
        i = j
    else:
        i += 1

# Also save CSV
rows = []
i = 0
while i < len(counts):
    if counts[i] >= 2:
        j = i
        while j < len(counts) and counts[j] >= 2:
            j += 1
        if j - i >= config.MIN_EPISODE_DURATION:
            for k in range(i, j):
                votes = ",".join(p for p in PROBES if masks[p][k])
                rows.append({
                    "window_id": int(ref.iloc[k]["Window_ID"]),
                    "narrative_progress": float(ref.iloc[k]["Narrative_Progress"]),
                    "voted_probes": votes,
                    "warriner_nei": float(neis["warriner"][k]),
                    "nrc_nei": float(neis["nrc"][k]),
                    "cr4_nrc_nei": float(neis["cr4_nrc"][k]),
                    "center_sentence": str(ref.iloc[k]["Center_Sentence"]),
                    "full_window_text": str(ref.iloc[k]["Full_Window_Text"]),
                })
        i = j
    else:
        i += 1
pd.DataFrame(rows).to_csv("results/trial_lenient_2of3_episodes.csv", index=False, encoding="utf-8-sig")
print(f"\n[save] results/trial_lenient_2of3_episodes.csv ({len(rows)} rows)")
