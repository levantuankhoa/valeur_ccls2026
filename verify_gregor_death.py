#!/usr/bin/env python3
"""Verify NEI behaviour at Gregor's death scene (W698-W705) vs chief clerk (W167-168)."""
import sys
import numpy as np
import pandas as pd
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from nei_plot import compute_nei

print(f"{'='*90}")
print("NEGATIVE CONTROL: Gregor's death (W698-W705) — should be V↓ A↓ D↓ = defeat = NEI=0")
print("POSITIVE: Chief clerk (W165-W175) — should be V↓ A↑ D↓ = entrapment = NEI>0")
print(f"{'='*90}\n")

for lex in ["warriner", "nrc", "cr4_nrc"]:
    df = pd.read_csv(f"results/windows_{lex}.csv")
    V = df["Valence_Smooth"].values
    A = df["Arousal_Smooth"].values
    D = df["Dominance_Smooth"].values
    nei, zV, zA, zD = compute_nei(V, A, D)
    baseV, baseA, baseD = np.median(V), np.median(A), np.median(D)

    print(f"--- {lex} (baseline median: V={baseV:.2f} A={baseA:.2f} D={baseD:.2f}) ---")
    print(f"\n  GREGOR'S DEATH zone (W698-W705):")
    print(f"  {'W':>4s} {'V':>5s} {'A':>5s} {'D':>5s}  {'zV':>5s} {'zA':>5s} {'zD':>5s}  {'NEI':>5s}  text")
    for i in range(697, min(706, len(df))):
        row = df.iloc[i]
        text = str(row["Center_Sentence"])[:60]
        print(f"  {int(row['Window_ID']):4d} {V[i]:5.2f} {A[i]:5.2f} {D[i]:5.2f}  "
              f"{zV[i]:5.2f} {zA[i]:5.2f} {zD[i]:5.2f}  {nei[i]:5.2f}  {text!r}")

    print(f"\n  CHIEF CLERK zone (W165-W175):")
    print(f"  {'W':>4s} {'V':>5s} {'A':>5s} {'D':>5s}  {'zV':>5s} {'zA':>5s} {'zD':>5s}  {'NEI':>5s}  text")
    for i in range(164, min(176, len(df))):
        row = df.iloc[i]
        text = str(row["Center_Sentence"])[:60]
        print(f"  {int(row['Window_ID']):4d} {V[i]:5.2f} {A[i]:5.2f} {D[i]:5.2f}  "
              f"{zV[i]:5.2f} {zA[i]:5.2f} {zD[i]:5.2f}  {nei[i]:5.2f}  {text!r}")
    print()
