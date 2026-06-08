#!/usr/bin/env python3
"""Merge 5 ridge diagnostics JSONs into a single all-probes file for upload bundle."""
import json
from pathlib import Path

PROBES = ["warriner", "nrc", "cr4_nrc", "cr4_nrcbert", "cr4_emo"]
out = {}
for p in PROBES:
    src = Path(f"models/ridge_{p}.diagnostics.json")
    out[p] = json.loads(src.read_text(encoding="utf-8"))

dest = Path("../paper1_writing/upload_bundle/diagnostics_all_probes.json")
dest.parent.mkdir(parents=True, exist_ok=True)
dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
print(f"[save] {dest}  ({dest.stat().st_size} bytes)")
