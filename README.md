# Valeur

**V**AD **a**ffective tr**a**jectories for **li**terary nar**r**ative — a
computational pipeline that projects narrative prose into
Valence / Arousal / Dominance space and operationalizes the clinical
*entrapment* state (Gilbert & Allan, 1998) as a window-level
**Narrative Entrapment Index (NEI)**.

Three supervision sources are currently supported:

| Probe | Training data | Scale | n\_train |
|---|---|---|---|
| `warriner` | Warriner et al. (2013) word norms | 1–9 | 13,915 |
| `nrc` | NRC VAD Lexicon v2.1 (Mohammad, 2025) | −1 to +1 | 44,728 |
| `cr4_nrc` | CR4-NarrEmote citizen annotations (Piper & Budac, 2025) | −1 to +1 | ~26,500 passages |

> This repository accompanies the CCLS 2026 poster **"Operationalizing
> Narrative Entrapment: Predictive Affective Trajectories via Contextual
> Sentence Embeddings in Kafka's *The Metamorphosis*"** and the forthcoming
> JCLS journal paper.

---

## Key results

**Within-text discrimination (headline finding):**

| Scene | Window | Probe | NEI | Mechanism |
|---|---|---|---|---|
| Chief clerk visit — Gregor's defensive monologue | W167–168 | Warriner | **6.73, 6.45** | V↓ A↑ D↓ all fire |
| Gregor's death — "empty and peaceful rumination" | W699–701 | Warriner | **0.00** | smoothed A < baseline → z_A = 0 → gate fails |

Same probe, same formula, same text. Two affectively opposite scenes correctly discriminated.

**Cross-lexicon D-divergence:**

| Level | V | A | D |
|---|---|---|---|
| Word-level (Mohammad 2025, 13,729 shared terms) | 0.81 | 0.61 | 0.33 |
| Trajectory-level — *Die Verwandlung* (this work) | 0.82 | 0.62 | **0.19** |
| Trajectory-level — *The Trial* (exploratory) | 0.81 | 0.70 | **0.10** |

Contextual application via SBERT amplifies the D-disagreement already present at lexicon level.

---

## Pipeline

```
    Lexicon norms  ─────────────────────────────────────────┐
    OR CR4 passages                                         │
         │                                                  │
         │  template "The word {w}."  /  passage text       │ spaCy sentence split
         ▼                                                  ▼
   ┌───────────┐  train_vad.py /        ┌─────────────────────────┐
   │   SBERT   │  train_cr4_probe.py    │  3-sentence sliding     │
   │ mpnet-768 │ ──► RidgeCV × 3 dims  │  windows (stride = 1)   │
   └───────────┘     → .joblib          └──────────┬──────────────┘
                                                   │ encode_vad.py
                                                   ▼
                                          ┌──────────────────┐
                                          │  Ridge project   │
                                          │  Savitzky–Golay  │
                                          │  → V/A/D signal  │
                                          └────────┬─────────┘
                                                   │ nei_plot.py
                                                   ▼
                                          ┌──────────────────┐
                                          │  gated-sum NEI   │
                                          │  episodes + ROIs │
                                          │  N-probe consensus│
                                          └──────────────────┘
```

---

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/levantuankhoa/valeur_ccls2026_poster.git
cd valeur_ccls2026_poster
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 2. Place lexicons + corpus in data/  (see data/README.md for download links)
#    data/warriner_2013.csv
#    data/unigrams-NRC-VAD-Lexicon-v2.1.txt
#    data/metamorphosis.txt

# 3. Run the full pipeline (2-lexicon main + EDA figure + F1 annotated figure)
bash run_all.sh
```

### CR4 probe (optional third probe)

```bash
# Requires: data/cr4_passages.csv  (prepared from CR4-NarrEmote release)
python prepare_cr4_passages.py
python train_cr4_all_targets.py   # trains cr4_nrc, cr4_nrcbert, cr4_emo variants
python encode_cr4_all.py          # encodes corpus with all CR4 probes
python compare_nway.py            # 5-probe comparison matrices
```

### The Trial exploratory transfer

```bash
# Requires: data/thetrial_clean.txt
python run_trial.py               # 3-probe NEI + consensus on The Trial
python trial_consensus_detail.py  # print episode detail + correct-exclusion check
```

---

## Script reference

### Core pipeline

| Script | Role |
|---|---|
| `config.py` | All hyperparameters and paths — single source of truth |
| `utils.py` | Lexicon loader, SBERT init, spaCy segmenter, window builder, Savitzky–Golay |
| `train_vad.py` | Stage 1 — Ridge probe training on lexicon word norms |
| `encode_vad.py` | Stage 2 — encode corpus windows via trained probe |
| `nei_plot.py` | Stage 3 — gated-sum NEI, episodes, ROIs, trajectory plot, consensus heatmap |
| `run_all.sh` | End-to-end orchestrator (steps 1–8) |

### CR4-NarrEmote integration

| Script | Role |
|---|---|
| `prepare_cr4_passages.py` | Aggregate CR4 citizen annotations from row-level to passage-level; book-stratified 70/15/15 split |
| `train_cr4_probe.py` | Ridge probe on CR4 single-sentence passages (NRC-target or NRCBERT-target) |
| `train_cr4_all_targets.py` | Train all three CR4 variants (cr4\_nrc, cr4\_nrcbert, cr4\_emo) in one pass |
| `encode_cr4_all.py` | Encode corpus with all CR4 probes; output same schema as `windows_*.csv` |

### Multi-probe analysis

| Script | Role |
|---|---|
| `compare_nway.py` | N-probe trajectory correlation matrices (V/A/D Pearson + NEI Spearman + Jaccard) |
| `compare_3way.py` | 3-probe (Warriner + NRC + CR4-NRC) focused comparison |
| `consolidate_diagnostics.py` | Merge all `*.diagnostics.json` into one summary table |

### Paper figures

| Script | Output | Paper section |
|---|---|---|
| `plot_eda_lexicons.py` | `results/eda_lexicons_scatter.png` | §3.3 Lexicon characterization |
| `plot_f1_annotated.py` | `results/F1_warriner_trajectory_annotated.png` | §4.4 Headline discrimination |

### Validation / diagnostics

| Script | Role |
|---|---|
| `verify_gregor_death.py` | Negative-control check: print per-window NEI for chief clerk + death zones across all probes |
| `run_trial.py` | Encode *The Trial* (3,644 windows) with 3 main probes; extract consensus episodes |
| `trial_consensus_detail.py` | Print Trial consensus episode detail + correct-exclusion finding (cathedral + execution) |

---

## Results directory

```
results/
├── warriner.{nei,episodes,rois}.csv    # per-probe summary artefacts
├── nrc.*                               # same for NRC
├── consensus.{nei_matrix.csv,heatmap.png}  # 2-probe consensus
├── compare_3way_*.csv / compare_3way_plot.png
├── compare_5way_*.csv / compare_5way_plot.png  # full 5-probe matrices
├── trial_3way_*.csv / trial_3way_plot.png      # The Trial transfer
├── trial_lenient_2of3_episodes.csv             # Trial consensus episodes
├── eda_lexicons_scatter.png                    # EDA paper figure
└── F1_warriner_trajectory_annotated.png        # F1 paper figure
```

Per-window CSVs (`windows_*.csv`, `trial_*.csv`) are gitignored (regeneratable). Diagnostic JSONs and summary CSVs are committed.

---

## The NEI formula

$$
\text{NEI}_i =
\begin{cases}
z^V_i + z^A_i + z^D_i & \text{if } z^V_i > \tau \;\wedge\; z^A_i > \tau \;\wedge\; z^D_i > \tau \\
0 & \text{otherwise}
\end{cases}
$$

where $z^V_i = \max\!\left(\frac{(\text{base}_V - V_i) - \mu}{\sigma}, 0\right)$ (and analogously for A↑, D↓), $\tau = 0.10$ (default), baseline = within-text median.

The gate enforces Gilbert & Allan's (1998) joint condition: *"strongly aroused flight motivation which is powerfully blocked"* — all three channels must be non-trivially active. Without the gate, near-independent V/A channels (NRC: r ≈ −0.08) produce ~74% false positives via naive summation.

`--method add` and `--method mult` are retained for ablation.

---

## Held-out probe diagnostics

| Probe | V (r) | A (r) | D (r) | n\_train |
|---|---|---|---|---|
| Warriner | 0.812 | 0.651 | 0.710 | 11,132 |
| NRC v2.1 | 0.761 | 0.634 | 0.695 |  35,781 |
| CR4-NRC | 0.677 | 0.513 | 0.591 | 26,537 passages |

All p < 0.002 (500-iteration permutation null). Full diagnostics in `models/*.diagnostics.json`.

---

## Method notes

- **Encoder:** `all-mpnet-base-v2` (768-d, frozen). The same encoder instance is used for both probe training (word templates / CR4 passages) and corpus inference (Kafka/Trial windows) — cross-region transfer within a shared manifold, not cross-domain.
- **Word template:** `"The word {w}."` wraps each lexicon item as a minimal sentence so it occupies the same 768-d manifold as the corpus windows.
- **CR4 single-sentence training → 3-sentence inference:** defensible because SBERT is length-agnostic (all inputs map to the same 768-d space) and Piper & Budac (2025) themselves find single-sentence context maximises annotation signal.
- **Window size:** 3 sentences, stride 1 (`config.py`).
- **Smoothing:** Savitzky–Golay, auto-sized odd window ≤ 21, polyorder 2.
- **Reproducibility:** `RANDOM_STATE = 42` seeded globally.

### Hardware / runtime

Reference: AMD Ryzen 7 7800X3D, 32 GB DDR5, Windows 11.

| Task | Time (CPU) |
|---|---|
| Full `bash run_all.sh` (2-probe pipeline + figures) | ~6–8 min |
| CR4 encode (38K passages, SBERT) | ~1–2 min |
| Trial full novel encode (3,644 windows) | ~2–3 min |

GPU optional; `utils.detect_device()` auto-selects CUDA when available.

---

## Citation

```bibtex
@inproceedings{khoa2026valeur,
  author    = {Lê, Văn Tuấn Khoa and Võ, Thị Phương Linh
               and Nguyễn, Thị Ngọc Trinh and Trương, Nguyễn Cát Ly},
  title     = {Operationalizing Narrative Entrapment: Predictive
               Affective Trajectories via Contextual Sentence Embeddings
               in Kafka's {\em The Metamorphosis}},
  booktitle = {Computational Literary Studies Conference (CCLS 2026), Potsdam},
  note      = {Poster session},
  year      = {2026}
}
```

---

## Authors

**Lê Văn Tuấn Khoa** (lead) — Faculty of Foreign Languages, Dalat University, Vietnam.

Co-authors: Võ Thị Phương Linh, Nguyễn Thị Ngọc Trinh, Trương Nguyễn Cát Ly.

## Acknowledgments

During the preparation of this work, the lead author used Claude (Anthropic)
to assist with code refactoring, documentation, and analysis. All content
was reviewed and edited by the lead author, who takes full responsibility for it.

## License

MIT — see [`LICENSE`](LICENSE).
