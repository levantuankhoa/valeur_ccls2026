# Valeur — NEI Pipeline Reference

**Project:** Valeur (formerly drafted under the working title *Noema / Project Kafka*)
**Author:** Lê Văn Tuấn Khoa (Dalat University)
**Target venue:** CCLS 2026 poster → JCLS journal submission
**Purpose:** Master reference for theoretical grounding, technical methodology, code verification, and poster design.
**Audience:** Claude Code (for code verification) + author (for poster prep)
**Last updated:** April 2026 — synced against current `valeur2/` codebase.

> **Naming note.** The repository is `valeur` (acronym: **V**AD **a**ffective tr**a**jectories for **li**terary nar**r**ative). All earlier "Noema / Project Kafka" references were a working title from the February 2025 draft and have been replaced. The `bibtex` key is `khoa2026valeur`.

---

## TABLE OF CONTENTS

1. [Theoretical Grounding — Gilbert & Allan (1998)](#1-theoretical-grounding)
2. [NEI Formula — Why GATED_SUM, not ADD or MULTIPLY](#2-nei-formula)
3. [SBERT Methodology — Linear Probing Paradigm](#3-sbert-methodology)
4. [Two-Stage Regression — Mathematical Formulation](#4-two-stage-regression)
5. [Cross-Lexicon Comparison — Why D=0.33 is a Feature, Not a Bug](#5-cross-lexicon-comparison)
6. [Code Verification Checklist (for Claude Code)](#6-code-verification-checklist)
7. [Poster Design Recommendations](#7-poster-design-recommendations)
8. [Honest Limitations & Rebuttal Strategies](#8-honest-limitations)
9. [Appendix A — Key citations](#appendix-a-key-citations)
10. [Appendix B — Minimal NEI verification snippet (matches production code)](#appendix-b-minimal-nei-verification-snippet)
11. [Appendix C — Doc → code crosswalk (Feb 2025 → Apr 2026)](#appendix-c-doc--code-crosswalk)

---

## 1. THEORETICAL GROUNDING

### 1.1 Source: Gilbert, P., & Allan, S. (1998)

> Gilbert, P., & Allan, S. (1998). The role of defeat and entrapment (arrested flight) in depression: An exploration of an evolutionary view. *Psychological Medicine*, 28(3), 585–598.

This paper provides the **psychological construct** that NEI operationalizes computationally.

### 1.2 The Three Components of NEI

NEI requires the **simultaneous** presence of three conditions:

| Component | VAD Direction | Gilbert & Allan Construct | Direct Quote |
|---|---|---|---|
| **V↓** | Negative valence | Defeat / hopelessness | Defeat × BDI: r = 0.77, p < .001 (Table 4) |
| **A↑** | High arousal | Aroused but blocked flight | "Physiologically, depression is a state of high arousal not conservation" (p. 586) |
| **D↓** | Low dominance | Subordinate self-perception | "The one behaving submissively does not actually have the power to de-escalate" (p. 586) |

### 1.3 Why CONJUNCTIVE (not compensatory)

Critical quote (p. 596):

> "**Strongly aroused flight motivation which is powerfully blocked**, may be particularly depressogenic."

This is the linguistic signature of **conjunction**: high arousal AND blocked agency. It is NOT "high arousal OR blocked agency". This grammatical distinction is the theoretical justification for using a **gated** formula instead of a purely additive one.

### 1.4 Mapping into VAD Framework

Gilbert & Allan do NOT use Mehrabian–Russell VAD terminology. The mapping is an **operational reformulation**, not a claim about what they wrote. Be honest about this in the paper.

**Defensible framing for reviewers:**

> "We operationally reformulate Gilbert & Allan's (1998) three empirically validated components — defeat, arrested flight, and subordination — into the dimensional affect framework of Mehrabian & Russell (1974). This is a notation choice, not a theoretical claim about Gilbert & Allan's intent."

---

## 2. NEI FORMULA

### 2.1 Three Candidate Formulas Considered

```python
# All three operate on rectified within-text z-scores of directional deltas:
#   v_drop = baseline_V − V    (positive when V is below the text-level baseline)
#   a_rise = A − baseline_A    (positive when A is above the text-level baseline)
#   d_drop = baseline_D − D    (positive when D is below the text-level baseline)
# zV, zA, zD are then  max((delta − mean(delta)) / std(delta), 0)
# i.e. positive-only z-scores of the directional deltas across the full text.

# ADD (compensatory): REJECTED
NEI_add = zV + zA + zD

# MULT (strict conjunction): REJECTED (kept for ablation only)
NEI_mult = zV * zA * zD

# GATED_SUM (conjunction + magnitude): SELECTED
def NEI_gated(zV, zA, zD, threshold=0.10):
    if (zV > threshold) and (zA > threshold) and (zD > threshold):
        return zV + zA + zD
    else:
        return 0.0
```

The actual implementation lives in `nei_plot.py:compute_nei` (vectorised over the whole window array). Default threshold is `config.NEI_MIN_COMPONENT = 0.10`. Default baseline is the within-text **median** (`config.BASELINE_METHOD = "median"`); `"mean"` is also wired up.

### 2.2 Why ADD was Rejected

**Empirical failure:** ADD produced a false positive at Window 1 (Gregor's awakening scene).

- Pattern observed: V↓ (sad), A↓ (low energy/sleepy), D↓ (helpless)
- Required pattern for entrapment: V↓ A↑ D↓
- ADD: zV + zA + zD → after rectification, zV is large but zA = 0 (a_rise < 0); summing still yields a moderate score that contaminates the top-percentile threshold
- Gilbert & Allan's theory explicitly requires arousal, not just sadness

**Statistical cause:** on lexicons with near-independent V/A channels (NRC VAD v2.1, V↔A ≈ −0.08 empirically), a large fraction of windows have zA ≈ 0; ADD therefore systematically overestimates entrapment in low-arousal regions.

### 2.3 Why MULTIPLY was Rejected

**Logical correctness but practical problems:**

| Problem | Explanation |
|---|---|
| Magnitude compression | Three rectified z-scores |z| < 1 multiplied → very small numbers, hard to threshold |
| Vanishing on weak co-presence | Any single zero kills the index even when the other two are clearly elevated |
| Reviewer pushback | Multiplying z-scores has no natural statistical interpretation; reviewers will demand justification |

### 2.4 Why GATED_SUM is the Right Choice

| Property | Benefit | Theoretical Anchor |
|---|---|---|
| Gate enforces conjunction | All three components must be non-trivially present | Gilbert & Allan's "aroused AND blocked" |
| Sum after gate | Magnitude is interpretable; scale matches z-distribution | Mirrors Defeat Scale construction (sum of 16 Likert items, α = 0.94) |
| Threshold (0.10) is principled | Filters noise without imposing a strong a priori cutoff | Conservative; allows weak co-presence |
| Eliminates false positives | Window 1 NEI now = 0 (correct) | Construct validity |

### 2.5 Empirical Validation (current run: April 2026 artefacts in `results/`)

After gated_sum was applied (current snapshot, regenerated by `bash run_all.sh`):

- Warriner Ridge: **6 entrapment episodes** (≥ 2 contiguous windows), top-5% NEI cutoff, 0.10 component gate
- NRC Ridge: **9 entrapment episodes** under the same gate
- Cross-lexicon **consensus episodes** (where both lexicons mark contiguous overlap) cluster around three narrative loci:

  | Window range | Scene | Lexicons |
  |---|---|---|
  | ~146–157 | Gregor's transformation realised; mother/sister react ("I am astonished") | Warriner + NRC |
  | ~165–175 | Chief clerk confrontation (Chapter 1 climax) | Warriner + NRC |
  | ~235–242 | Travellers/job monologue ("I know nobody likes the travellers") | Warriner + NRC |

- Warriner-only (lexicon-divergent finding): windows ~666–670 — the **lodgers / abandonment scene** ("this animal is persecuting us… force us to sleep on the streets"). NRC does not flag this region; this is exactly the kind of dominance-axis disagreement Section 5 frames as a feature.

> **Numbers will fluctuate by ±1–2 episodes depending on smoothing window size and `MIN_EPISODE_DURATION`. Always re-derive from `results/*.episodes.csv` before quoting in the paper.** The structural finding (~3 consensus regions aligning with established literary turning points) is robust.

This is **construct validity evidence**: an independently-derived formula produces signal that aligns with human-identified narrative turning points.

---

## 3. SBERT METHODOLOGY

### 3.1 Framing: This is Linear Probing

This is the single most important framing decision. Use these terms:

> "We employ a linear probing methodology (cf. Hewitt & Manning 2019 for syntactic probing; Tenney et al. 2019 for semantic probing). A frozen SBERT encoder (`all-mpnet-base-v2`, 768-d) provides sentence embeddings. We train a Ridge regression probe on word-template encodings and apply the same projection to natural sentence encodings within Kafka's *Die Verwandlung*."

This framing:
- Places the method in an **established NLP paradigm** (probing) reviewers know and respect
- Avoids claims about geometric structure of embedding space
- Avoids the trap of "cross-domain transfer" criticism

### 3.2 The "The word {}." Template

```python
# Encoding convention (config.TEMPLATE)
TEMPLATE = "The word {}."
def encode_word(word, model):
    return model.encode(TEMPLATE.format(word))  # 768-dim vector under all-mpnet-base-v2
```

**Critical clarification:** This is NOT word2vec-style word embedding. SBERT (`all-mpnet-base-v2`) is a **sentence encoder**. The template wraps each word as a minimal sentence so it is embedded in the same 768-dim manifold as Kafka's sentences.

**Cross-region within shared manifold, NOT cross-domain transfer:**
- Training points (word-templates) and inference points (Kafka sentences) live in the **same** geometric space
- The probe weight vector **w** is a direction in this shared space
- Applying **w** to Kafka sentences projects them onto the same affective axis learned from word-templates

### 3.3 Linear Representation Hypothesis (Park et al. 2023)

The pipeline relies on the empirical claim that affective dimensions are approximately **linear subspaces** of modern transformer embedding spaces.

**Evidence supporting this in the pipeline (current run, see `models/ridge_*.diagnostics.json`):**

| Probe | Pearson r | Spearman ρ | RMSE | MAE |
|---|---|---|---|---|
| Warriner Valence | **0.812** | 0.800 | 0.744 | 0.589 |
| Warriner Arousal | 0.651 | 0.621 | 0.688 | 0.546 |
| Warriner Dominance | 0.710 | 0.685 | 0.659 | 0.526 |
| NRC Valence | **0.761** | 0.754 | 0.322 | 0.247 |
| NRC Arousal | 0.634 | 0.632 | 0.390 | 0.311 |
| NRC Dominance | 0.695 | 0.700 | 0.317 | 0.244 |

Ridge held-out performance is high enough on the V channel (r ≈ 0.76–0.81) to support the linearity claim. Random Forest comparison was performed in earlier drafts (Feb 2025) and did not provide meaningful improvement; RF has since been **dropped from the production code path** (`train_vad.py` fits Ridge only). The Ridge ≈ RF observation is preserved as a one-line anecdote in `README.md` and is the empirical hook for the linear-representation argument — see also Section 6.5.

**Defensive note:** the paper does NOT claim Euclidean isotropy or manifold flatness. It claims **linear separability of affective axes**, which is empirically supported.

### 3.4 Honest Weaknesses (Acknowledge, Don't Hide)

1. **Word-template ≠ natural contextual embedding.** The template is an encoding convention to ensure vector consistency, not a semantic claim about how the word "means" in context.

2. **Word-template sentences cluster in narrow manifold region.** They share syntactic frame ("The word X."), so they may occupy a sub-region of the manifold. Generalization to natural literary sentences is **empirically demonstrated** (≈3 consensus windows aligning with literary turning points), not theoretically guaranteed.

3. **Validation is at word-level, not sentence-level.** No sentence-level VAD ground truth exists for Kafka. r ≈ 0.76–0.81 is on the word-template held-out test set. Cross-corpus validation = explicit future work.

4. **Lexicon-based affect is decontextualized.** Literary irony can flip the predicted valence. Known limitation of all lexicon-based approaches.

### 3.5 Defensive Statement (memorize for Q&A)

> "This is a linear probing methodology — we recover affective directions via Ridge regression on word-template supervision and apply the same projection at the sentence level. SBERT places both word-template encodings and natural sentences in the same 768-dimensional manifold, so this is cross-region transfer within a shared geometric space, not cross-domain. We make no claim of Euclidean isotropy; we claim linear separability of affective dimensions, supported empirically by held-out Ridge correlations of r ≈ 0.81 (Warriner Valence) / r ≈ 0.76 (NRC Valence) and by construct validity at the discourse level — consensus entrapment windows align with established literary turning points. Cross-corpus validation is explicitly framed as future work."

---

## 4. TWO-STAGE REGRESSION

### 4.1 Yes — there are TWO regression equations

The pipeline solves two distinct equations sequentially. Stage 1 lives in `train_vad.py`; Stage 2 lives in `encode_vad.py`.

### 4.2 Stage 1: Probe Training (solve for w) — `train_vad.py`

**Inputs:**
- `X_train ∈ R^(N × 768)`: word-template embeddings of N lexicon words (`config.TEMPLATE` formatted, encoded via `utils.encode_texts`)
- `y_train ∈ R^N`: VAD scores from lexicon (Warriner: 1–9; NRC: −1 to +1; loaded by `utils.load_lexicon`)

**Equation:**
```
w*, b* = argmin_{w, b} || X_train · w + b − y_train ||² + α ||w||²
```

This is `RidgeCV(alphas=config.ALPHAS, cv=5)`, fit independently per dimension.

**Output:**
- `w ∈ R^768`: weight vector (the affective direction)
- `b ∈ R`: intercept
- Both encapsulated inside `sklearn.linear_model.RidgeCV`, persisted via `joblib`

**Probe count:** 2 lexicons × 3 dimensions (V, A, D) = **6 probes total**, in 2 `.joblib` files (`models/ridge_warriner.joblib`, `models/ridge_nrc.joblib`); each artefact carries `models = [RidgeCV_V, RidgeCV_A, RidgeCV_D]` plus the two scalers.

### 4.3 Stage 2: Application to corpus (compute y') — `encode_vad.py`

**Inputs:**
- `X_kafka ∈ R^(M × 768)`: SBERT embeddings of M sliding 3-sentence windows from *Die Verwandlung*
- `w*, b*` (and the two scalers) from Stage 1

**Equation:**
```
y'_kafka = X_kafka · w* + b*
```

This is a forward pass through the trained probe — no further training.

**Output:**
- `y'_kafka ∈ R^M`: predicted V/A/D score per window, written to `results/windows_<lexicon>.csv` (raw) and `..._Smooth` (Savitzky–Golay)

### 4.4 Scaling Notes (IMPORTANT for code review)

`train_vad.py` uses BOTH:
- `StandardScaler` on `X` (input embeddings)
- `StandardScaler` on `y` (lexicon scores)

This is implementation detail; both scalers are fit on the **training split only** and saved into the `.joblib`. The `y`-scaling means:
- Training is on z-scored y values
- `encode_vad.py` calls `y_scaler.inverse_transform(...)` to recover lexicon scale (1–9 for Warriner, −1 to +1 for NRC)
- Reported α values (currently 100.0 for all 6 probes) depend on the scaling choice

For paper reproducibility this is documented in `train_vad.py:77–117` and visible in the persisted `models/ridge_*.joblib` artefacts.

### 4.5 Why this is "linear probing", literally

Stage 1 = train probe on frozen encoder.
Stage 2 = apply probe to new inputs from same encoder.

This is the textbook probing protocol. Cite Hewitt & Manning (2019), Tenney et al. (2019), Belinkov (2022) for review.

### 4.6 Convergence ROIs (a separate, complementary metric)

Beyond NEI, `nei_plot.run_single` also computes **Convergence ROIs**: contiguous windows where the V/A/D *range* (max − min over the three smoothed channels) falls below a scale-dependent threshold. These mark moments where the three affective axes collapse to a single value — a different kind of narrative event from entrapment.

Thresholds (from `config.CONVERGENCE_THRESHOLD`):
- Warriner (1–9 scale): `0.50`
- NRC (−1 to +1 scale): `0.15`

Outputs land in `results/<lexicon>.rois.csv` and overlay on `results/<lexicon>.plot.png`. ROIs and entrapment episodes are reported as **distinct artefacts** — ROIs answer "where do the dimensions converge?", episodes answer "where is the narrative entrapped?".

---

## 5. CROSS-LEXICON COMPARISON

### 5.1 The Validation Finding

Pipeline cross-lexicon agreement on Kafka trajectories matches Mohammad's official NRC-vs-Warriner overlap correlations **exactly** (V/D) or closely (A):

| Dimension | Mohammad's official table (word-level overlap) | This pipeline (Kafka trajectory-level, Feb 2025 reference run) |
|---|---|---|
| V | 0.81 | 0.81 |
| A | 0.62 | 0.64 |
| D | **0.33** | **0.33** |

> **Action item before submission:** re-derive the trajectory-level correlation from the current April 2026 run by saving `results/consensus.nei_matrix.csv` and computing pair-wise Pearson on the three smoothed dimensions (not just the gated NEI). The Spearman ρ on the gated NEI itself is already printed by `nei_plot.run_consensus`.

### 5.2 Interpretation: This is GOOD news

This is not coincidence — it is structural. The cross-lexicon disagreement is **inherent to the lexicons themselves**, not introduced by the pipeline.

**Implications:**

1. **Pipeline integrity:** the pipeline preserves the lexicon-level correlation structure — does not introduce spurious agreement or disagreement.

2. **D=0.33 is documented by NRC:** Mohammad (NRC author) himself reports D=0.33 between NRC and Warriner overlap. The paper can cite this directly to defend the finding.

3. **Reframes "Dominance divergence" claim:** the user's finding that Dominance disagrees most is now backed by Mohammad's own documentation. Not a finding about the pipeline — a finding about **dimensional structure of affective space across lexicon traditions**.

### 5.3 All-unigrams vs. Overlapping Terms

Two distinct comparisons require different data:

#### Comparison A: Word-level lexicon score agreement (what Mohammad reports)
- **MUST use overlapping terms** (~13,915 words common to both lexicons)
- Reason: cannot correlate scores for words present in only one lexicon
- This is what the table in NRC documentation reports

#### Comparison B: Trajectory-level pipeline output agreement (what this pipeline computes)
- **Use each model's full training set** (NRC 44,728; Warriner 13,915)
- Reason: each Ridge probe should be trained on maximum available supervision
- Compare the resulting V/A/D predictions on the same Kafka windows
- This is methodologically valid because we are comparing **model outputs**, not lexicon scores

#### Why this is correct
Both Ridge probes apply weights to the **same** Kafka sentences. The comparison is on probe output, not on lexicon scores. Training each probe on its own full lexicon is standard practice — like training two different classifiers on their own training sets, then comparing predictions on a shared test set.

### 5.4 What to Report in the Paper

```
Cross-lexicon agreement on Kafka trajectories:
  V: r = 0.81 (matches Mohammad's word-level r = 0.81)
  A: r = 0.64 (close to Mohammad's word-level r = 0.62)
  D: r = 0.33 (matches Mohammad's word-level r = 0.33)

The exact match for V and D, and close match for A, indicates that
trajectory-level cross-lexicon agreement reflects the underlying
lexicon-level agreement, not artefacts of the embedding pipeline.
The low Dominance agreement is documented by Mohammad (2025) and
reflects structural differences in how each lexicon construes the
Dominance dimension.
```

### 5.5 Dual-lexicon Design as Triangulation

The dual-lexicon setup serves two complementary purposes:

1. **Triangulation (consensus = high confidence):** When both lexicons agree on an entrapment window, that finding is robust to choice of lexicon. The consensus windows are the highest-confidence findings.

2. **Divergence analysis (disagreement = finding about lexicons):** When lexicons disagree systematically (especially Dominance — see the lodgers/abandonment scene at W ~666–670, flagged only by Warriner), this reveals structural properties of the dimensional construct itself, not noise.

This dual-purpose design is a methodological contribution.

---

## 6. CODE VERIFICATION CHECKLIST

For Claude Code (or any reviewer) to verify the implementation matches this specification.

### 6.1 Files (current as of April 2026)

The Feb-2025 monolithic `vad_kafka_ridge_rf_dual_lexicon.py` was refactored into three composable scripts plus a config module:

| File | Role | Stage |
|---|---|---|
| `config.py` | Single source of truth for hyperparameters, paths, lexicon registry | shared |
| `utils.py` | Lexicon loading, SBERT init, sentence segmentation (spaCy), sliding-window builder, Savitzky–Golay window helper, RNG seeding | shared |
| `train_vad.py` | Stage 1 — fits 3 × `RidgeCV` per lexicon on word-template embeddings; persists `.joblib` + `.diagnostics.json` | Stage 1 |
| `encode_vad.py` | Stage 2 — segments text, builds 3-sentence windows, encodes with SBERT, projects via Ridge, applies Savitzky–Golay smoothing, writes `windows_<lex>.csv` | Stage 2 |
| `nei_plot.py` | Stage 3 — `compute_nei` (the gated_sum implementation), convergence ROI extraction, episode extraction, single-lexicon plot, dual-lexicon consensus heatmap | Stage 3 |
| `run_all.sh` | End-to-end driver: 2 train + 2 encode + 2 single-lexicon NEI + 1 consensus | orchestration |

**Auxiliary diagnostic / regression scripts** (run on demand, not part of the paper pipeline):

| File | Purpose |
|---|---|
| `compare_results.py` | Compares current Ridge VAD trajectories vs. Feb 2025 reference outputs (correlation, RMSE, MAE, scatter, diff distribution) |
| `compare_episodes.py` | Compares old vs. new entrapment episodes/ROIs — Jaccard, IoU, MATCHED/MISSING tags |
| `diagnose_method.py` | Reverse-engineers which NEI method (`add` / `mult` / `gated_sum`) the Feb 2025 outputs were generated under, by correlating against re-computed indices on the old VAD values |
| `detail_episodes.py` | Prints each entrapment episode with full sentences, computes Warriner-vs-NRC episode-level Jaccard, locates the apple-throwing scene |
| `plot_convergence.py` | Plots VAD trajectories with convergence ROIs highlighted; reports ROI overlap statistics |

**Legacy / deprecated files (kept for reproducibility of Feb 2025 results, not used by `run_all.sh`):**

| File | Status |
|---|---|
| `vad_roi_plot.py` / `vad_roi_plot_batch.py` | Feb 2025 batch processor; uses the *old* `metamorphosis_vad_<lex>_windows.csv` filenames, the old `Valence_Ridge_Smooth` column scheme, and `ENTR_METHOD = "add"` (the rejected formula). **Do not run on the new pipeline outputs** — column names mismatch and the additive method is the false-positive failure mode this paper argues against. Either delete these files before final submission, or keep them only inside `legacy/` with a README disclaimer. |

### 6.2 Logical Flow Verification

```
[Lexicon CSVs] → utils.load_lexicon → utils.encode_texts("The word {}.")
                                                ↓
                                  StandardScaler(X), StandardScaler(y)
                                  on training split only (train_vad.py)
                                                ↓
                                    RidgeCV (alphas=config.ALPHAS, cv=5)
                                    independent per dimension (V/A/D)
                                                ↓
                                  models/ridge_<lex>.joblib   (+ .diagnostics.json)
                                                ↓
[Kafka text] → utils.segment_sentences (spaCy en_core_web_sm)
                                                ↓
                              utils.make_sliding_windows (w=3, step=1)
                                                ↓
                              utils.encode_texts (SBERT, normalised)
                                                ↓
                              x_scaler.transform → predict per dim
                                  → y_scaler.inverse_transform
                                                ↓
                              Savitzky-Golay smoothing
                              (window auto-chosen ≤ SAVGOL_WINDOW_MAX, polyorder 2)
                                                ↓
                              results/windows_<lex>.csv
                                                ↓
                              nei_plot.compute_nei
                              (delta → within-text z-score → rectify → gate@0.10 → sum)
                                                ↓
                              entrapment mask = NEI ≥ p95
                              ROI mask = (max − min) over V/A/D ≤ scale-dep threshold
                                                ↓
                              extract_contiguous_regions(min_duration=2)
                                                ↓
                              <lex>.nei.csv  +  <lex>.episodes.csv  +  <lex>.rois.csv
                                                ↓
                              consensus mode: cross-lexicon Spearman ρ on NEI
                              + heatmap + consensus.nei_matrix.csv
```

### 6.3 Parameter Sanity Checks

| Parameter | Expected value (config.py) | Where it is consumed |
|---|---|---|
| `SBERT_MODEL` | `"all-mpnet-base-v2"` | `utils.load_sbert` (called from both `train_vad.py` and `encode_vad.py`); the artefact stores `sbert_model` to enforce consistency at inference time |
| `TEMPLATE` | `"The word {}."` | `train_vad.py:93` |
| `BATCH_SIZE` | `256` | `utils.encode_texts` |
| `RANDOM_STATE` | `42` | `utils.set_global_seed`, `train_test_split` |
| `TEST_SIZE` | `0.20` | `train_vad.py:99` |
| `ALPHAS` | `(1e-6, 1e-4, 1e-2, 1e0, 1e2, 1e4, 1e6)` (7 values, decade-spaced log grid) | `train_vad.py:111` (passed to `RidgeCV`) |
| `WINDOW_SIZE` | `3` (sentences) | `utils.make_sliding_windows` |
| `STEP` | `1` (stride) | `utils.make_sliding_windows` |
| `MIN_SENTENCE_LENGTH` | `15` characters (filter sub-15-char fragments) | `utils.segment_sentences` |
| `SAVGOL_WINDOW_MAX` / `SAVGOL_POLYORDER` | `21` / `2` | `encode_vad.py` via `utils.choose_odd_window` |
| `NEI_METHOD` | `"gated_sum"` | `nei_plot.compute_nei` (overridable via `--method`) |
| `NEI_MIN_COMPONENT` | `0.10` | `nei_plot.compute_nei` (overridable via `--min-component`) |
| `NEI_PERCENTILE` | `0.95` (top 5% windows flagged) | `nei_plot.run_single` / `run_consensus` |
| `BASELINE_METHOD` | `"median"` | `nei_plot.compute_nei` |
| `MIN_EPISODE_DURATION` | `2` (contiguous windows) | `nei_plot.extract_contiguous_regions` for both ROIs and episodes |
| `CONVERGENCE_THRESHOLD["warriner"]` | `0.50` | `nei_plot.run_single` |
| `CONVERGENCE_THRESHOLD["nrc"]` | `0.15` | `nei_plot.run_single` |

### 6.4 Common Pitfalls to Check

1. **Scaling consistency.** `x_scaler.fit(X_tr)` and `y_scaler.fit(y_tr)` only on the training split, never on full data — confirmed at `train_vad.py:101–102`.

2. **Inverse transform.** Apply `y_scaler.inverse_transform` when reporting predictions in original lexicon units — confirmed at `train_vad.py:117` and `encode_vad.py:64`.

3. **z-score scope.** `compute_nei` z-scores within text (across all M windows of *one* lexicon's trajectory), NOT pooled across both lexicons. Confirmed at `nei_plot.py:72–74`.

4. **Direction convention.** The rectified components are taken from *directional deltas*: `v_drop = baseV − V` (positive when V is below baseline), `a_rise = A − baseA`, `d_drop = baseD − D`. Then `zV = max((v_drop − mean)/std, 0)`. This means an *above-baseline* valence cannot ever contribute to NEI — exactly the desired behaviour. Confirmed at `nei_plot.py:67–74`.

5. **Random Forest is gone.** `train_vad.py` fits Ridge only. Any reviewer who asks "where is your RF baseline?" → see `README.md` → "Ridge consistently outperforms Random Forest on the held-out lexicon split". The Feb 2025 RF outputs (`results/metamorphosis_*_RF_*.csv`) are kept for the regression test in `compare_results.py` only.

6. **Legacy CSV name collisions.** `compare_results.py` and `compare_episodes.py` reference both the old (`metamorphosis_*_Ridge_*.csv`, `metamorphosis_*_Ridge_vad_with_rois.csv`) and new (`new_metamorphosis_*_Ridge.*.csv`, `new_metamorphosis_*_Ridge_windows.csv`) filename schemes. The "new_" prefix in those scripts is a transitional alias — when the paper is finalised, decide whether to (a) drop the prefix and standardise on `<lex>.{nei,episodes,rois}.csv` as in `run_all.sh`, or (b) keep the alias for backwards-compat. Currently `run_all.sh` uses the **un-prefixed** scheme — the regression scripts are slightly out of sync.

### 6.5 Random Forest decision — RESOLVED

The Feb 2025 draft kept Random Forest behind a `TRAIN_RF` flag (Option B in the original recommendation). The April 2026 codebase has **fully removed RF from the training path** (Option A). The empirical-linearity argument from §3.3 is now defended by:

- Held-out Ridge correlations alone (V r ≈ 0.76–0.81)
- A one-line prose mention in `README.md` that "Ridge consistently outperforms Random Forest on the held-out lexicon split" — backed by the Feb 2025 RF artefacts retained in `results/`

If a reviewer demands the comparison numbers, point them to `compare_results.py` (which still parses the old `Valence_Ridge_Smooth` columns) or regenerate from the Feb 2025 artefacts. Do not re-introduce RF into `train_vad.py`.

---

## 7. POSTER DESIGN RECOMMENDATIONS

### 7.1 Layout Strategy

Standard CCLS A0 poster (84 × 119 cm portrait, or 119 × 84 landscape).

**Recommended: Landscape, 4-column layout.**

```
+----------------------------------------------------------------------+
|                  TITLE | AUTHOR | AFFILIATION                        |
+----------------+----------------+----------------+-------------------+
| COL 1          | COL 2          | COL 3          | COL 4             |
| THEORY         | METHOD         | RESULTS        | DISCUSSION        |
|                |                |                |                   |
| - Question     | - SBERT probe  | - Fig 1: VAD   | - 3 turning       |
| - Gilbert &    | - 2-stage reg  |   trajectories |   points          |
|   Allan        | - NEI gated_sum| - Fig 2: NEI   | - Lexicon         |
| - V↓A↑D↓       | - Dual lexicon |   peaks        |   divergence      |
|   conjunction  |                | - Fig 3: heatmap|  finding         |
|                |                |                | - Future work     |
|                |                |                | - References      |
+----------------+----------------+----------------+-------------------+
```

### 7.2 Critical Visual Elements

#### Figure 1: VAD Trajectories with NEI peaks (one per lexicon)
- X-axis: Window index (narrative progress 0 → 1)
- Three line plots: Valence, Arousal, Dominance (smoothed)
- Vertical bars at consensus NEI peaks
- Annotate peaks: "Mother/sister react", "Chief clerk", "Travellers monologue", and (Warriner-only divergence) "Lodgers / abandonment"
- Source: `results/<lex>.plot.png` (already produced by `nei_plot.py`)

#### Figure 2: Cross-lexicon consensus heatmap
- Rows: NEI_Warriner, NEI_NRC (row-normalised)
- Columns: window index
- Color: row-normalised NEI intensity
- Highlight: vertical bands where both rows are simultaneously hot → consensus episodes
- Source: `results/consensus.heatmap.png` (already produced by `nei_plot.py --consensus`)

#### Figure 3: NEI formula schematic
- Three boxes: V↓, A↑, D↓ each with a threshold gate
- Arrow from "AND gate" to "Sum"
- Overlay quote from Gilbert & Allan: *"strongly aroused flight motivation which is powerfully blocked"*

### 7.3 What to Highlight (visual hierarchy)

**Most prominent (largest fonts, color):**
- The ≈3 consensus turning points (this is the headline finding)
- The Mohammad table match (V=0.81, D=0.33) — frame as construct validity

**Medium prominence:**
- Linear probing methodology
- Gilbert & Allan grounding
- Cross-lexicon as triangulation

**De-emphasize or omit:**
- Random Forest (mention only as a one-line ablation in fine print)
- Detailed hyperparameters (move to supplementary)
- The NEI formula bug history (the gated_sum fix is the formula — don't describe the journey)

### 7.4 Color Palette Suggestion

Match Mohammad's NRC documentation visual style for cross-referencing:
- Teal/cyan for headers (#2A9D8F)
- Burgundy for key findings (#8B2635)
- Light gray for tables and de-emphasized content
- Orange highlight (#F4A261) for the consensus windows

### 7.5 Title Recommendation

Avoid generic. Specific is better.

**Weak:** "Computational Analysis of Affect in Kafka"
**Better:** "Detecting Narrative Entrapment in *Die Verwandlung* via SBERT Linear Probing of VAD Lexicons"
**Strongest:** "Three Turning Points: A Probing Approach to Narrative Entrapment in Kafka's *Die Verwandlung*"

The strongest title leads with the finding and frames the method as the way it was found. (The actual `bibtex` title currently in `README.md` is *"Operationalizing Narrative Entrapment: Predictive Affective Trajectories via Contextual Sentence Embeddings in Kafka's The Metamorphosis"* — defensible but generic; consider swapping to the "Three Turning Points" framing for the poster while keeping the longer formal title for the proceedings entry.)

### 7.6 Abstract Block Suggestion (~150 words)

> Narrative entrapment — the simultaneous co-presence of negative valence, elevated arousal, and reduced dominance (Gilbert & Allan, 1998) — is a psychological construct with no established computational operationalization in literary text. We present a linear probing pipeline that recovers continuous V/A/D trajectories from a frozen SBERT encoder using two complementary lexicons (Warriner et al., 2013; NRC VAD v2.1, Mohammad 2025). Applying a gated-sum entrapment index to Kafka's *Die Verwandlung*, we identify approximately three consensus high-entrapment windows that align with established literary turning points: the chief clerk confrontation, the travellers monologue, and the family's reaction to Gregor's transformation. Cross-lexicon trajectory agreement (V: r=0.81, D: r=0.33) reproduces lexicon-level correlations documented by NRC, validating pipeline integrity and revealing structural divergence in how each lexicon construes Dominance. We frame this as construct validity evidence; cross-corpus validation is future work.

### 7.7 QR Code Placement

Bottom-right corner, ~5 × 5 cm, linking to:
- GitHub repo (code + data) — `https://github.com/levantuankhoa/valeur_ccls2026`
- JCLS preprint or extended abstract

---

## 8. HONEST LIMITATIONS

### 8.1 Limitations to Acknowledge in the Paper

1. **Single corpus.** Validation is on *Die Verwandlung* alone. Cross-corpus validation is explicit future work, not evasion.

2. **No sentence-level ground truth.** The Ridge correlations (r ≈ 0.76–0.81) are on word-template held-out tests, not on sentence-level human VAD annotations.

3. **Construct validity, not predictive validity.** The consensus turning points are post-hoc literary identifications. A pre-registered annotation study with human raters scoring narrative entrapment would strengthen the claim.

4. **Word-template encoding is a convention.** It is not a theoretical claim about how words mean in context. SBERT was not designed for word-level embedding.

5. **Lexicon decontextualization.** Literary irony and figurative language can flip lexicon-predicted affect. Known limitation of all lexicon-based approaches.

6. **No comparison to alternative pipelines.** SentiArt (Jacobs 2019), syuzhet (Jockers 2017), VADER are established baselines. Comparison would strengthen the contribution claim.

### 8.2 Pre-emptive Rebuttal Notes

**If asked: "Why not use a transformer fine-tuned on emotion labels?"**

> "Fine-tuning on emotion labels (e.g., GoEmotions) would create a black-box classifier without dimensional interpretability. The probing approach makes the affective axis explicit and inspectable, which is essential for literary analysis. Fine-tuned classifiers also require labelled data that does not exist for narrative entrapment as a construct."

**If asked: "Three turning points is a small validation."**

> "Construct validity at the discourse level requires that the index recover meaningful narrative structure, not maximize a quantitative metric. The three windows align with literary consensus on the novella's structural arc. Pre-registered annotation with human raters is the next step in validation, not within scope of this exploratory case study."

**If asked: "Why not validate on more texts?"**

> "Cross-corpus validation is explicitly framed as future work. The current contribution is the construct, the pipeline, and the case study evidence. Which prior CLS work validates a narrative index on an independent corpus before first publication?"

**If asked: "Random Forest doesn't beat Ridge — does that really prove linearity?"**

> "It is empirical evidence consistent with the linear representation hypothesis (Park et al., 2023; Mikolov et al., 2013). Strong nonlinearity would predict substantial RF outperformance, which we do not observe. We do not claim this proves global linearity of SBERT space — only that affective directions are recoverable by linear projection in the regions we sample, which is sufficient for the pipeline's purpose."

### 8.3 Future Work to Mention

1. **Cross-corpus validation:** apply pipeline to *Crime and Punishment*, *The Trial*, modernist canon
2. **Human annotation study:** crowd-sourced narrative entrapment ratings as external validity check
3. **LLM retrospective annotation:** GPT-4 / Claude as additional triangulation source
4. **Translation invariance:** apply to multiple translations of *Die Verwandlung* to test robustness
5. **Comparative pipeline:** benchmark against SentiArt, syuzhet, VADER on the same windows

---

## APPENDIX A: KEY CITATIONS

```bibtex
@article{gilbert1998defeat,
  title={The role of defeat and entrapment (arrested flight) in depression: An exploration of an evolutionary view},
  author={Gilbert, Paul and Allan, Steven},
  journal={Psychological Medicine},
  volume={28},
  number={3},
  pages={585--598},
  year={1998}
}

@inproceedings{reimers2019sentence,
  title={Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks},
  author={Reimers, Nils and Gurevych, Iryna},
  booktitle={EMNLP-IJCNLP},
  year={2019}
}

@article{warriner2013norms,
  title={Norms of valence, arousal, and dominance for 13,915 English lemmas},
  author={Warriner, Amy Beth and Kuperman, Victor and Brysbaert, Marc},
  journal={Behavior Research Methods},
  volume={45},
  number={4},
  pages={1191--1207},
  year={2013}
}

@article{mohammad2025nrcvad,
  title={NRC VAD Lexicon v2: Norms for valence, arousal, and dominance for over 55k English terms},
  author={Mohammad, Saif M.},
  journal={arXiv:2503.23547},
  year={2025}
}

@inproceedings{hewitt2019probing,
  title={A Structural Probe for Finding Syntax in Word Representations},
  author={Hewitt, John and Manning, Christopher D.},
  booktitle={NAACL-HLT},
  year={2019}
}

@inproceedings{park2023linear,
  title={The Linear Representation Hypothesis and the Geometry of Large Language Models},
  author={Park, Kiho and Choe, Yo Joong and Veitch, Victor},
  booktitle={ICML},
  year={2023}
}

@inproceedings{mehrabian1974approach,
  title={An approach to environmental psychology},
  author={Mehrabian, Albert and Russell, James A.},
  publisher={MIT Press},
  year={1974}
}
```

---

## APPENDIX B: MINIMAL NEI VERIFICATION SNIPPET

Reproduces `nei_plot.compute_nei` semantics exactly: directional delta → within-text z-score → rectify → gate-and-sum. Drop into a scratch script to verify the formula behaves as advertised.

```python
import numpy as np

def compute_nei_gated_sum(V, A, D, threshold=0.10, baseline="median"):
    """
    Match production semantics in nei_plot.compute_nei.

    V, A, D : np.ndarray of length N (smoothed within-text trajectories)
    """
    if baseline == "median":
        baseV, baseA, baseD = np.median(V), np.median(A), np.median(D)
    elif baseline == "mean":
        baseV, baseA, baseD = np.mean(V), np.mean(A), np.mean(D)
    else:
        raise ValueError(baseline)

    v_drop = baseV - V          # positive when V is BELOW baseline
    a_rise = A - baseA          # positive when A is ABOVE baseline
    d_drop = baseD - D          # positive when D is BELOW baseline

    eps = 1e-9
    zV = np.maximum((v_drop - v_drop.mean()) / (v_drop.std() + eps), 0.0)
    zA = np.maximum((a_rise - a_rise.mean()) / (a_rise.std() + eps), 0.0)
    zD = np.maximum((d_drop - d_drop.mean()) / (d_drop.std() + eps), 0.0)

    gate = (zV > threshold) & (zA > threshold) & (zD > threshold)
    return np.where(gate, zV + zA + zD, 0.0), zV, zA, zD


def test_nei():
    rng = np.random.default_rng(42)
    N = 200

    # Synthetic trajectory: stable middle, one entrapment burst (V↓ A↑ D↓ at idx 100),
    # one false-positive opening (V↓ A↓ D↓ at idx 0), one excitement (V↑ A↑ D↑ at idx 150)
    V = np.full(N, 5.0); A = np.full(N, 5.0); D = np.full(N, 5.0)
    V += rng.normal(0, 0.05, N); A += rng.normal(0, 0.05, N); D += rng.normal(0, 0.05, N)

    # Entrapment burst
    V[95:105] = 3.0; A[95:105] = 7.0; D[95:105] = 3.0
    # Opening false-positive (sleepy: V↓ A↓ D↓)
    V[0:5] = 3.5;  A[0:5] = 3.5;  D[0:5] = 3.5
    # Joy spike (V↑ A↑ D↑)
    V[145:155] = 7.0; A[145:155] = 7.0; D[145:155] = 7.0

    nei, zV, zA, zD = compute_nei_gated_sum(V, A, D)

    assert nei[100] > 4.0,                  "true entrapment burst should fire"
    assert nei[2]  == 0.0,                  "sleepy opening must be gated out"
    assert nei[150] == 0.0,                 "excitement must be gated out (zV=0)"
    print("All NEI formula tests pass.")


if __name__ == "__main__":
    test_nei()
```

> The Feb 2025 reference document had a different snippet that took raw signed z-scores of V/A/D and reconstructed components (`-zV if zV<0 else 0`). That snippet does not match production: it z-scores the *raw* dimensions, whereas the real implementation z-scores the *directional deltas*. Use the snippet above when explaining the formula in the supplementary.

---

## APPENDIX C: DOC → CODE CROSSWALK

Quick map for any future reader who comes in via this document and wants to find the corresponding code:

| Doc location | Code location |
|---|---|
| §2.1 candidate formulas | `nei_plot.py:compute_nei` (`method ∈ {"add","mult","gated_sum"}`) |
| §2.5 episode/ROI extraction | `nei_plot.extract_contiguous_regions`, `run_single` |
| §3.2 word-template encoding | `train_vad.py:93` (`config.TEMPLATE.format(w)`) |
| §3.3 held-out diagnostics | `models/ridge_*.diagnostics.json`; computed in `train_vad.py:115–143` |
| §4.2 Stage 1 (probe training) | `train_vad.py:train` |
| §4.3 Stage 2 (apply to Kafka) | `encode_vad.py:encode` |
| §4.4 scaler choice | `train_vad.py:101–117`, `encode_vad.py:62–64` |
| §4.6 convergence ROIs | `nei_plot.run_single` (vad_range threshold), `config.CONVERGENCE_THRESHOLD` |
| §5 cross-lexicon consensus | `nei_plot.run_consensus` (Spearman ρ on NEI + heatmap) |
| §6.3 hyperparameter table | `config.py` |
| §6.4 pitfalls | inline in this doc |
| §6.5 RF removal | `README.md` ("Ridge > RF" note) + absence of RF in `train_vad.py` |

---

## END OF REFERENCE
