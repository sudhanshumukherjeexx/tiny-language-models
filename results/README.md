# Results

Everything in this folder was produced by code in this repository or transcribed from recorded
outputs, with its provenance. Values that have not been measured are left empty or marked
*TBD — run required*. Nothing is estimated or extrapolated.

| Path | What it is | Produced by | Status |
|---|---|---|---|
| `runs/tiny-27m/` | Main run: resolved config, metrics, environment, full training history and curve, selected-checkpoint metadata, test metrics, generation suite (with token ids), lexical metrics, rating sheet, memorization report | `scripts/train.py`, `evaluate.py`, `check_memorization.py` | measured (RTX 3050 Ti Laptop GPU) |
| `runs/legacy-colab-a100/` | The original notebook run: metrics, environment, training curve, all 14 samples | transcribed from the notebook's outputs | archived, with caveats |
| `data_integrity.json` | Exact/near-duplicate and cross-split leakage report for the prepared (decontaminated) splits | `scripts/check_data_integrity.py` | measured |
| `data_integrity_official_splits.json` | The same report on the raw official splits, which found the 29.9 % test-in-train overlap | `scripts/check_data_integrity.py` (before decontamination) | measured |
| `benchmarks/<hardware>.json` | Training and generation benchmarks, keyed by the hardware they ran on | `scripts/benchmark.py` | measured (RTX 3050 Ti Laptop GPU, laptop CPU) |
| `failure_analysis.md` | Failures of TinyLM-27M on the prompt suite (with run counts) and of the legacy samples | manual analysis of recorded outputs | done |
| `experiments.csv` | One row per run (blank = not measured) | `scripts/evaluate.py --record-index` | legacy + tiny-27m |
| `experiment_schema.json` | JSON Schema for `runs/<run>/metrics.json` | — | — |
| `ablations/` | Per-seed records and `summary.{json,md}` (5 variants × 3 seeds) | `scripts/run_ablation.py` | measured (RTX 3050 Ti Laptop GPU) |

## Protocol reminders

* **validation** = 0.5 % of the official train split (content hash). Used for checkpoint
  selection and ablation decisions.
* **test** = the official TinyStories validation split minus the 6,601 stories that have an
  exact copy in the training source (29.9 % of the split); 15,388 stories remain. Each final model is evaluated on it once,
  with `scripts/evaluate.py`. The legacy run used this split for checkpoint selection, so it has
  no clean test number.
* Perplexities are comparable only between runs that share a tokenizer (`tokenizer_sha256`).

## Memorization thresholds

`scripts/check_memorization.py` flags a text for manual review if any of these hold:

| Signal | Threshold | Why this level |
|---|---|---|
| TF-IDF cosine to the nearest training story | ≥ 0.8 | far above the held-out baseline (the smoke run's held-out maximum was 0.31) |
| Verbatim token n-gram found in training data | any 32-token span | ~25 words; held-out TinyStories text rarely shares spans this long |
| Exact shared word span with the nearest story | ≥ 30 words | about two full sentences |

Thresholds flag outputs for review; they are not a definition of memorization. Read the flagged
records and compare the generation distribution with the held-out baseline in
`memorization.png`.

## Near-duplicate parameters

MinHash with 128 permutations over 5-token shingles. LSH uses 16 bands × 8 rows, which puts the
detection threshold at about (1/16)^(1/8) ≈ 0.71. Candidate pairs are then verified at an
estimated Jaccard similarity ≥ 0.8. Cross-split containment is the fraction of a document's
13-token n-grams found anywhere in the training stream (the contamination test of Brown et al.,
2020, with BPE tokens instead of words).
