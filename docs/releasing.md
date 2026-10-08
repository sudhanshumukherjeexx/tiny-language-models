# Releasing

## Code release

1. `uv lock --check && uvx ruff check . && uvx ruff format --check . && uv run pytest` all pass.
2. Bump `version` in `pyproject.toml` and `src/tinylm/__init__.py`.
3. Tag and push: `git tag v0.2.0 && git push origin v0.2.0`, then create a GitHub release that
   lists what changed and which results (if any) it includes.

## Model release

Follow the checklist at the bottom of `model_card.md`. Checkpoints are never committed to this
repository; they go to the Hugging Face Hub (or Git LFS if that is ever chosen deliberately).

## Results release

A result enters `results/` only with its provenance:

* run directory files: `config.yaml`, `metrics.json`, `environment.json`, `training_history.jsonl`;
* evaluation files: `test_metrics.json`, `generation_samples.jsonl`, `lexical_metrics.json`;
* a row in `results/experiments.csv` (`scripts/evaluate.py --record-index results/experiments.csv`).

Numbers are copied into the README only from those files, labelled with the split (validation vs.
test) and the hardware they were measured on.
