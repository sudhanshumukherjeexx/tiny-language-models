# Legacy run: original Colab notebook (A100, 2026-08)

This folder archives the only full-scale training run that exists so far: the
original notebook execution. Its checkpoint lives on the author's Google Drive
and is **not** in this repository. Everything here was transcribed from the
notebook's printed outputs. Nothing was re-measured, and anything the
notebook did not record is `null`.

| File | Contents |
|---|---|
| `metrics.json` | Verified numbers, each with its caveat |
| `environment.json` | Hardware and library versions printed by the notebook |
| `training_curve.png` | The loss/perplexity figure embedded in the notebook (real `log_history`) |
| `generation_samples.jsonl` | All 14 generations printed by the notebook, with their exact decoding settings |

## What the evidence supports

| Quantity | Value | Status |
|---|---:|---|
| Parameters | 26,747,392 | Verified (and reproduced by `tests/test_model.py`) |
| Optimizer steps × tokens/step | 12,000 × 36,864 = 442,368,000 tokens | Verified from config + final step |
| Development-validation loss (official TinyStories validation split) | 1.4148 (PPL 4.12) | Verified, **but the same split selected the checkpoint** |
| Last logged training loss | 1.4274 | Verified |
| Untouched test loss | — | Not available (see below) |
| Training wall-clock time, throughput | — | Not recorded |

## Caveats that the original notebook did not make

1. **The run has two phases.** `training_curve.png` shows the training loss
   jumping from ~1.5 to ~5.4 at step ~4,000, and the validation loss rising
   from ~1.55 to ~1.95 before it recovers. An uninterrupted run with a fixed
   cosine schedule would not do this. The pattern fits a 4,000-step run
   (`USE_SUBSET=True` sets `MAX_STEPS=4000`) that was then resumed as the
   12,000-step full-corpus configuration. Resuming re-creates the LR schedule
   for the new horizon, so the learning rate jumps back up. If the tokenizer
   was retrained in between, token ids would also shift. The size of the
   spike suggests more than an LR change, but the notebook doesn't record
   enough to confirm either cause. The current trainer refuses to resume when
   the configuration changed (`tests/test_checkpointing.py`).
2. **"Final training loss: 0.0001 / Training finished in 0.4 minutes"** came
   from the last call to `trainer.train(resume_from_checkpoint=True)`. That
   call resumed from an end-of-training checkpoint and ran at most one step.
   Neither number describes the experiment. The current trainer accumulates
   time, tokens and best validation loss across resumes and reports the last
   invocation separately.
3. **Validation was also model selection, and it is now the test split.** The
   notebook used TinyStories' official validation split both to choose the
   checkpoint (`load_best_model_at_end`) and to report perplexity, so 1.4148 is
   a development number. The current protocol reserves that official split as
   the untouched TEST set, which means the legacy checkpoint can never get a
   clean test number. Evaluating it there would be mildly optimistic, because
   the split was used to pick among the checkpoints.
4. **About 30 % of the validation stories were training data.** `scripts/check_data_integrity.py`
   found that 6,582 of the 21,989 official validation stories (29.9 %) have an exact copy in
   the official train split, after case and whitespace normalization
   ([`data_integrity_official_splits.json`](../../data_integrity_official_splits.json)). The
   legacy run trained on the full official train split, so roughly a third of the stories
   behind 1.4148 were seen during training. This was measured with the current normalization;
   the legacy pipeline's ASCII stripping could change the count slightly.
5. **`missing keys: ['lm_head.weight']` is expected.** The model ties
   `lm_head.weight` to `embed_tokens.weight`, and safetensors stores a shared
   tensor once. The tie is restored on load.
   `tests/test_serialization.py` checks that the tensor is absent on disk and
   that weights are identical and tied after reload.
6. **Sampling settings.** The notebook's `generate()` helper defaulted to
   `top_k=50`, so "top-p 0.9" samples were really top-k 50 + top-p 0.9. The
   "Greedy" row also passed `temperature`/`top_p`, which transformers ignored
   with a warning. `generation_samples.jsonl` records the settings that were
   actually in effect. No per-sample seed was recorded, so these samples
   cannot be regenerated.
7. **Peak memory.** The notebook printed 4.49 GB peak allocated, but that
   measurement covers only the final resumed session. Training-memory numbers
   in this repository come from `scripts/benchmark.py` instead.

## Re-evaluating the legacy checkpoint (if you have it)

The legacy tokenizer is saved inside its `final_model` directory. Pack the
data with that tokenizer and the legacy normalization, then evaluate:

```bash
uv run python scripts/prepare_data.py --config configs/legacy/colab-notebook-2026-08.yaml \
    --set tokenizer.path=/path/to/final_model
uv run python scripts/evaluate.py --config configs/legacy/colab-notebook-2026-08.yaml \
    --set tokenizer.path=/path/to/final_model --checkpoint /path/to/final_model
```

Caveat 3 still applies to the resulting "test" number.
