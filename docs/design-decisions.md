# Design decisions

Each entry gives the choice, what it trades away, and the evidence. "Measured"
means a number produced by code in this repository. "Arithmetic" means it
follows from the configuration. "Literature" means it relies on the cited work
and was not tested here.

## Data

**TinyStories.** A ~27M-parameter model cannot cover the distribution of web
text. TinyStories (Eldan & Li, 2023) is ~2.1M synthetic stories in a
deliberately small vocabulary. On that narrow domain, models of this size
produce fluent text, so evaluation can probe coherence and memorization
instead of stopping at "not grammatical yet". *Cost:* the results say nothing
about general language modelling, world knowledge or reasoning, and the
synthetic, formulaic text makes memorization harder to tell apart from
fluency. That is why memorization is measured against a held-out baseline (below).

**Splits: official validation = TEST; validation = 0.5 % of train by content hash.**
TinyStories ships no test split. The legacy notebook used the official
validation split for both checkpoint selection and reporting. The current
protocol needs three roles, so the official split is never touched before final
evaluation. Validation is carved from train by hashing each normalized story,
which is deterministic, order-independent, and puts exact duplicates in the
same split. *Cost:* the legacy checkpoint can never be test-evaluated cleanly,
because it was selected on what is now the test split.

**Test decontamination.** The leakage analysis found that 29.9 % of the official
validation stories (6,582 / 21,989) have a normalized-exact copy in the official
train split. Test therefore excludes every story whose content hash appears in the
training source (`data.decontaminate_test: true`, count recorded in `text_meta.json`).
The cost is a smaller test set: 6,601 stories removed (6,582 with a copy in train, 19 in the
validation slice), leaving 15,388 and a break in comparability with
numbers reported on the raw official split, including this repository's own legacy
number. That is the right trade: a test set that is 30 % training data does not measure
generalization. Near-duplicates that are not exact copies are measured, not removed.

**Unicode is preserved (`strip_non_ascii: false`).** The legacy pipeline
deleted every non-ASCII character after mapping curly quotes and dashes. A
byte-level BPE can encode any byte, so that filter isn't needed. It also
silently corrupts words ("Café" → "Caf", which `tests/test_data.py` checks).
The current pipeline keeps NFKC normalization and the ASCII mapping for
typographic punctuation, which merges near-identical quote characters, and
otherwise leaves text intact. The legacy setting is still available as a
config flag for reproducing that run.

**Packing with a rolling remainder.** Stories are joined with one EOS token
each and cut into 512-token blocks. The legacy `datasets.map` implementation
discarded the ragged tail of every 1,000-story batch, so roughly one partial
block per batch was lost. `SequencePacker` carries the remainder forward, so
only the final remainder of each split is dropped (< 512 tokens). Token
accounting invariants are checked on every run (`PackingStats.validate`).
Attention is not masked at document boundaries. This is standard for
pretraining: the model learns that EOS starts a new, unrelated story.

## Tokenizer

**Custom 8,192-token byte-level BPE.** GPT-2's 50,257-token vocabulary would
need a 50,257 × 512 = 25.7M-parameter embedding table, as large as the rest
of the model (arithmetic), and most rows would never be trained on this
corpus. At 8,192 tokens the tied embedding is 4.19M parameters, 15.7 % of
the model (measured, `parameter_report`). *Cost:* a smaller vocabulary means
more tokens per story, which uses more of the context window and compute per
story. Measured compression on TinyStories: 4.09 characters per token, 220 tokens per
story on average.

**One special token for document boundaries.** BOS and EOS are the same
token (`<|endoftext|>`, id 0). At generation time it is prepended to the
prompt, matching what precedes every story in the packed training stream.

## Architecture

**~27M parameters (8 layers, d=512, 8 query heads, 2 KV heads, FFN 1408).**
This is large enough to produce fluent TinyStories text, and small enough to
pretrain on ~440M tokens with a single GPU in hours. At 442M training tokens,
the legacy run used ~16.5 tokens per parameter (arithmetic). That is close
to the ~20 tokens/parameter often quoted from Hoffmann et al. (2022) as
compute-optimal. That result was fitted on large general-domain models, and
nothing here shows it transfers to a 27M model on synthetic children's
stories. The number is context, not a claim of optimality.

**Hugging Face `LlamaForCausalLM` for the production model.** The
architecture classes are used as-is; tokenizer and weights are trained from
random initialization. This buys tested kernels, `generate()` with KV cache,
`save_pretrained`/Hub compatibility and SDPA support, at the price of a less
visible implementation. The visibility gap is closed by the educational
re-implementation, which `tests/test_model.py` verifies to be numerically
identical to it.

**RoPE instead of learned positions.** RoPE adds no parameters, makes
attention scores depend on relative offsets (`test_rope_scores_depend_only_on_relative_position`),
and frees 262K parameters that a learned 512 × 512 table would use. Whether it
helps *at this scale and context length* is an ablation question, not
something assumed.

**RMSNorm instead of LayerNorm.** RMSNorm needs one reduction instead of two
and has no bias (Zhang & Sennrich, 2019). At 27M parameters the speed
difference is small. It is kept for fidelity to the modern recipe, and the
ablation measures any quality difference.

**SwiGLU instead of GELU.** Shazeer (2020) reports better quality at matched
compute. A gated FFN has three matrices, so at equal parameter count its
hidden width is ~2/3 of a GELU FFN's (1152 vs 1728 in the matched ablation
configs).

**Grouped-query attention, 2 KV heads.** GQA cuts the KV cache by
`heads / kv_heads` = 4× (arithmetic). *At this scale the saving is small in
absolute terms.* At bf16 and full 512-token context, the cache is 2 MiB per
sequence with GQA versus 8 MiB with MHA (`kv_cache_table`). That matters
when serving many concurrent sequences, not for single-user demos. GQA also
removes 3.1M attention parameters, which the production config spends on a
wider FFN. GQA is used because it is part of the recipe under study, and the
ablation measures its quality cost at matched parameters.

**Tied input/output embeddings.** This saves 4.19M parameters (15.7 %). The
serialized checkpoint stores the shared tensor once, which is why loaders
report a "missing" `lm_head.weight`. The tie is verified after every reload.

**512-token context.** With the 8K tokenizer, TinyStories stories average 220 tokens, with a
median of 187 and a 95th percentile of 434 (measured on 2,000 validation stories,
`tokenizer_report.json`). So 512 tokens holds about 95 % of stories end to end. A longer context
would mostly add attention across unrelated, concatenated stories.

**`attn_implementation="sdpa"`.** This is PyTorch Scaled Dot Product
Attention. It may dispatch to a fused kernel such as FlashAttention when the
hardware, dtype and masking pattern are eligible, and otherwise uses the
memory-efficient or math kernel. The repository does not claim which kernel
ran.

## Training

**Hand-written training loop instead of `transformers.Trainer`.** The legacy
run used Trainer. When it resumed after a configuration change, the result
was a two-phase run, and the resumed call reported a 0.4-minute runtime and a
0.0001 loss for a 12,000-step experiment. The current loop (~300 lines)
accumulates experiment-level metrics across resumes and refuses to resume
after a configuration change. Its data order is a pure function of
(seed, position), which makes "interrupted training equals uninterrupted
training" a passing test (bitwise on CPU). *Cost:* there is more code to
maintain, and Trainer features such as distributed training are not
available. Multi-GPU is unnecessary at this scale.

**Optimizer settings (unchanged from the legacy run):** AdamW with β=(0.9, 0.95),
weight decay 0.1 on matrices only, peak LR 6e-4, 200 warmup steps, cosine to
0, gradient clipping at 1.0, 72 sequences (36,864 tokens) per step. These are
conventional small-GPT settings. They were not tuned in this repository, and
tuning them belongs on the validation split.

**Model selection on validation; one test evaluation.** `train.py` exports
the best-validation model to `best/`. `evaluate.py` evaluates it once on the
test split. Ablations never touch test.

## Evaluation

**Lexical metrics are diagnostics.** Distinct-n, repeated-n-gram fraction and
sentence-repetition ratio catch degenerate outputs cheaply. They don't
measure coherence, which is why the repository also exports a rubric sheet
for human rating and provides an optional LLM judge with every prompt and
raw response saved.

**Memorization is measured against a held-out baseline.** On formulaic text,
"this generation shares an 8-gram with training data" is expected even from a
model that copies nothing. In the smoke run, 98 % of held-out test stories
did. The memorization report therefore puts generations and held-out stories
through identical checks (corpus-wide verbatim n-grams, TF-IDF nearest
neighbour, longest shared span), and a generation stands out only relative to
that baseline.

**Separate educational implementation.** It makes the mechanisms readable
and individually testable: RoPE's relative-position property, GQA head
mapping, causal masking, and KV-cache equivalence. Because it is
parameter-name compatible with HF Llama, it doubles as a correctness oracle
(identical logits) and as the controlled model family for ablations, where
every variant must share one implementation. Comparing GPT-2, GPT-NeoX and
Llama classes would confound architecture with implementation details.
