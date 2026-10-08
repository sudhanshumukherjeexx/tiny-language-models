"""Evaluation: held-out loss/perplexity, a fixed generation suite, lexical metrics.

Lexical metrics are *diagnostics*, not a quality score: a model can have
high Distinct-2 and still write incoherent stories. They are reported next to
human or LLM-judge ratings (see ``tinylm.judge``) and the failure analysis.
"""

from __future__ import annotations

import logging
import math
import re
import statistics
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm

from tinylm.data import PackedDataset
from tinylm.generation import GenerationResult, generate, get_preset
from tinylm.utils import autocast_context, read_json

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- LM loss


@dataclass
class LMEvalResult:
    split: str
    loss: float  # mean next-token cross-entropy (nats/token)
    perplexity: float
    sequences: int
    predicted_tokens: int  # (block_size - 1) predictions per sequence

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@torch.no_grad()
def evaluate_loss(
    model: torch.nn.Module,
    dataset: PackedDataset,
    batch_size: int,
    device: torch.device,
    precision: str = "fp32",
    max_batches: int | None = None,
    split: str = "validation",
    progress: bool = False,
) -> LMEvalResult:
    """Token-weighted mean cross-entropy over (a prefix of) a packed split.

    Every packed sequence has the same length, so weighting each batch's mean
    loss by its number of predictions gives the exact corpus mean.
    """
    was_training = model.training
    model.eval()
    n = len(dataset)
    starts = list(range(0, n, batch_size))
    if max_batches is not None:
        starts = starts[:max_batches]
    total_loss, total_pred, total_seq = 0.0, 0, 0
    for start in tqdm(starts, desc=f"eval {split}", disable=not progress):
        ids = dataset.batch(np.arange(start, min(start + batch_size, n))).to(device)
        with autocast_context(device, precision):
            loss = model(input_ids=ids, labels=ids).loss
        n_pred = ids.shape[0] * (ids.shape[1] - 1)
        total_loss += loss.float().item() * n_pred
        total_pred += n_pred
        total_seq += ids.shape[0]
    model.train(was_training)
    if total_pred == 0:
        raise ValueError(f"Split {split!r} is empty; nothing to evaluate.")
    mean = total_loss / total_pred
    return LMEvalResult(split, mean, math.exp(mean), total_seq, total_pred)


# --------------------------------------------------------------------------- lexical metrics

_WORD = re.compile(r"[a-z0-9']+")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def distinct_n(texts: Iterable[str], n: int) -> float:
    """Corpus-level Distinct-n (Li et al., 2016): unique n-grams / total n-grams."""
    grams = [g for t in texts for g in ngrams(words(t), n)]
    return len(set(grams)) / len(grams) if grams else 0.0


def repeated_ngram_fraction(text: str, n: int = 4) -> float:
    """Share of a text's n-grams that already occurred earlier in the same text."""
    grams = ngrams(words(text), n)
    if not grams:
        return 0.0
    counts = Counter(grams)
    return sum(c - 1 for c in counts.values()) / len(grams)


def sentence_repetition_ratio(text: str) -> float:
    """Share of sentences that exactly repeat an earlier sentence (loop detector)."""
    sentences = [" ".join(words(s)) for s in _SENTENCE.split(text.strip())]
    sentences = [s for s in sentences if s]
    if len(sentences) < 2:
        return 0.0
    seen: set[str] = set()
    repeats = 0
    for s in sentences:
        repeats += s in seen
        seen.add(s)
    return repeats / len(sentences)


def lexical_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate lexical metrics over generation records (``completion`` texts)."""
    texts = [r["completion"] for r in records]
    if not texts:
        return {}
    lengths = [len(words(t)) for t in texts]
    return {
        "generations": len(texts),
        "mean_words": statistics.fmean(lengths),
        "mean_new_tokens": statistics.fmean(r["new_tokens"] for r in records),
        "eos_rate": statistics.fmean(float(r["finished_with_eos"]) for r in records),
        "distinct_1": distinct_n(texts, 1),
        "distinct_2": distinct_n(texts, 2),
        "repeated_4gram_fraction": statistics.fmean(repeated_ngram_fraction(t) for t in texts),
        "sentence_repetition_ratio": statistics.fmean(sentence_repetition_ratio(t) for t in texts),
    }


def summarize_by(records: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        groups.setdefault(str(r[key]), []).append(r)
    return {k: lexical_summary(v) for k, v in sorted(groups.items())}


# --------------------------------------------------------------------------- prompt suite


def load_prompt_suite(path: str | Path) -> dict[str, Any]:
    suite = read_json(path)
    ids = [p["id"] for p in suite["prompts"]]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate prompt ids in {path}")
    return suite


def run_generation_suite(
    model,
    tokenizer,
    suite: dict[str, Any],
    presets: list[str],
    seeds: list[int],
    max_new_tokens: int,
    metadata: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Generate every (prompt, preset, seed) combination with full provenance.

    Greedy decoding is deterministic, so it is run once per prompt regardless
    of how many seeds are requested. Prompts that do not fit in the model's
    context are not run; they are logged and listed in ``skipped``.

    Returns:
        (records, skipped) where ``skipped`` holds ``{"prompt_id", "reason"}`` items.
    """
    records: list[dict[str, Any]] = []
    max_ctx = model.config.max_position_embeddings
    usable, skipped = [], []
    for p in suite["prompts"]:
        n = len(tokenizer(p["prompt"], add_special_tokens=False)["input_ids"]) + 1  # + BOS
        if n >= max_ctx:
            reason = f"prompt is {n} tokens; context is {max_ctx}"
            logger.warning("Skipping prompt %s: %s", p["id"], reason)
            skipped.append({"prompt_id": p["id"], "reason": reason})
        else:
            usable.append(p)
    jobs = [
        (p, preset, seed)
        for p in usable
        for preset in presets
        for seed in (seeds[:1] if not get_preset(preset).do_sample else seeds)
    ]
    for prompt, preset, seed in tqdm(jobs, desc="generating"):
        result: GenerationResult = generate(
            model, tokenizer, prompt["prompt"], preset, max_new_tokens, seed
        )
        records.append(
            {
                "prompt_id": prompt["id"],
                "category": prompt["category"],
                "suite_version": suite.get("version"),
                **result.to_dict(include_ids=True),  # ids feed the memorization scan
                "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                **(metadata or {}),
            }
        )
    return records, skipped
