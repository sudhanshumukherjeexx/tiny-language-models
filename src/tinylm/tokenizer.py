"""Byte-level BPE tokenizer: training, loading, fingerprinting and diagnostics."""

from __future__ import annotations

import hashlib
import statistics
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from tinylm.config import TokenizerConfig

# Probes deliberately include text TinyStories rarely or never contains: a
# byte-level tokenizer must still round-trip it exactly (there is no <unk>).
UNICODE_PROBES = [
    "Once upon a time, there was a little girl named Lily.",
    'She said, "Hello!" and smiled.',
    "Antidisestablishmentarianism is a very long word.",
    "Curly “quotes”, an em—dash, and an ellipsis…",
    "Café, naïve, jalapeño, Zürich.",
    "Emoji: 🐱🌈 and CJK: 小猫",
    "Tabs\tand\nnewlines\n\nare kept.",
]


def train_tokenizer(
    texts: Iterable[list[str]],
    cfg: TokenizerConfig,
    model_max_length: int,
    length: int | None = None,
) -> PreTrainedTokenizerFast:
    """Train a byte-level BPE tokenizer and wrap it for the transformers ecosystem.

    Args:
        texts: iterable of *batches* of training strings (training split only).
        cfg: tokenizer settings (vocabulary size, special tokens).
        model_max_length: context length recorded in the tokenizer config.
        length: optional number of examples, used only for the progress bar.
    """
    backend = Tokenizer(models.BPE(unk_token=None))
    # add_prefix_space=True makes "Once" at the start of a story share a token
    # with " Once" mid-sentence; decoding therefore yields one leading space.
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    backend.decoder = decoders.ByteLevel()
    backend.post_processor = processors.ByteLevel(trim_offsets=True)
    trainer = trainers.BpeTrainer(
        vocab_size=cfg.vocab_size,
        min_frequency=cfg.min_frequency,
        special_tokens=[cfg.eos_token, cfg.pad_token],  # ids 0 and 1
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    backend.train_from_iterator(texts, trainer=trainer, length=length)
    return wrap_tokenizer(backend, cfg, model_max_length)


def wrap_tokenizer(
    backend: Tokenizer, cfg: TokenizerConfig, model_max_length: int
) -> PreTrainedTokenizerFast:
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token=cfg.eos_token,  # documents are delimited by a single EOS token
        eos_token=cfg.eos_token,
        pad_token=cfg.pad_token,
        model_max_length=model_max_length,
        # The WordPiece-era cleanup strips spaces before punctuation, which
        # corrupts byte-level BPE output. Persisted in tokenizer_config.json.
        clean_up_tokenization_spaces=False,
    )


def load_tokenizer(path: str | Path) -> PreTrainedTokenizerFast:
    path = Path(path)
    if not (path / "tokenizer.json").exists():
        raise FileNotFoundError(
            f"No tokenizer found at {path}. Run scripts/train_tokenizer.py first "
            "(or point tokenizer.path at an existing tokenizer directory)."
        )
    return AutoTokenizer.from_pretrained(str(path), clean_up_tokenization_spaces=False)


def tokenizer_fingerprint(path: str | Path) -> str:
    """SHA-256 of tokenizer.json: identifies the exact vocabulary and merges."""
    return hashlib.sha256((Path(path) / "tokenizer.json").read_bytes()).hexdigest()


def encode_batch(tokenizer: PreTrainedTokenizerFast, texts: list[str]) -> list[list[int]]:
    """Encode without special tokens using the Rust backend (parallel, ordered)."""
    return [
        e.ids for e in tokenizer.backend_tokenizer.encode_batch(texts, add_special_tokens=False)
    ]


def iter_batches(texts: Iterable[str], batch_size: int = 1000) -> Iterator[list[str]]:
    batch: list[str] = []
    for t in texts:
        batch.append(t)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def tokenizer_diagnostics(
    tokenizer: PreTrainedTokenizerFast, sample_texts: list[str]
) -> dict[str, Any]:
    """Compression and correctness statistics on a sample of held-out-style text."""
    ids = encode_batch(tokenizer, sample_texts)
    n_tokens = [len(x) for x in ids]
    n_chars = sum(len(t) for t in sample_texts)
    n_bytes = sum(len(t.encode("utf-8")) for t in sample_texts)
    total_tokens = sum(n_tokens)
    roundtrip = [
        tokenizer.decode(x).lstrip(" ") == t.lstrip(" ")
        for x, t in zip(ids, sample_texts, strict=True)
    ]
    probes = []
    for text in UNICODE_PROBES:
        pid = encode_batch(tokenizer, [text])[0]
        probes.append(
            {
                "text": text,
                "tokens": len(pid),
                "pieces": tokenizer.convert_ids_to_tokens(pid),
                "roundtrip_exact": tokenizer.decode(pid).lstrip(" ") == text,
            }
        )
    vocab = tokenizer.get_vocab()
    return {
        "vocab_size": len(tokenizer),
        "special_tokens": {
            "eos": [tokenizer.eos_token, tokenizer.eos_token_id],
            "bos": [tokenizer.bos_token, tokenizer.bos_token_id],
            "pad": [tokenizer.pad_token, tokenizer.pad_token_id],
        },
        "unk_token": tokenizer.unk_token,  # None: byte-level BPE has no unknowns
        "sample_stories": len(sample_texts),
        "characters_per_token": n_chars / total_tokens,
        "bytes_per_token": n_bytes / total_tokens,
        "tokens_per_story_mean": statistics.fmean(n_tokens),
        "tokens_per_story_median": statistics.median(n_tokens),
        "tokens_per_story_p95": sorted(n_tokens)[int(0.95 * (len(n_tokens) - 1))],
        "roundtrip_exact_fraction": sum(roundtrip) / len(roundtrip),
        "single_byte_tokens": sum(1 for t in vocab if len(t) == 1),
        "probes": probes,
    }
