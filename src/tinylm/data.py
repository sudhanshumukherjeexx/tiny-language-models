"""Data pipeline: load -> normalize -> split -> tokenize -> pack -> account.

Split protocol
--------------
TinyStories ships two splits. We use them as:

* ``test``       = the official ``validation`` split, never used for training,
                   model selection or hyper-parameter decisions;
* ``validation`` = a deterministic ~0.5 % of the official ``train`` split,
                   chosen by hashing each story's *normalized* text, used for
                   checkpoint selection and ablation decisions;
* ``train``      = the rest of the official ``train`` split.

Hash-based assignment is order-independent and seedless, and it places exact
duplicates (after normalization) in the same split, so exact copies cannot leak
between train and validation. Leakage between train and the official split is
*measured* (see ``tinylm.integrity``), not assumed away.

Packing
-------
Documents are tokenized, each followed by one EOS token, and concatenated into
a single stream that is cut into fixed ``block_size`` sequences by
``SequencePacker``. The packer carries its unfinished remainder across batches,
so only the final global remainder (< block_size tokens per split) is dropped.
"""

from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from tqdm.auto import tqdm

from tinylm.config import DataConfig, NormalizationConfig
from tinylm.tokenizer import encode_batch
from tinylm.utils import read_json, write_json

logger = logging.getLogger(__name__)

SPLITS = ("train", "validation", "test")
TOKEN_DTYPE = np.uint16  # vocab <= 65,535; halves disk/RAM versus int32

_WS = re.compile(r"[ \t]+")
_NL = re.compile(r"\n{3,}")
_NON_ASCII = re.compile(r"[^\x20-\x7E\n]")
_ASCII_PUNCT = str.maketrans(
    {
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "–": "-",
        "—": "-",
        "…": "...",
        " ": " ",
    }
)


# --------------------------------------------------------------------------- text


def normalize_text(text: str, cfg: NormalizationConfig) -> str:
    """Deterministic text normalization applied identically to every split."""
    if cfg.unicode_form != "none":
        text = unicodedata.normalize(cfg.unicode_form, text)
    if cfg.ascii_punctuation:
        text = text.translate(_ASCII_PUNCT)
    if cfg.strip_non_ascii:
        text = _NON_ASCII.sub("", text)
    if cfg.collapse_whitespace:
        text = _NL.sub("\n\n", _WS.sub(" ", text))
    return text.strip()


def dedup_key(text: str) -> str:
    """Canonical form used for exact-duplicate detection (case/space-insensitive)."""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def content_hash(text: str) -> int:
    """Stable 64-bit hash of :func:`dedup_key` (Python's ``hash`` is salted per process)."""
    return int.from_bytes(hashlib.blake2b(dedup_key(text).encode(), digest_size=8).digest(), "big")


def split_fingerprint(texts) -> str:
    """Order-sensitive hash of a split's normalized stories (identifies its exact contents)."""
    h = hashlib.blake2b(digest_size=16)
    for t in texts:
        h.update(content_hash(t).to_bytes(8, "big"))
    return h.hexdigest()


def is_validation(text: str, fraction: float) -> bool:
    return content_hash(text) % 1_000_000 < round(fraction * 1_000_000)


def _text_cache_key(cfg: DataConfig) -> dict[str, Any]:
    keys = (
        "dataset_name",
        "dataset_revision",
        "text_field",
        "source_train_split",
        "source_test_split",
        "validation_fraction",
        "min_chars",
        "max_train_examples",
        "subset_seed",
        "decontaminate_test",
    )
    return {k: getattr(cfg, k) for k in keys} | {"normalization": asdict(cfg.normalization)}


def resolve_dataset_revision(cfg: DataConfig) -> str | None:
    """Commit SHA of the dataset on the Hub, or None if it cannot be resolved."""
    try:
        from huggingface_hub import HfApi

        return HfApi().dataset_info(cfg.dataset_name, revision=cfg.dataset_revision).sha
    except Exception as err:  # network/auth errors vary by hub version; never fatal
        logger.warning("Could not resolve dataset revision (%s); recording null.", err)
        return cfg.dataset_revision


def build_text_splits(cfg: DataConfig, force: bool = False) -> DatasetDict:
    """Return normalized train/validation/test text, cached under ``data_dir/text``.

    The cache is reused only if it was built with identical data settings;
    otherwise an actionable error is raised instead of silently mixing data.
    """
    text_dir = Path(cfg.data_dir) / "text"
    meta_path = text_dir / "text_meta.json"
    key = _text_cache_key(cfg)
    if text_dir.exists() and not force:
        cached = read_json(meta_path) if meta_path.exists() else {}
        if cached.get("cache_key") != key:
            raise RuntimeError(
                f"{text_dir} was built with different data settings. Delete it or pass "
                "--force to rebuild (this invalidates the tokenizer and packed data)."
            )
        logger.info("Reusing normalized text splits from %s", text_dir)
        return load_from_disk(str(text_dir))

    revision = resolve_dataset_revision(cfg)
    logger.info("Loading %s (revision=%s)", cfg.dataset_name, revision)
    raw = load_dataset(cfg.dataset_name, revision=revision)
    norm = cfg.normalization

    def _clean(batch: dict[str, list[str]]) -> dict[str, list[str]]:
        return {"text": [normalize_text(t, norm) for t in batch[cfg.text_field]]}

    def _prepare(split: str) -> Dataset:
        ds = raw[split]
        ds = ds.map(_clean, batched=True, remove_columns=ds.column_names, desc=f"normalize {split}")
        return ds.filter(
            lambda b: [len(t) >= cfg.min_chars for t in b["text"]], batched=True, desc="min_chars"
        )

    source_train = _prepare(cfg.source_train_split)
    test = _prepare(cfg.source_test_split)
    frac = cfg.validation_fraction
    train_hashes = [content_hash(t) for t in tqdm(source_train["text"], desc="hash split")]
    is_val = [h % 1_000_000 < round(frac * 1_000_000) for h in train_hashes]
    contaminated = 0
    if cfg.decontaminate_test:
        seen = set(train_hashes)
        keep = [i for i, t in enumerate(test["text"]) if content_hash(t) not in seen]
        contaminated = len(test) - len(keep)
        test = test.select(keep)
        logger.info(
            "Removed %d test stories with an exact copy in the training source", contaminated
        )
    val_idx = [i for i, v in enumerate(is_val) if v]
    train_idx = [i for i, v in enumerate(is_val) if not v]
    train = source_train.select(train_idx)
    validation = source_train.select(val_idx)
    if cfg.max_train_examples is not None and cfg.max_train_examples < len(train):
        train = train.shuffle(seed=cfg.subset_seed).select(range(cfg.max_train_examples))

    splits = DatasetDict(
        train=train.flatten_indices(),
        validation=validation.flatten_indices(),
        test=test.flatten_indices(),
    )
    splits.save_to_disk(str(text_dir))
    write_json(
        meta_path,
        {
            "cache_key": key,
            "resolved_revision": revision,
            "source_rows": {
                cfg.source_train_split: len(raw[cfg.source_train_split]),
                cfg.source_test_split: len(raw[cfg.source_test_split]),
            },
            "dropped_below_min_chars": {
                cfg.source_train_split: len(raw[cfg.source_train_split]) - len(source_train),
                cfg.source_test_split: len(raw[cfg.source_test_split]) - len(test) - contaminated,
            },
            "test_removed_exact_copy_of_train": contaminated,
            "rows": {k: len(v) for k, v in splits.items()},
            "train_split_fingerprint": split_fingerprint(splits["train"]["text"]),
        },
    )
    return splits


# --------------------------------------------------------------------------- packing


class SequencePacker:
    """Cut a token stream into fixed-length blocks, carrying the remainder forward.

    ``add`` may be called with arbitrarily sized chunks; the concatenation of all
    returned blocks equals the concatenated input minus the final remainder.
    """

    def __init__(self, block_size: int):
        self.block_size = block_size
        self._buffer = np.empty(0, dtype=np.int64)
        self.tokens_in = 0
        self.tokens_out = 0

    def add(self, tokens: np.ndarray) -> np.ndarray:
        self.tokens_in += len(tokens)
        stream = np.concatenate([self._buffer, tokens.astype(np.int64, copy=False)])
        n_full = len(stream) // self.block_size * self.block_size
        self._buffer = stream[n_full:]
        self.tokens_out += n_full
        return stream[:n_full].reshape(-1, self.block_size)

    @property
    def remainder(self) -> int:
        return len(self._buffer)

    def finish(self) -> int:
        """Drop the final partial block; return how many tokens were discarded."""
        dropped = len(self._buffer)
        self._buffer = np.empty(0, dtype=np.int64)
        return dropped


@dataclass
class PackingStats:
    examples: int = 0
    characters: int = 0
    text_tokens: int = 0  # tokens produced by the tokenizer (no EOS)
    eos_tokens: int = 0  # one per document
    stream_tokens: int = 0  # text_tokens + eos_tokens
    packed_tokens: int = 0
    sequences: int = 0
    discarded_tokens: int = 0  # final global remainder only
    block_size: int = 0

    @property
    def retained_fraction(self) -> float:
        return self.packed_tokens / self.stream_tokens if self.stream_tokens else 0.0

    def validate(self) -> None:
        """Token accounting invariants; raises if any number does not add up."""
        checks = {
            "stream == text + eos": self.stream_tokens == self.text_tokens + self.eos_tokens,
            "eos == examples": self.eos_tokens == self.examples,
            "packed == sequences * block": self.packed_tokens == self.sequences * self.block_size,
            "stream == packed + discarded": self.stream_tokens
            == self.packed_tokens + self.discarded_tokens,
            "discarded < block": 0 <= self.discarded_tokens < self.block_size,
        }
        failed = [name for name, ok in checks.items() if not ok]
        if failed:
            raise AssertionError(f"Token accounting failed: {failed} for {self}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"retained_fraction": self.retained_fraction}


def pack_texts(
    texts: list[str] | Dataset,
    tokenizer,
    block_size: int,
    out_path: str | Path | None = None,
    batch_size: int = 1000,
    desc: str = "packing",
) -> tuple[PackingStats, np.ndarray | None]:
    """Tokenize documents, append EOS, and pack into ``block_size`` blocks.

    Blocks are streamed to ``out_path`` (raw ``uint16``) when given; otherwise
    they are returned as an array (useful for tests and small corpora).
    """
    if len(tokenizer) > np.iinfo(TOKEN_DTYPE).max + 1:
        raise ValueError(f"Vocabulary of {len(tokenizer)} does not fit in {TOKEN_DTYPE}.")
    eos = tokenizer.eos_token_id
    packer = SequencePacker(block_size)
    stats = PackingStats(block_size=block_size)
    blocks: list[np.ndarray] = []
    out_file = None
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        out_file = open(out_path, "wb")  # noqa: SIM115 - closed in finally
    try:
        n = len(texts)
        for start in tqdm(range(0, n, batch_size), desc=desc, unit_scale=batch_size):
            batch = texts[start : start + batch_size]
            batch = batch["text"] if isinstance(batch, dict) else batch
            ids = encode_batch(tokenizer, list(batch))
            lengths = np.fromiter((len(x) + 1 for x in ids), dtype=np.int64, count=len(ids))
            flat = np.empty(int(lengths.sum()), dtype=np.int64)
            ends = np.cumsum(lengths)
            for doc, end in zip(ids, ends, strict=True):
                flat[end - len(doc) - 1 : end - 1] = doc
            flat[ends - 1] = eos
            stats.examples += len(ids)
            stats.characters += sum(len(t) for t in batch)
            stats.text_tokens += int(lengths.sum()) - len(ids)
            stats.eos_tokens += len(ids)
            out = packer.add(flat)
            if len(out):
                if out_file is not None:
                    out_file.write(out.astype(TOKEN_DTYPE).tobytes())
                else:
                    blocks.append(out.astype(TOKEN_DTYPE))
    finally:
        if out_file is not None:
            out_file.close()
    stats.discarded_tokens = packer.finish()
    stats.stream_tokens = packer.tokens_in
    stats.packed_tokens = packer.tokens_out
    stats.sequences = packer.tokens_out // block_size
    stats.validate()
    array = None
    if out_path is None:
        array = np.concatenate(blocks) if blocks else np.empty((0, block_size), TOKEN_DTYPE)
    return stats, array


def prepare_packed_data(
    cfg: DataConfig, tokenizer, tokenizer_sha256: str, force: bool = False
) -> dict[str, Any]:
    """Pack every split to ``data_dir/<split>.bin`` and write ``meta.json``."""
    data_dir = Path(cfg.data_dir)
    # --force rebuilds the normalized text too, so data-setting changes take effect.
    splits = build_text_splits(cfg, force=force)
    text_meta = read_json(data_dir / "text" / "text_meta.json")
    tok_report = Path(tokenizer.name_or_path) / "tokenizer_report.json"
    if tok_report.exists():
        trained_on = read_json(tok_report).get("train_split_fingerprint")
        if trained_on and trained_on != text_meta["train_split_fingerprint"]:
            raise RuntimeError(
                "The tokenizer was trained on a different train split than the one being "
                "packed. Retrain it: scripts/train_tokenizer.py --force."
            )
    meta: dict[str, Any] = {
        "dataset": cfg.dataset_name,
        "dataset_revision": text_meta["resolved_revision"],
        "train_split_fingerprint": text_meta["train_split_fingerprint"],
        "block_size": cfg.block_size,
        "tokenizer_sha256": tokenizer_sha256,
        "vocab_size": len(tokenizer),
        "eos_token_id": tokenizer.eos_token_id,
        "token_dtype": np.dtype(TOKEN_DTYPE).name,
        "splits": {},
    }
    for split in SPLITS:
        out = data_dir / f"{split}.bin"
        if out.exists() and not force:
            raise FileExistsError(f"{out} exists; pass --force to rebuild packed data.")
        stats, _ = pack_texts(splits[split], tokenizer, cfg.block_size, out, desc=f"pack {split}")
        meta["splits"][split] = stats.to_dict()
        logger.info(
            "%-10s %9d stories -> %9d sequences (%s tokens, %.4f%% retained)",
            split,
            stats.examples,
            stats.sequences,
            f"{stats.packed_tokens:,}",
            100 * stats.retained_fraction,
        )
    write_json(data_dir / "meta.json", meta)
    return meta


# --------------------------------------------------------------------------- reading


class TextColumn:
    """List-like, memory-mapped view of a text column (avoids materializing 2M strings)."""

    def __init__(self, source: Dataset | list[str], field: str = "text"):
        self.source, self.field = source, field

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, key: int | slice) -> str | list[str]:
        item = self.source[key]
        if isinstance(self.source, Dataset):
            return item[self.field]
        return item


def load_text_split(data_dir: str | Path, split: str) -> TextColumn:
    text_dir = Path(data_dir) / "text"
    if not text_dir.exists():
        raise FileNotFoundError(
            f"No normalized text in {text_dir}; run scripts/prepare_data.py first."
        )
    return TextColumn(load_from_disk(str(text_dir))[split])


class PackedDataset(torch.utils.data.Dataset):
    """Memory-mapped fixed-length token blocks written by :func:`prepare_packed_data`."""

    def __init__(self, path: str | Path, block_size: int):
        self.path = Path(path)
        self.block_size = block_size
        tokens = np.memmap(self.path, dtype=TOKEN_DTYPE, mode="r")
        if len(tokens) % block_size:
            raise ValueError(f"{path} holds {len(tokens)} tokens, not a multiple of {block_size}.")
        self.blocks = tokens.reshape(-1, block_size)

    @classmethod
    def from_array(cls, blocks: np.ndarray) -> PackedDataset:
        obj = cls.__new__(cls)
        obj.path, obj.block_size, obj.blocks = None, blocks.shape[1], blocks
        return obj

    def __len__(self) -> int:
        return len(self.blocks)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.from_numpy(self.blocks[idx].astype(np.int64))

    def batch(self, indices: np.ndarray) -> torch.Tensor:
        """Gather a batch of blocks with a single fancy-indexing read."""
        return torch.from_numpy(self.blocks[indices].astype(np.int64))

    @property
    def num_tokens(self) -> int:
        return self.blocks.size


def load_packed_split(
    data_dir: str | Path, split: str, expected_tokenizer_sha256: str | None = None
) -> PackedDataset:
    """Open a packed split, verifying it was built with the expected tokenizer."""
    data_dir = Path(data_dir)
    meta_path = data_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"No packed data in {data_dir}. Run scripts/prepare_data.py with the same config."
        )
    meta = read_json(meta_path)
    if expected_tokenizer_sha256 and meta["tokenizer_sha256"] != expected_tokenizer_sha256:
        raise RuntimeError(
            f"Packed data in {data_dir} was built with a different tokenizer "
            f"({meta['tokenizer_sha256'][:12]} != {expected_tokenizer_sha256[:12]}). "
            "Re-run scripts/prepare_data.py --force."
        )
    return PackedDataset(data_dir / f"{split}.bin", meta["block_size"])
