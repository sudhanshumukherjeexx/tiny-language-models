import itertools

import numpy as np
import pytest

from tinylm.config import NormalizationConfig
from tinylm.data import (
    PackedDataset,
    PackingStats,
    SequencePacker,
    content_hash,
    is_validation,
    normalize_text,
    pack_texts,
)
from tinylm.tokenizer import encode_batch


def test_packer_emits_only_full_blocks():
    packer = SequencePacker(block_size=4)
    out = packer.add(np.arange(10))
    assert out.shape == (2, 4) and packer.remainder == 2


def test_rolling_remainder_is_carried_across_chunks():
    rng = np.random.default_rng(0)
    stream = rng.integers(0, 100, size=1003)
    packer = SequencePacker(block_size=16)
    chunks, start = [], 0
    while start < len(stream):  # arbitrary, uneven chunk sizes
        size = int(rng.integers(1, 40))
        chunks.append(packer.add(stream[start : start + size]))
        start += size
    blocks = np.concatenate(chunks)
    dropped = packer.finish()
    assert dropped == 1003 % 16
    np.testing.assert_array_equal(blocks.ravel(), stream[: len(stream) - dropped])


def test_pack_texts_accounting_and_content(stories, tokenizer):
    stats, blocks = pack_texts(stories, tokenizer, block_size=32, batch_size=7)
    stats.validate()
    ids = encode_batch(tokenizer, stories)
    expected = list(itertools.chain.from_iterable([*x, tokenizer.eos_token_id] for x in ids))
    assert stats.examples == len(stories) == stats.eos_tokens
    assert stats.text_tokens == sum(len(x) for x in ids)
    assert stats.stream_tokens == len(expected)
    assert stats.discarded_tokens == len(expected) % 32  # only the final global remainder
    np.testing.assert_array_equal(blocks.ravel(), expected[: stats.packed_tokens])
    assert blocks.shape[1] == 32


def test_packing_loses_less_than_per_batch_truncation(stories, tokenizer):
    """The legacy notebook dropped a remainder per map batch; now only one per split."""
    stats, _ = pack_texts(stories, tokenizer, block_size=32, batch_size=10)
    ids = encode_batch(tokenizer, stories)
    legacy_kept = sum(
        (sum(len(x) + 1 for x in ids[i : i + 10]) // 32) * 32 for i in range(0, len(ids), 10)
    )
    assert stats.packed_tokens >= legacy_kept
    assert stats.stream_tokens - stats.packed_tokens < 32


def test_accounting_validation_catches_errors():
    bad = PackingStats(
        examples=2,
        text_tokens=10,
        eos_tokens=2,
        stream_tokens=12,
        packed_tokens=8,
        sequences=2,
        discarded_tokens=3,
        block_size=4,
    )
    with pytest.raises(AssertionError):
        bad.validate()


def test_packed_dataset_roundtrip(tmp_path, stories, tokenizer):
    out = tmp_path / "train.bin"
    stats, _ = pack_texts(stories, tokenizer, block_size=32, out_path=out)
    ds = PackedDataset(out, 32)
    assert len(ds) == stats.sequences and ds.num_tokens == stats.packed_tokens
    batch = ds.batch(np.array([3, 0, 2]))
    assert batch.shape == (3, 32)
    np.testing.assert_array_equal(batch[0].numpy(), ds[3].numpy())


def test_normalization_preserves_unicode_by_default():
    text = "Café “quote” — ok…  extra   spaces\n\n\n\nnext"
    out = normalize_text(text, NormalizationConfig())
    assert "Café" in out and '"quote"' in out and "..." in out
    assert "\n\n\n" not in out and "  " not in out
    stripped = normalize_text(text, NormalizationConfig(strip_non_ascii=True))
    # The legacy ASCII filter silently corrupts words: "Café" becomes "Caf".
    assert stripped.isascii() and stripped.startswith("Caf ")


def test_hash_split_is_deterministic_and_duplicate_safe(stories):
    a = [is_validation(s, 0.2) for s in stories]
    assert a == [is_validation(s, 0.2) for s in stories]
    assert 0 < sum(a) < len(a)
    # Case/whitespace variants of one story hash identically -> same split.
    s = stories[0]
    assert content_hash(s) == content_hash("  " + s.upper().replace(" ", "  "))
    assert is_validation(s, 0.2) == is_validation(s.upper(), 0.2)


def test_train_validation_test_are_disjoint_by_construction(stories):
    val = {s for s in stories if is_validation(s, 0.3)}
    train = {s for s in stories if not is_validation(s, 0.3)}
    assert not (val & train)


def test_build_text_splits_decontaminates_test_and_keeps_splits_disjoint(
    stories, tmp_path, monkeypatch
):
    from datasets import Dataset, DatasetDict

    import tinylm.data as data
    from tinylm.config import DataConfig
    from tinylm.utils import read_json

    unique = sorted(set(stories))
    train_src = unique[:40]
    # Test = 10 new stories + 5 case/whitespace variants of training stories.
    novel = [f"A brand new story number {i} about a purple whale who sings." * 3 for i in range(10)]
    leaked = ["  " + s.upper() for s in train_src[:5]]
    raw = DatasetDict(
        train=Dataset.from_dict({"text": train_src}),
        validation=Dataset.from_dict({"text": novel + leaked}),
    )
    monkeypatch.setattr(data, "load_dataset", lambda *a, **k: raw)
    monkeypatch.setattr(data, "resolve_dataset_revision", lambda cfg: "test-rev")
    cfg = DataConfig(data_dir=str(tmp_path / "d"), validation_fraction=0.2, min_chars=10)

    splits = data.build_text_splits(cfg)
    meta = read_json(tmp_path / "d" / "text" / "text_meta.json")
    assert meta["test_removed_exact_copy_of_train"] == 5
    assert len(splits["test"]) == 10
    keys = {s: {data.content_hash(t) for t in splits[s]["text"]} for s in splits}
    assert not (keys["train"] & keys["validation"])
    assert not (keys["train"] & keys["test"]) and not (keys["validation"] & keys["test"])
    assert meta["train_split_fingerprint"] == data.split_fingerprint(splits["train"]["text"])
    # Cached text is reused for identical settings and rejected for changed ones.
    assert len(data.build_text_splits(cfg)["test"]) == 10
    with pytest.raises(RuntimeError, match="different data settings"):
        data.build_text_splits(DataConfig(**{**cfg.__dict__, "decontaminate_test": False}))
