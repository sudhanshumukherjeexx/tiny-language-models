"""Memorization scanning, retrieval, duplicate/leakage detection, configs, tracking."""

import numpy as np
import pytest

from tinylm.config import ConfigError, load_config
from tinylm.integrity import (
    MinHashLSH,
    containment,
    cross_split_exact,
    exact_duplicate_stats,
    hash_texts,
    sample_documents,
    split_documents,
)
from tinylm.memorization import (
    TfidfRetriever,
    analyze,
    longest_common_span,
    ngram_hashes,
    scan_corpus_for_ngrams,
)
from tinylm.tracking import aggregate_seeds, index_row, upsert_index


def test_ngram_scan_finds_matches_across_chunk_boundaries():
    rng = np.random.default_rng(0)
    stream = rng.integers(2, 500, size=10_000).astype(np.uint16)
    planted = stream[4_990:5_030].astype(np.int64)  # straddles the 5,000-token chunk edge
    novel = rng.integers(2, 500, size=40)
    found = scan_corpus_for_ngrams(
        stream, [planted, novel], ns=(8, 16, 32), chunk_tokens=5_000, progress=False
    )
    for n in (8, 16, 32):
        assert found[n][0].all()  # every window of the planted span is in the corpus
        assert not found[n][1].any()


def test_ngram_hashes_are_consistent():
    t = np.array([5, 6, 7, 5, 6, 7])
    h = ngram_hashes(t, 3)
    assert len(h) == 4 and h[0] == h[3] and h[0] != h[1]


def test_longest_common_span():
    n, span = longest_common_span("the cat sat on the red mat", "a dog sat on the red rug")
    assert n == 4 and span == "sat on the red"


def test_retriever_finds_the_matching_story(stories):
    retriever = TfidfRetriever(stories[:100])
    query = stories[42]
    batches = [(0, stories[:150]), (150, stories[150:])]
    best = retriever.search([query], batches, top_k=1, progress=False)[0][0]
    assert stories[best[1]] == query and best[0] == pytest.approx(1.0)


def test_analyze_flags_copied_text_and_not_novel_text(stories, tokenizer):
    from tinylm.data import pack_texts

    _, blocks = pack_texts(stories, tokenizer, block_size=32)
    copied_ids = tokenizer(stories[3], add_special_tokens=False)["input_ids"]
    novel = "Seventeen purple elephants debated quantum chromodynamics over breakfast tea."
    items = [
        {"source": "generation", "item_id": "copy", "text": stories[3], "token_ids": copied_ids},
        {
            "source": "generation",
            "item_id": "novel",
            "text": novel,
            "token_ids": tokenizer(novel, add_special_tokens=False)["input_ids"],
        },
    ]
    recs = analyze(
        items, blocks.ravel(), stories, ns=(8, 16, 32), idf_sample_size=100, batch_size=64
    )
    copy, new = recs
    assert copy.flagged and copy.ngram_overlap[32] == 1.0 and copy.similarity > 0.99
    assert not new.flagged and new.ngram_overlap[8] == 0.0


def test_exact_duplicates_and_cross_split():
    a = hash_texts(["Hello  world", "hello world", "Other"])
    assert exact_duplicate_stats(a)["redundant_copies"] == 1
    b = hash_texts(["HELLO WORLD", "new"])
    assert cross_split_exact(a, b)["with_copy_in_reference"] == 1


def test_document_splitting_and_containment():
    stream = np.array([5, 6, 7, 0, 8, 9, 0, 5, 6, 7, 0], dtype=np.uint16)
    docs = split_documents(stream, 0)
    assert [d.tolist() for d in docs] == [[5, 6, 7], [8, 9], [5, 6, 7]]
    assert containment(stream, [np.array([5, 6, 7]), np.array([9, 9, 9])], n=2).tolist() == [
        1.0,
        0.0,
    ]
    sampled = sample_documents(stream, 0, 2, np.random.default_rng(0))
    assert len(sampled) == 2 and all(0 not in d for d in sampled)


def test_minhash_detects_near_duplicates():
    rng = np.random.default_rng(1)
    base = rng.integers(2, 1000, 200)
    near = base.copy()
    near[100] = 3  # one token edited
    unrelated = rng.integers(2, 1000, 200)
    lsh = MinHashLSH()
    report = lsh.near_duplicates([base, near, unrelated])
    assert report["verified_pairs"] == 1 and report["documents_with_near_duplicate"] == 2
    cross = lsh.cross_near_duplicates([base, unrelated], [near, rng.integers(2, 1000, 200)])
    assert cross["with_near_duplicate"] == 1


def test_config_extends_overrides_and_unknown_keys(tmp_path):
    (tmp_path / "base.yaml").write_text(
        "name: base\ntraining:\n  max_steps: 100\n  warmup_steps: 10\n"
    )
    (tmp_path / "child.yaml").write_text("extends: base.yaml\nname: child\n")
    cfg = load_config(tmp_path / "child.yaml", ["training.seed=7"])
    assert cfg.name == "child" and cfg.training.max_steps == 100 and cfg.training.seed == 7
    assert cfg.model.hidden_size == 512  # untouched default
    (tmp_path / "bad.yaml").write_text("training:\n  learning_rat: 0.1\n")
    with pytest.raises(ConfigError, match="learning_rat"):
        load_config(tmp_path / "bad.yaml")


def test_experiment_index_never_invents_values(tmp_path):
    row = index_row({"run_id": "r1", "name": "m", "seed": 0, "parameters": 10})
    assert row["test_loss"] is None and row["tokens_per_second"] is None
    path = tmp_path / "experiments.csv"
    upsert_index(path, row)
    upsert_index(
        path, index_row({"run_id": "r1", "name": "m", "seed": 0}, {"loss": 1.5, "perplexity": 4.48})
    )
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 2 and "1.5" in lines[1]


def test_seed_aggregation():
    agg = aggregate_seeds([{"x": 1.0}, {"x": 3.0}, {"x": None}], ["x"])
    assert (
        agg["x"]["mean"] == 2.0 and agg["x"]["n"] == 2 and agg["x"]["std"] == pytest.approx(2**0.5)
    )
