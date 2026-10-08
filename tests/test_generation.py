import logging

import pytest

from tinylm import TinyLM
from tinylm.evaluation import (
    distinct_n,
    lexical_summary,
    repeated_ngram_fraction,
    run_generation_suite,
    sentence_repetition_ratio,
)
from tinylm.generation import PRESETS, build_generation_config, custom_preset, generate
from tinylm.modeling import build_model


@pytest.fixture
def model(model_cfg):
    return build_model(model_cfg).eval()


def test_presets_pass_only_relevant_parameters(tokenizer):
    greedy = PRESETS["greedy"].generation_kwargs()
    assert greedy == {"do_sample": False}
    for name, preset in PRESETS.items():
        kw = preset.generation_kwargs()
        if preset.do_sample:
            assert {"temperature", "top_p", "top_k"} <= kw.keys(), name  # top_k explicit
    cfg = build_generation_config(PRESETS["balanced"], tokenizer, 50)
    assert cfg.max_new_tokens == 50 and cfg.top_k == 0 and cfg.max_length in (None, 20)


def test_custom_preset_zero_temperature_is_greedy():
    assert not custom_preset(temperature=0).do_sample
    assert custom_preset(temperature=0.7, top_k=40).generation_kwargs()["top_k"] == 40


def test_generation_produces_tokens_and_metadata(model, tokenizer):
    res = generate(model, tokenizer, "Once upon a time", "balanced", max_new_tokens=10, seed=0)
    assert 0 < res.new_tokens <= 10 and len(res.token_ids) == res.new_tokens
    assert res.text.startswith("Once upon a time")
    assert res.settings["max_new_tokens"] == 10 and res.settings["top_k"] == 0


def test_seeded_sampling_is_reproducible(model, tokenizer):
    a = generate(model, tokenizer, "Tom saw", "creative", 15, seed=123)
    b = generate(model, tokenizer, "Tom saw", "creative", 15, seed=123)
    c = generate(model, tokenizer, "Tom saw", "creative", 15, seed=124)
    assert a.token_ids == b.token_ids
    assert a.token_ids != c.token_ids  # different seed, different sample (random-init model)


def test_greedy_is_deterministic_and_emits_no_flag_warnings(model, tokenizer, caplog):
    with caplog.at_level(logging.WARNING):
        a = generate(model, tokenizer, "Lily", "greedy", 12)
        b = generate(model, tokenizer, "Lily", "greedy", 12, seed=7)
    assert a.token_ids == b.token_ids
    noisy = [
        r.getMessage()
        for r in caplog.records
        if "not valid" in r.getMessage() or "max_length" in r.getMessage()
    ]
    assert not noisy, noisy


def test_kv_cache_does_not_change_greedy_output(model, tokenizer):
    a = generate(model, tokenizer, "The cat", "greedy", 12, use_cache=True)
    b = generate(model, tokenizer, "The cat", "greedy", 12, use_cache=False)
    assert a.token_ids == b.token_ids


def test_context_window_is_respected(model, tokenizer):
    res = generate(model, tokenizer, "Once upon a time", "greedy", max_new_tokens=500)
    assert res.prompt_tokens + res.new_tokens <= model.config.max_position_embeddings
    with pytest.raises(ValueError, match="Shorten the prompt"):
        generate(model, tokenizer, "word " * 100, "greedy", max_new_tokens=5)


def test_tinylm_rejects_preset_plus_sampling_args(model, tokenizer):
    lm = TinyLM(model, tokenizer)
    with pytest.raises(ValueError):
        lm.generate("Hi", preset="greedy", temperature=0.5)


def test_lexical_metrics():
    assert distinct_n(["a b c d"], 1) == 1.0
    assert distinct_n(["a a a a"], 1) == 0.25
    assert distinct_n(["a b", "a b"], 2) == 0.5
    assert repeated_ngram_fraction("one two three four one two three four", n=4) == pytest.approx(
        1 / 5
    )
    assert sentence_repetition_ratio("She smiled. He ran. She smiled.") == pytest.approx(1 / 3)
    assert sentence_repetition_ratio("Only one sentence.") == 0.0


def test_generation_suite_records_provenance(model, tokenizer):
    suite = {
        "version": 9,
        "prompts": [
            {"id": "p1", "category": "x", "prompt": "Once"},
            {"id": "p2", "category": "y", "prompt": "Tom"},
        ],
    }
    records, skipped = run_generation_suite(
        model, tokenizer, suite, ["greedy", "balanced"], [0, 1], 6, metadata={"checkpoint": "test"}
    )
    assert skipped == []
    # greedy once per prompt, balanced once per seed
    assert len(records) == 2 * (1 + 2)
    r = records[0]
    assert {
        "prompt_id",
        "category",
        "preset",
        "seed",
        "settings",
        "token_ids",
        "checkpoint",
        "suite_version",
        "finished_with_eos",
    } <= r.keys()
    summary = lexical_summary(records)
    assert summary["generations"] == 6 and 0 <= summary["distinct_2"] <= 1


def test_exported_model_generates_greedy_and_sampled_without_flag_warnings(
    model, tokenizer, tmp_path, caplog
):
    from tinylm.checkpointing import export_model

    export_model(model, tokenizer, tmp_path / "m")
    lm = TinyLM.from_pretrained(tmp_path / "m", device="cpu")
    with caplog.at_level(logging.WARNING):
        lm.generate("Once", max_new_tokens=5, preset="greedy")
        lm.generate("Once", max_new_tokens=5, temperature=0.8, top_p=0.9, seed=0)
    noisy = [r.getMessage() for r in caplog.records if "not valid" in r.getMessage()]
    assert not noisy, noisy
