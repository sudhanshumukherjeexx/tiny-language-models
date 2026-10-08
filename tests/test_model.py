import math
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from tinylm.config import ConfigError, ModelConfig, load_config
from tinylm.educational import ReferenceDecoder
from tinylm.modeling import (
    build_model,
    initialization_loss,
    kv_cache_bytes,
    parameter_report,
    weights_are_tied,
)

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.mark.parametrize("arch", ["llama", "reference"])
def test_logits_shape_and_backward(model_cfg, arch):
    model = build_model(replace(model_cfg, architecture=arch))
    ids = torch.randint(0, model_cfg.vocab_size, (3, 16))
    out = model(input_ids=ids, labels=ids)
    assert out.logits.shape == (3, 16, model_cfg.vocab_size)
    out.loss.backward()
    grads = [p.grad for p in model.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)


@pytest.mark.parametrize("arch", ["llama", "reference"])
def test_causality_future_tokens_do_not_change_past_logits(model_cfg, arch):
    model = build_model(replace(model_cfg, architecture=arch)).eval()
    ids = torch.randint(0, model_cfg.vocab_size, (1, 20))
    changed = ids.clone()
    changed[0, 12:] = (changed[0, 12:] + 1) % model_cfg.vocab_size
    with torch.no_grad():
        a, b = model(input_ids=ids).logits, model(input_ids=changed).logits
    torch.testing.assert_close(a[:, :12], b[:, :12])
    assert not torch.allclose(a[:, 12:], b[:, 12:])


@pytest.mark.parametrize("arch", ["llama", "reference"])
def test_initial_loss_is_near_ln_vocab(model_cfg, arch):
    torch.manual_seed(0)
    model = build_model(replace(model_cfg, architecture=arch, vocab_size=8192))
    loss, expected = initialization_loss(model, 8192)
    assert expected == pytest.approx(math.log(8192))
    assert abs(loss - expected) < 0.25


def test_production_parameter_count_matches_legacy_run():
    cfg = load_config(CONFIGS / "tiny-27m.yaml")
    report = parameter_report(build_model(cfg.model))
    assert report.total == 26_747_392  # value printed by the original notebook
    assert report.tied_embeddings
    assert report.by_group["Token embedding (tied LM head)"] == 8192 * 512
    assert report.by_group["Attention"] == 8 * (512 * 512 * 2 + 512 * 128 * 2)
    assert report.by_group["Feed-forward"] == 8 * 3 * 512 * 1408
    assert len(report.by_layer) == 8 and len(set(report.by_layer)) == 1
    assert report.memory_mb["bf16"] == pytest.approx(report.total * 2 / 1024**2)


@pytest.mark.parametrize("name", ["baseline-gpt-style", "rope", "rmsnorm", "swiglu", "gqa"])
def test_ablation_variants_are_parameter_matched(name):
    cfg = load_config(CONFIGS / "ablations" / f"{name}.yaml")
    total = parameter_report(build_model(cfg.model)).total
    assert abs(total - 26_747_392) / 26_747_392 < 0.001


def test_tied_weights(model_cfg):
    assert weights_are_tied(build_model(model_cfg))
    untied = build_model(replace(model_cfg, tie_word_embeddings=False))
    assert not weights_are_tied(untied)


def test_reference_decoder_matches_hf_llama(model_cfg):
    hf = build_model(model_cfg).eval()
    ref = ReferenceDecoder(replace(model_cfg, architecture="reference")).eval()
    ref.load_state_dict(hf.state_dict(), strict=True)  # identical parameter names
    ids = torch.randint(0, model_cfg.vocab_size, (2, 24))
    with torch.no_grad():
        a, b = hf(input_ids=ids, labels=ids), ref(ids, labels=ids)
    torch.testing.assert_close(a.logits, b.logits, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(a.loss, b.loss)


def test_config_validation():
    with pytest.raises(ConfigError):
        ModelConfig(hidden_size=100, num_attention_heads=8).validate()
    with pytest.raises(ConfigError):
        ModelConfig(num_attention_heads=8, num_key_value_heads=3).validate()
    with pytest.raises(ConfigError, match="reference"):
        ModelConfig(architecture="llama", norm="layernorm").validate()


def test_kv_cache_formula_and_gqa_saving():
    gqa = kv_cache_bytes(1, 512, 8, 2, 64, 2)
    mha = kv_cache_bytes(1, 512, 8, 8, 64, 2)
    assert gqa == 2 * 8 * 1 * 512 * 2 * 64 * 2 == 2 * 1024**2  # 2 MiB per sequence
    assert mha / gqa == 4
