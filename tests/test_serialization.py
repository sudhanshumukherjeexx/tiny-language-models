import logging
from dataclasses import replace

import torch
from safetensors import safe_open

from tinylm import TinyLM
from tinylm.checkpointing import export_model
from tinylm.educational import ReferenceDecoder
from tinylm.modeling import build_model, load_model, weights_are_tied


def _logits(model, ids):
    with torch.no_grad():
        return model.eval()(input_ids=ids).logits


def test_export_reload_gives_identical_logits_and_tied_weights(
    model_cfg, tokenizer, tmp_path, caplog
):
    model = build_model(model_cfg)
    ids = torch.randint(0, model_cfg.vocab_size, (2, 12))
    export_model(model, tokenizer, tmp_path / "m", {"note": "test"})
    with caplog.at_level(logging.WARNING):
        reloaded = load_model(tmp_path / "m")
    assert not [r for r in caplog.records if "missing" in r.getMessage().lower()]
    torch.testing.assert_close(_logits(model, ids), _logits(reloaded, ids), atol=0, rtol=0)
    assert weights_are_tied(reloaded)
    # Tied embeddings are stored once: there is intentionally no lm_head tensor on disk.
    with safe_open(str(tmp_path / "m" / "model.safetensors"), "pt") as f:
        assert not any(k.startswith("lm_head") for k in f.keys())  # noqa: SIM118 - safe_open is not a dict
    assert (tmp_path / "m" / "tokenizer.json").exists()
    assert (tmp_path / "m" / "generation_config.json").exists()


def test_tied_weights_stay_tied_through_training_update(model_cfg, tmp_path):
    model = build_model(model_cfg)
    ids = torch.randint(0, model_cfg.vocab_size, (2, 12))
    model(input_ids=ids, labels=ids).loss.backward()
    torch.optim.SGD(model.parameters(), lr=0.1).step()
    model.save_pretrained(tmp_path / "m")
    reloaded = load_model(tmp_path / "m")
    assert weights_are_tied(reloaded)
    torch.testing.assert_close(reloaded.lm_head.weight, model.model.embed_tokens.weight)


def test_reference_model_roundtrip(model_cfg, tmp_path):
    model = ReferenceDecoder(
        replace(
            model_cfg,
            architecture="reference",
            norm="layernorm",
            position_encoding="learned",
            ffn="gelu",
        )
    )
    model.save_pretrained(tmp_path / "r")
    reloaded = load_model(tmp_path / "r")
    assert isinstance(reloaded, ReferenceDecoder) and weights_are_tied(reloaded)
    ids = torch.randint(0, model_cfg.vocab_size, (1, 10))
    torch.testing.assert_close(_logits(model, ids), _logits(reloaded, ids), atol=0, rtol=0)


def test_tinylm_api_loads_export(model_cfg, tokenizer, tmp_path):
    export_model(build_model(model_cfg), tokenizer, tmp_path / "m")
    lm = TinyLM.from_pretrained(tmp_path / "m", device="cpu")
    text = lm.generate("Once upon a time", max_new_tokens=8, temperature=0.8, seed=1)
    assert text.startswith("Once upon a time")
    assert lm.metadata["parameters"] > 0 and lm.metadata["kv_heads"] == 2
