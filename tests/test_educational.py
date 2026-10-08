"""Mechanism-level tests for the framework-free components."""

from dataclasses import replace

import pytest
import torch
from transformers.models.llama.modeling_llama import LlamaRMSNorm, LlamaRotaryEmbedding

from tinylm.educational import (
    CausalSelfAttention,
    DecoderBlock,
    ReferenceDecoder,
    RMSNorm,
    RotaryEmbedding,
    apply_rope,
    causal_mask,
    repeat_kv,
)
from tinylm.modeling import llama_config


def test_rmsnorm_normalizes_rms_and_is_scale_invariant():
    norm = RMSNorm(32)
    x = torch.randn(4, 7, 32) * 5 + 3
    y = norm(x)
    rms = y.pow(2).mean(-1).sqrt()
    torch.testing.assert_close(rms, torch.ones_like(rms), atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(norm(10 * x), y, atol=1e-5, rtol=1e-4)


def test_rmsnorm_matches_hf():
    ours, theirs = RMSNorm(32, 1e-5), LlamaRMSNorm(32, eps=1e-5)
    with torch.no_grad():
        ours.weight.uniform_()
        theirs.weight.copy_(ours.weight)
    x = torch.randn(2, 5, 32)
    torch.testing.assert_close(ours(x), theirs(x))


def test_rope_rejects_odd_head_dim():
    with pytest.raises(ValueError):
        RotaryEmbedding(15)


def test_rope_position_zero_is_identity_and_norm_is_preserved():
    rope = RotaryEmbedding(16)
    cos, sin = rope(torch.arange(10), torch.float32)
    assert cos.shape == sin.shape == (10, 16)
    x = torch.randn(1, 2, 10, 16)
    y = apply_rope(x, cos, sin)
    torch.testing.assert_close(y[..., 0, :], x[..., 0, :])
    torch.testing.assert_close(y.norm(dim=-1), x.norm(dim=-1))  # rotations preserve length


def test_rope_scores_depend_only_on_relative_position():
    rope = RotaryEmbedding(16)
    q, k = torch.randn(16), torch.randn(16)

    def score(m: int, n: int) -> torch.Tensor:
        cos, sin = rope(torch.tensor([m, n]), torch.float32)
        qm = apply_rope(q[None, None, None], cos[0:1], sin[0:1])
        kn = apply_rope(k[None, None, None], cos[1:2], sin[1:2])
        return (qm * kn).sum()

    torch.testing.assert_close(score(3, 1), score(10, 8), atol=1e-5, rtol=1e-5)
    assert not torch.allclose(score(3, 1), score(3, 2))


def test_rope_matches_hf(model_cfg):
    hf = LlamaRotaryEmbedding(llama_config(model_cfg))
    ours = RotaryEmbedding(model_cfg.head_dim, model_cfg.rope_theta)
    pos = torch.arange(12)
    hcos, hsin = hf(torch.zeros(1, 12, model_cfg.head_dim), pos[None])
    cos, sin = ours(pos, torch.float32)
    torch.testing.assert_close(cos, hcos[0])
    torch.testing.assert_close(sin, hsin[0])


def test_repeat_kv_maps_groups_of_query_heads_to_one_kv_head():
    kv = torch.arange(2).float().view(1, 2, 1, 1).expand(1, 2, 3, 4)
    out = repeat_kv(kv, 4)
    assert out.shape == (1, 8, 3, 4)
    assert out[0, :, 0, 0].tolist() == [0, 0, 0, 0, 1, 1, 1, 1]


def test_causal_mask_with_and_without_cache():
    m = causal_mask(4, 4, torch.device("cpu"))
    assert torch.equal(m, torch.tril(torch.ones(4, 4, dtype=torch.bool)))
    m2 = causal_mask(2, 5, torch.device("cpu"))  # 2 new queries after 3 cached keys
    assert m2.tolist() == [[True, True, True, True, False], [True, True, True, True, True]]


@pytest.mark.parametrize("n_kv", [1, 2, 4])
def test_eager_and_sdpa_attention_agree(n_kv):
    torch.manual_seed(0)
    eager = CausalSelfAttention(32, 4, n_kv, use_rope=False, implementation="eager")
    sdpa = CausalSelfAttention(32, 4, n_kv, use_rope=False, implementation="sdpa")
    sdpa.load_state_dict(eager.state_dict())
    x = torch.randn(2, 9, 32)
    torch.testing.assert_close(eager(x)[0], sdpa(x)[0], atol=1e-5, rtol=1e-5)


def test_gqa_cache_stores_only_kv_heads():
    attn = CausalSelfAttention(32, 4, 1, use_rope=False)
    _, (k, v) = attn(torch.randn(1, 5, 32))
    assert k.shape == v.shape == (1, 1, 5, 8)  # 1 KV head, not 4


def test_block_gradient_flow():
    block = DecoderBlock(32, 4, 2, 64)
    rope = RotaryEmbedding(8)(torch.arange(6), torch.float32)
    x = torch.randn(2, 6, 32, requires_grad=True)
    out, _ = block(x, rope)
    out.sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
    assert all(p.grad is not None for p in block.parameters())


@pytest.mark.parametrize("pos", ["rope", "learned"])
def test_kv_cache_generation_matches_full_recompute(model_cfg, pos):
    model = ReferenceDecoder(
        replace(model_cfg, architecture="reference", position_encoding=pos)
    ).eval()
    prompt = torch.randint(0, model_cfg.vocab_size, (2, 5))
    cached = model.generate_greedy(prompt, 10, use_cache=True)
    full = model.generate_greedy(prompt, 10, use_cache=False)
    assert torch.equal(cached, full)


def test_context_limit_is_enforced(model_cfg):
    model = ReferenceDecoder(replace(model_cfg, architecture="reference"))
    with pytest.raises(ValueError, match="max_position_embeddings"):
        model(torch.zeros(1, model_cfg.max_position_embeddings + 1, dtype=torch.long))
