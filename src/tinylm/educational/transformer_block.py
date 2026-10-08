"""A pre-norm decoder block:  x + attn(norm(x)),  then  x + ffn(norm(x)).

Pre-norm (normalizing the *input* of each sub-layer) keeps an un-normalized
residual path from the embeddings to the output, which is what makes deep
transformers train stably without careful warmup tricks.
"""

from __future__ import annotations

import torch
from torch import nn

from tinylm.educational.attention import CausalSelfAttention, KVCache
from tinylm.educational.rmsnorm import RMSNorm
from tinylm.educational.swiglu import GeluMLP, SwiGLUMLP


def make_norm(kind: str, dim: int, eps: float) -> nn.Module:
    if kind == "rmsnorm":
        return RMSNorm(dim, eps)
    if kind == "layernorm":
        return nn.LayerNorm(dim, eps=eps)
    raise ValueError(f"unknown norm {kind!r}")


class DecoderBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        n_heads: int,
        n_kv_heads: int,
        intermediate_size: int,
        norm: str = "rmsnorm",
        ffn: str = "swiglu",
        use_rope: bool = True,
        norm_eps: float = 1e-5,
        attn_implementation: str = "sdpa",
    ):
        super().__init__()
        self.input_layernorm = make_norm(norm, hidden_size, norm_eps)
        self.self_attn = CausalSelfAttention(
            hidden_size, n_heads, n_kv_heads, use_rope, attn_implementation
        )
        self.post_attention_layernorm = make_norm(norm, hidden_size, norm_eps)
        mlp_cls = {"swiglu": SwiGLUMLP, "gelu": GeluMLP}[ffn]
        self.mlp = mlp_cls(hidden_size, intermediate_size)

    def forward(
        self,
        x: torch.Tensor,
        rope: tuple[torch.Tensor, torch.Tensor] | None = None,
        past_kv: KVCache | None = None,
    ) -> tuple[torch.Tensor, KVCache]:
        attn_out, cache = self.self_attn(self.input_layernorm(x), rope, past_kv)
        x = x + attn_out
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, cache
