"""Causal self-attention with grouped-query attention (GQA) and a KV cache.

Multi-head attention (MHA) gives every query head its own key/value head.
Grouped-query attention (Ainslie et al., 2023) lets ``n_heads / n_kv_heads``
query heads *share* one key/value head. Setting ``n_kv_heads == n_heads``
recovers MHA; ``n_kv_heads == 1`` is multi-query attention.

The saving is in the KV cache, which stores K and V for every past token
during generation: its size scales with ``n_kv_heads``, not ``n_heads``. The
cache below therefore stores the *un-repeated* KV heads; heads are expanded
only transiently for the attention product.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from tinylm.educational.rope import apply_rope

KVCache = tuple[torch.Tensor, torch.Tensor]


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """(batch, kv_heads, seq, d) -> (batch, kv_heads * n_rep, seq, d).

    KV head ``j`` serves query heads ``j * n_rep ... (j + 1) * n_rep - 1``.
    """
    return x if n_rep == 1 else x.repeat_interleave(n_rep, dim=1)


def causal_mask(q_len: int, k_len: int, device: torch.device) -> torch.Tensor:
    """Boolean mask (q_len, k_len), True where attention is allowed.

    Queries are the *last* ``q_len`` positions of a ``k_len`` long sequence, so
    query i may attend to keys 0 .. (k_len - q_len + i).
    """
    offset = k_len - q_len
    q_pos = torch.arange(q_len, device=device)[:, None] + offset
    k_pos = torch.arange(k_len, device=device)[None, :]
    return k_pos <= q_pos


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        n_heads: int,
        n_kv_heads: int,
        use_rope: bool = True,
        implementation: str = "sdpa",
    ):
        super().__init__()
        if hidden_size % n_heads or n_heads % n_kv_heads:
            raise ValueError("hidden_size % n_heads and n_heads % n_kv_heads must both be 0")
        self.n_heads, self.n_kv_heads = n_heads, n_kv_heads
        self.head_dim = hidden_size // n_heads
        self.use_rope = use_rope
        self.implementation = implementation
        self.q_proj = nn.Linear(hidden_size, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * self.head_dim, hidden_size, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        rope: tuple[torch.Tensor, torch.Tensor] | None = None,
        past_kv: KVCache | None = None,
    ) -> tuple[torch.Tensor, KVCache]:
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        if self.use_rope:
            if rope is None:
                raise ValueError("RoPE attention needs (cos, sin) for the current positions")
            cos, sin = rope
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if past_kv is not None:  # append to the cache (stored with n_kv_heads heads)
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)
        new_cache = (k, v)

        n_rep = self.n_heads // self.n_kv_heads
        k_all, v_all = repeat_kv(k, n_rep), repeat_kv(v, n_rep)
        if self.implementation == "sdpa":
            # PyTorch Scaled Dot Product Attention. With is_causal=True (no
            # explicit mask) it may dispatch to a fused kernel such as
            # FlashAttention when the hardware, dtype and shapes are eligible.
            if past_kv is None:
                out = F.scaled_dot_product_attention(q, k_all, v_all, is_causal=True)
            else:
                mask = causal_mask(t, k_all.shape[2], x.device)
                out = F.scaled_dot_product_attention(q, k_all, v_all, attn_mask=mask)
        else:
            out = self._eager_attention(q, k_all, v_all, causal_mask(t, k_all.shape[2], x.device))
        out = out.transpose(1, 2).reshape(b, t, self.n_heads * self.head_dim)
        return self.o_proj(out), new_cache

    def _eager_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """softmax(q k^T / sqrt(d) + mask) v, written out explicitly."""
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = scores.float().softmax(dim=-1).to(q.dtype)
        return weights @ v
