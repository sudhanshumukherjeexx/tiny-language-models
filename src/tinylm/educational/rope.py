"""Rotary position embeddings (Su et al., 2021, "RoFormer").

RoPE encodes position by *rotating* query and key vectors instead of adding a
learned vector to the input. Each pair of channels (i, i + d/2) is rotated by
an angle ``position * theta**(-2i/d)``. Because a rotation by angle a followed
by the dot product with a key rotated by b depends only on a - b, attention
scores become a function of *relative* distance, and there are no position
parameters to learn.

This file uses the "rotate-half" channel pairing used by Hugging Face Llama
(pairs (i, i + d/2)), not the interleaved (2i, 2i + 1) pairing of the paper;
the two are equivalent up to a fixed permutation of the head dimensions.
"""

from __future__ import annotations

import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, theta: float = 10000.0):
        super().__init__()
        if head_dim % 2:
            raise ValueError(f"RoPE needs an even head_dim, got {head_dim}")
        exponent = torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim
        self.register_buffer("inv_freq", 1.0 / (theta**exponent), persistent=False)

    @torch.no_grad()
    def forward(
        self, positions: torch.Tensor, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (cos, sin) of shape (seq, head_dim) for integer ``positions``."""
        freqs = positions.float()[:, None] * self.inv_freq[None, :]  # (seq, d/2)
        angles = torch.cat([freqs, freqs], dim=-1)  # (seq, d): pair i with i + d/2
        return angles.cos().to(dtype), angles.sin().to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate ``x`` of shape (batch, heads, seq, head_dim)."""
    return x * cos + rotate_half(x) * sin
