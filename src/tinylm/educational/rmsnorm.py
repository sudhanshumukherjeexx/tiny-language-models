"""Root-mean-square layer normalization (Zhang & Sennrich, 2019).

LayerNorm re-centres *and* re-scales each vector. RMSNorm drops the mean
subtraction and the bias, keeping only re-scaling by the root mean square:

    y = x / sqrt(mean(x**2) + eps) * g

It is cheaper (one reduction instead of two) and, in practice, as stable for
pre-norm transformers. Statistics are computed in float32 even under mixed
precision, because squaring fp16/bf16 activations easily loses precision.
"""

from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * x.to(dtype)
