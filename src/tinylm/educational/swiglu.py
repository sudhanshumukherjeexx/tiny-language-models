"""Position-wise feed-forward networks.

* ``GeluMLP``   — the GPT-2 style FFN: down(gelu(up(x))). Two matrices.
* ``SwiGLUMLP`` — gated FFN (Shazeer, 2020): down(silu(gate(x)) * up(x)).
  Three matrices; the elementwise gate lets the network modulate each hidden
  unit multiplicatively. To keep parameters comparable, SwiGLU's hidden size
  is usually ~2/3 of a GELU FFN's (e.g. 8/3 * d instead of 4 * d).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class SwiGLUMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class GeluMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.gelu(self.up_proj(x)))
