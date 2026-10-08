"""Framework-free PyTorch implementations of the production model's components.

These modules exist to make the mechanisms inspectable and testable; the
production 27M model is trained with Hugging Face ``LlamaForCausalLM``. The
``ReferenceDecoder`` is additionally used as the controlled model family for
architecture ablations.
"""

from tinylm.educational.attention import CausalSelfAttention, causal_mask, repeat_kv
from tinylm.educational.model import ReferenceDecoder
from tinylm.educational.rmsnorm import RMSNorm
from tinylm.educational.rope import RotaryEmbedding, apply_rope, rotate_half
from tinylm.educational.swiglu import GeluMLP, SwiGLUMLP
from tinylm.educational.transformer_block import DecoderBlock

__all__ = [
    "CausalSelfAttention",
    "DecoderBlock",
    "GeluMLP",
    "RMSNorm",
    "ReferenceDecoder",
    "RotaryEmbedding",
    "SwiGLUMLP",
    "apply_rope",
    "causal_mask",
    "repeat_kv",
    "rotate_half",
]
