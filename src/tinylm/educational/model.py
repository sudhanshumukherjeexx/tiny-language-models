"""A complete decoder-only language model built from the educational components.

``ReferenceDecoder`` serves two purposes:

1. It shows every mechanism of the production model without framework
   abstractions. With the Llama recipe (RoPE + RMSNorm + SwiGLU + GQA) it uses
   the same parameter names and math as Hugging Face ``LlamaForCausalLM``, so
   HF weights load into it directly and produce the same logits (tested).
2. Its components can be switched one at a time (learned vs. rotary positions,
   LayerNorm vs. RMSNorm, GELU vs. SwiGLU, MHA vs. GQA), which makes it the
   model family for the controlled architecture ablations: every variant shares
   one code path, initialization scheme and training loop.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_model, save_model
from torch import nn

from tinylm.config import ModelConfig
from tinylm.educational.attention import KVCache
from tinylm.educational.rope import RotaryEmbedding
from tinylm.educational.transformer_block import DecoderBlock, make_norm


@dataclass
class DecoderOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None
    past_key_values: list[KVCache] | None = None


class _Backbone(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        if cfg.position_encoding == "learned":
            self.embed_positions = nn.Embedding(cfg.max_position_embeddings, cfg.hidden_size)
        self.layers = nn.ModuleList(
            DecoderBlock(
                cfg.hidden_size,
                cfg.num_attention_heads,
                cfg.num_key_value_heads,
                cfg.intermediate_size,
                norm=cfg.norm,
                ffn=cfg.ffn,
                use_rope=cfg.position_encoding == "rope",
                norm_eps=cfg.norm_eps,
                attn_implementation=cfg.attn_implementation,
            )
            for _ in range(cfg.num_hidden_layers)
        )
        self.norm = make_norm(cfg.norm, cfg.hidden_size, cfg.norm_eps)


class ReferenceDecoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        cfg.validate()
        self.config = cfg
        self.model = _Backbone(cfg)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.rotary = (
            RotaryEmbedding(cfg.head_dim, cfg.rope_theta)
            if cfg.position_encoding == "rope"
            else None
        )
        self.apply(self._init_weights)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def _init_weights(self, module: nn.Module) -> None:
        # Same scheme as HF Llama: N(0, initializer_range) for every matrix.
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        past_key_values: list[KVCache] | None = None,
        use_cache: bool = False,
        **_: object,
    ) -> DecoderOutput:
        _, t = input_ids.shape
        past_len = 0 if past_key_values is None else past_key_values[0][0].shape[2]
        if past_len + t > self.config.max_position_embeddings:
            raise ValueError(
                f"sequence length {past_len + t} exceeds "
                f"max_position_embeddings={self.config.max_position_embeddings}"
            )
        positions = torch.arange(past_len, past_len + t, device=input_ids.device)
        x = self.model.embed_tokens(input_ids)
        if self.config.position_encoding == "learned":
            x = x + self.model.embed_positions(positions)
        rope = self.rotary(positions, x.dtype) if self.rotary is not None else None

        new_cache: list[KVCache] = []
        for i, layer in enumerate(self.model.layers):
            past = None if past_key_values is None else past_key_values[i]
            x, cache = layer(x, rope, past)
            new_cache.append(cache)
        logits = self.lm_head(self.model.norm(x))

        loss = None
        if labels is not None:  # predict token i+1 from positions <= i
            loss = F.cross_entropy(
                logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return DecoderOutput(logits, loss, new_cache if use_cache else None)

    @torch.no_grad()
    def generate_greedy(self, input_ids: torch.Tensor, max_new_tokens: int, use_cache: bool = True):
        """Minimal greedy decoding; demonstrates (and tests) the KV cache."""
        cache, ids, step_input = None, input_ids, input_ids
        for _ in range(max_new_tokens):
            out = self(step_input if use_cache else ids, past_key_values=cache, use_cache=use_cache)
            next_id = out.logits[:, -1].argmax(dim=-1, keepdim=True)
            ids = torch.cat([ids, next_id], dim=1)
            cache, step_input = out.past_key_values, next_id
        return ids

    # -- serialization (mirrors the HF directory layout: config.json + model.safetensors)

    def save_pretrained(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        payload = {"tinylm_architecture": "reference", **asdict(self.config)}
        (path / "config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        save_model(self, str(path / "model.safetensors"))  # stores tied tensors once

    @classmethod
    def from_pretrained(cls, path: str | Path) -> ReferenceDecoder:
        path = Path(path)
        raw = json.loads((path / "config.json").read_text(encoding="utf-8"))
        raw.pop("tinylm_architecture", None)
        model = cls(ModelConfig(**raw))
        load_model(model, str(path / "model.safetensors"))
        return model
