"""Model construction, loading, parameter accounting and memory estimates."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn
from transformers import LlamaConfig, LlamaForCausalLM

from tinylm.config import ModelConfig
from tinylm.educational.model import ReferenceDecoder

BYTES_PER_DTYPE = {"fp32": 4, "bf16": 2, "fp16": 2}


def llama_config(cfg: ModelConfig, eos_token_id: int = 0, pad_token_id: int = 1) -> LlamaConfig:
    """Translate a :class:`ModelConfig` into a Hugging Face ``LlamaConfig``."""
    return LlamaConfig(
        vocab_size=cfg.vocab_size,
        hidden_size=cfg.hidden_size,
        intermediate_size=cfg.intermediate_size,
        num_hidden_layers=cfg.num_hidden_layers,
        num_attention_heads=cfg.num_attention_heads,
        num_key_value_heads=cfg.num_key_value_heads,
        max_position_embeddings=cfg.max_position_embeddings,
        hidden_act="silu",  # with Llama's gated MLP this is SwiGLU
        rms_norm_eps=cfg.norm_eps,
        rope_parameters={"rope_type": "default", "rope_theta": cfg.rope_theta},
        initializer_range=cfg.initializer_range,
        tie_word_embeddings=cfg.tie_word_embeddings,
        attention_bias=False,
        mlp_bias=False,
        bos_token_id=eos_token_id,  # a document boundary is a single EOS token
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        attn_implementation=cfg.attn_implementation,
    )


def build_model(cfg: ModelConfig, eos_token_id: int = 0, pad_token_id: int = 1) -> nn.Module:
    """Instantiate a randomly initialized model (never ``from_pretrained``)."""
    cfg.validate()
    if cfg.architecture == "llama":
        return LlamaForCausalLM(llama_config(cfg, eos_token_id, pad_token_id))
    return ReferenceDecoder(cfg)


def load_model(path: str | Path) -> nn.Module:
    """Load a saved model directory written by either architecture."""
    path = Path(path)
    config_file = path / "config.json"
    if not config_file.exists():
        raise FileNotFoundError(f"No config.json in {path}; is this a saved model directory?")
    if (
        json.loads(config_file.read_text(encoding="utf-8")).get("tinylm_architecture")
        == "reference"
    ):
        return ReferenceDecoder.from_pretrained(path)
    return LlamaForCausalLM.from_pretrained(str(path))


def save_model(model: nn.Module, path: str | Path) -> None:
    model.save_pretrained(str(path))


def weights_are_tied(model: nn.Module) -> bool:
    """True when the LM head and the input embedding share one tensor."""
    head = model.lm_head.weight
    emb = model.model.embed_tokens.weight
    return head is emb or head.data_ptr() == emb.data_ptr()


# --------------------------------------------------------------------------- accounting


@dataclass
class ParameterReport:
    total: int
    trainable: int
    by_group: dict[str, int]
    by_layer: list[int]
    tied_embeddings: bool
    memory_mb: dict[str, float] = field(default_factory=dict)

    def format(self) -> str:
        lines = [f"{'Component':<34}{'Params':>14}{'Share':>9}", "-" * 57]
        for name, n in sorted(self.by_group.items(), key=lambda kv: -kv[1]):
            lines.append(f"{name:<34}{n:>14,}{100 * n / self.total:>8.1f}%")
        lines += [
            "-" * 57,
            f"{'Total (unique tensors)':<34}{self.total:>14,}",
            f"{'Trainable':<34}{self.trainable:>14,}",
            f"{'Per decoder layer':<34}{self.by_layer[0] if self.by_layer else 0:>14,}",
            f"LM head: {'tied to token embedding (0 extra params)' if self.tied_embeddings else 'separate'}",
            "Parameter memory: " + ", ".join(f"{k} {v:,.1f} MB" for k, v in self.memory_mb.items()),
        ]
        return "\n".join(lines)


def _group(name: str) -> str:
    if "embed_tokens" in name:
        return "Token embedding (tied LM head)"
    if "embed_positions" in name:
        return "Learned position embedding"
    if "lm_head" in name:
        return "LM head (untied)"
    if any(k in name for k in ("q_proj", "k_proj", "v_proj", "o_proj")):
        return "Attention"
    if any(k in name for k in ("gate_proj", "up_proj", "down_proj")):
        return "Feed-forward"
    if "norm" in name:
        return "Normalization"
    return "Other"


def parameter_report(model: nn.Module) -> ParameterReport:
    """Count parameters by component; shared (tied) tensors are counted once."""
    by_group: dict[str, int] = defaultdict(int)
    by_layer: dict[int, int] = defaultdict(int)
    total = trainable = 0
    seen: set[int] = set()
    for name, p in model.named_parameters(remove_duplicate=False):
        if id(p) in seen:
            continue
        seen.add(id(p))
        n = p.numel()
        total += n
        trainable += n if p.requires_grad else 0
        by_group[_group(name)] += n
        parts = name.split(".")
        if "layers" in parts:
            by_layer[int(parts[parts.index("layers") + 1])] += n
    return ParameterReport(
        total=total,
        trainable=trainable,
        by_group=dict(by_group),
        by_layer=[by_layer[i] for i in sorted(by_layer)],
        tied_embeddings=weights_are_tied(model),
        memory_mb={k: total * b / 1024**2 for k, b in BYTES_PER_DTYPE.items()},
    )


def training_memory_estimate_mb(n_params: int) -> dict[str, float]:
    """Static memory for fp32 master weights + grads + AdamW (two moments).

    Activations are excluded: they scale with batch * sequence * hidden * layers
    and are measured by the benchmark instead of estimated.
    """
    mb = n_params * 4 / 1024**2
    return {"weights": mb, "gradients": mb, "adamw_moments": 2 * mb, "total": 4 * mb}


def kv_cache_bytes(
    batch_size: int,
    seq_len: int,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    dtype_bytes: int = 2,
) -> int:
    """Memory of the K and V caches: 2 * layers * batch * seq * kv_heads * head_dim * bytes."""
    return 2 * num_layers * batch_size * seq_len * num_kv_heads * head_dim * dtype_bytes


def kv_cache_table(cfg: ModelConfig, batch_sizes=(1, 8, 64), dtype: str = "bf16") -> list[dict]:
    """KV-cache size at full context for GQA (as configured) vs. MHA."""
    rows = []
    for b in batch_sizes:
        gqa = kv_cache_bytes(
            b,
            cfg.max_position_embeddings,
            cfg.num_hidden_layers,
            cfg.num_key_value_heads,
            cfg.head_dim,
            BYTES_PER_DTYPE[dtype],
        )
        mha = kv_cache_bytes(
            b,
            cfg.max_position_embeddings,
            cfg.num_hidden_layers,
            cfg.num_attention_heads,
            cfg.head_dim,
            BYTES_PER_DTYPE[dtype],
        )
        rows.append(
            {
                "batch_size": b,
                "seq_len": cfg.max_position_embeddings,
                "gqa_mb": gqa / 1024**2,
                "mha_mb": mha / 1024**2,
                "ratio": mha / gqa,
            }
        )
    return rows


@torch.no_grad()
def initialization_loss(
    model: nn.Module, vocab_size: int, seq_len: int = 64, batch: int = 4, seed: int = 0
) -> tuple[float, float]:
    """Cross-entropy of the untrained model on uniform random tokens vs ln(vocab).

    A correctly initialized LM is near-uniform over the vocabulary, so its loss
    should be close to ln(V). A much larger value means overconfident logits
    (initialization scale too large).
    """
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, vocab_size, (batch, seq_len), generator=g)
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    loss = model(input_ids=ids.to(device), labels=ids.to(device)).loss.item()
    model.train(was_training)
    return loss, math.log(vocab_size)
