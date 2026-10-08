"""Typed, validated experiment configuration loaded from YAML.

A config file may ``extends:`` another file (path relative to itself); the child
is deep-merged over the parent. Any field can also be overridden from the CLI
with ``--set section.key=value``. The fully resolved config is written into
every run directory so a result can always be traced to its exact settings.
"""

from __future__ import annotations

import dataclasses
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml


class ConfigError(ValueError):
    """Raised for invalid or inconsistent configuration values."""


@dataclass
class ModelConfig:
    # "llama": Hugging Face LlamaForCausalLM (production model).
    # "reference": the pure-PyTorch decoder in tinylm.educational, whose
    # components can be switched individually for controlled ablations.
    architecture: Literal["llama", "reference"] = "llama"
    vocab_size: int = 8192
    hidden_size: int = 512
    num_hidden_layers: int = 8
    num_attention_heads: int = 8
    num_key_value_heads: int = 2
    intermediate_size: int = 1408
    max_position_embeddings: int = 512
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    initializer_range: float = 0.02
    tie_word_embeddings: bool = True
    attn_implementation: Literal["sdpa", "eager"] = "sdpa"
    # Component switches (only "reference" may deviate from the Llama recipe).
    position_encoding: Literal["rope", "learned"] = "rope"
    norm: Literal["rmsnorm", "layernorm"] = "rmsnorm"
    ffn: Literal["swiglu", "gelu"] = "swiglu"

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    def validate(self) -> None:
        if self.hidden_size % self.num_attention_heads:
            raise ConfigError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads})."
            )
        if self.num_attention_heads % self.num_key_value_heads:
            raise ConfigError(
                f"num_attention_heads ({self.num_attention_heads}) must be a multiple of "
                f"num_key_value_heads ({self.num_key_value_heads}) for grouped-query attention."
            )
        if self.position_encoding == "rope" and self.head_dim % 2:
            raise ConfigError(f"RoPE needs an even head_dim, got {self.head_dim}.")
        llama_recipe = (self.position_encoding, self.norm, self.ffn) == (
            "rope",
            "rmsnorm",
            "swiglu",
        )
        if self.architecture == "llama" and not llama_recipe:
            raise ConfigError(
                "architecture=llama always uses RoPE + RMSNorm + SwiGLU. "
                "Set model.architecture=reference to ablate individual components."
            )


@dataclass
class NormalizationConfig:
    unicode_form: Literal["NFC", "NFKC", "none"] = "NFKC"
    # Map curly quotes / dashes / ellipsis / NBSP to ASCII equivalents.
    ascii_punctuation: bool = True
    # Legacy run only: delete every remaining non-ASCII character. Byte-level
    # BPE can encode any byte, so this is off by default (see design decisions).
    strip_non_ascii: bool = False
    collapse_whitespace: bool = True


@dataclass
class DataConfig:
    dataset_name: str = "roneneldan/TinyStories"
    # Pin a Hub commit for exact reproducibility; null means "latest" and the
    # resolved revision is recorded in the prepared-data metadata.
    dataset_revision: str | None = None
    text_field: str = "text"
    # TinyStories ships only train + validation. The official validation split
    # becomes the untouched TEST set; VALIDATION is carved out of train by a
    # content hash so that exact duplicates can never straddle train/validation.
    source_train_split: str = "train"
    source_test_split: str = "validation"
    validation_fraction: float = 0.005
    min_chars: int = 100
    block_size: int = 512
    # Documents are concatenated with EOS separators and cut into full blocks.
    # Padding-based batching is not implemented; the flag documents the choice.
    packing: bool = True
    # Drop test stories whose normalized text has an exact copy in the official
    # train split. ~30 % of TinyStories' official validation split does
    # (measured, results/data_integrity.json), so without this the "held-out"
    # test set would partly be training data.
    decontaminate_test: bool = True
    # Optional subset of the (post-split) training stories for cheap runs.
    max_train_examples: int | None = None
    subset_seed: int = 1337
    data_dir: str = "data/tinystories"
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)

    def validate(self) -> None:
        if not 0 < self.validation_fraction < 0.5:
            raise ConfigError("data.validation_fraction must be in (0, 0.5).")
        if self.block_size < 8:
            raise ConfigError("data.block_size must be at least 8.")
        if not self.packing:
            raise ConfigError(
                "data.packing=false is not supported; only packed sequences are implemented."
            )
        if self.max_train_examples is not None and self.max_train_examples <= 0:
            raise ConfigError("data.max_train_examples must be positive or null.")


@dataclass
class TokenizerConfig:
    vocab_size: int = 8192
    min_frequency: int = 2
    eos_token: str = "<|endoftext|>"
    pad_token: str = "<|pad|>"
    # Train the BPE merges on at most this many training stories (null = all).
    max_training_examples: int | None = None
    path: str = "data/tokenizer-8k"


@dataclass
class TrainingConfig:
    seed: int = 1337
    output_dir: str = "runs/tiny-27m"
    max_steps: int = 12000
    per_device_train_batch_size: int = 24
    gradient_accumulation_steps: int = 3
    eval_batch_size: int = 24
    learning_rate: float = 6e-4
    min_lr_ratio: float = 0.0
    lr_scheduler: Literal["cosine", "constant"] = "cosine"
    warmup_steps: int = 200
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    gradient_clip_norm: float = 1.0
    precision: Literal["auto", "fp32", "fp16", "bf16"] = "auto"
    gradient_checkpointing: bool = False
    log_every: int = 20
    eval_every: int = 250
    # Cap validation batches during training (null = the whole validation split).
    eval_max_batches: int | None = None
    save_every: int = 250
    keep_last_checkpoints: int = 2
    deterministic: bool = False

    def validate(self) -> None:
        positive = (
            "max_steps",
            "per_device_train_batch_size",
            "gradient_accumulation_steps",
            "eval_batch_size",
            "log_every",
            "eval_every",
            "save_every",
        )
        for name in positive:
            if getattr(self, name) <= 0:
                raise ConfigError(f"training.{name} must be positive.")
        if self.warmup_steps >= self.max_steps:
            raise ConfigError("training.warmup_steps must be smaller than training.max_steps.")
        if not 0 <= self.min_lr_ratio <= 1:
            raise ConfigError("training.min_lr_ratio must be within [0, 1].")
        if self.save_every % self.eval_every:
            raise ConfigError(
                "training.save_every must be a multiple of training.eval_every so every "
                "checkpoint carries a validation result for model selection."
            )

    @property
    def sequences_per_step(self) -> int:
        return self.per_device_train_batch_size * self.gradient_accumulation_steps


@dataclass
class EvaluationConfig:
    prompts_path: str = "evaluation/prompts.json"
    presets: list[str] = field(default_factory=lambda: ["greedy", "balanced"])
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    max_new_tokens: int = 200


@dataclass
class ExperimentConfig:
    name: str = "tiny-27m"
    description: str = ""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def validate(self) -> ExperimentConfig:
        self.model.validate()
        self.data.validate()
        self.training.validate()
        if self.model.vocab_size != self.tokenizer.vocab_size:
            raise ConfigError(
                f"model.vocab_size ({self.model.vocab_size}) must equal "
                f"tokenizer.vocab_size ({self.tokenizer.vocab_size})."
            )
        if self.data.block_size > self.model.max_position_embeddings:
            raise ConfigError(
                f"data.block_size ({self.data.block_size}) exceeds "
                f"model.max_position_embeddings ({self.model.max_position_embeddings})."
            )
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8")


# --------------------------------------------------------------------------- loading


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _read_yaml_with_extends(path: Path, _seen: tuple[Path, ...] = ()) -> dict:
    path = path.resolve()
    if path in _seen:
        raise ConfigError(f"Circular 'extends' chain: {' -> '.join(map(str, (*_seen, path)))}")
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    parent = raw.pop("extends", None)
    if parent is None:
        return raw
    return _deep_merge(_read_yaml_with_extends(path.parent / parent, (*_seen, path)), raw)


def _build(cls: type, values: dict[str, Any], prefix: str) -> Any:
    """Instantiate a (possibly nested) dataclass, rejecting unknown keys."""
    known = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(values) - set(known)
    if unknown:
        raise ConfigError(
            f"Unknown config key(s) under '{prefix or 'root'}': {sorted(unknown)}. "
            f"Valid keys: {sorted(known)}"
        )
    kwargs = {}
    for name, value in values.items():
        default = known[name].default_factory  # type: ignore[misc]
        nested = default() if default is not dataclasses.MISSING else None
        if dataclasses.is_dataclass(nested) and isinstance(value, dict):
            kwargs[name] = _build(type(nested), value, f"{prefix}.{name}".strip("."))
        else:
            kwargs[name] = value
    return cls(**kwargs)


def apply_overrides(raw: dict, overrides: list[str] | None) -> dict:
    """Apply ``a.b.c=value`` overrides; values are parsed as YAML scalars."""
    for item in overrides or []:
        if "=" not in item:
            raise ConfigError(f"Override {item!r} must look like section.key=value")
        dotted, value = item.split("=", 1)
        node = raw
        *parents, leaf = dotted.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(value)
    return raw


def load_config(path: str | Path, overrides: list[str] | None = None) -> ExperimentConfig:
    """Load, merge, override and validate an experiment config."""
    raw = apply_overrides(_read_yaml_with_extends(Path(path)), overrides)
    return _build(ExperimentConfig, raw, "").validate()


def config_from_dict(raw: dict[str, Any]) -> ExperimentConfig:
    return _build(ExperimentConfig, raw, "").validate()
