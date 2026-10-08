"""Text generation: explicit decoding presets and the ``TinyLM`` convenience API.

Presets pass *only* the parameters their decoding mode uses: greedy decoding
never receives ``temperature``/``top_p`` (transformers would ignore them and
warn), and sampling presets set ``top_k`` explicitly because leaving it unset
makes transformers apply a default top-k of 50. Length is bounded by
``max_new_tokens`` alone; ``max_length`` is never set alongside it.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from transformers import GenerationConfig, PreTrainedModel, set_seed

from tinylm.utils import read_json, resolve_device

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GenerationPreset:
    name: str
    do_sample: bool
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    repetition_penalty: float | None = None

    def generation_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"do_sample": self.do_sample}
        if self.do_sample:
            kwargs.update(
                temperature=self.temperature if self.temperature is not None else 1.0,
                top_p=self.top_p if self.top_p is not None else 1.0,
                top_k=self.top_k if self.top_k is not None else 0,  # 0 disables top-k
            )
        if self.repetition_penalty is not None:
            kwargs["repetition_penalty"] = self.repetition_penalty
        return kwargs


PRESETS: dict[str, GenerationPreset] = {
    p.name: p
    for p in (
        GenerationPreset("greedy", do_sample=False),
        GenerationPreset("low_temperature", do_sample=True, temperature=0.5),
        GenerationPreset("balanced", do_sample=True, temperature=0.8, top_p=0.9),
        GenerationPreset("creative", do_sample=True, temperature=1.1, top_p=0.95),
        GenerationPreset("top_k", do_sample=True, temperature=0.8, top_k=40),
        GenerationPreset("top_p", do_sample=True, temperature=1.0, top_p=0.9),
    )
}


def get_preset(name: str) -> GenerationPreset:
    if name not in PRESETS:
        raise KeyError(f"Unknown preset {name!r}; choose from {sorted(PRESETS)}")
    return PRESETS[name]


def custom_preset(
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    repetition_penalty: float | None = None,
) -> GenerationPreset:
    """Build a preset from user parameters; ``temperature == 0`` means greedy."""
    if temperature is not None and temperature <= 0:
        return GenerationPreset(
            "custom_greedy", do_sample=False, repetition_penalty=repetition_penalty
        )
    return GenerationPreset(
        "custom",
        True,
        temperature if temperature is not None else 1.0,
        top_p,
        top_k,
        repetition_penalty,
    )


def build_generation_config(
    preset: GenerationPreset,
    tokenizer,
    max_new_tokens: int,
    use_cache: bool = True,
    min_new_tokens: int | None = None,
) -> GenerationConfig:
    kwargs = preset.generation_kwargs()
    if min_new_tokens is not None:
        kwargs["min_new_tokens"] = min_new_tokens
    return GenerationConfig(
        max_new_tokens=max_new_tokens,
        use_cache=use_cache,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        **kwargs,
    )


@dataclass
class GenerationResult:
    prompt: str
    completion: str  # newly generated text only
    text: str  # prompt + completion
    prompt_tokens: int
    new_tokens: int
    finished_with_eos: bool
    preset: str
    seed: int | None
    settings: dict[str, Any] = field(default_factory=dict)
    token_ids: list[int] = field(default_factory=list, repr=False)  # new tokens only

    def to_dict(self, include_ids: bool = False) -> dict[str, Any]:
        d = asdict(self)
        if not include_ids:
            d.pop("token_ids")
        return d


@torch.no_grad()
def generate(
    model: PreTrainedModel,
    tokenizer,
    prompt: str,
    preset: GenerationPreset | str = "balanced",
    max_new_tokens: int = 200,
    seed: int | None = None,
    prepend_bos: bool = True,
    use_cache: bool = True,
) -> GenerationResult:
    """Generate one continuation of ``prompt``.

    ``prepend_bos`` places an EOS/BOS token before the prompt: in the packed
    training stream every story starts right after an EOS token, so this
    conditions the model on "a new story starts here".
    """
    if not hasattr(model, "generate"):
        raise TypeError(
            "generate() needs a Hugging Face model; ReferenceDecoder has generate_greedy()."
        )
    preset = get_preset(preset) if isinstance(preset, str) else preset
    ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    if prepend_bos:
        ids = [tokenizer.bos_token_id, *ids]
    max_ctx = model.config.max_position_embeddings
    if len(ids) >= max_ctx:
        raise ValueError(
            f"Prompt is {len(ids)} tokens; the model's context is {max_ctx} tokens. "
            "Shorten the prompt."
        )
    if len(ids) + max_new_tokens > max_ctx:
        new_budget = max(0, max_ctx - len(ids))
        logger.warning(
            "Prompt (%d tokens) + max_new_tokens (%d) exceeds the %d-token context; "
            "limiting to %d new tokens.",
            len(ids),
            max_new_tokens,
            max_ctx,
            new_budget,
        )
        max_new_tokens = new_budget
    gen_cfg = build_generation_config(preset, tokenizer, max_new_tokens, use_cache=use_cache)
    if seed is not None:
        set_seed(seed)
    device = next(model.parameters()).device
    input_ids = torch.tensor([ids], device=device)
    out = model.generate(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        generation_config=gen_cfg,
    )
    new = out[0, input_ids.shape[1] :].tolist()
    finished = bool(new) and new[-1] == tokenizer.eos_token_id
    completion = tokenizer.decode(new, skip_special_tokens=True)
    return GenerationResult(
        prompt=prompt,
        completion=completion,
        text=prompt + completion,  # byte-level BPE keeps the leading space of the continuation
        prompt_tokens=len(ids),
        new_tokens=len(new),
        finished_with_eos=finished,
        preset=preset.name,
        seed=seed,
        settings={
            **preset.generation_kwargs(),
            "max_new_tokens": max_new_tokens,
            "prepend_bos": prepend_bos,
            "use_cache": use_cache,
        },
        token_ids=new,
    )


class TinyLM:
    """Load a trained model directory (or Hub id) and generate stories.

    >>> lm = TinyLM.from_pretrained("runs/tiny-27m/final")      # doctest: +SKIP
    >>> lm.generate("Once upon a time", max_new_tokens=100, temperature=0.8, seed=42)
    """

    def __init__(self, model: PreTrainedModel, tokenizer, source: str | None = None):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.source = source

    @classmethod
    def from_pretrained(
        cls, path_or_id: str | Path, device: str = "auto", dtype: str | None = None
    ) -> TinyLM:
        from transformers import AutoModelForCausalLM

        from tinylm.tokenizer import load_tokenizer
        from tinylm.utils import DTYPES

        device_t = resolve_device(device)
        torch_dtype = DTYPES[dtype] if dtype else None
        model = AutoModelForCausalLM.from_pretrained(str(path_or_id), dtype=torch_dtype)
        if Path(path_or_id).exists():
            tokenizer = load_tokenizer(path_or_id)
        else:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                str(path_or_id), clean_up_tokenization_spaces=False
            )
        return cls(model.to(device_t), tokenizer, str(path_or_id))

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 200,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        repetition_penalty: float | None = None,
        preset: str | None = None,
        seed: int | None = None,
    ) -> str:
        """Return prompt + continuation. Without sampling arguments, uses ``balanced``."""
        return self.generate_full(
            prompt, max_new_tokens, temperature, top_p, top_k, repetition_penalty, preset, seed
        ).text

    def generate_full(
        self,
        prompt: str,
        max_new_tokens: int = 200,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        repetition_penalty: float | None = None,
        preset: str | None = None,
        seed: int | None = None,
    ) -> GenerationResult:
        sampling_args = (temperature, top_p, top_k, repetition_penalty)
        if preset is not None and any(a is not None for a in sampling_args):
            raise ValueError("Pass either a preset or explicit sampling arguments, not both.")
        chosen = (
            get_preset(preset)
            if preset
            else custom_preset(*sampling_args)
            if any(a is not None for a in sampling_args)
            else PRESETS["balanced"]
        )
        return generate(self.model, self.tokenizer, prompt, chosen, max_new_tokens, seed)

    @property
    def metadata(self) -> dict[str, Any]:
        from tinylm.modeling import parameter_report

        cfg = self.model.config
        meta = {
            "source": self.source,
            "parameters": parameter_report(self.model).total,
            "layers": cfg.num_hidden_layers,
            "hidden_size": cfg.hidden_size,
            "attention_heads": cfg.num_attention_heads,
            "kv_heads": cfg.num_key_value_heads,
            "context_length": cfg.max_position_embeddings,
            "vocab_size": cfg.vocab_size,
            "device": str(next(self.model.parameters()).device),
        }
        if self.source and (Path(self.source) / "tinylm_metadata.json").exists():
            meta["training"] = read_json(Path(self.source) / "tinylm_metadata.json")
        return meta
