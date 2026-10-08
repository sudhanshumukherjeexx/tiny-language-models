"""Checkpoints that make resumed training continue exactly where it stopped.

A checkpoint directory ``checkpoints/step-000250/`` holds:

* ``model.safetensors``       weights (tied tensors stored once)
* ``training_state.pt``       optimizer, LR scheduler, grad scaler, RNG states
* ``experiment_state.json``   cumulative experiment bookkeeping (see below)

``ExperimentState`` is what separates *full-experiment* metrics from the
metrics of a single process invocation: steps, tokens, wall-clock time, best
validation loss and peak memory accumulate across every resume, and each
invocation is recorded individually in ``invocations``.

Directories are written to a temporary name and renamed into place, so an
interruption during saving can never leave a half-written "latest" checkpoint.
"""

from __future__ import annotations

import logging
import os
import random
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_model, save_model
from transformers import GenerationConfig

from tinylm.utils import read_json, write_json

logger = logging.getLogger(__name__)

CHECKPOINT_PREFIX = "step-"


@dataclass
class ExperimentState:
    run_id: str
    global_step: int = 0
    sequences_seen: int = 0  # position in the deterministic data order
    tokens_seen: int = 0
    train_seconds: float = 0.0  # time inside optimizer steps, summed over invocations
    wall_clock_seconds: float = 0.0  # including evaluation and checkpoint IO
    best_val_loss: float | None = None
    best_step: int | None = None
    last_train_loss: float | None = None
    peak_vram_allocated_mb: float | None = None
    peak_vram_reserved_mb: float | None = None
    invocations: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ExperimentState:
        return cls(**d)


# --------------------------------------------------------------------------- RNG


def capture_rng_state() -> dict[str, Any]:
    np_state = np.random.get_state()
    state: dict[str, Any] = {
        "python": random.getstate(),
        # numpy's state holds an ndarray; store it as a tensor so the checkpoint
        # loads with torch.load(weights_only=True).
        "numpy": (np_state[0], torch.from_numpy(np_state[1].copy()), *np_state[2:]),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    name, keys, *rest = state["numpy"]
    np.random.set_state((name, keys.numpy(), *rest))
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(state["cuda"])
        except RuntimeError as err:  # e.g. resumed on a machine with fewer GPUs
            logger.warning("CUDA RNG state not restored (%s); sampling may differ.", err)


# --------------------------------------------------------------------------- save / load


def checkpoint_dir(output_dir: str | Path, step: int) -> Path:
    return Path(output_dir) / "checkpoints" / f"{CHECKPOINT_PREFIX}{step:06d}"


def list_checkpoints(output_dir: str | Path) -> list[Path]:
    root = Path(output_dir) / "checkpoints"
    if not root.exists():
        return []
    found = [
        p
        for p in root.iterdir()
        if p.is_dir()
        and p.name.startswith(CHECKPOINT_PREFIX)
        and (p / "experiment_state.json").exists()
    ]
    return sorted(found, key=lambda p: int(p.name[len(CHECKPOINT_PREFIX) :]))


def latest_checkpoint(output_dir: str | Path) -> Path | None:
    ckpts = list_checkpoints(output_dir)
    return ckpts[-1] if ckpts else None


def save_checkpoint(
    output_dir: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler | None,
    state: ExperimentState,
    keep_last: int | None = None,
) -> Path:
    final = checkpoint_dir(output_dir, state.global_step)
    tmp = final.with_name(final.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    save_model(model, str(tmp / "model.safetensors"))
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "rng": capture_rng_state(),
        },
        tmp / "training_state.pt",
    )
    write_json(tmp / "experiment_state.json", state.to_dict())
    if final.exists():
        shutil.rmtree(final)
    os.replace(tmp, final)
    if keep_last:
        for old in list_checkpoints(output_dir)[:-keep_last]:
            shutil.rmtree(old)
    return final


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: torch.amp.GradScaler | None = None,
    restore_rng: bool = True,
) -> ExperimentState:
    """Restore everything saved by :func:`save_checkpoint` into live objects."""
    path = Path(path)
    # strict=True: every parameter must be present; tied tensors are handled by
    # safetensors' shared-tensor bookkeeping, so no spurious "missing lm_head".
    load_model(model, str(path / "model.safetensors"), strict=True)
    training_state = torch.load(path / "training_state.pt", map_location="cpu", weights_only=True)
    if optimizer is not None:
        optimizer.load_state_dict(training_state["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(training_state["scheduler"])
    if scaler is not None and training_state["scaler"] is not None:
        scaler.load_state_dict(training_state["scaler"])
    if restore_rng:
        restore_rng_state(training_state["rng"])
    return ExperimentState.from_dict(read_json(path / "experiment_state.json"))


# --------------------------------------------------------------------------- export


def default_generation_config(tokenizer, max_new_tokens: int = 200) -> GenerationConfig:
    """Generation defaults stored with exported models: special tokens and length only.

    Sampling parameters are deliberately *not* stored. transformers merges the
    saved config into every ``generate`` call, so saved temperature/top-p values
    would leak into greedy requests (and trigger "flags are not valid" warnings).
    Decoding is chosen per call via ``tinylm.generation.PRESETS``. Length is
    bounded by ``max_new_tokens`` only, never alongside ``max_length``.
    """
    return GenerationConfig(
        max_new_tokens=max_new_tokens,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )


def export_model(
    model: torch.nn.Module, tokenizer, path: str | Path, metadata: dict[str, Any] | None = None
) -> Path:
    """Save a self-contained, ``from_pretrained``-loadable model directory."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    model.save_pretrained(str(tmp))
    if tokenizer is not None:
        tokenizer.save_pretrained(str(tmp))
        if hasattr(model, "generation_config"):
            default_generation_config(tokenizer).save_pretrained(str(tmp))
    if metadata:
        write_json(tmp / "tinylm_metadata.json", metadata)
    if path.exists():
        shutil.rmtree(path)
    os.replace(tmp, path)
    return path
