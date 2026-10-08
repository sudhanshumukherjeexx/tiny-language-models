"""Pretraining loop: AdamW, warmup + cosine LR, mixed precision, accumulation,
clipping, validation-based model selection, and exact resume.

Why a hand-written loop instead of ``transformers.Trainer``? The project needs
three things Trainer does not provide out of the box: (1) experiment-level
metrics that accumulate correctly across resumes (a resumed Trainer reports the
*last invocation's* runtime and an averaged loss that is meaningless when it ran
for a handful of steps), (2) a refusal to resume when the configuration has
silently changed, and (3) a data order that is a pure function of
(seed, sequences consumed), which makes interrupted-vs-uninterrupted training
testable. The loop is ~250 lines; every technique is visible.

Data order: sequences are visited in a fresh seeded permutation per epoch.
The position in that order is ``ExperimentState.sequences_seen``, so a resumed
run consumes exactly the batches the uninterrupted run would have.
"""

from __future__ import annotations

import logging
import math
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from tinylm.checkpointing import (
    ExperimentState,
    export_model,
    latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from tinylm.config import ExperimentConfig, TrainingConfig
from tinylm.data import PackedDataset
from tinylm.evaluation import evaluate_loss
from tinylm.modeling import parameter_report
from tinylm.utils import (
    append_jsonl,
    autocast_context,
    environment_info,
    git_commit,
    hardware_name,
    read_jsonl,
    resolve_precision,
    seed_everything,
    write_json,
)

logger = logging.getLogger(__name__)

# Fields that may legitimately differ when resuming (e.g. a smaller micro-batch
# with more accumulation after an OOM keeps the optimization identical).
RESUMABLE_CHANGES = {
    "training.per_device_train_batch_size",
    "training.gradient_accumulation_steps",
    "training.eval_batch_size",
    "training.gradient_checkpointing",
    "training.log_every",
    "training.keep_last_checkpoints",
    "training.output_dir",
    "description",
}


# --------------------------------------------------------------------------- pieces


def lr_multiplier(step: int, cfg: TrainingConfig) -> float:
    """Linear warmup to the peak LR, then cosine decay to ``min_lr_ratio`` * peak."""
    if step < cfg.warmup_steps:
        return (step + 1) / cfg.warmup_steps
    if cfg.lr_scheduler == "constant":
        return 1.0
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return cfg.min_lr_ratio + (1.0 - cfg.min_lr_ratio) * cosine


def build_optimizer(model: torch.nn.Module, cfg: TrainingConfig, device: torch.device):
    """AdamW with weight decay on matrices only (not norms, biases)."""
    decay, no_decay, seen = [], [], set()
    for p in model.parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        (decay if p.ndim >= 2 else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(
        groups,
        lr=cfg.learning_rate,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        eps=cfg.adam_eps,
        fused=device.type == "cuda",
    )


def batch_indices(n_sequences: int, start: int, batch_size: int, seed: int) -> np.ndarray:
    """Indices of the batch beginning at global position ``start`` of the data order."""
    out = []
    pos = start
    while len(out) < batch_size:
        epoch, offset = divmod(pos, n_sequences)
        perm = _epoch_permutation(n_sequences, seed, epoch)
        take = min(batch_size - len(out), n_sequences - offset)
        out.extend(perm[offset : offset + take])
        pos += take
    return np.asarray(out, dtype=np.int64)


_PERM_CACHE: dict[tuple[int, int, int], np.ndarray] = {}


def _epoch_permutation(n: int, seed: int, epoch: int) -> np.ndarray:
    key = (n, seed, epoch)
    if key not in _PERM_CACHE:
        _PERM_CACHE.clear()  # keep at most one epoch in memory
        _PERM_CACHE[key] = np.random.default_rng([seed, epoch]).permutation(n)
    return _PERM_CACHE[key]


def _flatten(d: dict, prefix: str = "") -> dict[str, Any]:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


def config_diff(old: dict, new: dict) -> dict[str, tuple[Any, Any]]:
    a, b = _flatten(old), _flatten(new)
    return {k: (a.get(k), b.get(k)) for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)}


# --------------------------------------------------------------------------- trainer


class PretrainingRun:
    """One experiment: owns the run directory, model, optimizer and bookkeeping."""

    def __init__(
        self,
        cfg: ExperimentConfig,
        model: torch.nn.Module,
        train_data: PackedDataset,
        val_data: PackedDataset,
        device: torch.device,
        tokenizer=None,
        run_metadata: dict[str, Any] | None = None,
    ):
        self.cfg, self.tcfg = cfg, cfg.training
        self.model = model.to(device)
        self.train_data, self.val_data = train_data, val_data
        self.device = device
        self.tokenizer = tokenizer
        self.run_metadata = run_metadata or {}
        self.out = Path(self.tcfg.output_dir)
        self.precision = resolve_precision(self.tcfg.precision, device)
        self.optimizer = build_optimizer(model, self.tcfg, device)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda s: lr_multiplier(s, self.tcfg)
        )
        self.scaler = torch.amp.GradScaler("cuda") if self.precision == "fp16" else None
        if self.tcfg.gradient_checkpointing:
            if not hasattr(model, "gradient_checkpointing_enable"):
                raise ValueError("gradient_checkpointing is only supported for architecture=llama")
            model.gradient_checkpointing_enable()
            model.config.use_cache = False
        self.state: ExperimentState | None = None

    # -- setup / resume ------------------------------------------------------

    def _start(self, resume: str | bool) -> ExperimentState:
        self.out.mkdir(parents=True, exist_ok=True)
        cfg_path = self.out / "config.yaml"
        ckpt = latest_checkpoint(self.out) if resume else None
        if isinstance(resume, (str, Path)) and resume not in ("auto", True):
            ckpt = Path(resume)
        new_cfg = self.cfg.to_dict()

        if ckpt is None:
            if (self.out / "metrics.json").exists() or (
                self.out / "training_history.jsonl"
            ).exists():
                raise FileExistsError(
                    f"{self.out} already contains a run but no checkpoint to resume from. "
                    "Use a new training.output_dir or delete the old run."
                )
            seed_everything(self.tcfg.seed)
            self.cfg.save(cfg_path)
            run_id = (
                f"{self.cfg.name}-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
            )
            return ExperimentState(run_id=run_id)

        old_cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        diff = {
            k: v for k, v in config_diff(old_cfg, new_cfg).items() if k not in RESUMABLE_CHANGES
        }
        if diff:
            raise RuntimeError(
                "Refusing to resume: the configuration changed in ways that alter the "
                f"experiment: {diff}. Start a new output_dir, or revert the change."
            )
        state = load_checkpoint(ckpt, self.model, self.optimizer, self.scheduler, self.scaler)
        logger.info(
            "Resumed %s from %s (step %d, %s tokens, best val %.4f)",
            state.run_id,
            ckpt.name,
            state.global_step,
            f"{state.tokens_seen:,}",
            state.best_val_loss or float("nan"),
        )
        # Drop history lines logged after this checkpoint: those steps will be redone.
        hist = self.out / "training_history.jsonl"
        kept = [r for r in read_jsonl(hist) if r["step"] <= state.global_step]
        hist.unlink(missing_ok=True)
        if kept:
            append_jsonl(hist, kept)
        self.cfg.save(cfg_path)
        return state

    # -- main loop -------------------------------------------------------------

    def train(
        self, resume: str | bool = "auto", stop_after_steps: int | None = None
    ) -> dict[str, Any]:
        """Train to ``max_steps``, or for at most ``stop_after_steps`` steps in this
        invocation (then checkpoint and return; the next call resumes exactly)."""
        state = self.state = self._start(resume)
        t = self.tcfg
        if state.global_step >= t.max_steps:
            logger.info(
                "Experiment already completed %d steps; nothing to train.", state.global_step
            )
            return self._finalize()
        stop_at = (
            t.max_steps
            if stop_after_steps is None
            else min(t.max_steps, state.global_step + stop_after_steps)
        )
        bs, accum, block = (
            t.per_device_train_batch_size,
            t.gradient_accumulation_steps,
            self.train_data.block_size,
        )
        n_train = len(self.train_data)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        self._inv = {
            "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "start_step": state.global_step,
            "hardware": hardware_name(self.device),
            "precision": self.precision,
            "micro_batch": bs,
            "grad_accum": accum,
            "git_commit": git_commit(),
            "train_seconds": 0.0,
            "tokens": 0,
        }
        self._inv_start = time.perf_counter()
        write_json(self.out / "environment.json", environment_info(self.device))
        logger.info(
            "Training %s on %s (%s), steps %d -> %d, %d sequences x %d tokens per step",
            state.run_id,
            self._inv["hardware"],
            self.precision,
            state.global_step,
            stop_at,
            bs * accum,
            block,
        )

        window_loss, window_steps, window_tokens, window_time = 0.0, 0, 0, 0.0
        self.model.train()
        try:
            while state.global_step < stop_at:
                step_start = time.perf_counter()
                step_loss = 0.0
                for _ in range(accum):
                    idx = batch_indices(n_train, state.sequences_seen, bs, t.seed)
                    ids = self.train_data.batch(idx).to(self.device, non_blocking=True)
                    with autocast_context(self.device, self.precision):
                        loss = self.model(input_ids=ids, labels=ids).loss
                    scaled = loss / accum
                    (self.scaler.scale(scaled) if self.scaler else scaled).backward()
                    step_loss += loss.detach().float().item() / accum
                    state.sequences_seen += bs
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), t.gradient_clip_norm
                )
                lr = self.scheduler.get_last_lr()[0]
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                dt = time.perf_counter() - step_start

                tokens = bs * accum * block
                state.global_step += 1
                state.tokens_seen += tokens
                state.train_seconds += dt
                self._inv["train_seconds"] += dt
                self._inv["tokens"] += tokens
                window_loss += step_loss
                window_steps += 1
                window_tokens += tokens
                window_time += dt
                if not math.isfinite(step_loss):
                    raise FloatingPointError(
                        f"Non-finite training loss at step {state.global_step}"
                    )
                step = state.global_step
                last = step == stop_at  # end of run, or end of this invocation

                if step % t.log_every == 0 or last:
                    state.last_train_loss = window_loss / window_steps
                    self._log(
                        {
                            "step": state.global_step,
                            "train_loss": state.last_train_loss,
                            "lr": lr,
                            "grad_norm": float(grad_norm),
                            "tokens_seen": state.tokens_seen,
                            "tokens_per_second": window_tokens / window_time,
                            "train_seconds": state.train_seconds,
                        }
                    )
                    window_loss, window_steps, window_tokens, window_time = 0.0, 0, 0, 0.0

                if step % t.eval_every == 0 or step == t.max_steps:
                    self._evaluate_and_select()
                if step % t.save_every == 0 or last:
                    self._record_memory()
                    status = (
                        "completed" if step == t.max_steps else ("stopped" if last else "running")
                    )
                    save_checkpoint(
                        self.out,
                        self.model,
                        self.optimizer,
                        self.scheduler,
                        self.scaler,
                        self._snapshot(status),
                        t.keep_last_checkpoints,
                    )
        except KeyboardInterrupt:
            logger.warning(
                "Interrupted at step %d; resume from the last checkpoint.", state.global_step
            )
            raise
        self._record_memory()
        final = self._snapshot("completed" if state.global_step >= t.max_steps else "stopped")
        self.state = final
        return self._finalize()

    # -- helpers ---------------------------------------------------------------

    def _log(self, record: dict[str, Any]) -> None:
        append_jsonl(self.out / "training_history.jsonl", record)
        if "train_loss" in record:
            logger.info(
                "step %6d | loss %.4f | lr %.2e | gnorm %.2f | %.0f tok/s",
                record["step"],
                record["train_loss"],
                record["lr"],
                record["grad_norm"],
                record["tokens_per_second"],
            )

    def _evaluate_and_select(self) -> None:
        state = self.state
        result = evaluate_loss(
            self.model,
            self.val_data,
            self.tcfg.eval_batch_size,
            self.device,
            self.precision,
            self.tcfg.eval_max_batches,
            "validation",
        )
        improved = state.best_val_loss is None or result.loss < state.best_val_loss
        self._log(
            {
                "step": state.global_step,
                "val_loss": result.loss,
                "val_perplexity": result.perplexity,
                "val_predicted_tokens": result.predicted_tokens,
            }
        )
        logger.info(
            "step %6d | val loss %.4f | val ppl %.3f%s",
            state.global_step,
            result.loss,
            result.perplexity,
            "  (best)" if improved else "",
        )
        if improved:
            state.best_val_loss, state.best_step = result.loss, state.global_step
            export_model(
                self.model,
                self.tokenizer,
                self.out / "best",
                {
                    "selected_by": "validation_loss",
                    "step": state.global_step,
                    "validation_loss": result.loss,
                    "run_id": state.run_id,
                    **self.run_metadata,
                },
            )

    def _record_memory(self) -> None:
        if self.device.type != "cuda":
            return
        alloc = torch.cuda.max_memory_allocated(self.device) / 1024**2
        reserved = torch.cuda.max_memory_reserved(self.device) / 1024**2
        s = self.state
        s.peak_vram_allocated_mb = max(alloc, s.peak_vram_allocated_mb or 0.0)
        s.peak_vram_reserved_mb = max(reserved, s.peak_vram_reserved_mb or 0.0)

    def _snapshot(self, status: str) -> ExperimentState:
        """Experiment state including the current invocation so far.

        The live state keeps only *finished* invocations; each checkpoint stores
        a copy with the in-progress invocation appended, so a run that crashes
        later is still accounted for up to its last checkpoint.
        """
        wall = time.perf_counter() - self._inv_start
        inv = self._inv | {
            "end_step": self.state.global_step,
            "status": status,
            "wall_clock_seconds": wall,
            "tokens_per_second": self._inv["tokens"] / self._inv["train_seconds"]
            if self._inv["train_seconds"]
            else None,
        }
        snap = ExperimentState.from_dict(self.state.to_dict())
        snap.invocations = [*snap.invocations, inv]
        snap.wall_clock_seconds = sum(i["wall_clock_seconds"] for i in snap.invocations)
        return snap

    def _finalize(self) -> dict[str, Any]:
        s = self.state
        params = parameter_report(self.model).total
        best_dir = self.out / "best"
        metrics = {
            "run_id": s.run_id,
            "name": self.cfg.name,
            "status": "completed" if s.global_step >= self.tcfg.max_steps else "partial",
            "git_commit": git_commit(),
            "seed": self.tcfg.seed,
            "hardware": sorted({i["hardware"] for i in s.invocations}),
            "precision": self.precision,
            "parameters": params,
            "steps": s.global_step,
            "training_tokens": s.tokens_seen,
            "tokens_per_parameter": s.tokens_seen / params,
            "wall_clock_seconds": s.wall_clock_seconds,
            "train_seconds": s.train_seconds,
            "tokens_per_second": s.tokens_seen / s.train_seconds if s.train_seconds else None,
            "final_train_loss": s.last_train_loss,
            "best_validation_loss": s.best_val_loss,
            "best_validation_perplexity": math.exp(s.best_val_loss)
            if s.best_val_loss is not None
            else None,
            "best_step": s.best_step,
            "best_checkpoint": str(best_dir) if best_dir.exists() else None,
            "peak_vram_allocated_mb": s.peak_vram_allocated_mb,
            "peak_vram_reserved_mb": s.peak_vram_reserved_mb,
            "invocations": s.invocations,
            **self.run_metadata,
        }
        write_json(self.out / "metrics.json", metrics)
        current = s.invocations[-1]
        logger.info(
            "This invocation: steps %d->%d, %.1f s, %s tokens. Full experiment: %d steps, "
            "%.1f s training time, %s tokens, best val loss %.4f at step %s.",
            current["start_step"],
            current["end_step"],
            current["wall_clock_seconds"],
            f"{current['tokens']:,}",
            s.global_step,
            s.train_seconds,
            f"{s.tokens_seen:,}",
            s.best_val_loss or float("nan"),
            s.best_step,
        )
        return metrics


def train_from_config(
    cfg: ExperimentConfig,
    resume: str | bool = "auto",
    device: str = "auto",
    stop_after_steps: int | None = None,
) -> dict[str, Any]:
    """Load tokenizer + packed data, build a randomly initialized model, and train."""
    from tinylm.data import load_packed_split
    from tinylm.modeling import build_model, initialization_loss
    from tinylm.tokenizer import load_tokenizer, tokenizer_fingerprint
    from tinylm.utils import read_json, resolve_device

    dev = resolve_device(device)
    tokenizer = load_tokenizer(cfg.tokenizer.path)
    sha = tokenizer_fingerprint(cfg.tokenizer.path)
    if len(tokenizer) > cfg.model.vocab_size:
        raise ValueError(
            f"Tokenizer has {len(tokenizer)} tokens but model.vocab_size={cfg.model.vocab_size}."
        )
    train = load_packed_split(cfg.data.data_dir, "train", sha)
    val = load_packed_split(cfg.data.data_dir, "validation", sha)
    if train.block_size != cfg.data.block_size:
        raise ValueError(
            f"Packed block size {train.block_size} != data.block_size {cfg.data.block_size}; "
            "re-run scripts/prepare_data.py --force."
        )
    data_meta = read_json(Path(cfg.data.data_dir) / "meta.json")

    seed_everything(cfg.training.seed)  # before init, so weights depend only on the seed
    model = build_model(cfg.model, tokenizer.eos_token_id, tokenizer.pad_token_id)
    report = parameter_report(model)
    logger.info("Model parameters:\n%s", report.format())
    init_loss, ln_v = initialization_loss(model, cfg.model.vocab_size)
    logger.info("Initialization check: loss %.4f vs ln(vocab) %.4f", init_loss, ln_v)
    if abs(init_loss - ln_v) > 0.5:
        raise RuntimeError("Untrained loss is far from ln(vocab): check initializer_range.")
    plan = training_summary(cfg, len(train), train.block_size)
    logger.info(
        "Plan: %s tokens/step x %d steps = %s tokens (%.2f epochs of %s packed tokens)",
        f"{plan['tokens_per_step']:,}",
        plan["max_steps"],
        f"{plan['planned_training_tokens']:,}",
        plan["epochs"],
        f"{train.num_tokens:,}",
    )

    run = PretrainingRun(
        cfg,
        model,
        train,
        val,
        dev,
        tokenizer,
        run_metadata={
            "dataset": data_meta["dataset"],
            "dataset_revision": data_meta["dataset_revision"],
            "tokenizer_sha256": sha,
            "train_split_tokens": train.num_tokens,
            "planned_epochs": plan["epochs"],
            "initialization_loss": init_loss,
            "architecture": cfg.model.architecture,
        },
    )
    return run.train(resume, stop_after_steps)


def training_summary(
    cfg: ExperimentConfig, n_train_sequences: int, block_size: int
) -> dict[str, Any]:
    """Token budget implied by a config, printed before training starts."""
    t = cfg.training
    tokens_per_step = t.sequences_per_step * block_size
    total = tokens_per_step * t.max_steps
    return {
        "sequences_per_step": t.sequences_per_step,
        "tokens_per_step": tokens_per_step,
        "max_steps": t.max_steps,
        "planned_training_tokens": total,
        "epochs": total / (n_train_sequences * block_size),
        "training_config": asdict(t),
    }
