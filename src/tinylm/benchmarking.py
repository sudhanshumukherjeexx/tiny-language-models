"""Throughput, latency and memory benchmarks, recorded per *measured* hardware.

Training benchmarks use uniformly random token ids: the cost of a forward and
backward pass does not depend on token values, and this keeps the benchmark
independent of the dataset. Every record carries the hardware it ran on;
numbers from one GPU are never reported for another.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

import torch

from tinylm.config import ModelConfig, TrainingConfig
from tinylm.modeling import build_model, parameter_report
from tinylm.training import build_optimizer
from tinylm.utils import (
    DTYPES,
    autocast_context,
    environment_info,
    hardware_name,
    resolve_precision,
)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _reset_peak(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def _peaks(device: torch.device) -> dict[str, float | None]:
    if device.type != "cuda":
        return {"peak_allocated_mb": None, "peak_reserved_mb": None}
    return {
        "peak_allocated_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
        "peak_reserved_mb": torch.cuda.max_memory_reserved(device) / 1024**2,
    }


def _timed(fn: Callable[[], Any], device: torch.device, repeats: int, warmup: int) -> list[float]:
    for _ in range(warmup):
        fn()
    _sync(device)
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        _sync(device)
        times.append(time.perf_counter() - start)
    return times


@dataclass
class TrainBenchResult:
    hardware: str
    precision: str
    micro_batch: int
    grad_accum: int
    seq_len: int
    gradient_checkpointing: bool
    step_time_s_mean: float
    step_time_s_std: float
    tokens_per_second: float
    sequences_per_second: float
    peak_allocated_mb: float | None
    peak_reserved_mb: float | None
    parameters: int
    measured_steps: int


def benchmark_training(
    model_cfg: ModelConfig,
    device: torch.device,
    precision: str = "auto",
    micro_batch: int = 8,
    grad_accum: int = 1,
    seq_len: int = 512,
    steps: int = 10,
    warmup: int = 3,
    gradient_checkpointing: bool = False,
) -> TrainBenchResult:
    """Time full optimizer steps (forward, backward, clip, AdamW) on random tokens."""
    precision = resolve_precision(precision, device)
    torch.manual_seed(0)
    model = build_model(model_cfg).to(device).train()
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
    tcfg = TrainingConfig()
    optimizer = build_optimizer(model, tcfg, device)
    scaler = torch.amp.GradScaler("cuda") if precision == "fp16" else None
    ids = torch.randint(0, model_cfg.vocab_size, (micro_batch, seq_len), device=device)

    def step() -> None:
        for _ in range(grad_accum):
            with autocast_context(device, precision):
                loss = model(input_ids=ids, labels=ids).loss / grad_accum
            (scaler.scale(loss) if scaler else loss).backward()
        if scaler:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    _reset_peak(device)
    times = _timed(step, device, steps, warmup)
    mean = statistics.fmean(times)
    seqs = micro_batch * grad_accum
    result = TrainBenchResult(
        hardware=hardware_name(device),
        precision=precision,
        micro_batch=micro_batch,
        grad_accum=grad_accum,
        seq_len=seq_len,
        gradient_checkpointing=gradient_checkpointing,
        step_time_s_mean=mean,
        step_time_s_std=statistics.stdev(times) if len(times) > 1 else 0.0,
        tokens_per_second=seqs * seq_len / mean,
        sequences_per_second=seqs / mean,
        parameters=parameter_report(model).total,
        measured_steps=steps,
        **_peaks(device),
    )
    return result


@dataclass
class GenerationBenchResult:
    hardware: str
    dtype: str
    use_cache: bool
    batch_size: int
    prompt_tokens: int
    new_tokens: int
    time_to_first_token_s: float
    total_latency_s: float
    decode_tokens_per_second: float  # tokens after the first, per sequence
    end_to_end_tokens_per_second: float
    peak_allocated_mb: float | None
    peak_reserved_mb: float | None
    repeats: int


@torch.no_grad()
def benchmark_generation(
    model: torch.nn.Module,
    device: torch.device,
    dtype: str = "fp32",
    use_cache: bool = True,
    batch_size: int = 1,
    prompt_tokens: int = 32,
    new_tokens: int = 128,
    repeats: int = 5,
    warmup: int = 2,
) -> GenerationBenchResult:
    """Greedy generation of exactly ``new_tokens`` tokens (EOS disabled).

    Time to first token = prefill + one decoding step, measured with
    ``max_new_tokens=1``; decode throughput uses the remaining tokens.
    """
    from transformers import GenerationConfig

    model = model.to(device=device, dtype=DTYPES[dtype]).eval()
    vocab = model.config.vocab_size
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(2, vocab, (batch_size, prompt_tokens), generator=g).to(device)
    mask = torch.ones_like(ids)

    def run(n: int) -> None:
        cfg = GenerationConfig(
            max_new_tokens=n,
            min_new_tokens=n,
            do_sample=False,
            use_cache=use_cache,
            eos_token_id=None,
            pad_token_id=1,
        )
        model.generate(input_ids=ids, attention_mask=mask, generation_config=cfg)

    _reset_peak(device)
    ttft = statistics.median(_timed(lambda: run(1), device, repeats, warmup))
    total = statistics.median(_timed(lambda: run(new_tokens), device, repeats, warmup))
    decode = (new_tokens - 1) / max(total - ttft, 1e-9)
    return GenerationBenchResult(
        hardware=hardware_name(device),
        dtype=dtype,
        use_cache=use_cache,
        batch_size=batch_size,
        prompt_tokens=prompt_tokens,
        new_tokens=new_tokens,
        time_to_first_token_s=ttft,
        total_latency_s=total,
        decode_tokens_per_second=decode,
        end_to_end_tokens_per_second=new_tokens / total,
        repeats=repeats,
        **_peaks(device),
    )


def benchmark_record(
    kind: str, result: Any, device: torch.device, notes: str = ""
) -> dict[str, Any]:
    return {
        "kind": kind,
        "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **asdict(result),
        "environment": environment_info(device),
        "notes": notes,
    }


def format_training_table(records: list[dict[str, Any]]) -> str:
    rows = [
        "| Hardware | Precision | Micro-batch | Grad accum | Seq len | tok/s | Step time (s) | Peak alloc (MB) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in records:
        if r["kind"] != "training":
            continue
        peak = (
            f"{r['peak_allocated_mb']:,.0f}" if r["peak_allocated_mb"] is not None else "n/a (CPU)"
        )
        rows.append(
            f"| {r['hardware']} | {r['precision']} | {r['micro_batch']} | {r['grad_accum']} | "
            f"{r['seq_len']} | {r['tokens_per_second']:,.0f} | {r['step_time_s_mean']:.3f} | {peak} |"
        )
    return "\n".join(rows)


def format_generation_table(records: list[dict[str, Any]]) -> str:
    rows = [
        "| Hardware | dtype | KV cache | Batch | Prompt | New tokens | TTFT (ms) | Decode tok/s | Total (s) | Peak alloc (MB) |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in records:
        if r["kind"] != "generation":
            continue
        peak = (
            f"{r['peak_allocated_mb']:,.0f}" if r["peak_allocated_mb"] is not None else "n/a (CPU)"
        )
        rows.append(
            f"| {r['hardware']} | {r['dtype']} | {'on' if r['use_cache'] else 'off'} | "
            f"{r['batch_size']} | {r['prompt_tokens']} | {r['new_tokens']} | "
            f"{1000 * r['time_to_first_token_s']:.1f} | {r['decode_tokens_per_second']:,.0f} | "
            f"{r['total_latency_s']:.2f} | {peak} |"
        )
    return "\n".join(rows)
