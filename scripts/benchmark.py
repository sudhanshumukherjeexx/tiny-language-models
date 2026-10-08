"""Measure training throughput/memory and generation latency on *this* machine.

Results are appended to results/benchmarks/<hardware>.json, keyed by the
hardware they were measured on. Generation timing does not depend on the
weights (EOS is disabled and exactly --new-tokens are produced), so a
randomly initialized model is used unless --checkpoint is given.

Examples:
    uv run python scripts/benchmark.py --config configs/tiny-27m.yaml --mode all
    uv run python scripts/benchmark.py --config configs/tiny-27m.yaml --mode training \\
        --precisions bf16 fp16 --micro-batches 4 8 16
    uv run python scripts/benchmark.py --config configs/tiny-27m.yaml --mode generation --device cpu
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import torch

from tinylm.benchmarking import (
    benchmark_generation,
    benchmark_record,
    benchmark_training,
    format_generation_table,
    format_training_table,
)
from tinylm.cli import load, make_parser
from tinylm.modeling import build_model, kv_cache_table, load_model
from tinylm.plotting import plot_benchmarks
from tinylm.utils import hardware_name, read_json, resolve_device, write_json

logger = logging.getLogger("benchmark")


def _is_oom(err: RuntimeError) -> bool:
    return isinstance(err, torch.OutOfMemoryError) or "out of memory" in str(err).lower()


def main() -> None:
    parser = make_parser(__doc__)
    parser.add_argument("--mode", choices=["training", "generation", "all"], default="all")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--precisions",
        nargs="+",
        default=None,
        help="Training: fp32/fp16/bf16 (default: all supported on the device).",
    )
    parser.add_argument("--micro-batches", type=int, nargs="+", default=[8])
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--dtypes", nargs="+", default=None, help="Generation weight dtypes.")
    parser.add_argument("--prompt-tokens", type=int, default=32)
    parser.add_argument("--new-tokens", type=int, default=256)
    parser.add_argument("--checkpoint", help="Benchmark generation with these weights instead.")
    parser.add_argument("--out-dir", default="results/benchmarks")
    parser.add_argument("--notes", default="")
    args = parser.parse_args()
    cfg = load(args)
    device = resolve_device(args.device)
    hw = hardware_name(device)
    supported = (
        ["fp32"]
        + (["fp16"] if device.type == "cuda" else [])
        + (["bf16"] if device.type == "cpu" or torch.cuda.is_bf16_supported() else [])
    )
    records = []

    if args.mode in ("training", "all"):
        for precision in args.precisions or supported:
            for mb in args.micro_batches:
                try:
                    res = benchmark_training(
                        cfg.model,
                        device,
                        precision,
                        mb,
                        args.grad_accum,
                        cfg.data.block_size,
                        args.steps,
                    )
                except RuntimeError as err:
                    if not _is_oom(err):
                        raise
                    logger.warning("OOM: %s micro-batch %d does not fit on %s", precision, mb, hw)
                    records.append(
                        {
                            "kind": "training_oom",
                            "hardware": hw,
                            "precision": precision,
                            "micro_batch": mb,
                            "seq_len": cfg.data.block_size,
                        }
                    )
                    torch.cuda.empty_cache()
                    continue
                records.append(benchmark_record("training", res, device, args.notes))
                logger.info("%s %s mb=%d: %.0f tok/s", hw, precision, mb, res.tokens_per_second)

    if args.mode in ("generation", "all"):
        dtypes = args.dtypes or (["fp32", "fp16", "bf16"] if device.type == "cuda" else ["fp32"])
        for dtype in dtypes:
            if dtype == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
                continue
            for use_cache in (True, False):
                torch.manual_seed(0)
                model = load_model(args.checkpoint) if args.checkpoint else build_model(cfg.model)
                res = benchmark_generation(
                    model, device, dtype, use_cache, 1, args.prompt_tokens, args.new_tokens
                )
                note = args.notes + (" random-init weights" if not args.checkpoint else "")
                records.append(benchmark_record("generation", res, device, note.strip()))
                logger.info(
                    "%s %s cache=%s: TTFT %.1f ms, decode %.0f tok/s",
                    hw,
                    dtype,
                    use_cache,
                    1000 * res.time_to_first_token_s,
                    res.decode_tokens_per_second,
                )
                del model

    slug = re.sub(r"[^a-z0-9]+", "-", hw.lower()).strip("-")
    out = Path(args.out_dir) / f"{slug}.json"
    existing = read_json(out)["records"] if out.exists() else []
    write_json(
        out,
        {
            "hardware": hw,
            "records": existing + records,
            "kv_cache_estimate_bf16": kv_cache_table(cfg.model),
        },
    )
    print("\n" + format_training_table(records))
    print("\n" + format_generation_table(records))
    ooms = [r for r in records if r["kind"] == "training_oom"]
    if ooms:
        print(
            "\nDid not fit: " + ", ".join(f"{r['precision']} mb={r['micro_batch']}" for r in ooms)
        )
    if any(r["kind"] == "generation" for r in records):
        plot_benchmarks(
            [r for r in existing + records if r["kind"] == "generation"],
            Path(args.out_dir) / f"{slug}-generation.png",
        )
    print(f"\nAppended {len(records)} records to {out}")


if __name__ == "__main__":
    main()
