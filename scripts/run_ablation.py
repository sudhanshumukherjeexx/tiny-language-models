"""Run the controlled architecture ablations over several seeds and summarize them.

Each (variant, seed) trains with the shared reduced budget in
configs/ablations/_budget.yaml, then its validation-selected model is scored
on the FULL validation split. Test data is not touched: ablations are
modelling decisions, which belong to validation.

Examples:
    uv run python scripts/run_ablation.py --seeds 0 1 2
    uv run python scripts/run_ablation.py --configs configs/ablations/rope.yaml --seeds 0
    uv run python scripts/run_ablation.py --summarize-only
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
from pathlib import Path

import torch

from tinylm.config import load_config
from tinylm.data import load_packed_split
from tinylm.evaluation import evaluate_loss
from tinylm.modeling import load_model
from tinylm.tracking import aggregate_seeds
from tinylm.training import train_from_config
from tinylm.utils import read_json, resolve_device, resolve_precision, setup_logging, write_json

logger = logging.getLogger("ablation")
DEFAULT_ORDER = ["baseline-gpt-style", "rope", "rmsnorm", "swiglu", "gqa"]
METRICS = [
    "parameters",
    "final_train_loss",
    "full_validation_loss",
    "full_validation_perplexity",
    "tokens_per_second",
    "peak_vram_allocated_mb",
    "wall_clock_seconds",
]


def run_one(config: Path, seed: int, out_root: Path, device: str) -> dict:
    name = config.stem
    run_dir = Path("runs/ablations") / name / f"seed-{seed}"
    cfg = load_config(
        config, [f"training.seed={seed}", f"training.output_dir={run_dir.as_posix()}"]
    )
    metrics = train_from_config(cfg, "auto", device)
    dev = resolve_device(device)
    model = load_model(run_dir / "best").to(dev)
    val = load_packed_split(cfg.data.data_dir, "validation", metrics["tokenizer_sha256"])
    result = evaluate_loss(
        model,
        val,
        cfg.training.eval_batch_size,
        dev,
        resolve_precision(cfg.training.precision, dev),
        split="validation",
    )
    record = {k: v for k, v in metrics.items() if k != "invocations"} | {
        "variant": name,
        "config": str(config),
        "seed": seed,
        "full_validation_loss": result.loss,
        "full_validation_perplexity": result.perplexity,
        "full_validation_tokens": result.predicted_tokens,
        "model": {
            k: getattr(cfg.model, k)
            for k in (
                "position_encoding",
                "norm",
                "ffn",
                "num_key_value_heads",
                "intermediate_size",
            )
        },
    }
    write_json(out_root / name / f"seed-{seed}.json", record)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return record


def summarize(out_root: Path) -> dict:
    summary = {}
    for name in DEFAULT_ORDER + sorted(p.name for p in out_root.iterdir() if p.is_dir()):
        if name in summary or not (out_root / name).is_dir():
            continue
        runs = [read_json(p) for p in sorted((out_root / name).glob("seed-*.json"))]
        if runs:
            summary[name] = {
                "seeds": [r["seed"] for r in runs],
                "model": runs[0]["model"],
                "hardware": sorted({h for r in runs for h in r["hardware"]}),
                "training_tokens": runs[0]["training_tokens"],
                **aggregate_seeds(runs, METRICS),
            }
    write_json(out_root / "summary.json", summary)
    lines = [
        "| Variant | Position | Norm | FFN | KV heads | Params | Seeds | Val loss (mean ± std) | Val PPL | tok/s | Peak VRAM (MB) |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    def fmt(m: dict, digits: int = 4) -> str:
        if m["mean"] is None:
            return "—"
        return f"{m['mean']:.{digits}f}" + (
            f" ± {m['std']:.{digits}f}" if m["std"] is not None else ""
        )

    for name, s in summary.items():
        m = s["model"]
        lines.append(
            f"| {name} | {m['position_encoding']} | {m['norm']} | {m['ffn']} | {m['num_key_value_heads']} | "
            f"{s['parameters']['mean']:,.0f} | {len(s['seeds'])} | {fmt(s['full_validation_loss'])} | "
            f"{fmt(s['full_validation_perplexity'], 3)} | {fmt(s['tokens_per_second'], 0)} | "
            f"{fmt(s['peak_vram_allocated_mb'], 0)} |"
        )
    table = "\n".join(lines)
    (out_root / "summary.md").write_text(table + "\n", encoding="utf-8")
    return {"summary": summary, "table": table}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        type=Path,
        default=[Path(f"configs/ablations/{n}.yaml") for n in DEFAULT_ORDER],
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--out-dir", type=Path, default=Path("results/ablations"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    setup_logging()
    if not args.summarize_only:
        for config in args.configs:
            for seed in args.seeds:
                if (args.out_dir / config.stem / f"seed-{seed}.json").exists():
                    logger.info("Skipping %s seed %d (result exists)", config.stem, seed)
                    continue
                run_one(config, seed, args.out_dir, args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    result = summarize(args.out_dir)
    print(result["table"])
    print(json.dumps({k: v["seeds"] for k, v in result["summary"].items()}, indent=2))


if __name__ == "__main__":
    main()
