"""Generate site/data.js from the recorded result files (no hand-typed numbers).

python site/build_data.py
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"


def load(path: str):
    return json.loads((RES / path).read_text(encoding="utf-8"))


def jsonl(path: str) -> list[dict]:
    return [
        json.loads(x) for x in (RES / path).read_text(encoding="utf-8").splitlines() if x.strip()
    ]


def main() -> None:
    metrics = load("runs/tiny-27m/metrics.json")
    test = load("runs/tiny-27m/evaluation/test_metrics.json")
    history = jsonl("runs/tiny-27m/training_history.jsonl")
    lexical = load("runs/tiny-27m/evaluation/lexical_metrics.json")
    memo = load("runs/tiny-27m/evaluation/memorization/memorization_summary.json")["summary"]
    raw = load("data_integrity_official_splits.json")
    clean = load("data_integrity.json")
    legacy = load("runs/legacy-colab-a100/metrics.json")

    train = [r for r in history if "train_loss" in r]
    val = [r for r in history if "val_loss" in r]
    curve = {
        "train": [
            [r["step"], round(r["train_loss"], 4)]
            for r in train
            if r["step"] % 100 == 0 or r["step"] < 100
        ],
        "val": [[r["step"], round(r["val_loss"], 4)] for r in val],
    }

    order = ["baseline-gpt-style", "rope", "rmsnorm", "swiglu", "gqa"]
    labels = {
        "baseline-gpt-style": "GPT-style baseline",
        "rope": "+ RoPE",
        "rmsnorm": "+ RMSNorm",
        "swiglu": "+ SwiGLU",
        "gqa": "+ GQA",
    }
    ablations = []
    for v in order:
        runs = [
            json.loads(p.read_text(encoding="utf-8"))
            for p in sorted((RES / "ablations" / v).glob("seed-*.json"))
        ]
        losses = [r["full_validation_loss"] for r in runs]
        ablations.append(
            {
                "id": v,
                "label": labels[v],
                "seeds": losses,
                "mean": statistics.fmean(losses),
                "std": statistics.stdev(losses),
                "tokens_per_second": statistics.fmean(r["tokens_per_second"] for r in runs),
                "params": runs[0]["parameters"],
                "model": runs[0]["model"],
                "training_tokens": runs[0]["training_tokens"],
            }
        )

    bench = []
    for f in sorted((RES / "benchmarks").glob("*.json")):
        for r in json.loads(f.read_text(encoding="utf-8"))["records"]:
            hw = "RTX 3050 Ti Laptop" if "3050" in r["hardware"] else "Core i5-11300H CPU"
            if r["kind"] == "generation":
                bench.append(
                    {
                        "kind": "gen",
                        "hw": hw,
                        "dtype": r["dtype"],
                        "cache": r["use_cache"],
                        "decode": r["decode_tokens_per_second"],
                        "ttft_ms": 1000 * r["time_to_first_token_s"],
                    }
                )
            elif r["kind"] == "training":
                bench.append(
                    {
                        "kind": "train",
                        "hw": hw,
                        "precision": r["precision"],
                        "mb": r["micro_batch"],
                        "tps": r["tokens_per_second"],
                        "peak": r["peak_allocated_mb"],
                    }
                )

    meta = load("data_meta.json")
    gens = jsonl("runs/tiny-27m/evaluation/generation_samples.jsonl")
    example = next(g for g in gens if g["prompt_id"] == "open-05" and g["preset"] == "greedy")

    data = {
        "run": {
            k: metrics[k]
            for k in (
                "parameters",
                "steps",
                "training_tokens",
                "wall_clock_seconds",
                "train_seconds",
                "tokens_per_second",
                "best_validation_loss",
                "best_validation_perplexity",
                "peak_vram_allocated_mb",
                "tokens_per_parameter",
                "hardware",
                "final_train_loss",
                "initialization_loss",
            )
        },
        "test": {k: test[k] for k in ("loss", "perplexity", "sequences", "predicted_tokens")},
        "legacy": {"validation_loss": legacy["validation_loss"]},
        "curve": curve,
        "lexical": lexical["by_preset"],
        "memorization": memo,
        "leakage": {
            "raw_test_docs": raw["exact_cross_split"]["test_in_train"]["documents"],
            "raw_test_in_train": raw["exact_cross_split"]["test_in_train"][
                "with_copy_in_reference"
            ],
            "raw_containment_mean": raw["containment"]["test"]["mean"],
            "clean_test_docs": clean["exact_cross_split"]["test_in_train"]["documents"],
            "clean_test_in_train": clean["exact_cross_split"]["test_in_train"][
                "with_copy_in_reference"
            ],
            "clean_containment_mean": clean["containment"]["test"]["mean"],
            "val_containment_mean": clean["containment"]["validation"]["mean"],
            "train_redundant_rate": clean["exact_within_split"]["train"]["redundant_copy_rate"],
        },
        "splits": meta["splits"],
        "ablations": ablations,
        "bench": bench,
        "example": {
            "prompt": example["prompt"],
            "completion": example["completion"].strip(),
            "new_tokens": example["new_tokens"],
        },
    }
    out = Path(__file__).with_name("data.js")
    out.write_text(
        "// Generated by site/build_data.py from results/. Do not edit by hand.\n"
        f"window.TINYLM = {json.dumps(data, indent=1)};\n",
        encoding="utf-8",
        newline="\n",  # identical bytes on every OS (CI regenerates and diffs this file)
    )
    print(f"wrote {out} ({out.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
