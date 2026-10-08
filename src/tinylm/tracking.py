"""Experiment index (results/experiments.csv) and multi-seed aggregation."""

from __future__ import annotations

import csv
import statistics
from pathlib import Path
from typing import Any

INDEX_COLUMNS = [
    "run_id",
    "model_name",
    "seed",
    "parameters",
    "training_tokens",
    "steps",
    "validation_loss",
    "test_loss",
    "test_perplexity",
    "tokens_per_second",
    "peak_vram_mb",
    "wall_clock_seconds",
    "hardware",
    "checkpoint_path",
    "git_commit",
]


def index_row(metrics: dict[str, Any], test: dict[str, Any] | None = None) -> dict[str, Any]:
    """Map a run's metrics.json (and optional test result) onto the index columns.

    Missing values stay empty: the index never contains estimated numbers.
    """
    return {
        "run_id": metrics.get("run_id"),
        "model_name": metrics.get("name"),
        "seed": metrics.get("seed"),
        "parameters": metrics.get("parameters"),
        "training_tokens": metrics.get("training_tokens"),
        "steps": metrics.get("steps"),
        "validation_loss": metrics.get("best_validation_loss"),
        "test_loss": (test or {}).get("loss"),
        "test_perplexity": (test or {}).get("perplexity"),
        "tokens_per_second": metrics.get("tokens_per_second"),
        "peak_vram_mb": metrics.get("peak_vram_allocated_mb"),
        "wall_clock_seconds": metrics.get("wall_clock_seconds"),
        "hardware": "; ".join(metrics.get("hardware") or []),
        "checkpoint_path": metrics.get("best_checkpoint"),
        "git_commit": metrics.get("git_commit"),
    }


def upsert_index(path: str | Path, row: dict[str, Any]) -> None:
    """Insert or replace (by run_id) a row of the experiment index CSV."""
    path = Path(path)
    rows: list[dict[str, Any]] = []
    if path.exists():
        with path.open(newline="", encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f) if r["run_id"] != row["run_id"]]
    rows.append({k: ("" if row.get(k) is None else row.get(k)) for k in INDEX_COLUMNS})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=INDEX_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def mean_std(values: list[float]) -> dict[str, float | int | None]:
    vals = [v for v in values if v is not None]
    if not vals:
        return {"mean": None, "std": None, "n": 0}
    return {
        "mean": statistics.fmean(vals),
        "std": statistics.stdev(vals) if len(vals) > 1 else None,
        "n": len(vals),
    }


def aggregate_seeds(runs: list[dict[str, Any]], keys: list[str]) -> dict[str, dict[str, Any]]:
    """Mean and sample standard deviation over seeds for each metric in ``keys``."""
    return {k: mean_std([r.get(k) for r in runs]) for k in keys}
