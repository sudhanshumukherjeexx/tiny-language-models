"""Figures drawn only from recorded data (training history, memorization records)."""

from __future__ import annotations

import math
import textwrap
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless: scripts and CI never need a display
import matplotlib.pyplot as plt

from tinylm.utils import read_jsonl

_TRAIN, _VAL, _ACCENT = "#3b6ea8", "#c0504d", "#5a9e6f"


def plot_training_history(
    history_path: str | Path,
    out_path: str | Path,
    vocab_size: int | None = None,
    title: str | None = None,
) -> Path:
    """Train/validation loss, learning rate and throughput versus optimizer step."""
    rows = read_jsonl(history_path)
    train = [r for r in rows if "train_loss" in r]
    val = [r for r in rows if "val_loss" in r]
    if not train:
        raise ValueError(f"No training records in {history_path}")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    ax = axes[0]
    ax.plot(
        [r["step"] for r in train],
        [r["train_loss"] for r in train],
        color=_TRAIN,
        lw=1.2,
        label="train (window mean)",
    )
    if val:
        ax.plot(
            [r["step"] for r in val],
            [r["val_loss"] for r in val],
            color=_VAL,
            lw=2,
            marker="o",
            ms=3,
            label="validation",
        )
    if vocab_size:
        ax.axhline(math.log(vocab_size), ls=":", c="gray", label="ln(vocab): uniform guess")
    ax.set(xlabel="optimizer step", ylabel="cross-entropy (nats/token)", title="Loss")
    ax.grid(alpha=0.3)
    ax.legend()
    axes[1].plot([r["step"] for r in train], [r["lr"] for r in train], color=_ACCENT)
    axes[1].set(xlabel="optimizer step", ylabel="learning rate", title="Learning-rate schedule")
    axes[1].grid(alpha=0.3)
    axes[2].plot(
        [r["step"] for r in train], [r["tokens_per_second"] for r in train], color=_TRAIN, lw=1
    )
    axes[2].set(xlabel="optimizer step", ylabel="tokens / second", title="Training throughput")
    axes[2].grid(alpha=0.3)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_memorization(records: list[dict[str, Any]], out_path: str | Path) -> Path:
    """Nearest-neighbour similarity and verbatim n-gram overlap: generations vs. held-out text."""
    sources = sorted({r["source"] for r in records})
    colors = {"generation": _TRAIN, "heldout_test": _ACCENT}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for s in sources:
        sims = [r["similarity"] for r in records if r["source"] == s]
        axes[0].hist(
            sims,
            bins=30,
            range=(0, 1),
            alpha=0.6,
            color=colors.get(s),
            label=f"{s} (n={len(sims)})",
            density=True,
        )
    axes[0].set(
        xlabel="TF-IDF cosine to nearest training story",
        ylabel="density",
        title="Nearest-neighbour similarity",
    )
    axes[0].legend()
    ns = sorted({int(n) for r in records for n in r["ngram_overlap"]})
    width = 0.8 / max(1, len(sources))
    for i, s in enumerate(sources):
        rs = [r for r in records if r["source"] == s]
        means = [sum(r["ngram_overlap"][str(n)] for r in rs) / len(rs) for n in ns]
        axes[1].bar(
            [j + i * width for j in range(len(ns))], means, width, color=colors.get(s), label=s
        )
    axes[1].set_xticks(
        [j + width * (len(sources) - 1) / 2 for j in range(len(ns))], [f"{n}-gram" for n in ns]
    )
    axes[1].set(
        ylabel="share of n-grams found in training data", title="Verbatim token n-gram overlap"
    )
    axes[1].legend()
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def short_hardware(name: str) -> str:
    """Compact axis label, e.g. 'RTX 3050 Ti Laptop GPU' or 'CPU i5-11300H'."""
    for junk in ("NVIDIA ", "GeForce ", "(R)", "(TM)", "11th Gen ", "Intel ", "Core "):
        name = name.replace(junk, "")
    if name.startswith("CPU ("):
        name = "CPU " + name[5:].split(",")[0].split("@")[0].strip()
    return " ".join(name.split())


def plot_benchmarks(records: list[dict[str, Any]], out_path: str | Path) -> Path:
    """Decode throughput with and without KV cache, per hardware and dtype."""
    gen = [r for r in records if r["kind"] == "generation"]
    if not gen:
        raise ValueError("No generation benchmark records to plot")
    labels = sorted({(r["hardware"], r["dtype"]) for r in gen})
    fig, ax = plt.subplots(figsize=(max(6, 2.2 * len(labels)), 4))
    for i, (hw, dt) in enumerate(labels):
        for j, cache in enumerate((False, True)):
            vals = [
                r["decode_tokens_per_second"]
                for r in gen
                if r["hardware"] == hw and r["dtype"] == dt and r["use_cache"] == cache
            ]
            if vals:
                ax.bar(
                    i + (j - 0.5) * 0.38,
                    max(vals),
                    0.38,
                    color=_ACCENT if cache else _VAL,
                    label=("KV cache on" if cache else "KV cache off") if i == 0 else None,
                )
    ax.set_xticks(
        range(len(labels)),
        [f"{textwrap.fill(short_hardware(hw), 18)}\n{dt}" for hw, dt in labels],
        fontsize=9,
    )
    ax.set(ylabel="decode tokens / second (batch 1)", title="Generation throughput (measured)")
    ax.legend()
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
