"""Plot loss, learning rate and throughput from a run's training_history.jsonl.

Example:
    uv run python scripts/plot_training.py --run-dir runs/tiny-27m --out assets/training_curve.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from tinylm.plotting import plot_training_history


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--out", type=Path, help="Default: <run-dir>/training_curves.png")
    args = parser.parse_args()
    cfg = yaml.safe_load((args.run_dir / "config.yaml").read_text(encoding="utf-8"))
    out = plot_training_history(
        args.run_dir / "training_history.jsonl",
        args.out or args.run_dir / "training_curves.png",
        cfg["model"]["vocab_size"],
        cfg["name"],
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
