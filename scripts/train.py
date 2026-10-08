"""Pretrain a model from random initialization (resumes automatically from the latest checkpoint).

Examples:
    uv run python scripts/train.py --config configs/tiny-27m.yaml
    uv run python scripts/train.py --config configs/tiny-27m-t4.yaml --set training.per_device_train_batch_size=12 \\
        --set training.gradient_accumulation_steps=6
"""

from __future__ import annotations

import json
from pathlib import Path

from tinylm.cli import load, make_parser
from tinylm.plotting import plot_training_history
from tinylm.training import train_from_config


def main() -> None:
    parser = make_parser(__doc__)
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Fail instead of resuming if the output directory has checkpoints.",
    )
    parser.add_argument(
        "--resume-from", default=None, help="Resume from this specific checkpoint directory."
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:1, ...")
    parser.add_argument(
        "--stop-after-steps",
        type=int,
        default=None,
        help="Train at most this many steps in this invocation, checkpoint, and exit "
        "(useful for time-limited sessions; re-run to continue).",
    )
    args = parser.parse_args()
    cfg = load(args)
    resume = args.resume_from or (False if args.no_resume else "auto")
    metrics = train_from_config(cfg, resume, args.device, args.stop_after_steps)
    out = Path(cfg.training.output_dir)
    plot_training_history(
        out / "training_history.jsonl", out / "training_curves.png", cfg.model.vocab_size, cfg.name
    )
    summary = {k: v for k, v in metrics.items() if k != "invocations"}
    print(json.dumps(summary, indent=2))
    print(f"\nSelected (best-validation) model: {metrics['best_checkpoint']}")
    print("Next: scripts/evaluate.py evaluates it once on the untouched test split.")


if __name__ == "__main__":
    main()
