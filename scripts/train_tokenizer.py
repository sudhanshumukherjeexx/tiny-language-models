"""Train the byte-level BPE tokenizer on the TRAIN split and write diagnostics.

Example:
    uv run python scripts/train_tokenizer.py --config configs/tiny-27m.yaml
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from tinylm.cli import load, make_parser
from tinylm.data import build_text_splits
from tinylm.tokenizer import (
    iter_batches,
    load_tokenizer,
    tokenizer_diagnostics,
    tokenizer_fingerprint,
    train_tokenizer,
)
from tinylm.utils import read_json, write_json

logger = logging.getLogger("train_tokenizer")


def main() -> None:
    parser = make_parser(__doc__)
    parser.add_argument("--force", action="store_true", help="Overwrite an existing tokenizer.")
    parser.add_argument(
        "--diagnostic-stories",
        type=int,
        default=2000,
        help="Validation stories used for compression/round-trip diagnostics.",
    )
    args = parser.parse_args()
    cfg = load(args)
    out = Path(cfg.tokenizer.path)
    if (out / "tokenizer.json").exists() and not args.force:
        raise SystemExit(
            f"{out} already contains a tokenizer; pass --force to retrain "
            "(packed data built with the old one must then be rebuilt)."
        )

    splits = build_text_splits(cfg.data)
    train = splits["train"]
    n = (
        len(train)
        if cfg.tokenizer.max_training_examples is None
        else min(len(train), cfg.tokenizer.max_training_examples)
    )
    corpus = train.select(range(n))
    logger.info(
        "Training a %d-token byte-level BPE on %s training stories",
        cfg.tokenizer.vocab_size,
        f"{n:,}",
    )
    start = time.perf_counter()
    tokenizer = train_tokenizer(
        iter_batches(corpus["text"]), cfg.tokenizer, cfg.data.block_size, length=n
    )
    seconds = time.perf_counter() - start
    tokenizer.save_pretrained(str(out))

    reloaded = load_tokenizer(out)
    sample = splits["validation"].select(
        range(min(args.diagnostic_stories, len(splits["validation"])))
    )["text"]
    diag = tokenizer_diagnostics(reloaded, list(sample))
    same_ids = all(
        tokenizer(t, add_special_tokens=False)["input_ids"]
        == reloaded(t, add_special_tokens=False)["input_ids"]
        for t in sample[:200]
    )
    report = {
        "tokenizer_sha256": tokenizer_fingerprint(out),
        "training_stories": n,
        "train_split_fingerprint": read_json(Path(cfg.data.data_dir) / "text" / "text_meta.json")[
            "train_split_fingerprint"
        ],
        "training_characters": sum(len(t) for t in corpus["text"]),
        "training_seconds": seconds,
        "trainer": {
            "algorithm": "byte-level BPE",
            "min_frequency": cfg.tokenizer.min_frequency,
            "deterministic": "merges depend only on the corpus and its order; no random seed",
        },
        "save_reload_identical_ids": same_ids,
        "diagnostics_split": "validation",
        **diag,
    }
    write_json(out / "tokenizer_report.json", report)
    printable = {k: v for k, v in report.items() if k != "probes"}
    print(json.dumps(printable, indent=2))
    for p in diag["probes"]:
        print(f"{p['tokens']:>3} tokens  roundtrip={p['roundtrip_exact']!s:<5}  {p['text']!r}")
    if not same_ids or diag["roundtrip_exact_fraction"] < 1.0:
        raise SystemExit("Tokenizer failed save/reload or round-trip checks; see report above.")


if __name__ == "__main__":
    main()
