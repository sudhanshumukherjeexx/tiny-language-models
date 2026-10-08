"""Duplicate and leakage report for the prepared train/validation/test splits.

Example:
    uv run python scripts/check_data_integrity.py --config configs/tiny-27m.yaml \\
        --out results/data_integrity.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from tinylm.cli import load, make_parser
from tinylm.data import SPLITS, TOKEN_DTYPE, load_text_split
from tinylm.integrity import (
    MinHashLSH,
    containment,
    containment_summary,
    cross_split_exact,
    exact_duplicate_stats,
    hash_texts,
    sample_documents,
    split_documents,
)
from tinylm.utils import read_json, write_json

logger = logging.getLogger("integrity")


def main() -> None:
    parser = make_parser(__doc__)
    parser.add_argument("--out", default="results/data_integrity.json")
    parser.add_argument(
        "--ngram", type=int, default=13, help="Token n-gram length for containment."
    )
    parser.add_argument(
        "--near-dup-sample",
        type=int,
        default=100_000,
        help="Documents per split for MinHash near-duplicate search.",
    )
    parser.add_argument(
        "--containment-sample",
        type=int,
        default=None,
        help="Limit containment to this many val/test documents (default: all).",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    cfg = load(args)
    data_dir = Path(cfg.data.data_dir)
    meta = read_json(data_dir / "meta.json")
    eos = meta["eos_token_id"]
    rng = np.random.default_rng(args.seed)
    report: dict = {
        "dataset": meta["dataset"],
        "dataset_revision": meta["dataset_revision"],
        "tokenizer_sha256": meta["tokenizer_sha256"],
        "split_protocol": {
            "train": f"official '{cfg.data.source_train_split}' minus hash-held-out validation",
            "validation": f"{cfg.data.validation_fraction:.3%} of official train by content hash",
            "test": f"official '{cfg.data.source_test_split}' split",
        },
    }

    logger.info("Exact duplicates (normalized-text hash)")
    hashes = {s: hash_texts(load_text_split(data_dir, s)) for s in SPLITS}
    report["exact_within_split"] = {s: exact_duplicate_stats(h) for s, h in hashes.items()}
    report["exact_cross_split"] = {
        "validation_in_train": cross_split_exact(hashes["train"], hashes["validation"]),
        "test_in_train": cross_split_exact(hashes["train"], hashes["test"]),
        "test_in_validation": cross_split_exact(hashes["validation"], hashes["test"]),
    }

    logger.info("Cross-split %d-gram containment against the full training stream", args.ngram)
    train_stream = np.memmap(data_dir / "train.bin", dtype=TOKEN_DTYPE, mode="r")
    docs = {}
    for split in ("validation", "test"):
        stream = np.memmap(data_dir / f"{split}.bin", dtype=TOKEN_DTYPE, mode="r")
        d = split_documents(stream, eos)
        if args.containment_sample and len(d) > args.containment_sample:
            d = [d[i] for i in rng.choice(len(d), args.containment_sample, replace=False)]
        docs[split] = d
        values = containment(train_stream, d, args.ngram)
        report.setdefault("containment", {})[split] = containment_summary(values) | {
            "ngram_tokens": args.ngram
        }

    logger.info("MinHash/LSH near duplicates (sample of %d per split)", args.near_dup_sample)
    lsh = MinHashLSH(seed=args.seed)
    sample_train = sample_documents(train_stream, eos, args.near_dup_sample, rng)
    report["near_duplicates"] = {
        "train_sample": lsh.near_duplicates(sample_train)
        | {"population": "random sample of train"},
        "test": lsh.near_duplicates(docs["test"][: args.near_dup_sample]),
        "test_vs_train_sample": lsh.cross_near_duplicates(sample_train, docs["test"]),
    }
    write_json(args.out, report)
    printable = json.loads(json.dumps(report))
    for v in printable["near_duplicates"].values():
        v.pop("examples", None)
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
