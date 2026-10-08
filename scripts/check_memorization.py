"""Compare generated stories with the training data (and held-out stories as a baseline).

Input is the generation_samples.jsonl written by scripts/evaluate.py, which
stores the generated token ids, so no re-tokenization is involved.

Example:
    uv run python scripts/check_memorization.py --config configs/tiny-27m.yaml \\
        --generations runs/tiny-27m/evaluation/generation_samples.jsonl
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tinylm.cli import load, make_parser
from tinylm.data import TOKEN_DTYPE, load_text_split
from tinylm.memorization import MemorizationThresholds, analyze, summarize
from tinylm.plotting import plot_memorization
from tinylm.tokenizer import encode_batch, load_tokenizer
from tinylm.utils import append_jsonl, read_json, read_jsonl, write_json


def main() -> None:
    parser = make_parser(__doc__)
    parser.add_argument(
        "--generations", required=True, help="generation_samples.jsonl from evaluate.py"
    )
    parser.add_argument("--out-dir", help="Default: next to the generations file, in memorization/")
    parser.add_argument(
        "--heldout-samples",
        type=int,
        default=200,
        help="Held-out TEST stories analyzed identically as the baseline (0 to skip).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=200,
        help="Truncate held-out stories to this many tokens (match generation length).",
    )
    parser.add_argument("--ngrams", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    cfg = load(args)

    gens = read_jsonl(args.generations)
    if not gens or "token_ids" not in gens[0]:
        raise SystemExit("Generations must include token_ids (produced by scripts/evaluate.py).")
    items = [
        {
            "source": "generation",
            "item_id": f"{g['prompt_id']}|{g['preset']}|{g['seed']}",
            "text": g["completion"],
            "token_ids": g["token_ids"],
        }
        for g in gens
    ]

    if args.heldout_samples:
        tokenizer = load_tokenizer(cfg.tokenizer.path)
        test = load_text_split(cfg.data.data_dir, "test")
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(test), size=min(args.heldout_samples, len(test)), replace=False)
        texts = [test[int(i)] for i in idx]
        for i, ids in zip(idx, encode_batch(tokenizer, texts), strict=True):
            ids = ids[: args.max_tokens]
            items.append(
                {
                    "source": "heldout_test",
                    "item_id": f"test-{int(i)}",
                    "text": tokenizer.decode(ids),
                    "token_ids": ids,
                }
            )

    meta = read_json(Path(cfg.data.data_dir) / "meta.json")
    stream = np.memmap(Path(cfg.data.data_dir) / "train.bin", dtype=TOKEN_DTYPE, mode="r")
    train_texts = load_text_split(cfg.data.data_dir, "train")
    thresholds = MemorizationThresholds()
    records = analyze(items, stream, train_texts, args.ngrams, thresholds, seed=args.seed)

    out = Path(args.out_dir or Path(args.generations).parent / "memorization")
    out.mkdir(parents=True, exist_ok=True)
    (out / "memorization_records.jsonl").unlink(missing_ok=True)
    append_jsonl(out / "memorization_records.jsonl", [r.to_dict() for r in records])
    summary = {
        "summary": summarize(records),
        "thresholds": thresholds.__dict__,
        "ngram_sizes_tokens": args.ngrams,
        "train_tokens_scanned": len(stream),
        "train_stories_searched": len(train_texts),
        "tokenizer_sha256": meta["tokenizer_sha256"],
        "generations_file": str(args.generations),
        "retrieval": "TF-IDF cosine, word 1-2-grams, 2^20 hashed features, IDF from 100k-story sample",
    }
    write_json(out / "memorization_summary.json", summary)
    plot_memorization([r.to_dict() for r in records], out / "memorization.png")
    print(json.dumps(summary["summary"], indent=2))
    flagged = [r for r in records if r.flagged and r.source == "generation"]
    print(
        f"\n{len(flagged)} generation(s) flagged for manual review (see memorization_records.jsonl)."
    )
    for r in sorted(flagged, key=lambda r: -r.similarity)[:5]:
        print(
            f"- {r.item_id}: {'; '.join(r.flag_reasons)}\n  shared span: {r.longest_common_span[:160]!r}"
        )


if __name__ == "__main__":
    main()
