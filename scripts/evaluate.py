"""Final evaluation of a selected checkpoint: untouched TEST loss, generation suite, lexical metrics.

The checkpoint should already have been chosen on the validation split
(train.py exports it to <output_dir>/best). Run this once per final model;
do not use its test numbers to make further modelling decisions.

Examples:
    uv run python scripts/evaluate.py --config configs/tiny-27m.yaml
    uv run python scripts/evaluate.py --config configs/tiny-27m.yaml --checkpoint runs/tiny-27m/best \\
        --judge anthropic --record-index results/experiments.csv
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from tinylm.cli import load, make_parser
from tinylm.data import load_packed_split
from tinylm.evaluation import evaluate_loss, load_prompt_suite, run_generation_suite, summarize_by
from tinylm.judge import aggregate, export_manual_sheet, judge_available, judge_records
from tinylm.modeling import load_model, parameter_report
from tinylm.tokenizer import load_tokenizer, tokenizer_fingerprint
from tinylm.tracking import index_row, upsert_index
from tinylm.utils import (
    append_jsonl,
    environment_info,
    git_commit,
    read_json,
    resolve_device,
    resolve_precision,
    seed_everything,
    write_json,
)

logger = logging.getLogger("evaluate")


def main() -> None:
    parser = make_parser(__doc__)
    parser.add_argument(
        "--checkpoint", help="Model directory (default: <training.output_dir>/best)."
    )
    parser.add_argument(
        "--out-dir", help="Where to write results (default: <checkpoint>/../evaluation)."
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", default="fp32", help="fp32 (default, exact), bf16 or fp16.")
    parser.add_argument("--skip-test", action="store_true", help="Skip test-split loss.")
    parser.add_argument("--skip-generation", action="store_true", help="Skip the generation suite.")
    parser.add_argument(
        "--judge",
        choices=["none", "anthropic"],
        default="none",
        help="Optional LLM judge (needs ANTHROPIC_API_KEY and `uv sync --extra judge`).",
    )
    parser.add_argument(
        "--record-index", help="Append/update this experiments CSV with the result."
    )
    args = parser.parse_args()
    cfg = load(args)

    ckpt = Path(args.checkpoint or Path(cfg.training.output_dir) / "best")
    out = Path(args.out_dir or ckpt.parent / "evaluation")
    device = resolve_device(args.device)
    precision = resolve_precision(args.precision, device)
    model = load_model(ckpt).to(device).eval()
    tokenizer = load_tokenizer(ckpt if (ckpt / "tokenizer.json").exists() else cfg.tokenizer.path)
    ckpt_meta = (
        read_json(ckpt / "tinylm_metadata.json") if (ckpt / "tinylm_metadata.json").exists() else {}
    )
    data_meta = read_json(Path(cfg.data.data_dir) / "meta.json")
    tok_sha = ckpt_meta.get("tokenizer_sha256") or tokenizer_fingerprint(cfg.tokenizer.path)
    if not ckpt_meta:
        logger.warning(
            "%s has no tinylm_metadata.json (e.g. a legacy export); assuming it uses the "
            "tokenizer at %s.",
            ckpt,
            cfg.tokenizer.path,
        )
    provenance = {
        "checkpoint": str(ckpt),
        "checkpoint_metadata": ckpt_meta,
        "dataset": data_meta["dataset"],
        "dataset_revision": data_meta["dataset_revision"],
        "tokenizer_sha256": tok_sha,
        "parameters": parameter_report(model).total,
        "git_commit": git_commit(),
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "device": str(device),
        "precision": precision,
        "environment": environment_info(device),
    }
    out.mkdir(parents=True, exist_ok=True)

    test_result = None
    if not args.skip_test:
        test = load_packed_split(cfg.data.data_dir, "test", tok_sha)
        test_result = evaluate_loss(
            model,
            test,
            cfg.training.eval_batch_size,
            device,
            precision,
            split="test",
            progress=True,
        )
        write_json(
            out / "test_metrics.json",
            {
                **test_result.to_dict(),
                **provenance,
                "test_split_source": cfg.data.source_test_split,
                "test_split_stats": data_meta["splits"]["test"],
            },
        )
        print(
            f"TEST  loss {test_result.loss:.4f}  perplexity {test_result.perplexity:.3f}  "
            f"({test_result.predicted_tokens:,} predicted tokens, {test_result.sequences:,} sequences)"
        )

    if not args.skip_generation:
        seed_everything(0)
        suite = load_prompt_suite(cfg.evaluation.prompts_path)
        records, skipped = run_generation_suite(
            model,
            tokenizer,
            suite,
            cfg.evaluation.presets,
            cfg.evaluation.seeds,
            cfg.evaluation.max_new_tokens,
            metadata={"checkpoint": str(ckpt), "tokenizer_sha256": tok_sha, "device": str(device)},
        )
        samples = out / "generation_samples.jsonl"
        samples.unlink(missing_ok=True)
        append_jsonl(samples, records)
        lexical = {
            "by_preset": summarize_by(records, "preset"),
            "by_category": summarize_by(records, "category"),
            "suite_version": suite.get("version"),
            "skipped_prompts": skipped,
            "note": "Lexical diagnostics, not a quality score.",
        }
        write_json(out / "lexical_metrics.json", lexical)
        export_manual_sheet(records, out / "manual_rating_sheet.csv")
        print(json.dumps(lexical["by_preset"], indent=2))

        if args.judge == "anthropic":
            if not judge_available():
                raise SystemExit("--judge anthropic needs ANTHROPIC_API_KEY (see .env.example).")
            from tinylm.judge import AnthropicJudge

            results = judge_records(AnthropicJudge(), records)
            (out / "judge_results.jsonl").unlink(missing_ok=True)
            append_jsonl(out / "judge_results.jsonl", [r.to_dict() for r in results])
            write_json(out / "judge_summary.json", aggregate(results))
            print(json.dumps(aggregate(results), indent=2))
        else:
            print(f"No judge configured: rate {out / 'manual_rating_sheet.csv'} by hand if needed.")

    if args.record_index:
        run_metrics_path = ckpt.parent / "metrics.json"
        if not run_metrics_path.exists():
            raise SystemExit(f"--record-index needs {run_metrics_path} (written by train.py).")
        upsert_index(
            args.record_index,
            index_row(read_json(run_metrics_path), test_result.to_dict() if test_result else None),
        )
    print(f"\nResults written to {out}")


if __name__ == "__main__":
    main()
