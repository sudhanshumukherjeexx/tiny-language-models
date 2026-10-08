"""Tokenize and pack train/validation/test into fixed-length blocks, with token accounting.

Example:
    uv run python scripts/prepare_data.py --config configs/tiny-27m.yaml
"""

from __future__ import annotations

from tinylm.cli import load, make_parser
from tinylm.data import prepare_packed_data
from tinylm.tokenizer import load_tokenizer, tokenizer_fingerprint


def main() -> None:
    parser = make_parser(__doc__)
    parser.add_argument(
        "--force", action="store_true", help="Rebuild packed .bin files if present."
    )
    args = parser.parse_args()
    cfg = load(args)
    tokenizer = load_tokenizer(cfg.tokenizer.path)
    meta = prepare_packed_data(
        cfg.data, tokenizer, tokenizer_fingerprint(cfg.tokenizer.path), args.force
    )

    header = (
        f"{'split':<11}{'stories':>11}{'characters':>15}{'text tokens':>14}{'packed':>14}"
        f"{'sequences':>11}{'dropped':>9}{'retained':>10}"
    )
    print("\n" + header + "\n" + "-" * len(header))
    for split, s in meta["splits"].items():
        print(
            f"{split:<11}{s['examples']:>11,}{s['characters']:>15,}{s['text_tokens']:>14,}"
            f"{s['packed_tokens']:>14,}{s['sequences']:>11,}{s['discarded_tokens']:>9,}"
            f"{100 * s['retained_fraction']:>9.4f}%"
        )
    print(
        f"\nWrote {cfg.data.data_dir}/{{train,validation,test}}.bin and meta.json "
        f"(dataset revision {meta['dataset_revision']})."
    )


if __name__ == "__main__":
    main()
