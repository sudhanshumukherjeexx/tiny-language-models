"""Generate text from a trained model.

Examples:
    uv run python scripts/generate.py --model runs/tiny-27m/best --prompt "Once upon a time"
    uv run python scripts/generate.py --model runs/tiny-27m/best --preset greedy
    uv run python scripts/generate.py --model runs/tiny-27m/best --temperature 0.7 --top-k 40 --seed 3 -n 3
"""

from __future__ import annotations

import argparse
import json

from tinylm import PRESETS, TinyLM
from tinylm.utils import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model", required=True, help="Saved model directory or Hugging Face Hub id."
    )
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--preset", choices=sorted(PRESETS), help="Named decoding preset.")
    parser.add_argument("--temperature", type=float, help="Sampling temperature (0 = greedy).")
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--repetition-penalty", type=float)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0, help="Base seed; sample i uses seed + i.")
    parser.add_argument("-n", "--num-samples", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--json", action="store_true", help="Print full results with metadata as JSON lines."
    )
    args = parser.parse_args()
    setup_logging()

    lm = TinyLM.from_pretrained(args.model, device=args.device)
    for i in range(args.num_samples):
        result = lm.generate_full(
            args.prompt,
            args.max_new_tokens,
            args.temperature,
            args.top_p,
            args.top_k,
            args.repetition_penalty,
            args.preset,
            args.seed + i,
        )
        if args.json:
            print(json.dumps(result.to_dict()))
        else:
            print(
                f"--- sample {i} | preset={result.preset} seed={result.seed} "
                f"new_tokens={result.new_tokens} eos={result.finished_with_eos}"
            )
            print(result.text, end="\n\n")


if __name__ == "__main__":
    main()
