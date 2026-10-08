"""Gradio demo for a trained TinyLM model.

Local:
    uv sync --extra app
    TINYLM_MODEL=runs/tiny-27m/best uv run python app/app.py

The model location comes from --model or the TINYLM_MODEL environment
variable (a local directory or a Hugging Face Hub id).
"""

from __future__ import annotations

import argparse
import os

import gradio as gr

from tinylm import TinyLM

EXAMPLES = [
    "Once upon a time",
    "Tom and Sara were playing in the garden when they found",
    "The little cat was very hungry, so she",
    "It started to rain, so Mia",
]


def build_demo(lm: TinyLM) -> gr.Blocks:
    meta = lm.metadata
    info = (
        f"**{meta['parameters']:,} parameters** · {meta['layers']} layers · d={meta['hidden_size']} · "
        f"{meta['attention_heads']} query / {meta['kv_heads']} KV heads · "
        f"{meta['context_length']}-token context · {meta['vocab_size']:,}-token byte-level BPE · "
        f"running on `{meta['device']}`"
    )

    def infer(prompt, max_new_tokens, temperature, top_p, top_k, seed):
        prompt = (prompt or "").strip() or "Once upon a time"
        result = lm.generate_full(
            prompt,
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=int(top_k),
            seed=int(seed),
        )
        status = (
            f"{result.new_tokens} new tokens · "
            f"{'ended with end-of-story token' if result.finished_with_eos else 'stopped at the length limit'}"
            f" · preset `{result.preset}`"
        )
        return result.text, status

    with gr.Blocks(title="TinyLM") as demo:
        gr.Markdown(
            "# TinyLM\nA ~27M-parameter Llama-style model pretrained from random initialization "
            "on TinyStories (simple children's stories). It continues text; it is not an "
            "assistant and has no knowledge beyond that domain.\n\n" + info
        )
        with gr.Row():
            with gr.Column(scale=1):
                prompt = gr.Textbox(label="Prompt", value=EXAMPLES[0], lines=3)
                max_new = gr.Slider(16, 400, value=200, step=8, label="Max new tokens")
                temperature = gr.Slider(
                    0.0, 1.5, value=0.8, step=0.05, label="Temperature (0 = greedy)"
                )
                top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-p")
                top_k = gr.Slider(0, 200, value=0, step=1, label="Top-k (0 = off)")
                seed = gr.Number(value=42, precision=0, label="Seed")
                button = gr.Button("Generate", variant="primary")
            with gr.Column(scale=1):
                output = gr.Textbox(label="Story", lines=16)
                status = gr.Markdown()
        gr.Examples(EXAMPLES, inputs=prompt)
        button.click(infer, [prompt, max_new, temperature, top_p, top_k, seed], [output, status])
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.environ.get("TINYLM_MODEL"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--share", action="store_true", help="Create a temporary public link.")
    args = parser.parse_args()
    if not args.model:
        raise SystemExit(
            "Set --model or TINYLM_MODEL to a trained model directory or Hub id "
            "(e.g. runs/tiny-27m/best after scripts/train.py)."
        )
    build_demo(TinyLM.from_pretrained(args.model, device=args.device)).launch(share=args.share)


if __name__ == "__main__":
    main()
