---
title: TinyLM
emoji: 📖
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
license: mit
---

# TinyLM demo

Story continuation with a ~27M-parameter Llama-style model pretrained from
random initialization on TinyStories.

## Run locally

```bash
uv sync --extra app
TINYLM_MODEL=runs/tiny-27m/best uv run python app/app.py
```

## Deploy to Hugging Face Spaces

This directory is a complete Space (the YAML header above is the Space config).

1. Upload the trained model to the Hub first (see `model_card.md`).
2. Create a Gradio Space, then copy `app.py`, `requirements.txt` and this
   `README.md` into it.
3. In the Space settings, add the variable `TINYLM_MODEL=<your-hf-user>/<model-repo>`.

The Space runs on CPU; a 27M-parameter model generates at interactive speed
there (see `results/benchmarks/` for CPU measurements on one laptop CPU; Space
hardware will differ).
