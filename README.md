# Tiny Language Models

Build a ~27M parameter language model from a random initialization and train it on [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories), using Hugging Face `transformers`. No pretrained weights are downloaded — every parameter is trained from scratch in this notebook.

## Architecture

Uses the decoder-only design that Llama 3, Mistral, Qwen 3, and Gemma converged on:

| Component | Choice |
|---|---|
| Normalization | RMSNorm (pre-norm) |
| Positional encoding | RoPE |
| FFN activation | SwiGLU |
| Attention | Grouped-Query Attention (GQA) |
| Linear biases | None |
| Embeddings | Tied |
| Attention kernel | Flash Attention (`attn_implementation="sdpa"`) |
| Generation | KV cache |

Built with `LlamaConfig` + `LlamaForCausalLM(config)` — not `.from_pretrained()`.

## What's inside the notebook

| Part | Contents |
|---|---|
| 1 | Load, inspect, and clean the TinyStories dataset |
| 2 | Train an 8,192-token BPE tokenizer, pack the corpus into fixed-length blocks |
| 3 | Build the Transformer (RMSNorm, RoPE, GQA, SwiGLU) |
| 4 | Training loop via `Trainer` — mixed precision, gradient accumulation, cosine LR, checkpointing |
| 5 | Surviving free-tier Colab GPUs (VRAM budgeting, disconnect recovery) |
| 6 | Evaluation — loss curves, perplexity, sampling strategies, KV-cache generation |

## Running it

Open `tiny_language_model.ipynb` in Google Colab (T4 GPU recommended).

- `USE_SUBSET = True` — quick pass, ~300K stories, ~45 min
- `USE_SUBSET = False` — full corpus, ~2.1M stories, ~4 hours

## Where to go next

- **Scale up**: `hidden_size=768, num_hidden_layers=12, num_attention_heads=12, num_key_value_heads=4` (~90M params, still fits a T4 with `LOW_MEMORY=True`)
- **Change the data**: the pipeline is dataset-agnostic — point Part 1 at any text corpus
- **Try other architectures**: swap `LlamaConfig` for `MistralConfig`, `Qwen3Config`, or `MixtralConfig` with one line changed
- **Instruction tuning**: fine-tune with `trl`'s `SFTTrainer` on prompt/response pairs

## References

| Config field | Paper |
|---|---|
| `LlamaForCausalLM` | Attention Is All You Need — Vaswani et al., 2017 |
| RMSNorm | Zhang & Sennrich, 2019 |
| RoPE | RoFormer — Su et al., 2021 |
| SwiGLU (`hidden_act="silu"`) | GLU Variants Improve Transformer — Shazeer, 2020 |
| GQA (`num_key_value_heads`) | Ainslie et al., 2023 |
| Flash Attention (`attn_implementation="sdpa"`) | Dao et al., 2022 |
| TinyStories | Eldan & Li, 2023 |
