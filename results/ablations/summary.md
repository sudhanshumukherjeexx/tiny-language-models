| Variant | Position | Norm | FFN | KV heads | Params | Seeds | Val loss (mean ± std) | Val PPL | tok/s | Peak VRAM (MB) |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| baseline-gpt-style | learned | layernorm | gelu | 8 | 26,756,096 | 3 | 2.2208 ± 0.0019 | 9.215 ± 0.018 | 26197 ± 935 | 2474 ± 2 |
| rope | rope | layernorm | gelu | 8 | 26,756,096 | 3 | 2.0848 ± 0.0042 | 8.043 ± 0.033 | 22284 ± 733 | 2493 ± 0 |
| rmsnorm | rope | rmsnorm | gelu | 8 | 26,747,392 | 3 | 2.0887 ± 0.0028 | 8.075 ± 0.023 | 19559 ± 81 | 2765 ± 0 |
| swiglu | rope | rmsnorm | swiglu | 8 | 26,747,392 | 3 | 2.0600 ± 0.0183 | 7.847 ± 0.144 | 19220 ± 679 | 2969 ± 0 |
| gqa | rope | rmsnorm | swiglu | 2 | 26,747,392 | 3 | 2.0268 ± 0.0143 | 7.590 ± 0.109 | 19057 ± 64 | 3095 ± 0 |
