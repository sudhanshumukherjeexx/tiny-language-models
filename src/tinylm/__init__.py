"""TinyLM Lab: pretraining, evaluating and understanding a ~27M-parameter language model.

Quick start::

    from tinylm import TinyLM
    lm = TinyLM.from_pretrained("runs/tiny-27m/best")
    print(lm.generate("Once upon a time", max_new_tokens=100, temperature=0.8, seed=42))
"""

from tinylm.generation import PRESETS, TinyLM

__version__ = "0.2.0"
__all__ = ["PRESETS", "TinyLM", "__version__"]
