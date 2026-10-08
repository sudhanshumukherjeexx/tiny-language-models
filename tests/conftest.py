"""Tiny, deterministic fixtures so the whole suite runs on CPU in seconds."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from tinylm.config import ExperimentConfig, ModelConfig, TokenizerConfig
from tinylm.data import PackedDataset, pack_texts
from tinylm.tokenizer import train_tokenizer

NAMES = ["Lily", "Tom", "Sara", "Ben", "Mia", "Max"]
ANIMALS = ["cat", "dog", "bird", "frog", "duck"]
PLACES = ["park", "garden", "forest", "beach"]
OBJECTS = ["ball", "kite", "box", "hat", "toy"]


def make_stories(n: int = 300, seed: int = 0) -> list[str]:
    rng = random.Random(seed)
    stories = []
    for _ in range(n):
        name, animal, place, obj = (rng.choice(x) for x in (NAMES, ANIMALS, PLACES, OBJECTS))
        stories.append(
            f"Once upon a time, {name} went to the {place}. {name} saw a little {animal} with a "
            f'{obj}. "Can I play?" asked {name}. The {animal} was happy and they played all day. '
            f"At the end, {name} said thank you and went home."
        )
    return stories


@pytest.fixture(scope="session")
def stories() -> list[str]:
    return make_stories()


@pytest.fixture(scope="session")
def tokenizer(stories):
    cfg = TokenizerConfig(vocab_size=400, min_frequency=2)
    return train_tokenizer([stories], cfg, model_max_length=64)


@pytest.fixture
def model_cfg(tokenizer) -> ModelConfig:
    return ModelConfig(
        vocab_size=400,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
        max_position_embeddings=64,
    )


@pytest.fixture
def packed(stories, tokenizer) -> tuple[PackedDataset, PackedDataset]:
    _, train = pack_texts(stories[:250], tokenizer, block_size=32)
    _, val = pack_texts(stories[250:], tokenizer, block_size=32)
    return PackedDataset.from_array(train), PackedDataset.from_array(val)


@pytest.fixture
def experiment(model_cfg, tmp_path) -> ExperimentConfig:
    cfg = ExperimentConfig(name="test", model=model_cfg)
    cfg.tokenizer.vocab_size = model_cfg.vocab_size
    cfg.data.block_size = 32
    t = cfg.training
    t.output_dir = str(tmp_path / "run")
    t.max_steps, t.warmup_steps = 12, 2
    t.per_device_train_batch_size, t.gradient_accumulation_steps, t.eval_batch_size = 4, 2, 8
    t.log_every, t.eval_every, t.save_every = 2, 3, 3
    t.precision, t.learning_rate = "fp32", 3e-3
    return cfg.validate()


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
