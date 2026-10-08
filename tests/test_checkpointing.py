import copy
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from tinylm.checkpointing import (
    ExperimentState,
    capture_rng_state,
    latest_checkpoint,
    list_checkpoints,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint,
)
from tinylm.modeling import build_model, weights_are_tied
from tinylm.training import PretrainingRun, batch_indices, build_optimizer, lr_multiplier
from tinylm.utils import read_jsonl, seed_everything


def _run(cfg, packed, tokenizer):
    seed_everything(cfg.training.seed)
    model = build_model(cfg.model)
    return PretrainingRun(cfg, model, packed[0], packed[1], torch.device("cpu"), tokenizer)


def _state_dict(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def test_interrupted_training_matches_uninterrupted(experiment, packed, tokenizer, tmp_path):
    """5 steps + stop + resume for 7 steps == 12 uninterrupted steps (bitwise on CPU)."""
    full_cfg = copy.deepcopy(experiment)
    full_cfg.training.output_dir = str(tmp_path / "full")
    full = _run(full_cfg, packed, tokenizer)
    full_metrics = full.train()

    split_cfg = copy.deepcopy(experiment)
    split_cfg.training.output_dir = str(tmp_path / "split")
    first = _run(split_cfg, packed, tokenizer)
    partial = first.train(stop_after_steps=5)
    assert partial["status"] == "partial" and partial["steps"] == 5
    del first

    # Fresh process-like restart: new model with different random init, resumed.
    torch.manual_seed(999)
    resumed = PretrainingRun(
        split_cfg,
        build_model(split_cfg.model),
        packed[0],
        packed[1],
        torch.device("cpu"),
        tokenizer,
    )
    metrics = resumed.train(resume="auto")

    for (k, a), b in zip(
        _state_dict(full.model).items(), _state_dict(resumed.model).values(), strict=True
    ):
        torch.testing.assert_close(a, b, atol=0, rtol=0, msg=f"parameter {k} differs")
    assert metrics["steps"] == full_metrics["steps"] == 12
    assert metrics["training_tokens"] == full_metrics["training_tokens"] == 12 * 8 * 32
    assert metrics["best_validation_loss"] == pytest.approx(full_metrics["best_validation_loss"])
    assert [i["start_step"] for i in metrics["invocations"]] == [0, 5]
    assert metrics["train_seconds"] >= metrics["invocations"][-1]["train_seconds"]

    history = read_jsonl(Path(split_cfg.training.output_dir) / "training_history.jsonl")
    val_steps = [r["step"] for r in history if "val_loss" in r]
    assert val_steps == sorted(set(val_steps))  # no duplicated evaluations after resume


def test_resume_discards_history_written_after_last_checkpoint(experiment, packed, tokenizer):
    run = _run(experiment, packed, tokenizer)
    run.train(stop_after_steps=6)  # checkpoints at steps 3 and 6
    hist = Path(experiment.training.output_dir) / "training_history.jsonl"
    with hist.open("a") as f:  # simulate a crash after step 6 was logged further
        f.write('{"step": 8, "train_loss": 1.0, "lr": 0.0, "grad_norm": 0.0}\n')
    resumed = PretrainingRun(
        experiment,
        build_model(experiment.model),
        packed[0],
        packed[1],
        torch.device("cpu"),
        tokenizer,
    )
    resumed.train()
    steps = [r["step"] for r in read_jsonl(hist) if "train_loss" in r]
    assert steps == sorted(steps) and steps.count(8) == 1


def test_resume_refuses_changed_configuration(experiment, packed, tokenizer):
    _run(experiment, packed, tokenizer).train(stop_after_steps=3)
    changed = copy.deepcopy(experiment)
    changed.training.learning_rate = 1e-2
    with pytest.raises(RuntimeError, match="Refusing to resume"):
        _run(changed, packed, tokenizer).train()
    allowed = copy.deepcopy(experiment)  # micro-batch/accumulation trade is allowed
    allowed.training.per_device_train_batch_size, allowed.training.gradient_accumulation_steps = (
        2,
        4,
    )
    _run(allowed, packed, tokenizer).train(stop_after_steps=1)


def test_checkpoint_restores_optimizer_scheduler_step_and_rng(model_cfg, experiment, tmp_path):
    model = build_model(model_cfg)
    opt = build_optimizer(model, experiment.training, torch.device("cpu"))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_multiplier(s, experiment.training))
    ids = torch.randint(0, model_cfg.vocab_size, (2, 16))
    for _ in range(3):
        model(input_ids=ids, labels=ids).loss.backward()
        opt.step()
        sched.step()
        opt.zero_grad()
    state = ExperimentState(run_id="x", global_step=3, sequences_seen=24, tokens_seen=768)
    random.seed(5)
    np.random.seed(5)
    torch.manual_seed(5)
    path = save_checkpoint(tmp_path, model, opt, sched, None, state)
    expected_draws = (random.random(), np.random.rand(), torch.rand(1).item())

    model2 = build_model(model_cfg)
    opt2 = build_optimizer(model2, experiment.training, torch.device("cpu"))
    sched2 = torch.optim.lr_scheduler.LambdaLR(
        opt2, lambda s: lr_multiplier(s, experiment.training)
    )
    restored = load_checkpoint(path, model2, opt2, sched2)
    assert (random.random(), np.random.rand(), torch.rand(1).item()) == expected_draws
    assert restored.global_step == 3 and restored.tokens_seen == 768
    assert sched2.last_epoch == sched.last_epoch and sched2.get_last_lr() == sched.get_last_lr()
    s1, s2 = opt.state_dict()["state"], opt2.state_dict()["state"]
    assert s1.keys() == s2.keys()
    for k in s1:
        torch.testing.assert_close(s1[k]["exp_avg_sq"], s2[k]["exp_avg_sq"])
    for a, b in zip(model.state_dict().values(), model2.state_dict().values(), strict=True):
        torch.testing.assert_close(a, b)
    assert weights_are_tied(model2)  # strict load kept the shared tensor shared


def test_rotation_keeps_last_checkpoints(model_cfg, experiment, tmp_path):
    model = build_model(model_cfg)
    opt = build_optimizer(model, experiment.training, torch.device("cpu"))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
    for step in (3, 6, 9):
        save_checkpoint(
            tmp_path, model, opt, sched, None, ExperimentState("x", global_step=step), keep_last=2
        )
    assert [p.name for p in list_checkpoints(tmp_path)] == ["step-000006", "step-000009"]
    assert latest_checkpoint(tmp_path).name == "step-000009"


def test_rng_capture_roundtrip():
    state = capture_rng_state()
    a = (random.random(), np.random.rand(), torch.rand(2))
    restore_rng_state(state)
    b = (random.random(), np.random.rand(), torch.rand(2))
    assert a[0] == b[0] and a[1] == b[1] and torch.equal(a[2], b[2])


def test_data_order_is_a_function_of_position():
    a = np.concatenate([batch_indices(10, s, 4, 7) for s in range(0, 40, 4)])
    b = np.concatenate([batch_indices(10, s, 2, 7) for s in range(0, 40, 2)])
    assert np.array_equal(a, b)  # independent of micro-batch size
    assert sorted(a[:10]) == list(range(10))  # each epoch is a permutation
    assert not np.array_equal(a[:10], a[10:20])  # reshuffled per epoch


def test_lr_schedule_warmup_and_cosine(experiment):
    t = experiment.training
    assert lr_multiplier(0, t) == pytest.approx(1 / t.warmup_steps)
    assert lr_multiplier(t.warmup_steps - 1, t) == pytest.approx(1.0)
    assert lr_multiplier(t.max_steps, t) == pytest.approx(t.min_lr_ratio)


def test_rerunning_a_finished_experiment_changes_nothing(experiment, packed, tokenizer):
    first = _run(experiment, packed, tokenizer).train()
    again = _run(experiment, packed, tokenizer).train()
    assert again["steps"] == first["steps"]
    assert len(again["invocations"]) == len(first["invocations"]) == 1
    assert again["training_tokens"] == first["training_tokens"]
