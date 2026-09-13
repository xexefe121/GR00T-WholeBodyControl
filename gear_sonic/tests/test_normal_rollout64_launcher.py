"""Rollout64 is a bounded separate lineage, not a silent edit to normal16."""

from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("mjlab")

from gear_sonic.scripts.train_g1_true23_normal_lora import agent_configuration as original_configuration
from gear_sonic.scripts.train_g1_true23_normal_rollout64 import (
    agent_configuration,
    install_returns_capture,
    run,
)


def test_only_rollout_length_changes_agent_configuration():
    args = SimpleNamespace(
        warm_start=Path("normal_init.pt"),
        source_checkpoint=Path("normal.pt"),
        seed=20260803,
        updates=100,
        rollout_steps=64,
    )
    new = asdict(agent_configuration(args))
    args.rollout_steps = 16
    old = asdict(original_configuration(args))
    assert new.pop("num_steps_per_env") == 64
    assert old.pop("num_steps_per_env") == 16
    assert new == old


@pytest.mark.parametrize(
    "updates,envs,steps", [(100, 128, 16), (101, 32, 64), (100, 128, 64), (2, 4, 8), (1, 32, 64)]
)
def test_only_declared_smoke_and_full_geometry_allowed(updates, envs, steps):
    with pytest.raises(ValueError, match="limited to"):
        run(SimpleNamespace(updates=updates, num_envs=envs, rollout_steps=steps))


def test_returns_capture_observes_one_real_forward_and_copies_storage():
    critic = torch.nn.Linear(3, 1)
    storage = SimpleNamespace(
        **{
            key: torch.arange(8, dtype=torch.float32).reshape(4, 2, 1)
            for key in ("values", "returns", "advantages", "rewards", "dones")
        }
    )
    calls = []

    def compute(obs):
        calls.append(critic(obs).detach())
        return 123

    runner = SimpleNamespace(alg=SimpleNamespace(critic=critic, storage=storage, compute_returns=compute))
    rows = install_returns_capture(runner)
    assert runner.alg.compute_returns(torch.ones(2, 3)) == 123
    assert len(calls) == len(rows) == 1
    np.testing.assert_array_equal(rows[0]["last_values"], calls[0].numpy())
    storage.values.zero_()
    assert rows[0]["values"].sum() == 28
    assert not critic._forward_hooks


def test_capture_hook_removed_on_compute_exception():
    critic = torch.nn.Linear(3, 1)

    def compute(obs):
        critic(obs)
        raise RuntimeError("test failure")

    runner = SimpleNamespace(alg=SimpleNamespace(critic=critic, compute_returns=compute))
    rows = install_returns_capture(runner)
    with pytest.raises(RuntimeError, match="test failure"):
        runner.alg.compute_returns(torch.ones(2, 3))
    assert rows == [] and not critic._forward_hooks
