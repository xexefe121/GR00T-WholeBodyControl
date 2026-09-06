"""Synthetic admission/gradient checks, not physical standing evidence."""

import numpy as np
import pytest
import torch

from gear_sonic.scripts.fit_g1_true23_standing_lora import standing_loss, validate_standing_labels
from gear_sonic.utils.g1_23dof_safe_target_transform import (
    safe_target_transform_numpy,
    safe_target_transform_torch,
)


def labels():
    raw = np.zeros((500, 23), dtype=np.float32)
    time = np.arange(5001) * 0.002
    result = dict(
        raw_native23=raw,
        target_hardware23=safe_target_transform_numpy(raw)[1],
        encoder267=np.zeros((500, 267), dtype=np.float32),
        history930=np.zeros((500, 930), dtype=np.float32),
        physics_dt=np.array([0.002]),
        physics_engine_pre_time_s=time[:-1],
        physics_engine_post_time_s=time[1:],
        physics_engine_warning_counts=np.zeros((5000, 8), dtype=np.int64),
    )
    for key, width in (
        ("pre_qpos", 30),
        ("post_qpos", 30),
        ("pre_qvel", 29),
        ("post_qvel", 29),
        ("effort", 23),
        ("generalized_actuator_force", 23),
    ):
        result["physics_" + key] = np.zeros((5000, width))
    return result


def test_complete_numeric_evidence_accepted():
    validate_standing_labels(labels())


@pytest.mark.parametrize("key", ["encoder267", "history930", "raw_native23", "target_hardware23"])
def test_short_episode_is_not_a_standing_teacher(key):
    batch = labels()
    batch[key] = batch[key][:-1]
    with pytest.raises(ValueError, match="500 finite rows"):
        validate_standing_labels(batch)


@pytest.mark.parametrize("key", ["physics_pre_qpos", "physics_post_qvel", "physics_effort"])
def test_nonfinite_physics_rejected(key):
    batch = labels()
    batch[key][1, 0] = np.nan
    with pytest.raises(ValueError, match="5000 finite"):
        validate_standing_labels(batch)


def test_unrepresentable_relabel_rejected():
    batch = labels()
    batch["target_hardware23"][10, 10] += 0.1
    with pytest.raises(AssertionError):
        validate_standing_labels(batch)


def test_raw_action_limit_cannot_be_raised():
    batch = labels()
    batch["raw_native23"][0, 0] = 10.01
    with pytest.raises(ValueError, match="action bound"):
        validate_standing_labels(batch)


@pytest.mark.parametrize("bad_clock", [False, True])
def test_engine_warning_or_clock_reset_rejected(bad_clock):
    batch = labels()
    if bad_clock:
        batch["physics_engine_post_time_s"][-1] = 0
    else:
        batch["physics_engine_warning_counts"][-1, 0] = 1
    with pytest.raises(ValueError, match="continuous engine"):
        validate_standing_labels(batch)


def test_force_clipping_cannot_be_hidden():
    batch = labels()
    batch["physics_generalized_actuator_force"][3, 0] = 0.5
    with pytest.raises(AssertionError):
        validate_standing_labels(batch)


def test_exact_labels_zero_loss_and_outside_raw_still_has_correcting_gradient():
    label_raw = torch.zeros((2, 23), dtype=torch.float32)
    label_target = safe_target_transform_torch(label_raw)[1]
    assert standing_loss(label_raw, label_raw, label_target).item() == 0
    raw = torch.full((2, 23), 11.0, dtype=torch.float32, requires_grad=True)
    loss = standing_loss(raw, label_raw, label_target)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(raw.grad).all() and torch.all(raw.grad > 0)
