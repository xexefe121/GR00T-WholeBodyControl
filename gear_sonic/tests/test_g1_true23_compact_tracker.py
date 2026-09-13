"""Focused native feature/action/learner checks, without physical transport."""

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from gear_sonic.utils.g1_true23_compact_features import HW_IN_PAD29, make_goal, pack_observation, quaternion_matrix
from gear_sonic.trl.mjlab.native23_compact_actor import CompactNative23Actor, reachable_source_bounds
from gear_sonic.utils.g1_23dof_contract import MUJOCO_TO_ISAACLAB_DOF
from gear_sonic.utils.g1_23dof_safe_target_transform import (
    SAFE_TARGET_DEFAULT_Q_HARDWARE,
    safe_target_transform_numpy,
)
from gear_sonic.utils.g1_true23_source_action_codec import SOURCE_SCALE_NATIVE_IL23, source_scaled_precompensation
from gear_sonic.scripts.train_g1_true23_compact_tracker import audit_reward


def sample(n=4):
    history = torch.arange(n * 930, dtype=torch.float32).reshape(n, 930) / 1000
    q = torch.tensor(SAFE_TARGET_DEFAULT_Q_HARDWARE).expand(n, -1).clone()
    identity = torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(n, -1)
    goal = make_goal(
        q,
        torch.zeros(n, 23),
        torch.zeros(n, 9),
        identity,
        identity,
        torch.zeros(n, 21),
        torch.zeros(n, 2, 3),
        torch.ones(n, 2) * 0.76,
    )
    return history, goal


def test_latest_physical_slots_and_q_reference():
    history, goal = sample()
    packed = pack_observation(history, goal)
    assert packed.shape == (4, 165)
    torch.testing.assert_close(packed[:, :3], history[:, 27:30], rtol=0, atol=0)
    for i, start in enumerate((30, 320, 610)):
        expected = torch.stack([history[:, start + 9 * 29 + idx] for idx in HW_IN_PAD29], -1)
        torch.testing.assert_close(packed[:, 3 + 23 * i : 3 + 23 * (i + 1)], expected, rtol=0, atol=0)
    torch.testing.assert_close(packed[:, 75:], goal, rtol=0, atol=0)
    torch.testing.assert_close(packed[:, 75:98], torch.tensor(SAFE_TARGET_DEFAULT_Q_HARDWARE).expand(4, -1))


def test_orientation_and_foot_frame():
    half = np.sqrt(0.5)
    quat = torch.tensor([[half, 0.0, 0.0, half]], dtype=torch.float32)
    expected = torch.tensor([[[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]])
    torch.testing.assert_close(quaternion_matrix(quat), expected, atol=2e-7, rtol=0)
    history, goal = sample(1)
    goal = make_goal(
        goal[:, :23],
        torch.zeros(1, 23),
        torch.zeros(1, 9),
        quat,
        quat,
        torch.zeros(1, 21),
        torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]),
        torch.ones(1, 2),
    )
    torch.testing.assert_close(goal[:, 82:88], torch.tensor([[0.0, -1.0, 0.0, 1.0, 0.0, 0.0]]), atol=2e-7, rtol=0)


def test_zero_residual_targets_and_learned_change():
    history, goal = sample()
    obs = TensorDict({"native_controller": pack_observation(history, goal)}, batch_size=[4])
    actor = CompactNative23Actor(obs, {"actor": ["native_controller"]}, "actor", 23)
    actor.update_normalization(obs)
    before = actor(obs).detach().clone()
    torch.testing.assert_close(before, torch.zeros_like(before), atol=1e-6, rtol=0)
    for raw in before.numpy():
        target = safe_target_transform_numpy(source_scaled_precompensation(raw)[0])[1]
        np.testing.assert_allclose(target, SAFE_TARGET_DEFAULT_Q_HARDWARE, atol=5e-7, rtol=0)
    optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)
    loss = (actor(obs) - 0.1).square().mean()
    loss.backward()
    optimizer.step()
    assert float((actor(obs) - before).detach().abs().max()) > 0.001
    assert sum(p.numel() for p in actor.parameters()) < 250_000
    replay = CompactNative23Actor(obs, {"actor": ["native_controller"]}, "actor", 23)
    replay.load_state_dict(actor.state_dict())
    torch.testing.assert_close(replay(obs), actor(obs), atol=0, rtol=0)


def test_non_default_reference_targets_are_preserved_when_reachable():
    history, goal = sample()
    goal[:, :23] += 0.03
    obs = TensorDict({"native_controller": pack_observation(history, goal)}, batch_size=[4])
    actor = CompactNative23Actor(obs, {"actor": ["native_controller"]}, "actor", 23)
    expected = np.full(23, 0.03)[list(MUJOCO_TO_ISAACLAB_DOF)] / np.asarray(SOURCE_SCALE_NATIVE_IL23)
    np.testing.assert_allclose(actor(obs).detach().numpy(), np.tile(expected, (4, 1)), atol=2e-6, rtol=0)


def test_bounds_and_gaussian_likelihood_are_finite():
    history, goal = sample()
    obs = TensorDict({"native_controller": pack_observation(history, goal)}, batch_size=[4])
    actor = CompactNative23Actor(obs, {"actor": ["native_controller"]}, "actor", 23)
    action = actor(obs, stochastic_output=True)
    assert action.shape == (4, 23)
    assert torch.isfinite(actor.get_output_log_prob(action)).all()
    assert torch.isfinite(actor.output_entropy).all()
    lower, upper = reachable_source_bounds()
    assert np.all(lower < upper) and np.max(np.abs(np.r_[lower, upper])) < 9
    assert torch.all(actor.output_std > 0.03) and torch.all(actor.output_std < 0.5)


def test_bad_feature_shapes_and_quaternions_fail():
    history, goal = sample()
    with pytest.raises(ValueError):
        pack_observation(history[:, :929], goal)
    with pytest.raises(ValueError):
        quaternion_matrix(torch.zeros(4, 4))
    history[0, 0] = float("nan")
    with pytest.raises(ValueError):
        pack_observation(history, goal)


def test_reward_audit_detects_wrong_timeout_bootstrap():
    row = {
        "weighted_base_components": torch.ones(2, 2),
        "base_reward": torch.ones(2) * 2,
        "shaping_reward": torch.zeros(2),
        "world_quality_bonus": torch.zeros(2),
        "foot_precision_bonus": torch.zeros(2),
        "timeouts": torch.tensor([False, True]),
        "terminated": torch.tensor([False, False]),
    }
    values = torch.tensor([3.0, 4.0])
    rewards = torch.tensor([2.0, 2.0])
    stored = rewards + 0.99 * values * row["timeouts"]
    assert max(audit_reward(row, rewards, stored, values, 0.99).values()) < 1e-6
    with pytest.raises(ValueError):
        audit_reward(row, rewards, rewards, values, 0.99)
