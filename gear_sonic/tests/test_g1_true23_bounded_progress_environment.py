"""Use original29/native23 geometry and the actual inherited task config."""

from dataclasses import asdict

import numpy as np
import pytest
import torch

from gear_sonic.envs.mjlab import sonic_true23_bounded_progress as bounded, sonic_true23_original_intent as intent
from gear_sonic.tests.test_g1_true23_original_intent_environment import make_cfg, reference as original_reference


@pytest.fixture
def model_reference(tmp_path):
    return original_reference.__wrapped__(tmp_path)


def test_real_config_leaves_every_nonreward_field_unchanged(model_reference):
    spec, _, _ = model_reference
    before = intent.configure_original_intent_environment(make_cfg(spec), spec)
    after = bounded.configure_environment(before)
    old, new = asdict(before), asdict(after)
    old.pop("rewards")
    new.pop("rewards")
    assert old == new
    assert after.terminations["ee_body_pos"].params["threshold"] == 0.25
    assert after.terminations == before.terminations


def test_potential_uses_actual_current_state_and_q1_not_q2_with_unchanged_original_intent(model_reference):
    spec, command, env = model_reference
    command.robot_anchor_pos_w = command.robot_body_pos_w[:, 0].clone()
    phi, parts = bounded.world_potential(env, spec)
    desired = command.motion.body_pos_w[[10, 10], 0] + command._env.scene.env_origins
    expected_root = 10 * (desired - command.robot_anchor_pos_w).square().sum(-1)
    torch.testing.assert_close(parts[:, 0], expected_root)
    np.testing.assert_allclose(phi.numpy(), -np.log1p(parts.numpy().sum(-1)), atol=1e-6)
    original_target = intent.task_states(env, spec)[0].clone()
    command.robot_anchor_pos_w[:, 0] += 0.3
    moved_phi, moved_parts = bounded.world_potential(env, spec)
    torch.testing.assert_close(moved_parts[:, 0], torch.full((2,), 0.9))
    assert (moved_phi < phi).all()
    torch.testing.assert_close(intent.task_states(env, spec)[0], original_target)
    command.motion.body_pos_w[11:] += 100
    command._original_intent_cache["position_w"][11:] += 100
    final_phi, final_parts = bounded.world_potential(env, spec)
    torch.testing.assert_close(final_phi, moved_phi, rtol=0, atol=0)
    torch.testing.assert_close(final_parts, moved_parts, rtol=0, atol=0)
