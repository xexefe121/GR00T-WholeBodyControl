from dataclasses import replace
from pathlib import Path

import pytest
import torch

from gear_sonic.envs.mjlab.sonic_true23_stage_one_actuation import (
    apply_stage_one_actuation_profile,
    projection_cost_contract,
    requested_projection_cost,
)
from gear_sonic.utils.g1_23dof_contract import HARDWARE_23_ACTION_SCALE
from gear_sonic.utils.g1_true23_actuation_profile import SIM_CONFIG, NativeSupportActuationProfile

ROOT = Path(__file__).resolve().parents[2]


def test_zero_projection_is_zero_cost_and_normalization_is_per_joint():
    applied = torch.zeros(2, 23)
    assert torch.count_nonzero(requested_projection_cost(applied, applied)) == 0
    requested = torch.tensor(HARDWARE_23_ACTION_SCALE)[None].repeat(2, 1)
    requested[1] *= 2
    torch.testing.assert_close(requested_projection_cost(requested, applied), torch.tensor([1.0, 4.0]))


def test_nonfinite_cost_does_not_contaminate_other_environments():
    requested = torch.zeros(3, 23)
    requested[0, 0], requested[1, 0] = float("nan"), float("inf")
    cost = requested_projection_cost(requested, torch.zeros_like(requested))
    torch.testing.assert_close(cost, torch.tensor([100 / 23, 100 / 23, 0.0]))
    assert requested_projection_cost(torch.full_like(requested, 1e30), torch.zeros_like(requested)).max() == 100


def test_projection_cost_does_not_broadcast_or_silently_reorder():
    with pytest.raises(ValueError, match="equal"):
        requested_projection_cost(torch.zeros(23), torch.zeros(1, 23))
    with pytest.raises(ValueError, match="equal"):
        requested_projection_cost(torch.zeros(1, 22), torch.zeros(1, 22))


@pytest.mark.parametrize("weight", [True, -1, 101, float("nan"), float("inf"), "2"])
def test_invalid_cost_weights_are_rejected(weight):
    with pytest.raises(ValueError, match="projection penalty weight"):
        projection_cost_contract(weight)


def test_reward_config_is_opt_in_and_preserves_tracking_and_mechanics():
    from gear_sonic.envs.mjlab.sonic_true23_causal_multimotion_v14 import make_causal_multimotion_v14_env_cfg

    profile = replace(
        NativeSupportActuationProfile.from_sim_config(ROOT / SIM_CONFIG), consistent_controller_state=True
    )
    original = make_causal_multimotion_v14_env_cfg(motion_file="test.npz", num_envs=2, play=True)
    baseline = apply_stage_one_actuation_profile(original, profile)
    penalized = apply_stage_one_actuation_profile(original, profile, projection_penalty_weight=2)
    assert "requested_projection_l2" not in original.rewards
    assert "requested_projection_l2" not in baseline.rewards
    assert penalized.rewards["requested_projection_l2"].weight == -2
    assert set(penalized.rewards) == set(baseline.rewards) | {"requested_projection_l2"}
    for name, reward in baseline.rewards.items():
        assert penalized.rewards[name] == reward
    assert penalized.actions["joint_pos"].profile == baseline.actions["joint_pos"].profile
    assert penalized.terminations == baseline.terminations
    assert penalized.sim == baseline.sim
    assert penalized.actions["joint_pos"].record_requested_projection
    assert not baseline.actions["joint_pos"].record_requested_projection
    assert projection_cost_contract(0)["enabled"] is False
    assert projection_cost_contract(2)["controller_and_limits_changed"] is False
    with pytest.raises(ValueError, match="stateful V2"):
        apply_stage_one_actuation_profile(
            original, replace(profile, consistent_controller_state=False), projection_penalty_weight=2
        )
