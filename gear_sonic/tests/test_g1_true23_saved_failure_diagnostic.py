from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gear_sonic.utils.g1_true23_saved_failure_diagnostic import (
    contiguous_runs,
    fixed_observation_branch_outputs,
    limit_events,
    phase_path_metrics,
)


@pytest.mark.parametrize(
    "mask,expected",
    [([], []), ([0, 0], []), ([1, 1], [(0, 2)]), ([1, 0, 1, 1, 0], [(0, 1), (2, 4)])],
)
def test_contiguous_runs(mask, expected):
    assert contiguous_runs(np.asarray(mask, dtype=bool)) == expected


def path_fixture():
    n = 10
    q = np.zeros((n + 1, 30))
    q[:, 3] = 1
    q[:, 0] = np.arange(n + 1) * 0.02
    ref = np.zeros((n + 11, 36))
    ref[:, 3] = 1
    ref[10:, 0] = np.arange(n + 1) * 0.04
    velocity = np.zeros((n * 10, 29))
    velocity[:, 0] = 1
    phases = [dict(name="source_motion", control_start=0, control_stop=n)]
    return q, ref, phases, velocity


def test_path_integral_uses_q2_without_reanchoring():
    args = path_fixture()
    args[0][:, 1] = 0.3
    result = phase_path_metrics(*args)
    phase = result["phases"][0]
    assert result["translation_integral_max_abs_error_m"] < 1e-15
    assert phase["xy_velocity_least_squares_gain"] == pytest.approx(0.5)
    assert phase["end_error_xyz_m"] == pytest.approx([-0.2, 0.3, 0])
    assert phase["accumulated_velocity_mismatch_xyz_m"] == pytest.approx([-0.2, 0, 0])
    assert phase["root_q2_p95_m"] > 0.3


def test_path_rejects_forged_velocity_or_incomplete_phase():
    q, ref, phases, velocity = path_fixture()
    with pytest.raises(ValueError, match="integral"):
        phase_path_metrics(q, ref, phases, velocity * 0.5)
    phases[0]["control_stop"] = 9
    with pytest.raises(ValueError, match="incomplete"):
        phase_path_metrics(q, ref, phases, velocity)


def limit_fixture():
    n = 20
    trace = {
        "physics_post_qpos": np.zeros((n, 30)),
        "physics_pre_qpos": np.zeros((n, 30)),
        "physics_pre_qvel": np.zeros((n, 29)),
        "physics_post_qvel": np.zeros((n, 29)),
        "physics_time": np.column_stack((np.arange(n), np.arange(n) + 1)) * 0.002,
        "engine_actuator_force23": np.zeros((n, 23)),
        "requested_torque23": np.zeros((n, 23)),
        "target23": np.zeros((n // 10, 23)),
    }
    ranges = np.tile([-1.0, 1.0], (23, 1))
    names = [f"joint{i}" for i in range(23)]
    return trace, ranges, names


def test_lower_excursion_with_inward_braking_is_not_outward_motor_push():
    trace, ranges, names = limit_fixture()
    trace["physics_post_qpos"][3:6, 12] = [-1.1, -1.2, -1.15]
    trace["physics_pre_qvel"][3:6, 11] = [-0.5, -0.2, 0.1]
    trace["engine_actuator_force23"][:, 5] = 3
    trace["requested_torque23"][:, 5] = 3
    events = limit_events(trace, ranges, names)
    assert len(events) == 1
    event = events[0]
    assert event["start_substep"] == 3 and event["stop_substep_exclusive"] == 6
    assert event["peak"]["substep"] == 4
    assert event["peak"]["outward_motor_torque_nm"] == -3
    assert event["maximum_excess_rad"] == pytest.approx(0.2)
    assert event["motor_saturated_fraction"] == event["motor_outward_fraction"] == 0


def test_upper_excursion_with_saturation_and_boundary_touch():
    trace, ranges, names = limit_fixture()
    trace["physics_post_qpos"][0:3, 18] = [1.0, 1.01, 1.02]
    trace["engine_actuator_force23"][:, 11] = 4
    trace["requested_torque23"][:, 11] = 8
    event = limit_events(trace, ranges, names)[0]
    assert event["start_substep"] == 1
    assert event["side"] == "upper"
    assert event["motor_saturated_fraction"] == event["motor_outward_fraction"] == 1


def test_invalid_target_or_nonfinite_state_rejected():
    trace, ranges, names = limit_fixture()
    trace["target23"][0, 0] = 2
    with pytest.raises(ValueError, match="outside physical"):
        limit_events(trace, ranges, names)
    trace["target23"][0, 0] = 0
    trace["physics_post_qpos"][0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        limit_events(trace, ranges, names)


def test_branch_ablation_does_not_mutate_weights_or_observations():
    generator = torch.Generator().manual_seed(39)
    layers = torch.nn.Sequential(torch.nn.Linear(994, 8), torch.nn.ELU(), torch.nn.Linear(8, 23))
    actor = SimpleNamespace(
        core=SimpleNamespace(decoder=SimpleNamespace(module=layers)),
        root_conditioner=torch.nn.Linear(9, 8, bias=False),
        pose_lora_a=torch.randn(2, 994, generator=generator),
        pose_lora_b=torch.randn(8, 2, generator=generator),
    )
    x, root = torch.randn(3, 994, generator=generator), torch.randn(3, 9, generator=generator)
    saved = [v.clone() for v in (x, root, actor.pose_lora_a, actor.pose_lora_b)]
    result = fixed_observation_branch_outputs(actor, x, root)
    first = layers[0](x)
    pose = x @ actor.pose_lora_a.T @ actor.pose_lora_b.T
    expected = layers[2](layers[1](first + actor.root_conditioner(root) + pose))
    torch.testing.assert_close(result["full"], expected)
    torch.testing.assert_close(result["no_root"], layers[2](layers[1](first + pose)))
    torch.testing.assert_close(result["no_pose"], layers[2](layers[1](first + actor.root_conditioner(root))))
    for current, before in zip((x, root, actor.pose_lora_a, actor.pose_lora_b), saved, strict=True):
        assert torch.equal(current, before)
    root[:, :3] = 0
    result = fixed_observation_branch_outputs(actor, x, root)
    assert torch.equal(result["full"], result["no_root_position"])
