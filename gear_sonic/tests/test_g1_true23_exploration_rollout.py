import argparse
from types import SimpleNamespace

import numpy as np
import pytest

from gear_sonic.scripts.audit_g1_true23_exploration_rollout import (
    _noise_scale,
    audit_reset_contacts,
    lift_reset_floor_overlap,
    projection_failure_details,
    scale_reset_perturbations,
    summarize_episodes,
)


def test_counts_done_episodes_and_keeps_unfinished_runs_censored():
    done = np.array([[True, False], [False, False], [True, True], [False, False]])
    guard = np.array([[True, False], [False, False], [True, False], [False, False]])
    timeout = done & ~guard
    invalid = np.zeros_like(done)
    invalid[0, 0] = True
    report = summarize_episodes(done, {"stage_one_actuation_guard": guard, "timeout": timeout}, invalid)
    assert report["sampled_transitions"] == 8 and report["completed_episodes"] == 3
    assert report["completed_episode_mean_steps"] == 2
    assert report["completed_episode_maximum_steps"] == 3
    assert report["one_step_completed_episodes"] == 1
    assert report["unfinished_episode_lengths_steps"] == [1, 1]
    assert report["invalid_controller_at_episode_start"] == 1
    assert report["guard_terminations_from_invalid_episode_start"] == 1
    assert report["guard_terminations_without_invalid_start_that_step"] == 1


def test_no_done_is_not_a_full_clip_success():
    empty = np.zeros((4, 2), dtype=bool)
    report = summarize_episodes(empty, {"guard": empty}, empty)
    assert report["completed_episode_mean_steps"] is None
    assert report["completed_episodes"] == 0
    assert report["unfinished_episode_lengths_steps"] == [4, 4]


def test_overlapping_causes_are_counted_nonexclusively():
    done = np.ones((1, 1), dtype=bool)
    report = summarize_episodes(done, {"first": done, "second": done}, done)
    assert report["completed_episodes"] == 1
    assert report["termination_counts_nonexclusive"] == {"first": 1, "second": 1}


def test_missing_or_wrong_shape_term_evidence_fails():
    done = np.ones((2, 2), dtype=bool)
    with pytest.raises(ValueError, match="union"):
        summarize_episodes(done, {"guard": ~done}, done)
    with pytest.raises(ValueError, match="shape"):
        summarize_episodes(done, {"guard": np.ones((1, 2), dtype=bool)}, done)
    with pytest.raises(ValueError, match="boolean"):
        summarize_episodes(done.astype(int), {"guard": done}, done)


@pytest.mark.parametrize("value", ["-1", "1.1", "nan", "inf"])
def test_invalid_noise_scale_fails(value):
    with pytest.raises(argparse.ArgumentTypeError):
        _noise_scale(value)


@pytest.mark.parametrize("value", ["0", "0.25", "1"])
def test_zero_to_full_noise_scale_supported(value):
    assert _noise_scale(value) == float(value)


def test_failure_detail_does_not_mutate_or_hide_torch_numpy_disagreement():
    from gear_sonic.utils.g1_23dof_safe_target_transform import SAFE_TARGET_DEFAULT_Q_HARDWARE

    q = np.asarray(SAFE_TARGET_DEFAULT_Q_HARDWARE)
    before = q.copy()
    profile = SimpleNamespace(kp=np.ones(23), kd=np.ones(23), effort=np.ones(23), timestep_s=0.002, slew_rad_s=5.0)
    accepted = projection_failure_details(q, q, q, np.zeros(23), profile)
    assert not accepted["numpy_confirms_empty_interval"] and not accepted["joints"]
    rejected = projection_failure_details(q, q, q, np.ones(23) * 100, profile)
    assert rejected["numpy_confirms_empty_interval"] and rejected["joints"]
    assert all(row["instantaneous_minimum_slew_rad_s"] is None for row in rejected["joints"])
    np.testing.assert_array_equal(q, before)


def test_contact_audit_reports_geometric_penetration_without_changing_positions():
    import mujoco

    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><geom name="floor" type="plane" size="2 2 .1"/>'
        '<body pos="0 0 1"><freejoint/><geom name="ball" type="sphere" size=".1" mass="1"/>'
        "</body></worldbody></mujoco>"
    )
    states = np.tile(model.qpos0, (2, 1))
    states[0, 2] = 0.04
    before = states.copy()
    report = audit_reset_contacts(model, states)
    assert report["rows"][0]["minimum_floor_contact_distance_m"] == pytest.approx(-0.06)
    assert report["rows"][0]["floor_penetration_contact_count"] == 1
    assert report["rows"][1]["minimum_floor_contact_distance_m"] is None
    assert not report["dynamics_or_impulse_causation_proven"]
    np.testing.assert_array_equal(states, before)


@pytest.mark.parametrize("welded_plane", [False, True])
def test_floor_lift_preserves_articulation_and_does_not_lower_airborne_state(welded_plane):
    import mujoco

    plane = '<geom name="floor" type="plane" size="2 2 .1"/>'
    if welded_plane:
        plane = f'<body name="terrain">{plane}</body>'
    model = mujoco.MjModel.from_xml_string(
        f"<mujoco><worldbody>{plane}"
        '<body pos="0 0 1"><freejoint/><geom type="sphere" size=".1" mass="1"/>'
        '<body pos=".2 0 0"><joint/><geom type="sphere" size=".02" mass="1"/>'
        "</body></body></worldbody></mujoco>"
    )
    states = np.tile(model.qpos0, (2, 1)).astype(np.float32)
    states[0, 2] = 0.04
    states[:, -1] = 0.2
    before = states.copy()
    lifted, report = lift_reset_floor_overlap(model, states)
    np.testing.assert_array_equal(states, before)
    np.testing.assert_array_equal(lifted[:, [0, 1, 3, 4, 5, 6, 7]], before[:, [0, 1, 3, 4, 5, 6, 7]])
    np.testing.assert_array_equal(lifted[1], before[1])
    assert report["actual_lifts_m"] == pytest.approx([0.06001, 0])
    assert all(row["floor_penetration_contact_count"] == 0 for row in report["after"]["rows"])
    assert not report["dynamic_contact_consistency_proven"]


@pytest.mark.parametrize("plane", ["", 'quat=".92388 .38268 0 0"'])
def test_floor_lift_rejects_deep_or_tilted_floor_case(plane):
    import mujoco

    model = mujoco.MjModel.from_xml_string(
        f'<mujoco><worldbody><geom type="plane" size="2 2 .1" {plane}/>'
        '<body pos="0 0 -.3"><freejoint/><geom type="sphere" size=".1" mass="1"/>'
        "</body></worldbody></mujoco>"
    )
    with pytest.raises(ValueError, match="lift bound|horizontal"):
        lift_reset_floor_overlap(model, np.array([model.qpos0]))


def test_floor_lift_rejects_movable_mocap_floor():
    import mujoco

    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body mocap="true"><geom type="plane" size="2 2 .1"/></body>'
        '<body pos="0 0 1"><freejoint/><geom type="sphere" size=".1" mass="1"/>'
        "</body></worldbody></mujoco>"
    )
    with pytest.raises(ValueError, match="welded"):
        lift_reset_floor_overlap(model, np.array([model.qpos0]))


def test_scale_reset_perturbations_does_not_disable_observation_noise_or_change_reference():
    cfg = SimpleNamespace(
        pose_range={"z": (-0.1, 0.1)},
        velocity_range={"x": (-0.5, 0.5)},
        joint_position_range=(-0.1, 0.1),
        motion_file="unchanged.npz",
        enable_corruption=True,
    )
    scale_reset_perturbations(cfg, 0)
    assert cfg.pose_range == {"z": (0, 0)} and cfg.velocity_range == {"x": (0, 0)}
    assert cfg.joint_position_range == (0, 0)
    assert cfg.motion_file == "unchanged.npz" and cfg.enable_corruption
