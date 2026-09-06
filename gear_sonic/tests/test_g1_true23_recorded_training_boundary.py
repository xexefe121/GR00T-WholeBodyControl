"""Offline witness validation and production history/controller regression."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from gear_sonic.envs.mjlab import sonic_true23 as obs
from gear_sonic.utils.g1_23dof_safe_target_transform import (
    SAFE_TARGET_DEFAULT_Q_HARDWARE,
    safe_target_transform_numpy,
)
from gear_sonic.utils.g1_true23_actuation_profile import SIM_CONFIG, NativeSupportActuationProfile
from gear_sonic.utils.g1_true23_clean_mujoco_teleop import DEFAULT_NATIVE, MJ_TO_NATIVE, NATIVE_TO_IL29
from gear_sonic.utils.g1_true23_projected_controller_state import applied_target_native_numpy
from gear_sonic.utils.g1_true23_recorded_training_boundary import (
    comparison,
    recorded_control_states,
    training_actuation,
    training_observations,
)
from gear_sonic.utils.g1_true23_step1b_mujoco import term_major_history


def fixture(complete=12, partial=4, terminal=True):
    steps = complete * 10 + (partial if terminal else 0)
    count = complete + int(terminal)
    qpos = np.zeros((steps, 30))
    qpos[:, 2:4] = (0.8, 1)
    qpos[:, 7:] = SAFE_TARGET_DEFAULT_Q_HARDWARE
    qpos[:, 7] += np.arange(steps) * 0.00001
    qvel = np.zeros((steps, 29))
    qvel[:, 6] = np.arange(steps) * 0.00002
    previous = qpos[:, 7:] + 0.001
    end_q = qpos[-1].copy()
    end_q[7] += 0.00001
    end_dq = qvel[-1].copy()
    end_dq[6] += 0.00002
    controls = qpos[np.arange(complete) * 10]
    completed_q = qpos[complete * 10] if partial and terminal else end_q
    arrays = dict(
        policy_encoder267=np.zeros((count, 267), np.float32),
        policy_raw23=np.zeros((count, 23), np.float32),
        physics_phase=np.ones(steps, np.int64),
        physics_pre_qpos=qpos,
        physics_pre_qvel=qvel,
        qpos=np.concatenate((controls, completed_q[None])),
        terminal_active_qpos=end_q,
        terminal_active_qvel=end_dq,
        terminal_active_target=previous[-1].copy(),
        actuation_q=qpos[:, 7:].copy(),
        actuation_dq=qvel[:, 6:].copy(),
        actuation_previous_target=previous.copy(),
        actuation_requested=previous.copy(),
        actuation_target=previous.copy(),
        actuation_effort=np.zeros((steps, 23)),
    )
    result = dict(
        completed_transitions=complete,
        completed_active_physics_steps=steps,
        failure={"type": "TargetIntersectionError"} if terminal else None,
    )
    return arrays, result


@pytest.mark.parametrize("partial,terminal", [(0, True), (4, True), (9, True), (0, False)])
def test_every_call_including_terminal_uses_actual_control_state(partial, terminal):
    arrays, result = fixture(partial=partial, terminal=terminal)
    states = recorded_control_states(arrays, result)
    assert len(states["qpos"]) == 12 + int(terminal)
    np.testing.assert_array_equal(states["qpos"][:12], arrays["physics_pre_qpos"][:120:10])
    if terminal:
        np.testing.assert_array_equal(
            states["qpos"][-1], arrays["physics_pre_qpos"][120] if partial else arrays["terminal_active_qpos"]
        )
    assert states["startup_target_reconstruction_rows"] == 0


@pytest.mark.parametrize("fault", ["phase", "count", "nan", "actuation", "saved_control"])
def test_trace_drift_rejected(fault):
    arrays, result = fixture()
    if fault == "phase":
        arrays["physics_phase"][1] = 2
    elif fault == "count":
        result["completed_active_physics_steps"] += 1
    elif fault == "nan":
        arrays["physics_pre_qpos"][5, 7] = np.nan
    elif fault == "actuation":
        arrays["actuation_q"][0, 0] += 0.1
    else:
        arrays["qpos"][0, 0] += 0.1
    with pytest.raises((ValueError, AssertionError)):
        recorded_control_states(arrays, result)


def test_real_training_functions_and_buffer_reconstruct_temporal_order():
    arrays, result = fixture()
    states = recorded_control_states(arrays, result)
    names = list(obs._REQUIRED_REFERENCE_BODY_NAMES)
    frames = 40
    motion = dict(
        joint_pos=np.tile(SAFE_TARGET_DEFAULT_Q_HARDWARE, (frames, 1)).astype(np.float32),
        body_pos_w=np.zeros((frames, len(names), 3), np.float32),
        body_quat_w=np.zeros((frames, len(names), 4), np.float32),
    )
    motion["body_quat_w"][..., 0] = 1
    values = training_observations(motion, states, names)
    history = []
    expected = []
    for qpos, qvel, target in zip(states["qpos"], states["qvel"], states["previous_target"]):
        q29, dq29, action29 = np.zeros((3, 29))
        q29[NATIVE_TO_IL29] = qpos[7:][MJ_TO_NATIVE] - DEFAULT_NATIVE
        dq29[NATIVE_TO_IL29] = qvel[6:][MJ_TO_NATIVE]
        action29[NATIVE_TO_IL29] = applied_target_native_numpy(target)
        frame = np.concatenate((qvel[3:6], q29, dq29, action29, [0, 0, -1])).astype(np.float32)
        history = [frame.copy() for _ in range(10)] if not history else [*history[1:], frame]
        expected.append(term_major_history(history))
    np.testing.assert_allclose(values["history930"], expected, atol=1e-7, rtol=0)
    assert not np.array_equal(values["history930"][1], values["history930"][-1])
    np.testing.assert_array_equal(values["encoder267"][:, 261:], np.tile([1, 0, 0, 1, 0, 0], (13, 1)))
    # Stored policy arrays were dummy zeros; reconstruction cannot echo them.
    assert np.count_nonzero(values["encoder267"]) > 0


def test_training_projection_preserves_rejection_and_zero_effort_label():
    arrays, result = fixture()
    root = Path(__file__).resolve().parents[2]
    profile = replace(
        NativeSupportActuationProfile.from_sim_config(root / SIM_CONFIG), consistent_controller_state=True
    )
    arrays["terminal_active_target"][:] = 100
    output = training_actuation(arrays, result, profile)
    assert not output["invalid"].any()
    assert output["terminal"]["rejected_by_training"]
    assert output["terminal"]["training_zero_effort_until_50hz_reset_is_not_a_physical_return"]
    assert output["terminal"]["training_output_effort"] == [0.0] * 23
    target = safe_target_transform_numpy(arrays["policy_raw23"][0])[1]
    np.testing.assert_allclose(output["requested_per_control"], np.tile(target, (13, 1)), atol=2e-7, rtol=0)


def test_comparison_does_not_hide_small_changes_or_nonfinite_values():
    a, b = np.zeros((2, 3)), np.zeros((2, 3))
    b[1, 2] = 1e-7
    report = comparison(a, b, 1e-6)
    assert report["changed_elements"] == 1 and report["within_tolerance"]
    assert comparison(a, b, 0)["rows_outside_tolerance"] == [1]
    b[0, 0] = np.inf
    with pytest.raises(ValueError):
        comparison(a, b, 1)


def test_startup_history_uses_recorded_pd_effort_not_policy_inputs():
    arrays, result = fixture()
    q = arrays["physics_pre_qpos"][:100].copy()
    dq = arrays["physics_pre_qvel"][:100].copy()
    target = q[:, 7:] + 0.002
    effort = 40 * (target - q[:, 7:]) - 2 * dq[:, 6:]
    arrays["physics_phase"] = np.concatenate((np.zeros(100, np.int64), arrays["physics_phase"]))
    arrays["physics_pre_qpos"] = np.concatenate((q, arrays["physics_pre_qpos"]))
    arrays["physics_pre_qvel"] = np.concatenate((dq, arrays["physics_pre_qvel"]))
    arrays["physics_effort"] = np.concatenate((effort, arrays["actuation_effort"]))
    result["startup_hold"] = dict(
        completed_physics_steps=100,
        standing_screen_passed=True,
        gain_kp_hardware=[40.0] * 23,
        gain_kd_hardware=[2.0] * 23,
    )
    states = recorded_control_states(arrays, result)
    assert states["startup_target_reconstruction_rows"] == 9
    np.testing.assert_array_equal(states["warmup_qpos"], q[10::10])
    np.testing.assert_allclose(states["warmup_target"], target[9:90:10], atol=1e-15, rtol=0)
    np.testing.assert_array_equal(states["startup_last_quaternion"], q[90, 3:7])
    result["startup_hold"]["standing_screen_passed"] = False
    with pytest.raises(ValueError, match="successful recorded startup"):
        recorded_control_states(arrays, result)
