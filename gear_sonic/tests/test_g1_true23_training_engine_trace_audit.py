from types import SimpleNamespace

import numpy as np
import pytest

from gear_sonic.scripts.audit_g1_true23_training_engine_lifecycle import audit_trace, compare_prefix


def fixture_trace(completed=2, partial=0):
    steps = completed * 10 + partial
    actions = completed + int(partial > 0)
    q = np.zeros((steps + 1, 30), np.float32)
    q[:, 3] = 1
    v = np.zeros((steps + 1, 29), np.float32)
    times = np.arange(steps + 1, dtype=np.float32) * np.float32(0.002)
    return dict(
        qpos=q[: completed + 1].copy(), qvel=v[: completed + 1].copy(),
        physics_pre_qpos=q[:-1].copy(), physics_post_qpos=q[1:].copy(),
        physics_pre_qvel=v[:-1].copy(), physics_post_qvel=v[1:].copy(),
        requested_torque23=np.zeros((steps, 23), np.float32),
        applied_torque23=np.zeros((steps, 23), np.float32),
        physics_time=np.stack((times[:-1], times[1:]), axis=1),
        target23=np.zeros((actions, 23), np.float32),
        batch32_raw_repeats=np.zeros((actions, 8, 23), np.float32),
        batch32_token_repeats=np.zeros((actions, 8, 64), np.float32),
        released_model_raw23=np.zeros((actions, 23), np.float32),
        decoder994=np.zeros((actions, 994), np.float32),
    )


PROFILE = SimpleNamespace(kp=(40,) * 23, kd=(2,) * 23, effort=(25,) * 23)


@pytest.mark.parametrize("partial", [0, 1, 9, 10])
def test_complete_controls_and_any_stopped_partial_control_preserved(partial):
    result = audit_trace(fixture_trace(partial=partial), 2, PROFILE)
    assert result["actual_substeps"] == 20 + partial
    assert result["state_continuity_and_control_boundaries_exact"]
    assert result["float32_native_PD_and_effort_saturation_exact"]


@pytest.mark.parametrize("key", ["qpos", "physics_pre_qpos", "requested_torque23", "applied_torque23"])
def test_wrong_saved_state_or_torque_rejected(key):
    trace = fixture_trace()
    trace[key][1, 0] += np.float32(0.1)
    with pytest.raises(AssertionError):
        audit_trace(trace, 2, PROFILE)


def test_nonfinite_measurement_rejected():
    trace = fixture_trace()
    trace["physics_post_qvel"][0, 0] = np.nan
    with pytest.raises(ValueError):
        audit_trace(trace, 2, PROFILE)


def test_skipped_clock_step_rejected():
    trace = fixture_trace()
    trace["physics_time"][3, 0] += np.float32(0.001)
    with pytest.raises(AssertionError):
        audit_trace(trace, 2, PROFILE)


def test_repeated_inference_is_measured_not_required_exact():
    trace = fixture_trace()
    trace["batch32_raw_repeats"][0, 2, 1] = np.float32(0.2)
    result = audit_trace(trace, 2, PROFILE)
    assert result["raw_repeat_max_abs_difference"] == float(np.float32(0.2))


def test_prefix_comparison_does_not_extend_shorter_run():
    gpu, cpu = fixture_trace(2), fixture_trace(3)
    gpu["qpos"][2, 0] = 0.02
    gpu["qpos"][1, 7] = 0.002
    gpu["decoder994"][1, 0] = 0.1
    result = compare_prefix(gpu, cpu)
    assert result["shared_control_boundary_states"] == 3
    assert result["first_boundary_root_difference_over_1cm"] == 2
    assert result["first_boundary_joint_difference_over_1mrad"] == 1
    assert result["first_different_token_control"] == 1
