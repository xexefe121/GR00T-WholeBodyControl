"""Causal native23 per-frame IK, real FK, strict history and rejection contracts."""

from __future__ import annotations

import copy
import math
import time
from pathlib import Path

import numpy as np
import pytest

from gear_sonic.utils import g1_23dof_task_space_retarget as ik
from gear_sonic.utils.g1_23dof_contract import HARDWARE_23_JOINT_NAMES
from gear_sonic.utils.g1_23dof_safe_target_transform import SAFE_TARGET_DEFAULT_Q_HARDWARE
from gear_sonic.utils.g1_true23_generalist_causal import CausalNative23Retargeter, CausalRetargetConfig


@pytest.fixture(scope="module")
def models():
    root = Path(__file__).resolve().parents[2]
    return ik.load_models(root / ik.DEFAULT_SOURCE_MODEL, root / ik.DEFAULT_TARGET_MODEL)


def _core(models, *, initialized=True):
    core = CausalNative23Retargeter(*models)
    if initialized:
        core.initialize(
            joint_names=HARDWARE_23_JOINT_NAMES,
            joint_pos=np.asarray(SAFE_TARGET_DEFAULT_Q_HARDWARE),
            joint_vel=np.zeros(23),
            timestamp_s=0.0,
        )
    return core


def _frame(models, tick=1):
    names = ik._model_layout(models[0]).joint_names
    default = dict(zip(HARDWARE_23_JOINT_NAMES, SAFE_TARGET_DEFAULT_Q_HARDWARE, strict=True))
    return {
        "joint_names": names,
        "joint_pos": np.asarray([default.get(name, 0.0) for name in names]),
        "root_pos_w": np.asarray([0.0, 0.0, 0.8]),
        "root_quat_wxyz": np.asarray([1.0, 0.0, 0.0, 0.0]),
        "contact_flags": np.asarray([True, True]),
        "timestamp_s": tick * 0.02,
        "now_s": tick * 0.02,
    }


def test_real_models_follow_ten_single_standing_frames(models):
    core = _core(models)
    for tick in range(1, 11):
        result = core.step(**_frame(models, tick))
        assert result.accepted, result.diagnostics
        assert result.status == "accepted"
        assert result.reference.joint_pos.shape == (23,)
        assert result.reference.source_task_pos_w.shape == (len(ik.DEFAULT_TASKS), 3)
        assert tuple(result.reference.joint_names) == tuple(HARDWARE_23_JOINT_NAMES)
        assert result.diagnostics["active_joint_count"] == 23
        assert result.diagnostics["constraint_relaxation_count"] == 0
        assert result.diagnostics["joint_velocity_abs_max_rad_s"] <= 8 + 1e-7
        assert result.diagnostics["joint_acceleration_abs_max_rad_s2"] <= 80 + 1e-6
        assert result.diagnostics["task_position_error_m"]["right_hand"] == pytest.approx(0.084, abs=1e-6)
        assert not result.diagnostics["root_pose_optimized"]
        assert not result.diagnostics["controller_qualified"]
        assert not result.diagnostics["hardware_authorized"]
    assert core.state_snapshot()["last_accepted_timestamp_s"] == pytest.approx(0.2)
    assert core.timing_summary()["calls"] == 10
    assert not core.timing_summary()["scheduled_50hz_execution_verified"]


def test_requires_explicit_reference_initialization(models):
    result = _core(models, initialized=False).step(**_frame(models))
    assert result.status == "reinitialize_required"
    assert not result.accepted and result.reference is None


def test_joint_name_reordering_matches_same_single_frame(models):
    original = _frame(models)
    reversed_frame = {
        **original,
        "joint_names": tuple(reversed(original["joint_names"])),
        "joint_pos": original["joint_pos"][::-1],
    }
    first = _core(models).step(**original)
    second = _core(models).step(**reversed_frame)
    assert first.accepted and second.accepted
    np.testing.assert_array_equal(first.reference.joint_pos, second.reference.joint_pos)


def test_prefix_invariant_when_future_streams_differ(models):
    first_stream = [_frame(models, tick) for tick in range(1, 9)]
    second_stream = copy.deepcopy(first_stream)
    second_stream[-1]["joint_pos"][second_stream[-1]["joint_names"].index("waist_yaw_joint")] = 1.0
    first, second = _core(models), _core(models)
    first_outputs = [first.step(**frame) for frame in first_stream]
    second_outputs = [second.step(**frame) for frame in second_stream]
    for first_result, second_result in zip(first_outputs[:-1], second_outputs[:-1], strict=True):
        assert first_result.accepted and second_result.accepted
        np.testing.assert_array_equal(first_result.reference.joint_pos, second_result.reference.joint_pos)
        np.testing.assert_array_equal(
            first_result.reference.achieved_task_pos_w, second_result.reference.achieved_task_pos_w
        )
    assert first_outputs[-1].accepted
    assert not second_outputs[-1].accepted and second_outputs[-1].reference is None


def test_never_invokes_full_clip_solver_or_contact_inference(models, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("future-dependent helper called")

    monkeypatch.setattr(ik, "retarget_trajectory", forbidden)
    monkeypatch.setattr(ik, "infer_foot_contacts", forbidden)
    monkeypatch.setattr(ik, "_solve_lower_root_path", forbidden)
    assert _core(models).step(**_frame(models)).accepted


@pytest.mark.parametrize(
    "timestamp,status",
    [
        (0.0, "cadence_violation"),
        (-0.02, "cadence_violation"),
        (0.04, "cadence_violation"),
        (0.021, "cadence_violation"),
    ],
)
def test_gaps_duplicates_backwards_require_reinitialize(models, timestamp, status):
    core = _core(models)
    before = core.state_snapshot()
    frame = _frame(models)
    frame.update(timestamp_s=timestamp, now_s=timestamp)
    result = core.step(**frame)
    assert result.status == status and result.reference is None
    after = core.state_snapshot()
    np.testing.assert_array_equal(before["joint_pos"], after["joint_pos"])
    assert after["last_accepted_timestamp_s"] == before["last_accepted_timestamp_s"]
    assert after["requires_reinitialize"]
    assert core.step(**_frame(models)).status == "reinitialize_required"
    core.initialize(
        joint_names=HARDWARE_23_JOINT_NAMES,
        joint_pos=before["joint_pos"],
        joint_vel=before["joint_vel"],
        timestamp_s=1.0,
    )
    frame = _frame(models)
    frame.update(timestamp_s=1.02, now_s=1.02)
    assert core.step(**frame).accepted


@pytest.mark.parametrize("now", [0.061, 0.01])
def test_stale_or_future_input_rejected_without_reference(models, now):
    frame = _frame(models)
    frame["now_s"] = now
    result = _core(models).step(**frame)
    assert result.status == "stale_or_future_input"
    assert not result.accepted and result.reference is None
    assert result.diagnostics["requires_reinitialize"]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda f: f.update(joint_names=f["joint_names"][:23]),
        lambda f: f.update(joint_names=["joint"] * 29),
        lambda f: f.update(joint_pos=np.zeros((2, 29))),
        lambda f: f["joint_pos"].__setitem__(0, 100.0),
        lambda f: f["joint_pos"].__setitem__(0, np.nan),
        lambda f: f.update(contact_flags=np.asarray([1, 1])),
        lambda f: f.update(contact_flags=np.asarray([True])),
        lambda f: f.update(root_pos_w=np.zeros((2, 3))),
        lambda f: f.update(root_quat_wxyz=np.zeros(4)),
        lambda f: f.update(timestamp_s=float("nan")),
        lambda f: f.update(now_s=True),
    ],
)
def test_invalid_single_frame_never_emits_zero_or_fallback(models, mutation):
    core = _core(models)
    before = core.state_snapshot()
    frame = _frame(models)
    mutation(frame)
    result = core.step(**frame)
    assert not result.accepted and result.reference is None
    assert result.status == "invalid_input"
    after = core.state_snapshot()
    assert after["requires_reinitialize"]
    np.testing.assert_array_equal(after["joint_pos"], before["joint_pos"])
    np.testing.assert_array_equal(after["joint_vel"], before["joint_vel"])


def test_sudden_infeasible_pose_preserves_last_accepted_state(models):
    core = _core(models)
    assert core.step(**_frame(models)).accepted
    before = core.state_snapshot()
    frame = _frame(models, tick=2)
    frame["joint_pos"][frame["joint_names"].index("waist_yaw_joint")] = 1.0
    result = core.step(**frame)
    assert result.status == "infeasible_reference"
    assert result.reference is None
    assert result.diagnostics["failures"]
    after = core.state_snapshot()
    np.testing.assert_array_equal(after["joint_pos"], before["joint_pos"])
    np.testing.assert_array_equal(after["joint_vel"], before["joint_vel"])
    assert after["last_accepted_timestamp_s"] == before["last_accepted_timestamp_s"]


def test_returned_reference_and_snapshot_cannot_mutate_internal_history(models):
    core = _core(models)
    first = core.step(**_frame(models))
    first.reference.joint_pos[:] = 100
    snapshot = core.state_snapshot()
    snapshot["joint_pos"][:] = -100
    second = core.step(**_frame(models, tick=2))
    assert second.accepted
    assert np.max(np.abs(second.reference.joint_pos)) < 5


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cadence_s": 0.01},
        {"cadence_tolerance_s": 0.02},
        {"maximum_joint_velocity_rad_s": 8.01},
        {"maximum_joint_acceleration_rad_s2": 80.01},
        {"maximum_foot_error_m": 0.051},
        {"maximum_hand_head_error_m": 0.101},
        {"maximum_source_age_s": float("nan")},
        {"maximum_ik_iterations": True},
    ],
)
def test_configuration_cannot_relax_limits_or_hide_gap(kwargs):
    with pytest.raises(ValueError):
        CausalRetargetConfig(**kwargs)


def test_initialization_checks_joint_bounds_and_velocity(models):
    core = _core(models, initialized=False)
    position = np.asarray(SAFE_TARGET_DEFAULT_Q_HARDWARE).copy()
    position[0] = 100
    with pytest.raises(ValueError, match="physical joint range"):
        core.initialize(
            joint_names=HARDWARE_23_JOINT_NAMES, joint_pos=position, joint_vel=np.zeros(23), timestamp_s=0.0
        )
    with pytest.raises(ValueError, match="velocity exceeds"):
        core.initialize(
            joint_names=HARDWARE_23_JOINT_NAMES,
            joint_pos=np.asarray(SAFE_TARGET_DEFAULT_Q_HARDWARE),
            joint_vel=np.full(23, 9.0),
            timestamp_s=0.0,
        )
    assert core.state_snapshot()["requires_reinitialize"]


def test_small_source_excursion_does_not_extrapolate_seed_drift(models):
    core = _core(models)
    for tick in range(1, 101):
        frame = _frame(models, tick)
        frame["joint_pos"][frame["joint_names"].index("left_shoulder_roll_joint")] += 0.03 * math.sin(
            tick * 0.02 * 2 * math.pi
        )
        result = core.step(**frame)
        assert result.accepted, {"tick": tick, **result.diagnostics}
    assert core.timing_summary()["accepted_calls"] == 100


def test_nonfinite_solver_diagnostics_reject_without_state_update(models, monkeypatch):
    real_solver = ik._solve_frame

    def invalid_diagnostics(*args, **kwargs):
        position, diagnostics = real_solver(*args, **kwargs)
        diagnostics["weighted_task_error_after"] = float("nan")
        return position, diagnostics

    monkeypatch.setattr(ik, "_solve_frame", invalid_diagnostics)
    core = _core(models)
    result = core.step(**_frame(models))
    assert not result.accepted and result.reference is None
    assert "nonfinite solver diagnostics" in result.diagnostics["error"]
    assert core.state_snapshot()["last_accepted_timestamp_s"] == 0


def test_live_timing_history_is_bounded_and_overall_deadlines_remain_counted(models):
    core = _core(models, initialized=False)
    core._finish(started_s=time.perf_counter() - 0.03, status="test_only_rejected")
    for _ in range(4100):
        core._finish(started_s=time.perf_counter(), status="test_only_rejected")
    report = core.timing_summary()
    assert report["calls"] == 4101
    assert report["percentile_window_calls"] == 4096
    assert report["calls_exceeding_20ms"] >= 1
    assert report["maximum_s"] >= 0.03
    assert not report["all_observed_calls_within_20ms"]
