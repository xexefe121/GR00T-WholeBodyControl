from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gear_sonic.utils.g1_23dof_safe_target_transform import SAFE_TARGET_DEFAULT_Q_HARDWARE
from gear_sonic.utils.g1_true23_actuation_profile import SIM_CONFIG, NativeSupportActuationProfile
from gear_sonic.utils.g1_true23_predictive_target_filter import (
    MujocoTargetPreview,
    PredictiveTargetError,
    filter_with_preview,
)
from gear_sonic.utils.g1_true23_sim_acquisition import (
    TargetIntersectionError,
    effort_feasible_target,
    effort_target_interval,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def values():
    profile = NativeSupportActuationProfile.from_sim_config(ROOT / SIM_CONFIG)
    q = np.asarray(SAFE_TARGET_DEFAULT_Q_HARDWARE)
    return q, np.zeros(23), *map(np.asarray, (profile.kp, profile.kd, profile.effort))


def test_joint_diagnostics_preserve_rejection_boundary(values):
    q, dq, kp, kd, effort = values
    dq[16] = 10
    with pytest.raises(TargetIntersectionError, match="empty effort/position/slew") as caught:
        effort_feasible_target(q, q, q, dq, kp, kd, effort, dt=0.002, slew_rate=5.0)
    row = caught.value.details[0]
    assert row["compact_index"] == 16 and row["joint"] == "left_elbow_joint"
    assert not row["infeasible_without_slew"]
    assert row["intersection_gap_rad"] > 0 and row["instantaneous_minimum_slew_rad_s"] > 5


@pytest.mark.parametrize("slew_rate", [5.0, None])
def test_intrinsically_empty_interval_has_no_finite_slew_solution(values, slew_rate):
    q, dq, kp, kd, effort = values
    dq[16] = 100
    with pytest.raises(TargetIntersectionError) as caught:
        effort_feasible_target(q, q, q, dq, kp, kd, effort, dt=0.002, slew_rate=slew_rate)
    row = caught.value.details[0]
    assert row["joint"] == "left_elbow_joint" and row["infeasible_without_slew"]
    assert row["instantaneous_minimum_slew_rad_s"] is None


def test_feasible_greedy_target_is_unchanged(values):
    q, dq, kp, kd, effort = values
    requested = q + 0.2
    target, report = filter_with_preview(requested, q, *values, dt=0.002, slew_rate=5.0, predict=lambda _: (q, dq))
    expected = effort_feasible_target(requested, q, *values, dt=0.002, slew_rate=5.0)
    np.testing.assert_array_equal(target, expected)
    assert report["preview_calls"] == 1 and report["linearization_iterations"] == 0
    assert not report["recursive_feasibility_proven"] and not report["hardware_authorized"]


def test_linear_preview_changes_target_before_slew_effort_conflict(values):
    q, dq, kp, kd, effort = values
    width = 0.2375 * effort / kp

    def predict(target):
        future = q.copy()
        future[10] += width[10] + 0.005 + 3 * (target[10] - q[10])
        return future, dq.copy()

    target, report = filter_with_preview(q + 0.2, q, *values, dt=0.002, slew_rate=5.0, predict=predict)
    assert report["greedy_minimum_next_interval_width_rad"] < 0
    assert report["linearization_iterations"] >= 1
    assert target[10] - q[10] == pytest.approx((0.005 - 1e-5) / 2, abs=1e-7)
    assert np.all(np.abs(target - q) <= 0.01 + 1e-12)
    future, velocity = predict(target)
    lower, upper = effort_target_interval(target, future, velocity, kp, kd, effort, dt=0.002, slew_rate=5.0)
    assert np.min(upper - lower) >= 1e-5 * 0.99
    np.testing.assert_allclose(target[np.arange(23) != 10], (q + 0.01)[np.arange(23) != 10])


def test_search_failure_is_explicit_not_false_global_infeasibility_proof(values):
    q, dq, _, _, _ = values
    with pytest.raises(PredictiveTargetError, match="bounded offline optimizer") as caught:
        filter_with_preview(q, q, *values, dt=0.002, slew_rate=5.0, predict=lambda _: (q + 10, dq))
    report = caught.value.details
    assert report["preview_calls"] <= 97
    assert report["accepted_minimum_next_interval_width_rad"] < 0
    assert report["limiting_joints"] and not report["recursive_feasibility_proven"]


@pytest.mark.parametrize("copy_protocol_only", [False, True])
def test_exact_mujoco_preview_preserves_live_state_and_matches_step(values, copy_protocol_only):
    from gear_sonic.utils.g1_true23_clean_mujoco_teleop import CleanTrue23MujocoController

    model = ROOT.parent / "GR00T-WholeBodyControl/gear_sonic/data/robots/g1/g1_23dof_rev_1_0.xml"
    if not model.is_file():
        pytest.skip("local model unavailable")
    q, dq, kp, kd, _ = values
    controller = CleanTrue23MujocoController(model_path=model, physics_path=ROOT / SIM_CONFIG, policy=None)
    controller.reset(
        base_position=np.array([0.0, 0.0, 0.8]),
        base_quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        joint_position_hardware=q,
    )
    before = {
        key: getattr(controller.data, key).copy()
        for key in ("qpos", "qvel", "ctrl", "qacc_warmstart", "qfrc_applied", "xfrc_applied")
    }
    if copy_protocol_only:
        controller.module = SimpleNamespace(MjData=controller.module.MjData, mj_step=controller.module.mj_step)
    preview = MujocoTargetPreview(controller, kp, kd)
    target = q + 0.005
    expected_q, expected_dq = preview(target)
    second_q, second_dq = preview(target)
    np.testing.assert_array_equal(second_q, expected_q)
    np.testing.assert_array_equal(second_dq, expected_dq)
    for key, value in before.items():
        np.testing.assert_array_equal(getattr(controller.data, key), value)
    assert controller.data.time == 0
    controller.data.ctrl[:] = kp * (target - q) - kd * dq
    controller.module.mj_step(controller.model, controller.data)
    np.testing.assert_array_equal(controller.data.qpos[7:], expected_q)
    np.testing.assert_array_equal(controller.data.qvel[6:], expected_dq)
