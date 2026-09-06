from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gear_sonic.utils.g1_23dof_safe_target_transform import SAFE_TARGET_DEFAULT_Q_HARDWARE
from gear_sonic.utils.g1_true23_actuation_profile import NativeSupportActuationProfile
from gear_sonic.utils.g1_true23_interior_target_filter import interior_effort_target, run_interior_case
from gear_sonic.utils.g1_true23_sim_acquisition import (
    TargetIntersectionError,
    effort_feasible_target,
    effort_target_interval,
)


@pytest.fixture
def values():
    root = Path(__file__).resolve().parents[2]
    profile = NativeSupportActuationProfile.from_sim_config(
        root / "gear_sonic/config/sim_validation/g1_23dof_mujoco_sim2sim.json"
    )
    q = np.array(SAFE_TARGET_DEFAULT_Q_HARDWARE)
    return q, q.copy(), q.copy(), np.zeros(23), *map(np.asarray, (profile.kp, profile.kd, profile.effort))


@pytest.mark.parametrize("fraction", [0.0, 0.25, 0.5, 0.9, 1.0])
def test_inner_target_keeps_outer_position_effort_and_slew_bounds(values, fraction):
    rng = np.random.default_rng(401)
    for _ in range(100):
        requested, previous, q, dq, kp, kd, effort = [value.copy() for value in values]
        q += rng.normal(0, 0.04, 23)
        dq += rng.normal(0, 0.4, 23)
        previous[:] = q + kd * dq / kp
        requested += rng.normal(0, 0.8, 23)
        target, report = interior_effort_target(
            requested, previous, q, dq, kp, kd, effort, dt=0.002, slew_rate=5, fraction=fraction
        )
        low, high = effort_target_interval(previous, q, dq, kp, kd, effort, dt=0.002, slew_rate=5)
        assert np.all(target >= low) and np.all(target <= high)
        assert np.all(np.abs(kp * (target - q) - kd * dq) <= 0.2375 * effort + 1e-10)
        assert np.all(np.abs(target - previous) <= 0.01 + 1e-12)
        assert not report["hard_limits_relaxed"] and not report["recursive_feasibility_proven"]
        if fraction == 1:
            np.testing.assert_array_equal(
                target, effort_feasible_target(requested, previous, q, dq, kp, kd, effort, dt=0.002, slew_rate=5)
            )


def test_inner_band_unavailable_uses_outer_minimum_effort_not_zero_torque_escape(values):
    requested, previous, q, dq, kp, kd, effort = values
    previous[10] += 0.2375 * effort[10] / kp[10] - 0.005
    requested[10] += 0.8
    target, report = interior_effort_target(*values, dt=0.002, slew_rate=5, fraction=0.5)
    assert target[10] == pytest.approx(previous[10] - 0.01, abs=1e-12)
    np.testing.assert_array_equal(target[np.arange(23) != 10], q[np.arange(23) != 10])
    assert report["inner_band_unavailable_joints"] == [10]
    assert 0.5 < report["maximum_outer_effort_ratio"] < 1


def test_initially_feasible_request_inside_inner_band_is_unchanged(values):
    values[0][16] += 0.001
    target, report = interior_effort_target(*values, dt=0.002, slew_rate=5, fraction=0.5)
    np.testing.assert_array_equal(target, values[0])
    assert report["changed_joints"] == 0 and not report["inner_band_unavailable_joints"]


@pytest.mark.parametrize("fraction", [0, 0.5, 1])
def test_outer_empty_intersection_still_rejects_with_exact_old_details(values, fraction):
    values[3][16] = 10
    with pytest.raises(TargetIntersectionError) as old:
        effort_feasible_target(*values, dt=0.002, slew_rate=5)
    with pytest.raises(TargetIntersectionError) as new:
        interior_effort_target(*values, dt=0.002, slew_rate=5, fraction=fraction)
    assert new.value.details == old.value.details


@pytest.mark.parametrize("fraction", [-0.01, 1.01, np.nan, np.inf, True, [0.5]])
def test_invalid_headroom_selection_cannot_change_limits(values, fraction):
    with pytest.raises(ValueError, match="fraction"):
        interior_effort_target(*values, dt=0.002, slew_rate=5, fraction=fraction)


@pytest.mark.parametrize(
    "missing", ["stateful_native_controller", "project_active_effort", "trace_active_actuation"]
)
def test_experiment_requires_recorded_stateful_actuation(missing):
    arguments = dict(stateful_native_controller=True, project_active_effort=True, trace_active_actuation=True)
    arguments[missing] = False
    with pytest.raises(ValueError):
        run_interior_case(fraction=0.9, **arguments)


@pytest.mark.parametrize("lifecycle", [False, True])
def test_real_native23_loop_keeps_baseline_and_records_every_phase(values, lifecycle):
    from gear_sonic.scripts import evaluate_g1_true23_deployment_envelope as envelope
    from gear_sonic.utils.g1_true23_clean_mujoco_teleop import CleanTrue23MujocoController

    root = Path(__file__).resolve().parents[2]
    assets = root.parent / "GR00T-WholeBodyControl"
    controller = CleanTrue23MujocoController(
        model_path=assets / envelope.MODEL, physics_path=root / envelope.PHYSICS, policy=None
    )
    q = np.asarray(SAFE_TARGET_DEFAULT_Q_HARDWARE)
    controller.reset(
        base_position=np.array([0, 0, 0.8]), base_quaternion_wxyz=np.array([1, 0, 0, 0]), joint_position_hardware=q
    )
    controller.module.mj_forward(controller.model, controller.data)
    motion = dict(
        fps=np.array([50.0]),
        joint_pos=np.tile(q, (15, 1)),
        joint_vel=np.zeros((15, 23)),
        body_pos_w=np.tile(controller.data.xpos[1:], (15, 1, 1)),
        body_quat_w=np.tile(controller.data.xquat[1:], (15, 1, 1)),
        body_lin_vel_w=np.zeros((15, 24, 3)),
        body_ang_vel_w=np.zeros((15, 24, 3)),
    )
    kp, kd = values[4], values[5]

    class Balance:
        def reset(self):
            pass

        def activate(self, q):
            pass

        def infer(self, **kwargs):
            return q.copy(), kp.copy(), kd.copy()

    options = dict(
        root=root,
        asset_root=assets,
        policy=SimpleNamespace(infer=lambda encoder, history: (np.zeros(23, dtype=np.float32), None)),
        motion=motion,
        kp=kp,
        kd=kd,
        joint_scale=np.ones(23),
        ankle_effort=35.0,
        slew_rate=5.0,
        initial_state="neutral" if lifecycle else "reference",
        maximum_steps=2,
        startup_hold_s=0.04 if lifecycle else 0,
        return_hold_s=0.04 if lifecycle else 0,
        transition_policy=Balance() if lifecycle else None,
        project_transition_effort=lifecycle,
        project_active_effort=True,
        stateful_native_controller=True,
        trace_active_actuation=True,
    )
    old_function = envelope.run_case
    old_report, old = envelope.run_case(fraction=1.0, **options)
    report, arrays = run_interior_case(fraction=1.0, **options)
    assert report["completed_transitions"] == old_report["completed_transitions"] == 2
    for key in old:
        np.testing.assert_array_equal(arrays[key], old[key])
    assert envelope.run_case is old_function
    assert report["actual_engine_audit"]["passed"]
    assert report["interior_effort_experiment"]["changed_joint_substeps"] == 0
    assert np.sum(arrays["physics_phase"] == 1) == 20
    assert np.sum(arrays["physics_phase"] == 0) == (20 if lifecycle else 0)
    assert np.sum(arrays["physics_phase"] == 2) == (20 if lifecycle else 0)
    assert arrays["physics_engine_post_time_s"][-1] == pytest.approx(0.12 if lifecycle else 0.04, abs=1e-12)
    assert np.max(np.abs(arrays["physics_effort"] - arrays["physics_generalized_actuator_force"])) < 1e-12
    assert len(report["compiled_native_model_sha256"]) == 64
