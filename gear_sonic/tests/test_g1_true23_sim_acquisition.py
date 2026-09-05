from types import SimpleNamespace

import numpy as np
import pytest

from gear_sonic.utils.g1_23dof_safe_target_transform import SAFE_TARGET_DEFAULT_Q_HARDWARE
from gear_sonic.utils.g1_true23_sim_acquisition import (
    align_reference_xy_yaw,
    effort_feasible_target,
    simulate_balance_transition,
)


def test_reference_alignment_is_one_rigid_transform_without_mutating_source():
    generator = np.random.default_rng(23)
    motion = {
        "body_pos_w": generator.normal(size=(12, 24, 3)),
        "body_quat_w": np.tile([1.0, 0.0, 0.0, 0.0], (12, 24, 1)),
        "body_lin_vel_w": generator.normal(size=(12, 24, 3)),
        "body_ang_vel_w": generator.normal(size=(12, 24, 3)),
        "joint_pos": np.tile(SAFE_TARGET_DEFAULT_Q_HARDWARE, (12, 1)),
        "joint_vel": np.zeros((12, 23)),
        "fps": np.array([50.0]),
    }
    before = {key: value.copy() for key, value in motion.items()}
    for value in motion.values():
        value.setflags(write=False)
    qpos = np.r_[4.0, 5.0, 0.8, np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5), SAFE_TARGET_DEFAULT_Q_HARDWARE]
    result, report = align_reference_xy_yaw(motion, qpos)
    np.testing.assert_allclose(result["body_pos_w"][10, 0, :2], qpos[:2])
    np.testing.assert_array_equal(result["body_pos_w"][..., 2], before["body_pos_w"][..., 2])
    np.testing.assert_allclose(
        np.linalg.norm(np.diff(result["body_pos_w"], axis=0), axis=-1),
        np.linalg.norm(np.diff(before["body_pos_w"], axis=0), axis=-1),
    )
    for key in before:
        np.testing.assert_array_equal(motion[key], before[key])
    assert not report["height_changed"] and not report["per_frame_recentering"]


class ConstantPolicy:
    def reset(self):
        self.query_count = 0

    def activate(self, q):
        self.initial_q = q.copy()

    def infer(self, **kwargs):
        self.query_count += 1
        return np.asarray(SAFE_TARGET_DEFAULT_Q_HARDWARE) + 0.05, np.ones(23), np.ones(23)


def test_balance_transition_preserves_cadence_slew_and_prior_executed_target():
    data = SimpleNamespace(
        qpos=np.r_[0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0, SAFE_TARGET_DEFAULT_Q_HARDWARE],
        qvel=np.zeros(29),
        ctrl=np.zeros(23),
    )
    physics_steps = []
    controller = SimpleNamespace(
        data=data,
        model=None,
        physics=SimpleNamespace(timestep_s=0.002, decimation=10, effort=np.ones(23) * 100),
        module=SimpleNamespace(mj_step=lambda model, data: physics_steps.append(1)),
        history=[np.r_[np.zeros(92), -1.0] for _ in range(10)],
        _policy_frame=lambda: np.r_[np.zeros(92), -1.0],
    )
    policy = ConstantPolicy()
    report, states, target = simulate_balance_transition(controller, policy, duration_s=0.04, slew_rate=0.5)
    assert policy.query_count == 2 and len(physics_steps) == 20
    assert states.shape == (3, 30)
    np.testing.assert_allclose(target, np.asarray(SAFE_TARGET_DEFAULT_Q_HARDWARE) + 0.02)
    assert np.count_nonzero(controller.previous_safe_native) == 23
    assert report["standing_screen_passed"] and report["existing_guard_screen_passed"]
    assert not report["hardware_authorized"] and not report["unitree_mode_transfer_simulated"]


@pytest.mark.parametrize("duration", [0.0, -0.1, 11.0, 0.01, float("nan")])
def test_balance_duration_rejected_before_policy(duration):
    controller = SimpleNamespace(physics=SimpleNamespace(timestep_s=0.002))
    with pytest.raises(ValueError, match="balance duration"):
        simulate_balance_transition(controller, ConstantPolicy(), duration_s=duration, slew_rate=5.0)


def test_target_projection_satisfies_guard_without_loosening_effort():
    q = np.asarray(SAFE_TARGET_DEFAULT_Q_HARDWARE)
    target = effort_feasible_target(
        q + 0.5, q, q, np.zeros(23), np.ones(23) * 100, np.ones(23), np.ones(23) * 2, dt=0.002, slew_rate=5.0
    )
    assert np.max(100 * np.abs(target - q)) <= 0.95 * 0.25 * 2 + 1e-12
    assert np.max(np.abs(target - q)) <= 0.01


def test_infeasible_braking_projection_fails_instead_of_saturating():
    q = np.asarray(SAFE_TARGET_DEFAULT_Q_HARDWARE)
    with pytest.raises(ValueError, match="empty effort"):
        effort_feasible_target(
            q, q, q, np.ones(23) * 100, np.ones(23), np.ones(23), np.ones(23), dt=0.002, slew_rate=5.0
        )


@pytest.mark.parametrize(
    "dt,slew", [(0.0, 5.0), (-0.002, 5.0), (float("nan"), 5.0), (0.002, 0.0), (0.002, -1.0), (0.002, float("inf"))]
)
def test_projection_rejects_invalid_timestep_or_slew(dt, slew):
    q = np.asarray(SAFE_TARGET_DEFAULT_Q_HARDWARE)
    with pytest.raises(ValueError):
        effort_feasible_target(q, q, q, np.zeros(23), np.ones(23), np.ones(23), np.ones(23), dt=dt, slew_rate=slew)


def test_projection_rejects_negative_damping():
    q = np.asarray(SAFE_TARGET_DEFAULT_Q_HARDWARE)
    with pytest.raises(ValueError):
        effort_feasible_target(
            q, q, q, np.zeros(23), np.ones(23), -np.ones(23), np.ones(23), dt=0.002, slew_rate=5.0
        )


def _fake_controller(callback):
    data = SimpleNamespace(
        qpos=np.r_[0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0, SAFE_TARGET_DEFAULT_Q_HARDWARE],
        qvel=np.zeros(29),
        ctrl=np.zeros(23),
    )
    return SimpleNamespace(
        data=data,
        model=None,
        physics=SimpleNamespace(timestep_s=0.002, decimation=10, effort=np.ones(23) * 100),
        module=SimpleNamespace(mj_step=callback),
        history=[np.r_[np.zeros(92), -1.0] for _ in range(10)],
        _policy_frame=lambda: np.r_[np.zeros(92), -1.0],
    )


def test_failed_partial_interval_retains_terminal_state_and_exact_elapsed_time():
    def impossible_braking(model, data):
        data.qvel[6:] = 1000.0
        data.qpos[2] = 0.7

    controller = _fake_controller(impossible_braking)
    report, states, target = simulate_balance_transition(
        controller,
        ConstantPolicy(),
        duration_s=0.04,
        slew_rate=5.0,
        project_effort=True,
    )
    assert report["failure"] == "empty effort/position/slew target intersection"
    assert report["completed_transitions"] == 0
    assert report["completed_physics_steps"] == report["partial_transition_substeps"] == 1
    assert report["elapsed_simulation_s"] == 0.002
    assert states.shape == (2, 30)
    np.testing.assert_array_equal(states[-1], controller.data.qpos)
    assert report["minimum_height_m"] == 0.7
    assert not report["standing_screen_passed"] and np.isfinite(target).all()


def test_immediate_projection_failure_reports_initial_tilt_without_fake_step():
    controller = _fake_controller(lambda *_: pytest.fail("must not step physics"))
    controller.data.qpos[3:7] = [np.cos(0.3), np.sin(0.3), 0.0, 0.0]
    controller.data.qvel[6:] = 1000.0
    report, states, _ = simulate_balance_transition(
        controller,
        ConstantPolicy(),
        duration_s=0.04,
        slew_rate=5.0,
        project_effort=True,
    )
    assert report["maximum_tilt_rad"] == pytest.approx(0.6)
    assert report["completed_physics_steps"] == report["completed_transitions"] == 0
    assert states.shape == (1, 30)
    assert not report["existing_guard_screen_passed"]
