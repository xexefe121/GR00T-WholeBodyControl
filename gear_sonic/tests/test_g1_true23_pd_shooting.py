from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from gear_sonic.scripts.prepare_g1_true23_contact_step_lifecycle import motion_from_poses
from gear_sonic.tests.test_g1_true23_contact_step_transition import native  # noqa: F401
from gear_sonic.utils.g1_23dof_safe_target_transform import (
    SAFE_TARGET_DEFAULT_Q_HARDWARE,
    safe_target_transform_numpy,
)
from gear_sonic.utils.g1_true23_generalist_benchmark import PHYSICS
from gear_sonic.utils.g1_true23_native_model_actuation import NativeModelActuationProfile
from gear_sonic.utils.g1_true23_pd_box_step import box_backward_pass
from gear_sonic.utils.g1_true23_pd_shooting import PdShootingPlant, state_difference
from gear_sonic.utils.g1_true23_pd_trajectory_optimizer import (
    CONTROL_WEIGHT,
    SLEW_WEIGHT,
    MotionObjective,
    backward_pass,
    canonical_target,
    feedback_trial,
)
from gear_sonic.utils.g1_true23_source_action_codec import source_scaled_precompensation


def plant_for(model):
    return PdShootingPlant(
        model, NativeModelActuationProfile.from_sim_config(Path(__file__).resolve().parents[2] / PHYSICS)
    )


def test_state_difference_exact_pose_identity_keeps_velocity_and_real_tiny_displacements(native):  # noqa: F811
    model, standing, _ = native
    pose = standing.copy()
    pose[3:7] = [0.9999875478269311, -2.6127032839387913e-6, -0.004990409227244953, -2.8293503807935156e-8]
    velocity = np.arange(model.nv, dtype=float) * 0.01
    result = state_difference(model, pose.copy(), np.zeros(model.nv), pose.copy(), velocity)
    np.testing.assert_array_equal(result[: model.nv], np.zeros(model.nv))
    np.testing.assert_array_equal(result[model.nv :], velocity)
    moved = pose.copy()
    moved[7] = np.nextafter(moved[7], np.inf)
    result = state_difference(model, pose, velocity, moved, velocity)
    assert result[6] == moved[7] - pose[7]
    assert result[6] > 0.0


def test_full_state_snapshot_preserves_exact_next_control_and_warmstart(native):  # noqa: F811
    model, standing, _ = native
    plant = plant_for(model)
    data = plant.initial_data(standing, np.zeros(29))
    for _ in range(10):
        plant.integrate_control(data, standing[7:])
    snapshot = plant.state(data)
    first, second = plant.scratch(snapshot), plant.scratch(snapshot)
    plant.integrate_control(first, standing[7:])
    plant.integrate_control(second, standing[7:])
    np.testing.assert_array_equal(plant.state(first), plant.state(second))
    assert first.time == pytest.approx(0.22)
    np.testing.assert_array_equal(plant.state(data), snapshot)
    plant.assert_unchanged()


def test_target_envelope_and_external_force_are_not_bypassed(native):  # noqa: F811
    model, standing, _ = native
    plant = plant_for(model)
    with pytest.raises(ValueError, match="codec envelope"):
        plant.validate_target(plant.upper + 0.1)
    data = plant.initial_data(standing, np.zeros(29))
    data.xfrc_applied[1, 0] = 1
    with pytest.raises(ValueError, match="external forces"):
        plant.integrate_control(data, standing[7:])
    assert data.time == 0


def test_existing_policy_codec_endpoints_are_not_rejected_by_extra_diagnostic_guard(native):  # noqa: F811
    model, _, _ = native
    plant = plant_for(model)
    for sign, expected in ((-1, plant.lower), (1, plant.upper)):
        source = np.full(23, sign * np.nextafter(np.float32(10), np.float32(0)), dtype=np.float32)
        inverse, _ = source_scaled_precompensation(source)
        _, emitted = safe_target_transform_numpy(inverse)
        np.testing.assert_array_equal(plant.validate_target(emitted), expected)


@pytest.mark.parametrize("contact", [False, True])
def test_chained_pd_control_derivative_matches_independent_complete_control_difference(native, contact):  # noqa: F811
    model, standing, _ = native
    plant = plant_for(model)
    pose = standing.copy()
    if not contact:
        pose[2] += 0.2
    data = plant.initial_data(pose, np.zeros(29))
    if contact:
        for _ in range(10):
            plant.integrate_control(data, standing[7:])
        assert data.ncon > 0
    snapshot = plant.state(data)
    a, b = plant.linearize_control(snapshot, standing[7:])
    assert a.shape == (58, 58) and b.shape == (58, 23)
    rng = np.random.default_rng(440)
    dx, du, epsilon = rng.normal(size=58), rng.normal(size=23), 1e-6
    plus, minus = plant.scratch(snapshot), plant.scratch(snapshot)
    mujoco.mj_integratePos(model, plus.qpos, dx[:29], epsilon)
    mujoco.mj_integratePos(model, minus.qpos, dx[:29], -epsilon)
    plus.qvel[:] += epsilon * dx[29:]
    minus.qvel[:] -= epsilon * dx[29:]
    plant.integrate_control(plus, standing[7:] + epsilon * du)
    plant.integrate_control(minus, standing[7:] - epsilon * du)
    finite = state_difference(model, minus.qpos, minus.qvel, plus.qpos, plus.qvel) / (2 * epsilon)
    np.testing.assert_allclose(a @ dx + b @ du, finite, atol=1e-4, rtol=3e-3)
    np.testing.assert_array_equal(plant.state(data), snapshot)
    plant.assert_unchanged()


def test_motion_objective_gradient_matches_finite_difference_with_rotated_root(native):  # noqa: F811
    model, standing, _ = native
    plant = plant_for(model)
    motion = motion_from_poses(model, np.tile(standing, (15, 1)))
    timeline = dict(
        total_requested_controls=4, phases=[dict(name="initial_standing", control_start=0, control_stop=4)]
    )
    objective = MotionObjective(plant, motion, timeline)
    rng = np.random.default_rng(20260908)
    pose, velocity = standing.copy(), rng.normal(0, 0.1, 29)
    mujoco.mj_integratePos(model, pose, rng.normal(0, 0.02, 29), 1.0)
    _, gradient, hessian = objective.state_cost(pose, velocity, 1, derivatives=True)
    direction, eps = rng.normal(size=58), 1e-6
    plus, minus = pose.copy(), pose.copy()
    mujoco.mj_integratePos(model, plus, direction[:29], eps)
    mujoco.mj_integratePos(model, minus, direction[:29], -eps)
    finite = (
        objective.state_cost(plus, velocity + eps * direction[29:], 1)
        - objective.state_cost(minus, velocity - eps * direction[29:], 1)
    ) / (2 * eps)
    np.testing.assert_allclose(gradient[:58] @ direction, finite, atol=2e-5, rtol=1e-6)
    np.testing.assert_allclose(hessian, hessian.T, atol=1e-12)
    assert np.linalg.eigvalsh(hessian).min() >= -1e-8


def test_zero_feedback_trial_preserves_complete_physical_trajectory(native):  # noqa: F811
    model, standing, _ = native
    plant = plant_for(model)
    initial = plant.state(plant.initial_data(standing, np.zeros(29)))
    targets = np.tile(canonical_target(standing[7:]), (4, 1))
    original = plant.rollout(initial, targets, physics_records=True)
    replay, actual = feedback_trial(plant, initial, original, targets, np.zeros((4, 23)), np.zeros((4, 23, 81)), 0)
    np.testing.assert_array_equal(actual, targets)
    assert set(original) == set(replay)
    for key in original:
        np.testing.assert_array_equal(replay[key], original[key])


@pytest.mark.parametrize("implementation", [backward_pass, box_backward_pass])
def test_backward_pass_matches_independent_dense_full_horizon_quadratic_solution(implementation):
    rng, count = np.random.default_rng(19482), 3
    controls = rng.normal(0, 0.1, (count, 23))
    seed = rng.normal(0, 0.1, (count, 23))
    a, b = np.zeros((count, 81, 81)), np.zeros((count, 81, 23))
    for i in range(count):
        a[i, :58, :58] = np.eye(58) * 0.9
        b[i, :58] = rng.normal(0, 0.03, (58, 23))
        b[i, 58:] = np.eye(23)
    local = dict(
        a=a,
        b=b,
        gradient=rng.normal(0, 0.1, (count, 81)),
        hessian=np.tile(np.eye(81), (count, 1, 1)),
        terminal_g=rng.normal(0, 0.1, 81),
        terminal_h=np.eye(81) * 2,
    )
    plant = SimpleNamespace(lower=np.full(23, -1e6), upper=np.full(23, 1e6))
    increments, feedback = implementation(plant, local, controls, seed, 0.0)
    state, closed_loop = np.zeros(81), []
    for i in range(count):
        delta = increments[i] + feedback[i] @ state
        closed_loop.append(delta)
        state = a[i] @ state + b[i] @ delta
    # Condense every state into all control perturbations, independently of Riccati recursion.
    mapping = np.zeros((81, count * 23))
    gradient, hessian = np.zeros(count * 23), np.zeros((count * 23, count * 23))
    for i in range(count):
        selector = np.zeros((23, count * 23))
        selector[:, 23 * i : 23 * (i + 1)] = np.eye(23)
        previous = controls[i - 1] if i else np.asarray(SAFE_TARGET_DEFAULT_Q_HARDWARE)
        difference = selector - mapping[58:]
        gradient += mapping.T @ local["gradient"][i]
        hessian += mapping.T @ local["hessian"][i] @ mapping
        gradient += CONTROL_WEIGHT * selector.T @ (controls[i] - seed[i])
        gradient += SLEW_WEIGHT * difference.T @ (controls[i] - previous)
        hessian += CONTROL_WEIGHT * selector.T @ selector + SLEW_WEIGHT * difference.T @ difference
        mapping = a[i] @ mapping + b[i] @ selector
    gradient += mapping.T @ local["terminal_g"]
    hessian += mapping.T @ local["terminal_h"] @ mapping
    expected = -np.linalg.solve(hessian, gradient)
    np.testing.assert_allclose(np.asarray(closed_loop).ravel(), expected, atol=2e-12, rtol=1e-11)
