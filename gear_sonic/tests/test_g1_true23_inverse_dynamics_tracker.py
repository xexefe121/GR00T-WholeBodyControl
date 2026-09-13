from pathlib import Path
from types import SimpleNamespace

import clarabel
import mujoco
import numpy as np
import pytest

from gear_sonic.scripts.diagnose_g1_true23_inverse_dynamics_tracking import TrackingPolicy
from gear_sonic.tests.test_diagnose_g1_true23_leg_error_feedback import Probe, inputs
from gear_sonic.tests.test_g1_true23_contact_step_transition import native  # noqa: F401
from gear_sonic.tests.test_g1_true23_root_feedback_campaign import receipt, validate
from gear_sonic.utils.g1_23dof_contract import MUJOCO_TO_ISAACLAB_DOF
from gear_sonic.utils.g1_23dof_safe_target_transform import (
    SAFE_TARGET_DEFAULT_Q_HARDWARE,
    safe_target_transform_numpy,
)
from gear_sonic.utils.g1_true23_generalist_benchmark import PHYSICS
from gear_sonic.utils.g1_true23_inverse_dynamics_tracker import (
    KIND,
    PD_FEEDFORWARD,
    PD_ONLY,
    InverseDynamicsTracker,
    desired_generalized_acceleration,
    foot_acceleration_tasks,
    pd_plus_feedforward_numpy,
    reachable_target_bounds,
    reference_candidate_corners,
    solve_tracker_qp,
    tracker_contract,
)
from gear_sonic.utils.g1_true23_native_model_actuation import NativeModelActuationProfile, native_model_pd_numpy
from gear_sonic.utils.g1_true23_source_action_codec import SOURCE_SCALE_NATIVE_IL23, source_scaled_precompensation


def profile():
    return NativeModelActuationProfile.from_sim_config(Path(__file__).resolve().parents[2] / PHYSICS)


def test_strict_qp_solves_box_and_equality_without_changing_tolerances():
    solution, report = solve_tracker_qp([1, 1], [-2, 0], [[1, 1], [1, 0]], [1, 0], [1, 0.75])
    np.testing.assert_allclose(solution, [0.75, 0.25], atol=1e-8, rtol=0)
    assert report["accepted"] and not report["static_regularization"]
    assert report["requested_feasibility_and_gap_tolerance"] == 1e-9


@pytest.mark.parametrize(
    "status,x",
    [
        (clarabel.SolverStatus.AlmostSolved, [0.5]),
        (clarabel.SolverStatus.Solved, [2.0]),
        (clarabel.SolverStatus.Solved, [np.nan]),
    ],
)
def test_strict_qp_rejects_reduced_accuracy_and_false_success(monkeypatch, status, x):
    answer = SimpleNamespace(status=status, x=x, iterations=1, solve_time=0.001, r_prim=0.0, r_dual=0.0)
    monkeypatch.setattr(clarabel, "DefaultSolver", lambda *args: SimpleNamespace(solve=lambda: answer))
    solution, report = solve_tracker_qp([1], [0], [[1]], [0], [1])
    assert solution is None and not report["accepted"]


def test_reachable_targets_survive_unchanged_source_and_native_codecs():
    lower, upper = reachable_target_bounds()
    order = np.asarray(MUJOCO_TO_ISAACLAB_DOF)
    for target in (lower, upper, (lower + upper) / 2, np.asarray(SAFE_TARGET_DEFAULT_Q_HARDWARE)):
        source = ((target - SAFE_TARGET_DEFAULT_Q_HARDWARE)[order] / SOURCE_SCALE_NATIVE_IL23).astype(np.float32)
        inverse, projection = source_scaled_precompensation(source)
        assert np.max(np.abs(projection)) < 1e-6
        # safe_target_transform_numpy accepts native ISAACLAB order.
        _, actual = safe_target_transform_numpy(inverse)
        np.testing.assert_allclose(actual, target, atol=1e-6, rtol=0)


def test_candidate_selection_rejects_airborne_foot_without_discarding_actual_stance_toes():
    gaps = np.array([[0.0001, 0.0002, 0.010, 0.012], [0.04, 0.04, 0.04, 0.04]])
    np.testing.assert_array_equal(reference_candidate_corners(gaps), [[True] * 4, [False] * 4])
    np.testing.assert_array_equal(reference_candidate_corners(np.full((2, 4), 0.003)), np.ones((2, 4), bool))
    with pytest.raises(ValueError):
        reference_candidate_corners(np.full((2, 4), np.nan))


def test_stationary_received_pose_has_zero_feedforward_and_short_window(native):  # noqa: F811
    model, standing, _ = native
    poses = np.tile(standing, (3, 1))
    desired, velocity, acceleration = desired_generalized_acceleration(model, standing, np.zeros(29), poses)
    np.testing.assert_allclose(desired, 0, atol=3e-12)
    np.testing.assert_allclose(velocity, 0, atol=3e-12)
    np.testing.assert_allclose(acceleration, 0, atol=3e-12)
    with pytest.raises(ValueError, match="exactly three received"):
        desired_generalized_acceleration(model, standing, np.zeros(29), np.tile(standing, (4, 1)))
    poses[0, 3:7] *= 2
    with pytest.raises(ValueError, match="unit-quaternion"):
        desired_generalized_acceleration(model, standing, np.zeros(29), poses)


def test_constant_received_translation_derivative_is_not_future_full_sequence(native):  # noqa: F811
    model, standing, _ = native
    poses = np.tile(standing, (3, 1))
    poses[:, 0] += [-0.002, 0, 0.002]
    measured_velocity = np.zeros(29)
    measured_velocity[0] = 0.1
    desired, velocity, _ = desired_generalized_acceleration(model, standing, measured_velocity, poses)
    np.testing.assert_allclose(desired, 0, atol=1e-10)
    np.testing.assert_allclose(velocity, measured_velocity, atol=1e-12)


def test_static_qp_preserves_physics_and_respects_original_effort_and_target_bounds(native):  # noqa: F811
    model, standing, _ = native
    tracker = InverseDynamicsTracker(model, profile())
    poses, measured, velocity = np.tile(standing, (3, 1)), standing.copy(), np.zeros(29)
    originals = [value.copy() for value in (poses, measured, velocity)]
    raw, record = tracker.infer(measured, velocity, poses)
    for before, after in zip(originals, (poses, measured, velocity), strict=True):
        np.testing.assert_array_equal(before, after)
    tracker.assert_original_unchanged()
    assert np.max(np.abs(record["tracker_force_residual29"])) < 1e-5
    torque = record["tracker_torque23"]
    assert np.all(np.abs(torque) <= np.asarray(profile().effort) + 1e-8)
    assert np.all(record["tracker_ray_weights32"] >= -1e-8)
    assert 0 < record["tracker_contact_count1"][0] <= 8
    target = record["tracker_target23"]
    lower, upper = reachable_target_bounds()
    assert np.all(target >= lower - 1e-8) and np.all(target <= upper + 1e-8)
    requested, applied, invalid, _ = native_model_pd_numpy(target, measured[7:], velocity[6:], profile())
    np.testing.assert_allclose(torque, requested, atol=1e-12)
    np.testing.assert_allclose(torque, applied, atol=1e-8)
    assert not invalid and raw.dtype == np.float32
    # Candidate forces exist only as records, never as control/external forces.
    np.testing.assert_array_equal(tracker.data.ctrl, np.zeros(23))
    np.testing.assert_array_equal(tracker.data.qfrc_applied, np.zeros(29))
    np.testing.assert_array_equal(tracker.data.xfrc_applied, np.zeros((model.nbody, 6)))


def test_world_foot_tasks_correct_base_translation_even_with_exact_joint_angles(native):  # noqa: F811
    model, standing, _ = native
    tracker = InverseDynamicsTracker(model, profile())
    actual, reference = mujoco.MjData(model), mujoco.MjData(model)
    actual.qpos[:], reference.qpos[:] = standing, standing
    actual.qpos[1] += 0.06
    for data in (actual, reference):
        mujoco.mj_fwdPosition(model, data)
        mujoco.mj_fwdVelocity(model, data)
    jacobian, acceleration, error = foot_acceleration_tasks(
        model, actual, reference, np.zeros(29), tracker.sole_geoms
    )
    np.testing.assert_allclose(error, np.tile([0, -0.06, 0], (8, 1)), atol=1e-15)
    np.testing.assert_allclose(acceleration.reshape(8, 3), error * 80, atol=1e-12)
    assert jacobian.shape == (24, 29)
    # Exact joint tracking cannot see this displacement; the world foot task can.
    np.testing.assert_array_equal(actual.qpos[7:], reference.qpos[7:])


def test_world_foot_feedforward_agrees_with_received_reference_dynamics(native):  # noqa: F811
    model, standing, _ = native
    tracker = InverseDynamicsTracker(model, profile())
    data = mujoco.MjData(model)
    data.qpos[:] = standing
    data.qvel[:] = np.random.default_rng(184).normal(0, 0.1, 29)
    reference_acceleration = np.random.default_rng(185).normal(0, 0.3, 29)
    mujoco.mj_fwdPosition(model, data)
    mujoco.mj_fwdVelocity(model, data)
    jacobian, target, error = foot_acceleration_tasks(
        model, data, data, reference_acceleration, tracker.sole_geoms
    )
    np.testing.assert_allclose(target, jacobian @ reference_acceleration, atol=1e-14)
    np.testing.assert_array_equal(error, np.zeros((8, 3)))


def test_real_standing_toes_are_retained_when_reference_toes_are_four_mm_high(native):  # noqa: F811
    model, standing, _ = native
    tracker = InverseDynamicsTracker(model, profile())
    settled = standing.copy()
    settled[2] -= 0.004
    _, record = tracker.infer(settled, np.zeros(29), np.tile(standing, (3, 1)))
    assert record["tracker_contact_count1"][0] == 8
    np.testing.assert_array_equal(np.sort(record["tracker_contact_geom8"]), np.sort(tracker.sole_geoms.ravel()))
    assert (
        np.max(tracker.reference.geom_xpos[tracker.sole_geoms, 2] - model.geom_size[tracker.sole_geoms, 0]) > 0.004
    )
    tracker.assert_original_unchanged()


def test_airborne_model_cannot_receive_fictitious_ground_or_root_actuator_force(native):  # noqa: F811
    model, standing, _ = native
    tracker = InverseDynamicsTracker(model, profile())
    floating = standing.copy()
    floating[2] += 0.4
    _, record = tracker.infer(floating, np.zeros(29), np.tile(floating, (3, 1)))
    assert record["tracker_contact_count1"][0] == 0
    np.testing.assert_array_equal(record["tracker_ray_weights32"], np.zeros(32))
    mass = np.zeros((29, 29))
    mujoco.mj_fullM(tracker.model, mass, tracker.data.qM)
    required = mass @ record["tracker_qdd29"] + tracker.data.qfrc_bias - tracker.data.qfrc_passive
    np.testing.assert_allclose(required[:6], np.zeros(6), atol=1e-8)
    assert record["tracker_qdd29"][2] < -8


def test_zero_wrapper_returns_bit_exact_policy_and_does_not_call_controller():
    class NoController:
        def infer(self, *args):
            raise AssertionError("zero control called QP")

        def assert_original_unchanged(self):
            pass

    policy = TrackingPolicy(Probe(), NoController(), 0)
    encoder, history, feedback, _ = inputs()
    poses = np.zeros((3, 30))
    poses[:, 3] = 1
    poses[1, 7:19] = encoder[12:24]
    policy.set_received_state(poses[1], np.zeros(29), poses)
    output, _ = policy.infer(encoder, history, feedback)
    records = policy.take_records()
    np.testing.assert_array_equal(output, records["uncorrected_model_raw23"][0])
    np.testing.assert_array_equal(records["tracker_received_qpos3x30"][0], poses)
    with pytest.raises(ValueError, match="fresh"):
        policy.infer(encoder, history, feedback)
    policy.set_received_state(poses[1], np.zeros(29), poses)
    with pytest.raises(ValueError, match="exactly one fresh"):
        policy.set_received_state(poses[1], np.zeros(29), poses)


def test_solver_rejection_preserves_network_call_without_fallback():
    class RejectedController:
        def infer(self, *args):
            raise ValueError("fixture infeasible")

        def assert_original_unchanged(self):
            pass

    policy = TrackingPolicy(Probe(), RejectedController(), 1)
    encoder, history, feedback, _ = inputs()
    poses = np.zeros((3, 30))
    poses[1, 7:19] = encoder[12:24]
    policy.set_received_state(poses[1], np.zeros(29), poses)
    with pytest.raises(ValueError, match="fixture infeasible"):
        policy.infer(encoder, history, feedback)
    records = policy.take_records()
    assert len(records["uncorrected_model_raw23"]) == 1
    assert "actual_tracker_source_raw23" not in records


@pytest.mark.parametrize("strength", [-1, 0.5, 2, float("inf"), float("nan"), True])
def test_only_explicit_zero_or_full_probe_allowed(strength):
    with pytest.raises(ValueError):
        tracker_contract(strength)


def test_counterfactual_and_standing_prefix_cannot_be_training_parent():
    report = receipt()
    report["kind"] = KIND
    report["inverse_dynamics_counterfactual"] = tracker_contract(1, 250)
    assert not report["inverse_dynamics_counterfactual"]["eligible_parent_for_training_continuation"]
    with pytest.raises(ValueError):
        validate(report)
    with pytest.raises(ValueError):
        tracker_contract(1, 500)


def test_zero_joint_feedforward_is_bit_exact_original_pd():
    rng = np.random.default_rng(439)
    target, q, dq = [rng.normal(size=23) for _ in range(3)]
    expected = native_model_pd_numpy(target, q, dq, profile())
    actual = pd_plus_feedforward_numpy(target, q, dq, profile(), np.zeros(23))
    for left, right in zip(expected, actual, strict=True):
        assert np.asarray(left).dtype == np.asarray(right).dtype
        assert np.asarray(left).tobytes() == np.asarray(right).tobytes()


def test_total_pd_plus_feedforward_retains_same_motor_effort_saturation():
    actuation = profile()
    effort, kp = np.asarray(actuation.effort), np.asarray(actuation.kp)
    target = 0.5 * effort / kp
    ff = 0.75 * effort
    requested, applied, invalid, cost = pd_plus_feedforward_numpy(
        target, np.zeros(23), np.zeros(23), actuation, ff
    )
    np.testing.assert_allclose(requested, 1.25 * effort, atol=1e-12)
    np.testing.assert_array_equal(applied, effort)
    assert not invalid and cost == pytest.approx(0.25**2)


@pytest.mark.parametrize("ff", [np.zeros(29), np.full(23, np.nan), np.full(23, np.inf), np.full(23, 140.0)])
def test_invalid_or_oversize_joint_feedforward_rejected(ff):
    with pytest.raises(ValueError, match="finite bounded native23"):
        pd_plus_feedforward_numpy(np.zeros(23), np.zeros(23), np.zeros(23), profile(), ff)


def test_ff_controller_preserves_target_envelope_and_exact_float32_emitted_torque(native):  # noqa: F811
    model, standing, _ = native
    actuation = profile()
    tracker = InverseDynamicsTracker(model, actuation, actuation_mode=PD_FEEDFORWARD)
    velocity = np.zeros(29)
    # An ankle moving quickly enough that damping cannot be cancelled within
    # the bounded target envelope exercises nonzero FF without a saved run.
    velocity[11] = -10.0
    raw, record = tracker.infer(standing, velocity, np.tile(standing, (3, 1)))
    inverse, projection = source_scaled_precompensation(raw)
    _, emitted = safe_target_transform_numpy(inverse)
    lower, upper = reachable_target_bounds()
    assert np.all(emitted >= lower - 1e-6) and np.all(emitted <= upper + 1e-6)
    assert np.max(np.abs(projection)) <= 1e-6
    ff = record["tracker_feedforward_torque23"]
    assert np.max(np.abs(ff)) > 1e-4  # This state exceeds the PD-only target authority.
    assert np.all(np.abs(ff) <= np.asarray(actuation.effort) + 1e-6)
    requested, applied, invalid, _ = pd_plus_feedforward_numpy(emitted, standing[7:], velocity[6:], actuation, ff)
    np.testing.assert_allclose(requested, record["tracker_torque23"], atol=1e-10, rtol=0)
    np.testing.assert_allclose(applied, record["tracker_torque23"], atol=1e-8, rtol=0)
    assert not invalid
    tracker.assert_original_unchanged()
    np.testing.assert_array_equal(tracker.data.qfrc_applied, np.zeros(29))


def test_feedforward_is_fresh_and_held_for_exactly_ten_physics_substeps():
    ff = np.linspace(-1, 1, 23)
    tracker = SimpleNamespace(
        infer=lambda *args: (np.zeros(23, np.float32), {"tracker_feedforward_torque23": ff.copy()}),
        assert_original_unchanged=lambda: None,
    )
    policy = TrackingPolicy(Probe(), tracker, 1, actuation_mode=PD_FEEDFORWARD)
    encoder, history, feedback, _ = inputs()
    poses = np.zeros((3, 30))
    poses[1, 7:19] = encoder[12:24]
    zeros, actuation = np.zeros(23), profile()
    with pytest.raises(ValueError, match="stale torque"):
        policy.apply_pd(zeros, zeros, zeros, actuation)
    policy.set_received_state(poses[1], np.zeros(29), poses)
    policy.infer(encoder, history, feedback)
    for _ in range(9):
        np.testing.assert_array_equal(policy.apply_pd(zeros, zeros, zeros, actuation)[0], ff)
    with pytest.raises(ValueError, match="exactly ten"):
        policy.set_received_state(poses[1], np.zeros(29), poses)
    policy.apply_pd(zeros, zeros, zeros, actuation)
    with pytest.raises(ValueError, match="stale torque"):
        policy.apply_pd(zeros, zeros, zeros, actuation)
    policy.set_received_state(poses[1], np.zeros(29), poses)
    tracker.infer = lambda *args: (_ for _ in ()).throw(ValueError("fixture rejected next QP"))
    with pytest.raises(ValueError, match="rejected next QP"):
        policy.infer(encoder, history, feedback)
    with pytest.raises(ValueError, match="stale torque"):
        policy.apply_pd(zeros, zeros, zeros, actuation)
    record = policy.take_records()
    np.testing.assert_array_equal(record["tracker_attempted_physics_feedforward_torque23"], np.tile(ff, (10, 1)))
    assert len(record["uncorrected_model_raw23"]) == 2
    assert len(record["actual_tracker_source_raw23"]) == 1


def test_feedforward_contract_discloses_changed_command_law_not_sonic_compatibility():
    contract = tracker_contract(1, actuation_mode=PD_FEEDFORWARD)
    assert contract["actuator_command_law_changed"]
    assert contract["existing_sonic_previous_action_history_does_not_encode_feedforward"]
    assert not contract["commanded_torque_converted_to_50hz_held_target_with_500hz_original_pd"]
    for key in (
        "drop_in_existing_sonic_policy_or_hardware_compatibility_proven",
        "eligible_parent_for_training_continuation",
        "hardware_authorized",
        "deployment_ready",
        "simulator_qualified",
    ):
        assert not contract[key]
    assert not tracker_contract(0, actuation_mode=PD_FEEDFORWARD)["actuator_command_law_changed"]
    assert not tracker_contract(1, actuation_mode=PD_ONLY)["actuator_command_law_changed"]
    with pytest.raises(ValueError, match="unknown"):
        tracker_contract(1, actuation_mode="hardware")
