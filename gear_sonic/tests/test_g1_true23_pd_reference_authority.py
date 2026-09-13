from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from gear_sonic.scripts import audit_g1_true23_pd_reference_authority as audit
from gear_sonic.tests.test_g1_true23_contact_step_transition import native  # noqa: F401


def fixture():
    required, contact = np.zeros(29), np.zeros((29, 1))
    required[2] = 3
    contact[2, 0] = contact[11, 0] = 1
    return required, contact, np.zeros(23)


def test_motor_feasible_can_be_pd_target_infeasible_without_root_actuation():
    required, contact, friction = fixture()
    motor = audit.bounded_force_support(required, contact, np.full(23, -35), np.full(23, 35), friction)
    lower, upper = np.full(23, -35), np.full(23, 35)
    lower[5] = -2
    pd = audit.bounded_force_support(required, contact, lower, upper, friction)
    assert motor["status"] == "conditional_feasible"
    assert motor["joint_torque23_nm"][5] == pytest.approx(-3)
    assert pd["status"] == "infeasible"
    summary = audit.summarize_rows([dict(motor=motor, pd=pd)])
    assert summary["motor_feasible_but_pd_infeasible"] == 1
    assert not summary["dynamic_feasibility_proven"]


def test_no_candidate_ground_cannot_actuate_floating_root():
    required, _, friction = fixture()
    result = audit.bounded_force_support(required, np.zeros((29, 0)), np.full(23, -35), np.full(23, 35), friction)
    assert result["status"] == "infeasible"


def test_unknown_stays_unknown_after_identical_lp_retry(monkeypatch):
    calls = []

    def unknown(*args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(status=4, message="fixture unknown")

    monkeypatch.setattr(audit, "linprog", unknown)
    required, contact, friction = fixture()
    result = audit.bounded_force_support(required, contact, np.full(23, -35), np.full(23, 35), friction)
    assert result["status"] == "unknown" and len(calls) == 2
    np.testing.assert_array_equal(calls[0]["A_eq"], calls[1]["A_eq"])
    assert calls[1]["options"]["presolve"] is False
    summary = audit.summarize_rows([dict(motor=result, pd=result)])
    assert summary["motor_unknown"] == summary["pd_unknown"] == 1
    assert summary["motor_infeasible"] == summary["pd_infeasible"] == 0


def test_false_success_is_rejected_by_original_force_audit(monkeypatch):
    monkeypatch.setattr(audit, "linprog", lambda *args, **kwargs: SimpleNamespace(status=0, x=np.zeros(47)))
    required, contact, friction = fixture()
    with pytest.raises(RuntimeError, match="independent original"):
        audit.bounded_force_support(required, contact, np.full(23, -35), np.full(23, 35), friction)


def test_empty_pd_interval_is_independently_infeasible():
    required, contact, friction = fixture()
    result = audit.bounded_force_support(required, contact, np.ones(23), np.zeros(23), friction)
    assert result["status"] == "infeasible" and result["solver_status"] is None


@pytest.mark.parametrize("bad", [np.ones(22), np.full(23, np.nan), np.full(23, -1)])
def test_invalid_friction_or_dimensions_rejected(bad):
    required, contact, _ = fixture()
    with pytest.raises(ValueError, match="finite native23"):
        audit.bounded_force_support(required, contact, np.full(23, -35), np.full(23, 35), bad)


def test_known_standing_reference_is_candidate_gap_sensitive_not_physically_impossible(native):  # noqa: F811
    model, standing, _ = native
    profile = audit.NativeModelActuationProfile.from_sim_config(
        Path(__file__).resolve().parents[2] / audit.PHYSICS
    )
    tracker = audit.InverseDynamicsTracker(model, profile)
    results, counts = [], []
    for gap in (0.004, 0.005):
        audit.configure_candidate_gap(tracker, gap)
        data, private = tracker.data, tracker.model
        data.qpos[:] = standing
        mujoco.mj_fwdPosition(private, data)
        mujoco.mj_fwdVelocity(private, data)
        contact, candidates = audit.floor_contact_map(private, data, tracker.plane, gap)
        counts.append(len(candidates))
        results.append(
            audit.bounded_force_support(
                data.qfrc_bias - data.qfrc_passive,
                contact,
                -tracker.effort,
                tracker.effort,
                private.dof_frictionloss[6:],
            )["status"]
        )
    assert counts == [4, 8]
    assert results == ["infeasible", "conditional_feasible"]
    assert tracker.data.time == 0
    tracker.assert_original_unchanged()
    with pytest.raises(ValueError, match="4 or 5 mm"):
        audit.configure_candidate_gap(tracker, 0.01)


@pytest.mark.parametrize("scale", [1.0, 2.0, 4.0])
def test_uniform_time_hypothesis_scales_velocity_and_acceleration_without_changing_poses(native, scale):  # noqa: F811
    model, standing, _ = native
    poses = np.tile(standing, (3, 1))
    poses[:, 0] += [0, 0.001, 0.004]
    poses[:, 7] += [0, 0.002, 0.006]
    before = poses.copy()
    velocity, acceleration = audit.pose_path_derivatives(model, poses, 0.02)
    slower_v, slower_a = audit.pose_time_derivatives(model, poses, scale)
    np.testing.assert_array_equal(poses, before)
    np.testing.assert_allclose(slower_v, velocity / scale, atol=1e-14, rtol=0)
    np.testing.assert_allclose(slower_a, acceleration / scale**2, atol=1e-14, rtol=0)
    with pytest.raises(ValueError, match="1x, 2x or 4x"):
        audit.pose_time_derivatives(model, poses, 0.5)


@pytest.mark.parametrize("reverse", [False, True])
def test_verified_feasible_witness_cannot_be_reported_infeasible_in_containing_envelope(reverse):
    witness = dict(
        status="conditional_feasible",
        joint_torque23_nm=np.zeros(23).tolist(),
        maximum_force_residual=1e-9,
        maximum_bound_violation=0,
    )
    negative = dict(status="infeasible", solver_status=2)
    motor, pd = (negative, witness) if reverse else (witness, negative)
    result = audit.reconcile_verified_witnesses(motor, pd, np.full(23, -2), np.full(23, 2), np.full(23, 35))
    assert all(row["status"] == "conditional_feasible" for row in result)
    assert result[0 if reverse else 1]["original_destination_solver_result"] == negative
    assert negative["status"] == "infeasible"  # Raw records retained, not mutated.
    witness["maximum_force_residual"] = 1.0
    with pytest.raises(ValueError, match="independently verified"):
        audit.reconcile_verified_witnesses(motor, pd, np.full(23, -2), np.full(23, 2), np.full(23, 35))


def test_motor_witness_outside_pd_envelope_does_not_override_pd_infeasibility():
    witness = dict(
        status="conditional_feasible",
        joint_torque23_nm=np.full(23, 3).tolist(),
        maximum_force_residual=1e-9,
        maximum_bound_violation=0,
    )
    negative = dict(status="infeasible", solver_status=2)
    _, pd = audit.reconcile_verified_witnesses(witness, negative, np.full(23, -2), np.full(23, 2), np.full(23, 35))
    assert pd == negative
