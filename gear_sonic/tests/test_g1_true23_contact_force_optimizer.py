"""Full-path optimizer contracts and native23 geometry, not deployment tests."""

from pathlib import Path

import mujoco
import numpy as np
import pytest
from scipy import sparse

from gear_sonic.utils import g1_true23_contact_force_optimizer as optimizer
from gear_sonic.utils.g1_23dof_safe_target_transform import SAFE_TARGET_DEFAULT_Q_HARDWARE
from gear_sonic.utils.g1_23dof_trajectory_projection import audit_trajectory_constraints
from gear_sonic.utils.g1_true23_contact_force_optimizer import ContactForceConfig, PatchedForceLinearization
from gear_sonic.utils.g1_true23_reference_floor import compiled_model_sha256


@pytest.mark.parametrize(
    "settings",
    [
        {"maximum_iterations": False},
        {"qp_maximum_iterations": 1.5},
        {"force_residual_cost": np.nan},
        {"minimum_trust_fraction": 0.5},
        {"joint_trust_rad": 0.13},
        {"root_trust_m": 0.016},
        {"patch_guard_m": 0.002},
        {"effort_limit_fraction": 1},
        {"linear_audit_tolerance": 1e-7},
    ],
)
def test_settings_cannot_relax_limits_or_discard_guard(settings):
    with pytest.raises(ValueError):
        ContactForceConfig(**settings)


def numerical_path_problem(frames=5):
    # A deliberately synthetic linear residual oracle, NOT floating-base dynamics.
    current = np.zeros((frames, 26))
    current[:, 3:] = np.arange(frames)[:, None] * np.linspace(0.0001, 0.0003, 23)
    target = current.copy()
    target[:, :3] += 0.0001
    forces = dict(
        required=(current - target).ravel(),
        jacobian=sparse.eye(current.size, format="csc"),
        force_map=sparse.csc_matrix((current.size, 0)),
        scale=np.ones(current.size),
        lower=np.empty(0),
        upper=np.empty(0),
        patch_values=np.empty(0),
        patch_jacobian=sparse.csc_matrix((0, current.size)),
    )
    return current, target, forces


def step(current, forces):
    return optimizer.contact_force_step(
        current,
        current - 0.1,
        current + 0.1,
        np.ones(26) * 5,
        np.ones(26) * 80,
        (current[1] - current[0]) / 0.02,
        np.empty(0),
        sparse.csc_matrix((0, current.size)),
        forces,
        config=ContactForceConfig(),
        trust_fraction=1 / 16,
    )


def test_actual_clarabel_full_path_step_keeps_nonstationary_frames_and_all_23_joints():
    current, target, forces = numerical_path_problem()
    original = current.copy()
    result, report = step(current, forces)
    assert report["accepted"] and report["path_variables"] == 5 * 26
    assert report["all_frames_have_independent_pose_variables"]
    assert report["pose_objective_center"] == "current_iterate"
    np.testing.assert_allclose(result, target, atol=1e-8)
    np.testing.assert_array_equal(current, original)
    assert len(np.unique(result[:, 3:], axis=0)) == 5
    audit = audit_trajectory_constraints(
        result,
        lower_bounds=current - 0.1,
        upper_bounds=current + 0.1,
        dt=0.02,
        max_velocity=5,
        max_acceleration=80,
        initial_velocity=(current[1] - current[0]) / 0.02,
        tolerance=2e-7,
    )
    assert audit.passed


def test_patch_rows_are_hard_even_when_residual_slacks_would_prefer_lift():
    current, target, forces = numerical_path_problem()
    rows = sparse.csc_matrix((-np.ones(5), (np.arange(5), np.arange(5) * 26 + 2)), shape=(5, current.size))
    forces.update(patch_values=np.full(5, 1e-5), patch_jacobian=rows)
    result, report = step(current, forces)
    assert report["accepted"] and report["hard_patch_rows"] == 5
    assert np.max(result[:, 2] - current[:, 2]) <= 1e-5 + 1e-10
    assert np.max(np.abs(result - target)) >= 8.9e-5
    assert report["maximum_temporary_generalized_force_residual"] >= 8.9e-5


def test_current_center_has_no_hidden_pullback_term(monkeypatch):
    current, _target, forces = numerical_path_problem()
    current += 0.04

    def inspect_cost(diagonal, linear_cost, matrix, lower, upper, **kwargs):
        np.testing.assert_array_equal(linear_cost[: current.size], 0)
        assert matrix.shape[1] == 3 * current.size  # poses plus positive/negative residuals
        return None, {"accepted": False, "status": "intentional_test_rejection"}

    monkeypatch.setattr(optimizer, "solve_box_qp", inspect_cost)
    result, report = step(current, forces)
    assert result is None and not report["accepted"]


def test_bounded_failed_qps_do_not_move_path_or_promote(monkeypatch):
    current, _target, force = numerical_path_problem()
    force.update(
        normalized_residual=np.ones(current.size),
        summed_normalized_force_residual_squared=130,
        maximum_absolute_generalized_force_residual=1,
        models=[],
    )

    class Contacts:
        def audit(self, path):
            return dict(passed=True, summed_violation_m=0, violated_frames=0)

        def evaluate(self, path):
            return np.empty(0), sparse.csc_matrix((0, current.size))

    class Forces:
        def evaluate(self, path, **kwargs):
            return force

    fractions = []

    def reject(*args, **kwargs):
        fractions.append(kwargs["trust_fraction"])
        return None, {"accepted": False, "status": "intentional_test_rejection"}

    monkeypatch.setattr(optimizer, "contact_force_step", reject)
    result, report = optimizer.restore_contact_force_trajectory(
        current,
        current - 0.1,
        current + 0.1,
        np.ones(26) * 5,
        np.ones(26) * 80,
        (current[1] - current[0]) / 0.02,
        Contacts(),
        Forces(),
    )
    np.testing.assert_array_equal(result, current)
    assert fractions == [1 / 16, 1 / 32, 1 / 64, 1 / 128, 1 / 256]
    assert report["failure"] and len(report["iterations"]) == 1
    assert not any(report[k] for k in ("teacher_accepted", "deployment_ready", "hardware_authorized"))


def test_native_geometry_patch_covers_every_frame_without_model_or_root_rotation_mutation():
    root = Path(__file__).resolve().parents[2]
    model = mujoco.MjModel.from_xml_path(str(root / "gear_sonic/data/robots/g1/g1_23dof_rev_1_0.xml"))
    data = mujoco.MjData(model)
    data.qpos[7:] = SAFE_TARGET_DEFAULT_Q_HARDWARE
    data.qpos[2] = 0.8
    mujoco.mj_fwdPosition(model, data)
    plane = model.geom("floor").id
    lowest = min(
        mujoco.mj_geomDistance(model, data, plane, geom, 3.0, None)
        for geom in range(model.ngeom)
        if model.geom_bodyid[geom] and (model.geom_contype[geom] or model.geom_conaffinity[geom])
    )
    data.qpos[2] += 0.0004 - lowest
    source = np.tile(data.qpos, (5, 1))
    source[:, 0] += np.arange(5) * 0.0001
    path = np.column_stack((np.zeros((5, 3)), source[:, 7:]))
    digest = compiled_model_sha256(model)
    original = source.copy()
    system = PatchedForceLinearization({"mesh": model}, source, np.ones(23) * 10)
    linear = system.evaluate(path)
    assert linear["patch_jacobian"].shape[1] == 5 * 26
    assert linear["patch_counts"]["mesh"] > 0
    assert {frame for _, frame, _ in linear["frozen_patches"]} == set(range(5))
    assert len({id(patch.data) for _, _, patch in linear["frozen_patches"]}) == 1
    assert system.audit_frozen_patches(path, linear) == 0
    lifted = path.copy()
    lifted[3, 2] += 0.01
    assert system.audit_frozen_patches(lifted, linear) > 0.008
    assert compiled_model_sha256(model) == digest
    np.testing.assert_array_equal(source, original)
    assert linear["required"].shape == (5 * 29,)


@pytest.mark.parametrize("consumed,should_accept", [(0.00002, True), (0.00005, False), (0.00006, False)])
def test_nonlinear_curvature_may_consume_guard_but_cannot_expand_candidate_band(
    monkeypatch, consumed, should_accept
):
    current, target, linear = numerical_path_problem()

    class Contacts:
        def audit(self, path):
            return dict(passed=True, summed_violation_m=0, violated_frames=0)

        def evaluate(self, path):
            return np.empty(0), sparse.csc_matrix((0, current.size))

    class Forces:
        def evaluate(self, path, **kwargs):
            residual = (path - target).ravel()
            return {
                **linear,
                "normalized_residual": residual,
                "summed_normalized_force_residual_squared": float(residual @ residual),
                "maximum_absolute_generalized_force_residual": float(np.max(np.abs(residual))),
                "models": [],
            }

        def audit_frozen_patches(self, path, linearization):
            return consumed

    monkeypatch.setattr(
        optimizer,
        "contact_force_step",
        lambda *args, **kwargs: (target.copy(), {"status": "Solved", "accepted": True}),
    )
    result, report = optimizer.restore_contact_force_trajectory(
        current,
        current - 0.1,
        current + 0.1,
        np.ones(26) * 5,
        np.ones(26) * 80,
        (current[1] - current[0]) / 0.02,
        Contacts(),
        Forces(),
        config=ContactForceConfig(maximum_iterations=1),
    )
    np.testing.assert_array_equal(result, target if should_accept else current)
    assert (report["failure"] is None) == should_accept
    trial = report["iterations"][0]["attempts"][0]["trials"][0]
    assert trial["frozen_patch_guard_consumed_m"] == consumed
    assert trial["frozen_patch_within_candidate_band"] == should_accept
    assert not report["deployment_ready"] and not report["teacher_accepted"]
