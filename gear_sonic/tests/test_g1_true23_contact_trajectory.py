import copy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
from scipy import sparse

from gear_sonic.scripts.refine_g1_true23_stance_contacts import refinement_problem, root_orientation_error
from gear_sonic.utils.g1_23dof_safe_target_transform import SAFE_TARGET_DEFAULT_Q_HARDWARE
from gear_sonic.utils.g1_23dof_task_space_retarget import build_mjlab_motion_arrays
from gear_sonic.utils.g1_23dof_trajectory_projection import audit_trajectory_constraints
from gear_sonic.utils.g1_true23_contact_trajectory import (
    ContactLinearization,
    ContactTrajectoryConfig,
    refine_contact_trajectory,
    solve_linearized_restoration,
)
from gear_sonic.utils.g1_true23_reference_floor import compiled_model_sha256
from gear_sonic.utils.g1_true23_stance_retarget import FEET, StanceRetargetConfig


@pytest.mark.parametrize(
    "kwargs",
    [
        {"maximum_iterations": 0},
        {"maximum_iterations": 1.5},
        {"root_trust_m": float("nan")},
        {"joint_trust_rad": True},
        {"support_gap_m": 0.02},
        {"audit_tolerance": 0.002},
    ],
)
def test_invalid_contact_settings_rejected(kwargs):
    with pytest.raises(ValueError):
        ContactTrajectoryConfig(**kwargs)


def test_root_rotation_comparison_accepts_sign_and_normalization_not_actual_rotation():
    original = np.array([[0.8, 0.1, 0.2, 0.3]])
    assert root_orientation_error(original, -original) == 0
    assert root_orientation_error(original, original * (1 + 1e-8)) < 1e-14
    changed = original.copy()
    changed[:, 1] += 0.001
    assert root_orientation_error(original, changed) > 2e-7
    with pytest.raises(ValueError, match="equal"):
        root_orientation_error(original, original[:, :3])


@pytest.fixture
def materials():
    root = Path(__file__).resolve().parents[2]
    model = mujoco.MjModel.from_xml_path(str(root / "gear_sonic/data/robots/g1/g1_23dof_rev_1_0.xml"))
    source = np.tile(model.qpos0, (4, 1))
    source[:, :3] = [0, 0, 0.76]
    source[:, 3:7] = [1, 0, 0, 0]
    source[:, 7:] = SAFE_TARGET_DEFAULT_Q_HARDWARE
    path = np.column_stack((np.zeros((4, 3)), source[:, 7:]))
    return model, source, path


def test_contact_distance_jacobian_preserves_model_and_matches_finite_difference(materials):
    model, source, path = materials
    before = compiled_model_sha256(model)
    contacts = ContactLinearization({"mesh": model}, source, [FEET] * 4, ContactTrajectoryConfig())
    path = path.copy()
    path[:, :3] += [0.01, -0.01, -0.03]
    path[:, 3:] += np.linspace(-0.012, 0.014, 23)
    values, jacobian = contacts.evaluate(path)
    assert jacobian.shape == (len(values), 4 * 26)
    # One frame perturbed: all other frames must have exactly zero response.
    for column in range(26):
        step = np.zeros_like(path)
        step[1, column] = 1e-6
        numeric = (
            contacts.evaluate(path + step, jacobian=False)[0] - contacts.evaluate(path - step, jacobian=False)[0]
        ) / 2e-6
        np.testing.assert_allclose(numeric, jacobian[:, 26 + column].toarray().ravel(), atol=2e-6, rtol=2e-5)
    assert compiled_model_sha256(model) == before


def test_contact_hypotheses_and_layout_cannot_be_silently_changed(materials):
    model, source, _path = materials
    cfg = ContactTrajectoryConfig()
    with pytest.raises(ValueError, match="every frame"):
        ContactLinearization({"mesh": model}, source, [FEET], cfg)
    with pytest.raises(ValueError, match="duplicate"):
        ContactLinearization({"mesh": model}, source, [(FEET[0], FEET[0])] * 4, cfg)
    other = copy.copy(model)
    other.jnt_qposadr[1] = 8
    with pytest.raises(ValueError, match="exact native23"):
        ContactLinearization({"mesh": other}, source, [FEET] * 4, cfg)


def _linear_problem():
    current = np.zeros((4, 26))
    lower, upper = np.full_like(current, -1.0), np.full_like(current, 1.0)
    velocity, acceleration, initial = np.ones(26) * 0.1, np.ones(26), np.zeros(26)
    jacobian = sparse.csc_matrix(([1.0], ([0], [26 + 2])), shape=(1, current.size))
    return current, lower, upper, velocity, acceleration, initial, jacobian


def test_contact_qp_couples_all_frames_without_losing_derivative_limits():
    current, lower, upper, velocity, acceleration, initial, jacobian = _linear_problem()
    # A per-frame 1 cm correction violates adjacent finite differences.
    independent = current.copy()
    independent[1, 2] = 0.01
    kwargs = dict(
        lower_bounds=lower,
        upper_bounds=upper,
        dt=0.02,
        max_velocity=velocity,
        max_acceleration=acceleration,
        initial_velocity=initial,
    )
    assert not audit_trajectory_constraints(independent, **kwargs).passed
    projected, report = solve_linearized_restoration(
        current,
        current,
        lower,
        upper,
        velocity,
        acceleration,
        initial,
        np.array([-0.01]),
        jacobian,
        config=ContactTrajectoryConfig(),
    )
    assert projected is not None, report
    assert projected[1, 2] >= 0.01 - 2e-7
    assert np.count_nonzero(projected[:, 2] > 0.0079) == 4
    assert audit_trajectory_constraints(projected, **kwargs, tolerance=2e-7).passed


def test_contact_slack_is_not_false_acceptance_of_infeasible_path():
    current, _lower, _upper, velocity, acceleration, initial, jacobian = _linear_problem()

    class ImpossibleContact:
        def evaluate(self, path):
            return np.asarray(jacobian @ path.ravel()).ravel() - 0.01, jacobian

        def audit(self, path):
            gap = max(0.0, 0.01 - float((jacobian @ path.ravel())[0]))
            return {
                "passed": gap <= 2e-7,
                "summed_violation_m": gap,
                "maximum_violation_m": gap,
                "violated_frames": int(gap > 2e-7),
            }

    output, report = refine_contact_trajectory(
        current,
        current,
        current,
        velocity,
        acceleration,
        initial,
        ImpossibleContact(),
        config=ContactTrajectoryConfig(maximum_iterations=2),
    )
    assert not report["contact_and_derivative_constraints_passed"]
    assert not report["temporary_contact_slack_permitted_in_final_acceptance"]
    assert not report["teacher_accepted"] and not report["deployment_ready"]
    assert report["failure"]
    np.testing.assert_allclose(output, current, atol=2e-7)


def test_reported_solver_success_cannot_override_independent_constraints(monkeypatch):
    import osqp

    class FalseSuccess:
        def setup(self, **kwargs):
            self.size = len(kwargs["q"])

        def warm_start(self, **_kwargs):
            pass

        def solve(self, **_kwargs):
            return SimpleNamespace(
                x=np.ones(self.size),
                info=SimpleNamespace(status="solved", status_val=1, iter=1, prim_res=0, dual_res=0, run_time=0),
            )

    monkeypatch.setattr(osqp, "OSQP", FalseSuccess)
    current, lower, upper, velocity, acceleration, initial, jacobian = _linear_problem()
    output, report = solve_linearized_restoration(
        current,
        current,
        lower,
        upper,
        velocity,
        acceleration,
        initial,
        np.array([-0.01]),
        jacobian,
        config=ContactTrajectoryConfig(),
    )
    assert output is None
    assert report["status"] == "independent_linear_constraint_audit_failed"


def test_incompatible_model_support_bands_remain_failed(materials):
    model, source, desired = materials
    root = Path(__file__).resolve().parents[2]
    spec = mujoco.MjSpec.from_file(str(root / "gear_sonic/data/robots/g1/g1_23dof_rev_1_0.xml"))
    spec.geom("floor").pos[2] += 0.004
    higher_floor = spec.compile()
    cfg = ContactTrajectoryConfig(maximum_iterations=3)
    contacts = ContactLinearization({"mesh": model, "higher_floor": higher_floor}, source, [FEET] * 4, cfg)
    distances, _ = contacts.evaluate(desired, jacobian=False)
    mesh_rows = [i for i, row in enumerate(contacts.row_labels) if row[1:3] == ("mesh", "floor")]
    higher_rows = [i for i, row in enumerate(contacts.row_labels) if row[1:3] == ("higher_floor", "floor")]
    np.testing.assert_allclose(distances[mesh_rows] - distances[higher_rows], 0.004, atol=1e-10)
    _, report = refine_contact_trajectory(
        desired,
        desired - 0.08,
        desired + 0.08,
        np.ones(26),
        np.ones(26) * 6,
        np.zeros(26),
        contacts,
        config=cfg,
    )
    # Above the higher floor necessarily exceeds the lower floor's 2 mm
    # support band; below it necessarily violates the higher floor constraint.
    assert not report["contact_and_derivative_constraints_passed"]
    assert report["after"]["violated_frames"] == 4
    assert report["after"]["maximum_violation_m"] > 0.001


def test_actual_mesh_floor_restoration_keeps_whole_clip_and_false_readiness(materials):
    model, source, desired = materials
    cfg = ContactTrajectoryConfig(maximum_iterations=4)
    contacts = ContactLinearization({"mesh": model}, source, [()] * 4, cfg)
    distances, _ = contacts.evaluate(desired, jacobian=False)
    source[:, 2] -= distances.min() + 0.008
    contacts = ContactLinearization({"mesh": model}, source, [()] * 4, cfg)
    before = compiled_model_sha256(model)
    output, report = refine_contact_trajectory(
        desired,
        desired - 0.08,
        desired + 0.08,
        np.ones(26),
        np.ones(26) * 6,
        np.zeros(26),
        contacts,
        config=cfg,
    )
    assert report["before"]["maximum_violation_m"] >= 0.0079
    assert report["contact_and_derivative_constraints_passed"], report
    assert output.shape == desired.shape
    assert contacts.audit(output)["passed"]
    assert report["temporal_audit"]["passed"]
    assert not report["dynamic_feasibility_proven"] and not report["teacher_accepted"]
    assert compiled_model_sha256(model) == before


def test_refinement_problem_keeps_original_box_and_all_support_phases(materials):
    model, source, _path = materials
    source = np.tile(source, (4, 1))
    motion = build_mjlab_motion_arrays(
        model,
        SimpleNamespace(
            root_pos_w=source[:, :3],
            root_quat_wxyz=source[:, 3:7],
            joint_pos_hardware=source[:, 7:],
            fps=50.0,
        ),
    )
    evidence = {
        "config": asdict(StanceRetargetConfig()),
        "initial_projection_velocity": [0] * 26,
        "solver_rows": [{"frame": frame, "support_bodies": list(FEET)} for frame in range(16)],
    }
    problem = refinement_problem(model, motion, motion, evidence)
    assert problem["desired"].shape == (16, 26)
    assert len(problem["supports"]) == 16
    np.testing.assert_array_equal(problem["lower"][:, :3], np.full((16, 3), -0.08))
    assert np.all(problem["upper"][:, 3:] <= source[:, 7:] + 0.6000001)
    bad = copy.deepcopy(evidence)
    bad["solver_rows"].pop(1)
    with pytest.raises(ValueError, match="every original frame"):
        refinement_problem(model, motion, motion, bad)
