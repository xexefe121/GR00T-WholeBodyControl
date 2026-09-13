from pathlib import Path

import mujoco
import numpy as np
import pytest

from gear_sonic.utils.g1_23dof_safe_target_transform import SAFE_TARGET_DEFAULT_Q_HARDWARE
from gear_sonic.utils.g1_true23_contact_conditioned_path import (
    ContactConditionedPath,
    fit_contact_conditioned_path,
    solve_elastic_task_step,
)
from gear_sonic.utils.g1_true23_reference_floor import compiled_model_sha256


@pytest.fixture
def problem():
    model = mujoco.MjModel.from_xml_path(
        str(Path(__file__).resolve().parents[1] / "data/robots/g1/g1_23dof_rev_1_0.xml")
    )
    poses = np.tile(model.qpos0, (5, 1))
    poses[:, 2] = 0.78
    poses[:, 7:] = SAFE_TARGET_DEFAULT_Q_HARDWARE
    return ContactConditionedPath(model, poses)


def test_full_control_and_half_time_grids_and_waist_identity(problem):
    assert problem.initial.shape == (5, 26)
    assert problem.template.shape == (9, 30)
    assert problem.mapping.shape == (9 * 26, 5 * 26)
    np.testing.assert_array_equal(problem.poses(problem.initial), problem.original)
    np.testing.assert_array_equal(problem.lower[:, 15], problem.initial[:, 15])
    np.testing.assert_array_equal(problem.upper[:, 15], problem.initial[:, 15])


def test_task_and_sole_jacobians_match_independent_finite_differences(problem):
    variables = problem.initial.copy()
    before = compiled_model_sha256(problem.model)
    state = problem.evaluate(variables)
    for frame, coordinate in ((0, 0), (2, 1), (2, 3), (2, 8), (4, 18), (4, 25)):
        plus, minus = variables.copy(), variables.copy()
        plus[frame, coordinate] += 1e-6
        minus[frame, coordinate] -= 1e-6
        positive, negative = problem.evaluate(plus), problem.evaluate(minus)
        column = frame * 26 + coordinate
        np.testing.assert_allclose(
            (positive["residual"] - negative["residual"]) / 2e-6,
            state["jacobian"][:, column].toarray().ravel(),
            atol=2e-7,
            rtol=0,
        )
        np.testing.assert_allclose(
            (positive["gaps"] - negative["gaps"]) / 2e-6,
            state["floor_jacobian"][:, column].toarray().ravel(),
            atol=2e-9,
            rtol=0,
        )
    assert compiled_model_sha256(problem.model) == before
    np.testing.assert_array_equal(variables, problem.initial)


def test_root_displacement_and_temporal_acceleration_cannot_escape_bounds(problem):
    assert np.max(problem.acceleration[:3]) == 0.5
    assert np.max(problem.acceleration[3:]) == 80
    bad = problem.initial.copy()
    bad[2, 0] = 0.031
    audit = problem.correction_audit(bad, problem.evaluate(bad))
    assert not audit["passed"]
    assert audit["linear_violation"] > 0


def test_root_attitude_is_never_an_optimization_variable(problem):
    changed = problem.initial.copy()
    changed[:, :3] += 0.01
    changed[:, 16:] += 0.01
    np.testing.assert_array_equal(problem.poses(changed)[:, 3:7], problem.original[:, 3:7])


def test_contact_feature_is_actual_nearest_sole_not_tie_in_flattened_target(problem):
    state = problem.evaluate(problem.initial)
    gaps = state["gaps"].reshape(-1, 2, 4)
    expected = [
        8 * frame + 4 * side + int(np.argmin(gaps[frame, side]))
        for frame, side in np.argwhere(problem.query_contacts)
    ]
    np.testing.assert_array_equal(problem.selected_points, expected)


def test_already_grounded_clear_reference_is_not_rewritten_or_qualified(problem):
    poses = problem.original.copy()
    data = mujoco.MjData(problem.model)
    data.qpos[:] = poses[0]
    mujoco.mj_forward(problem.model, data)
    gap = (data.geom_xpos[problem.geoms, 2] - problem.model.geom_size[problem.geoms, 0]).min()
    poses[:, 2] -= gap
    result, contacts, report = fit_contact_conditioned_path(problem.model, poses)
    np.testing.assert_array_equal(result, poses)
    assert contacts.all()
    assert not report["iterations"]
    assert not report["training_reference_accepted"]
    assert not report["dynamic_feasibility_proven"]
    assert not report["hardware_authorized"]


def test_explicit_moving_boundary_resolves_fixed_waist_first_acceleration_conflict(problem):
    poses = problem.original.copy()
    poses[:, 19] += np.linspace(-0.12, 0.12, len(poses))
    velocity = (poses[1, 7:] - poses[0, 7:]) / 0.02
    before = poses.copy(), velocity.copy()
    stationary = ContactConditionedPath(problem.model, poses)
    moving = ContactConditionedPath(problem.model, poses, initial_joint_velocity=velocity)
    stationary_audit = stationary.correction_audit(stationary.initial, stationary.evaluate(stationary.initial))
    moving_audit = moving.correction_audit(moving.initial, moving.evaluate(moving.initial))
    assert not stationary_audit["passed"]
    assert stationary_audit["linear_violation"] == pytest.approx(0.06 - 0.02**2 * 80 * 0.995)
    assert moving_audit["passed"]
    assert moving.initial_joint_velocity[12] == pytest.approx(3.0)
    assert np.max(moving.acceleration[3:]) == 80
    np.testing.assert_array_equal(moving.lower[:, 15], moving.initial[:, 15])
    np.testing.assert_array_equal(moving.upper[:, 15], moving.initial[:, 15])
    np.testing.assert_array_equal(poses, before[0])
    np.testing.assert_array_equal(velocity, before[1])
    velocity[:] = 0
    assert moving.initial_joint_velocity[12] == pytest.approx(3.0)


def test_moving_grounded_path_keeps_velocity_boundary_in_final_audit(problem):
    poses = problem.original.copy()
    data = mujoco.MjData(problem.model)
    data.qpos[:] = poses[0]
    mujoco.mj_forward(problem.model, data)
    gap = (data.geom_xpos[problem.geoms, 2] - problem.model.geom_size[problem.geoms, 0]).min()
    poses[:, 2] -= gap
    poses[:, 19] += np.linspace(-0.12, 0.12, len(poses))
    initial_velocity = (poses[1, 7:] - poses[0, 7:]) / 0.02
    output, contacts, report = fit_contact_conditioned_path(
        problem.model, poses, initial_joint_velocity=initial_velocity
    )
    np.testing.assert_array_equal(output, poses)
    np.testing.assert_array_equal(report["initial_joint_velocity_rad_s"], initial_velocity)
    assert contacts.all() and not report["iterations"]
    assert report["bounds"]["passed"]
    assert report["kind"] == "g1_true23_joint_contact_clearance_moving_source_boundary_v2"
    assert report["initial_velocity_boundary"] == "explicit_source_velocity_not_standing_acquisition"
    assert not report["standing_acquisition_or_velocity_matching_proven"]
    assert not report["training_reference_accepted"]
    assert not report["hardware_authorized"]


@pytest.mark.parametrize(
    "velocity",
    [
        np.zeros(22),
        np.zeros((1, 23)),
        np.full(23, np.nan),
        np.full(23, np.inf),
        np.full(23, True),
        np.full(23, "0"),
        np.full(23, 5.00001),
        np.full(23, -5.00001),
    ],
)
def test_invalid_or_expanded_initial_velocity_rejected(problem, velocity):
    with pytest.raises(ValueError, match="initial joint velocity"):
        ContactConditionedPath(problem.model, problem.original, initial_joint_velocity=velocity)


def test_cli_moving_source_diagnostic_requires_explicit_whole_path_mode(tmp_path):
    from gear_sonic.scripts.diagnose_g1_true23_stance_foot_cleanup import main

    with pytest.raises(SystemExit) as error:
        main(
            [
                "--asset-root",
                str(tmp_path),
                "--motion",
                str(tmp_path / "not_read.npz"),
                "--output-directory",
                str(tmp_path / "not_created"),
                "--source-initial-joint-velocity-diagnostic",
            ]
        )
    assert error.value.code == 2
    assert not (tmp_path / "not_created").exists()


def test_objective_mean_preserves_same_elastic_step_and_hard_limits():
    from scipy import sparse

    arguments = (
        np.array([-0.03, 0.01]),
        sparse.eye(2),
        sparse.eye(2),
        np.array([-0.05, -0.05]),
        np.array([0.05, 0.05]),
        [dict(indices=np.arange(2), scales=np.ones(2), radius=0.04)],
        sparse.csc_matrix([[1.0, 0.0]]),
        np.array([-0.01]),
    )
    summed, summed_report = solve_elastic_task_step(*arguments)
    mean, mean_report = solve_elastic_task_step(*arguments, objective_scale=1 / 667)
    assert summed_report["accepted"] and mean_report["accepted"]
    np.testing.assert_allclose(mean, summed, atol=1e-6, rtol=0)
    for report in (summed_report, mean_report):
        assert report["original_row_audit_tolerance"] == 1e-8
        assert report["independent_original_row_violation"] <= 1e-8
        assert report["independent_norm_violation"] <= 1e-8
        assert not report["hard_constraints_changed"]
        assert not report["objective_changed_from_task_lsq"]


def test_whole_path_uses_frame_normalization_without_changing_acceptance(problem, monkeypatch):
    from gear_sonic.utils import g1_true23_contact_conditioned_path as module

    captured = []

    def fail(*args, **kwargs):
        captured.append(kwargs["objective_scale"])
        return None, {"status": "test_no_step", "accepted": False}

    monkeypatch.setattr(module, "solve_elastic_task_step", fail)
    output, _, report = module.fit_contact_conditioned_path(problem.model, problem.original)
    assert captured == [1 / len(problem.original)]
    np.testing.assert_array_equal(output, problem.original)
    assert report["numerical_profile"] == "objective_mean_over_control_frames_v2"
    assert not report["relative_task_weights_and_constraints_changed"]
    assert not report["training_reference_accepted"]


@pytest.mark.parametrize("record_indeterminate", [False, True])
def test_cli_retains_geometry_before_support_failure_without_qualifying_reference(
    problem, tmp_path, monkeypatch, record_indeterminate
):
    import json
    from types import SimpleNamespace

    from gear_sonic.scripts import diagnose_g1_true23_stance_foot_cleanup as module

    poses = np.repeat(problem.original[:1], 12, axis=0)
    data = mujoco.MjData(problem.model)
    data.qpos[:] = poses[0]
    mujoco.mj_forward(problem.model, data)
    gap = (data.geom_xpos[problem.geoms, 2] - problem.model.geom_size[problem.geoms, 0]).min()
    poses[:, 2] -= gap
    source = module.ik.build_mjlab_motion_arrays(
        problem.model,
        SimpleNamespace(
            root_pos_w=poses[:, :3], root_quat_wxyz=poses[:, 3:7], joint_pos_hardware=poses[:, 7:], fps=50.0
        ),
    )
    motion = tmp_path / "motion.npz"
    np.savez_compressed(motion, **source)
    output = tmp_path / "diagnostic"
    monkeypatch.setattr(module, "prepare_true23_model", lambda *args: (None, problem.model, None))
    calls = []

    def unresolved_support(model, result, effort, **kwargs):
        calls.append(kwargs)
        if not kwargs["record_solver_failures"]:
            raise RuntimeError("support LP did not solve reliably: test Unknown")
        return dict(
            frames_with_no_support_solution=12,
            frames_with_indeterminate_support=12,
            frames_with_solution_above_effort_limits=0,
            frames_with_conditional_solution_within_effort_limits=0,
            rows=[dict(status="numerically_indeterminate_support", within_supplied_effort_limits=False)] * 12,
            dynamic_feasibility_proven=False,
        )

    monkeypatch.setattr(module, "audit_reference_support", unresolved_support)
    args = [
        "--asset-root",
        str(Path(__file__).resolve().parents[2]),
        "--motion",
        str(motion),
        "--output-directory",
        str(output),
        "--whole-path-contact-conditioning",
    ]
    if record_indeterminate:
        args.append("--record-support-solver-failures-diagnostic")
        assert module.main(args) == 0
        report = json.loads((output / "report.json").read_text())
        assert report["support_solver_failure_recording_diagnostic"]
        assert report["reference_support"]["frames_with_indeterminate_support"] == 12
        assert not report["training_reference_accepted"]
        assert not report["standing_acquisition_qualified"]
        assert not report["hardware_authorized"]
        assert not report["deployment_ready"]
    else:
        with pytest.raises(RuntimeError, match="test Unknown"):
            module.main(args)
        assert not (output / "report.json").exists()
    assert calls == [dict(reference_dynamics=True, record_solver_failures=record_indeterminate)]
    geometry = json.loads((output / "geometry_only.json").read_text())
    assert not geometry["support_audit_complete"]
    assert not geometry["training_reference_accepted"]
    assert not geometry["hardware_authorized"]
    assert not geometry["deployment_ready"]
    for key in ("output", "hypothesis"):
        path = output / geometry[key]["path"]
        assert module.sha256_file(path) == geometry[key]["sha256"]
    with np.load(output / "contact_conditioned.diagnostic.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["joint_pos"], source["joint_pos"])
    with pytest.raises(FileExistsError, match="refuses overwrite"):
        module.main(args)
    assert len(calls) == 1


@pytest.mark.parametrize(
    "scenario,expected_factors", [("base", [1.0]), ("fallback", [1.0, 0.1, 0.01]), ("all_fail", [1.0, 0.1, 0.01])]
)
def test_stalled_step_backoff_only_tightens_internal_boxes(problem, monkeypatch, scenario, expected_factors):
    from gear_sonic.utils import g1_true23_contact_conditioned_path as module

    poses = problem.original.copy()
    data = mujoco.MjData(problem.model)
    data.qpos[:] = poses[0]
    mujoco.mj_forward(problem.model, data)
    gap = (data.geom_xpos[problem.geoms, 2] - problem.model.geom_size[problem.geoms, 0]).min()
    poses[:, 2] -= gap + 0.0001
    before = poses.copy()
    reference = ContactConditionedPath(problem.model, poses)
    start, n = reference.operator.shape[0], reference.initial.size
    trust = np.tile(np.r_[np.full(3, 0.03), np.full(23, 0.3)], reference.count)
    calls = []
    identity = compiled_model_sha256(problem.model)

    def proposed_step(*args, **kwargs):
        lower, upper = args[3], args[4]
        factor = upper[start] / 0.03
        np.testing.assert_allclose(lower[start : start + n], -factor * trust, rtol=0, atol=1e-16)
        np.testing.assert_allclose(upper[start : start + n], factor * trust, rtol=0, atol=1e-16)
        fixed_rows = np.r_[np.arange(start), np.arange(start + n, len(lower))]
        if calls:
            np.testing.assert_array_equal(lower[fixed_rows], calls[0][1][fixed_rows])
            np.testing.assert_array_equal(upper[fixed_rows], calls[0][2][fixed_rows])
        calls.append((factor, lower.copy(), upper.copy()))
        assert kwargs["objective_scale"] == 1 / len(poses)
        step = np.zeros_like(reference.initial)
        correct_direction = scenario == "base" or (scenario == "fallback" and len(calls) == 3)
        step[:, 2] = 0.0002 if correct_direction else -0.0002
        return step.ravel(), dict(status="test_feasible_step", accepted=True)

    monkeypatch.setattr(module, "solve_elastic_task_step", proposed_step)
    output, _, report = module.fit_contact_conditioned_path(problem.model, poses)
    np.testing.assert_allclose([call[0] for call in calls], expected_factors)
    np.testing.assert_array_equal(poses, before)
    np.testing.assert_array_equal(output[:, 3:], poses[:, 3:])
    assert compiled_model_sha256(problem.model) == identity
    assert len(report["iterations"]) == 1
    row = report["iterations"][0]
    assert row["accepted"] == (scenario != "all_fail")
    if row["accepted"]:
        np.testing.assert_allclose(output[:, 2], poses[:, 2] + 0.0002, atol=1e-14, rtol=0)
        assert row["trust_region_scale"] == pytest.approx(expected_factors[-1])
        assert row["merit_after"] < row["merit_before"]
        assert report["bounds"]["passed"]
    else:
        np.testing.assert_array_equal(output, poses)
        assert row["selected_trust_trial"] is None
    assert report["step_selection_profile"] == "stalled_nonlinear_step_trust_backoff_v1"
    assert not report["final_path_or_collision_acceptance_relaxed"]
    assert not report["hardware_authorized"] and not report["training_reference_accepted"]


@pytest.mark.parametrize("invalid", ["wrong_shape", "nonfinite", "outside_trust_box"])
def test_forged_solver_step_cannot_escape_intermediate_trust_bounds(problem, monkeypatch, invalid):
    from gear_sonic.utils import g1_true23_contact_conditioned_path as module

    def bad_step(*args, **kwargs):
        step = np.zeros(problem.initial.size)
        if invalid == "wrong_shape":
            step = step[:-1]
        elif invalid == "nonfinite":
            step[0] = np.nan
        else:
            step[0] = 0.030001
        return step, dict(status="forged", accepted=True)

    monkeypatch.setattr(module, "solve_elastic_task_step", bad_step)
    with pytest.raises(ValueError, match="contact step"):
        module.fit_contact_conditioned_path(problem.model, problem.original)
