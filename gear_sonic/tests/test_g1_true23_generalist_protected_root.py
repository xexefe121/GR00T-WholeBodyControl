"""Hard protected norms use actual native23 FK, never adjusted acceptance gates."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from gear_sonic.utils import g1_23dof_task_space_retarget as ik
from gear_sonic.utils.g1_23dof_contract import HARDWARE_23_JOINT_NAMES
from gear_sonic.utils.g1_23dof_safe_target_transform import SAFE_TARGET_DEFAULT_Q_HARDWARE
from gear_sonic.utils.g1_true23_generalist_protected_root import (
    audit_norms,
    residual_groups,
    solve_box_soc,
    fit_protected_task_path,
)
from gear_sonic.utils.g1_true23_generalist_retarget import (
    AdaptationLimits,
    _reduce_excursion,
    _resample,
    _restore_retained_diagnostic,
    adapt_offline_motion,
    refine_retained_protected_motion,
    validate_named_motion,
)
from gear_sonic.utils.g1_true23_original_task_trajectory import OriginalTaskPath


def test_soc_signs_enforce_norm_not_more_objective_weight():
    # Unconstrained minimizer [2,0] must lie on the exact unit disk boundary.
    group = {"indices": np.array([0, 1]), "scales": np.ones(2), "radius": 1.0}
    solution, report = solve_box_soc(np.ones(2), [-2.0, 0.0], sparse.eye(2), [-10.0, -10.0], [10.0, 10.0], [group])
    assert report["accepted"]
    np.testing.assert_allclose(solution, [1.0, 0.0], atol=1e-7)
    assert np.linalg.norm(solution) <= 1 + 1e-8


def test_soc_infeasible_problem_returns_no_solution_or_relaxed_pose():
    group = {"indices": np.array([0]), "scales": np.ones(1), "radius": 0.1}
    solution, report = solve_box_soc([1.0], [0.0], sparse.eye(1), [1.0], [2.0], [group])
    assert solution is None and not report["accepted"]
    assert "Infeasible" in report["status"]


@pytest.mark.parametrize("radius", [-1.0, np.nan, np.inf])
def test_soc_invalid_radius_rejected(radius):
    group = {"indices": np.array([0]), "scales": np.ones(1), "radius": radius}
    with pytest.raises(ValueError, match="norm constraint"):
        solve_box_soc([1.0], [0.0], sparse.eye(1), [-1.0], [1.0], [group])


def test_norm_audit_reports_exact_overlapping_frame_categories():
    groups = [
        {"name": "left", "frame": 3, "indices": np.array([0, 1]), "scales": np.ones(2), "radius": 1.0},
        {"name": "com", "frame": 3, "indices": np.array([2]), "scales": np.ones(1), "radius": 0.1},
    ]
    report = audit_norms(np.array([1.0, 1.0, 0.2]), groups)
    assert not report["passed"]
    assert report["categories"]["left"]["failed_frames"] == [3]
    assert report["categories"]["com"]["failed_frames"] == [3]


@pytest.fixture(scope="module")
def retained():
    root = Path(__file__).resolve().parents[2]
    source_model, target_model = ik.load_models(root / ik.DEFAULT_SOURCE_MODEL, root / ik.DEFAULT_TARGET_MODEL)
    names = ik._model_layout(source_model).joint_names
    defaults = dict(zip(HARDWARE_23_JOINT_NAMES, SAFE_TARGET_DEFAULT_Q_HARDWARE, strict=True))
    pose = np.array([defaults.get(name, 0.0) for name in names])
    pose[names.index("waist_roll_joint")] = 0.28
    source = {
        "joint_names": np.asarray(names),
        "joint_pos": np.tile(pose, (4, 1)),
        "root_pos_w": np.tile([0.0, 0.0, 0.8], (4, 1)),
        "root_quat_wxyz": np.tile([1.0, 0.0, 0.0, 0.0], (4, 1)),
        "fps": np.array([50.0]),
        "timestamps_s": np.arange(4) / 50,
        "contact_flags": np.ones((4, 2), dtype=bool),
    }
    result = adapt_offline_motion(
        source_model=source_model,
        target_model=target_model,
        arrays=source,
        source_role="requested_choreography",
        root_reference_refinement=True,
        limits=AdaptationLimits(duration_scales=(2.0,), excursion_scales=(0.9,)),
    )
    assert result.diagnostic_arrays is not None
    return source_model, target_model, source, result


def restored(retained):
    source_model, target_model, source, result = retained
    normalized = validate_named_motion(source, source_model)
    sampled, _, _ = _resample(normalized, 2.0, AdaptationLimits())
    candidate = _reduce_excursion(sampled, 0.9)
    config = ik.RetargetConfig(**result.report["ik_config"])
    return _restore_retained_diagnostic(
        source_model, target_model, candidate, result.diagnostic_arrays, config
    ), candidate


def test_restored_rejected_baselines_are_independently_recomputed(retained):
    baseline, _ = restored(retained)
    report = retained[3].report["attempts"][0]["ik_summary"]
    assert np.mean(baseline.expert_valid_mask()) == report["expert_valid_frame_fraction"]
    assert "joint_pos" not in retained[3].diagnostic_arrays


def test_tampered_retained_baseline_cannot_reset_com_budget(retained):
    source_model, target_model, source, result = retained
    normalized = validate_named_motion(source, source_model)
    sampled, _, _ = _resample(normalized, 2.0, AdaptationLimits())
    candidate = _reduce_excursion(sampled, 0.9)
    altered = {key: value.copy() for key, value in result.diagnostic_arrays.items()}
    altered["diagnostic_task_whole_robot_com_position_error_before_m"][:] += 0.02
    with pytest.raises(ValueError, match="baseline fails independent"):
        _restore_retained_diagnostic(
            source_model, target_model, candidate, altered, ik.RetargetConfig(**result.report["ik_config"])
        )


def test_task_soc_cost_norm_matches_original_per_frame_expert_budget(retained):
    baseline, candidate = restored(retained)
    # Constant real-model seed is already temporally feasible and inside ROM.
    seed = retained[3].diagnostic_arrays["diagnostic_fixed_root_qpos_native23"][:, 7:]
    problem = OriginalTaskPath(
        retained[0],
        retained[1],
        np.column_stack((candidate["root_pos_w"], candidate["root_quat_wxyz"], candidate["joint_pos"])),
        seed,
    )
    groups, row_count = residual_groups(problem, baseline)
    assert row_count == 39 and len(groups) == 6 * len(seed)
    residual, _, _ = problem.evaluate(problem.initial)
    for frame in range(len(seed)):
        group = next(g for g in groups if g["name"] == "original_weighted_cost" and g["frame"] == frame)
        # At the identical fixed-root seed, compare a direct actual FK call.
        layout, data = ik._model_layout(retained[1]), ik.mujoco.MjData(retained[1])
        ik._set_configuration(
            retained[1],
            data,
            layout,
            candidate["root_pos_w"][frame],
            candidate["root_quat_wxyz"][frame],
            seed[frame],
        )
        j, r, *_ = ik._task_linearization(
            retained[1],
            data,
            layout,
            problem.tasks,
            problem.targets[frame],
            (True, True),
            baseline.config.contact_weight_multiplier,
            np.arange(23),
        )
        assert np.linalg.norm(residual[group["indices"]] * group["scales"]) ** 2 == pytest.approx(
            ik._weighted_task_error(j, r), abs=1e-12
        )


def test_actual_model_hard_refinement_keeps_all_original_gates(retained):
    source_model, target_model = retained[:2]
    source = deepcopy(retained[2])
    source["joint_pos"][:, list(source["joint_names"]).index("waist_roll_joint")] = 0
    baseline = ik.retarget_trajectory(
        source_model=source_model,
        target_model=target_model,
        root_pos_w=source["root_pos_w"],
        root_quat_wxyz=source["root_quat_wxyz"],
        source_joint_pos_hardware=source["joint_pos"],
        fps=50,
        contact_flags=source["contact_flags"],
        config=ik.RetargetConfig(enable_lower_root_feasibility=False, optimize_lower_body=True),
    )
    problem = OriginalTaskPath(
        source_model,
        target_model,
        np.column_stack((source["root_pos_w"], source["root_quat_wxyz"], source["joint_pos"])),
        baseline.joint_pos_hardware,
    )
    initial = problem.initial.copy()
    initial[:, 6 + list(ik._model_layout(target_model).joint_names).index("left_ankle_roll_joint")] += 0.02
    variables, fit = fit_protected_task_path(problem, initial, baseline)
    assert not fit["before_protected_audit"]["passed"]
    assert fit["after_protected_audit"]["passed"], fit
    assert not fit["constraint_slack_variables_used"]
    assert not fit["protected_constraints_are_objective_penalties"]
    assert fit["path_constraints"]["passed"]
    assert variables.shape == initial.shape
    assert problem.audit(problem.serialized_variables(problem.serialize(variables)))["passed"]


def test_changed_approved_configuration_rejected_before_solving(retained):
    report = deepcopy(retained[3].report)
    report["accepted"] = False
    report["ik_config"]["valid_max_com_regression_m"] = 0.1
    with pytest.raises(ValueError, match="unchanged protected"):
        refine_retained_protected_motion(
            source_model=retained[0],
            target_model=retained[1],
            arrays=retained[2],
            stored=retained[3].diagnostic_arrays,
            forensic_report=report,
        )
