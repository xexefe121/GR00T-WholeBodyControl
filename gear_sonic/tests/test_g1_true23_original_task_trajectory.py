from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from gear_sonic.utils import g1_true23_original_task_trajectory as task_fit
from gear_sonic.utils.g1_23dof_safe_target_transform import SAFE_TARGET_DEFAULT_Q_HARDWARE
from gear_sonic.utils.g1_23dof_task_space_retarget import (
    DEFAULT_SOURCE_MODEL,
    DEFAULT_TARGET_MODEL,
    _model_layout,
    load_models,
    safe_target_joint_bounds,
)
from gear_sonic.utils.g1_true23_reference_floor import compiled_model_sha256


@pytest.fixture
def material():
    root = Path(__file__).resolve().parents[2]
    source, target = load_models(root / DEFAULT_SOURCE_MODEL, root / DEFAULT_TARGET_MODEL)
    low, high = safe_target_joint_bounds(target, native_action_clip=9.5, safe_limit_guard_rad=0.05)
    seed = np.tile(np.clip(SAFE_TARGET_DEFAULT_Q_HARDWARE, low, high), (6, 1))
    source_pose = np.tile(source.qpos0, (6, 1))
    source_pose[:, :7] = [0, 0, 0.76, 1, 0, 0, 0]
    layout = _model_layout(source)
    for index, name in enumerate(_model_layout(target).joint_names):
        source_pose[:, layout.qpos_addresses[layout.joint_names.index(name)]] = seed[:, index]
    for name, value in (("waist_pitch_joint", 0.18), ("waist_roll_joint", -0.04)):
        source_pose[:, layout.qpos_addresses[layout.joint_names.index(name)]] = value
    return source, target, source_pose, seed


@pytest.mark.parametrize("vector", [np.zeros(3), np.array([1e-8, 2e-8, -1e-8]), np.array([0.2, -0.13, 0.07])])
def test_so3_coordinate_jacobian_matches_world_rotation_difference(vector):
    base = Rotation.from_rotvec(vector)
    actual = []
    for coordinate in range(3):
        step = np.zeros(3)
        step[coordinate] = 1e-6
        plus = (Rotation.from_rotvec(vector + step) * base.inv()).as_rotvec()
        minus = (Rotation.from_rotvec(vector - step) * base.inv()).as_rotvec()
        actual.append((plus - minus) / 2e-6)
    jac = task_fit.so3_left_jacobian(vector)
    np.testing.assert_allclose(jac, np.array(actual).T, atol=1e-9, rtol=1e-8)
    np.testing.assert_allclose(task_fit.so3_left_jacobian_inverse(vector) @ jac, np.eye(3), atol=1e-12)


def test_all_pose_jacobian_columns_and_frame_isolation(material):
    source, target, raw, seed = material
    before = compiled_model_sha256(source), compiled_model_sha256(target)
    original = raw.copy(), seed.copy()
    problem = task_fit.OriginalTaskPath(source, target, raw, seed)
    x = problem.initial.copy()
    x[:, :6] = [0.01, -0.02, 0.01, 0.12, -0.08, 0.05]
    x[:, 6:] += 0.01
    residual, jac, _ = problem.evaluate(x)
    numerical = []
    for column in range(29):
        step = np.zeros_like(x)
        step[0, column] = 1e-6
        plus = problem.evaluate(x + step, derivatives=False)[0]
        minus = problem.evaluate(x - step, derivatives=False)[0]
        numerical.append((plus - minus) / 2e-6)
    np.testing.assert_allclose(jac[:, :29].toarray(), np.array(numerical).T, atol=3e-6, rtol=2e-6)
    assert jac.shape[1] == 29 * len(seed) and len(residual) % len(seed) == 0
    np.testing.assert_array_equal(raw, original[0])
    np.testing.assert_array_equal(seed, original[1])
    assert before == (compiled_model_sha256(source), compiled_model_sha256(target))


def test_whole_path_fit_compensates_missing_waist_without_authorization(material):
    problem = task_fit.OriginalTaskPath(*material, config=task_fit.OriginalTaskConfig(maximum_iterations=6))
    output, report = task_fit.fit_original_task_path(problem)
    assert output.shape == (6, 29)
    assert report["after"]["weighted_task_squared_error"] < report["before"]["weighted_task_squared_error"]
    assert np.max(np.abs(output[:, 3:6])) > 0.01
    assert report["path_constraints"]["passed"]
    assert report["frames_dropped"] == 0 and report["native_joint_count"] == 23
    for iteration in report["iterations"]:
        if iteration["accepted"]:
            assert iteration["qp"]["independent_original_absolute_row_violation"] <= 1e-8
            assert iteration["qp"]["objective_scale"] == pytest.approx(1 / 6)
    assert all(report[flag] is False for flag in ("teacher_accepted", "hardware_authorized", "deployment_ready"))
    motion = problem.serialize(output)
    saved = problem.serialized_variables(motion)
    assert problem.audit(saved)["passed"]
    np.testing.assert_allclose(saved, output, atol=1e-7, rtol=0)


def test_rotation_l1_bound_is_not_a_per_component_box(material):
    problem = task_fit.OriginalTaskPath(*material)
    x = problem.initial.copy()
    x[:, 3:6] = 0.2
    # Every coordinate is less than 0.45; their sum must still fail.
    assert not problem.audit(x)["passed"]


def test_relative_rotation_bound_does_not_allow_excess_absolute_tilt(material):
    source, target, raw, seed = material
    raw[:, 3:7] = Rotation.from_euler("y", 0.4).as_quat()[[3, 0, 1, 2]]
    problem = task_fit.OriginalTaskPath(source, target, raw, seed)
    x = problem.initial.copy()
    x[:, 4] = 0.2
    report = problem.audit(x)
    assert report["temporal"]["passed"]
    assert report["maximum_root_rotation_l1_rad"] == pytest.approx(0.2)
    assert report["maximum_absolute_base_tilt_rad"] == pytest.approx(0.6)
    assert report["absolute_base_tilt_limit_rad"] == 0.5
    assert not report["absolute_base_tilt_passed"]
    assert not report["passed"]


def test_absolute_tilt_jacobian_and_frame_isolation(material):
    source, target, raw, seed = material
    raw[:, 3:7] = Rotation.from_euler("xyz", [0.12, 0.25, 0.4]).as_quat()[[3, 0, 1, 2]]
    problem = task_fit.OriginalTaskPath(source, target, raw, seed)
    x = problem.initial.copy()
    x[:, 3:6] = [0.08, -0.03, 0.1]
    cosine, jacobian = problem.base_tilt_linearization(x)
    numerical = []
    for column in range(29):
        step = np.zeros_like(x)
        step[0, column] = 1e-6
        plus = problem.base_tilt_linearization(x + step, derivatives=False)[0]
        minus = problem.base_tilt_linearization(x - step, derivatives=False)[0]
        numerical.append((plus - minus) / 2e-6)
    np.testing.assert_allclose(jacobian[:, :29].toarray(), np.array(numerical).T, atol=2e-9, rtol=1e-7)
    assert jacobian.shape == (len(seed), 29 * len(seed))
    np.testing.assert_allclose(
        cosine,
        problem.qpos(x)[:, 3] ** 2
        + problem.qpos(x)[:, 6] ** 2
        - problem.qpos(x)[:, 4] ** 2
        - problem.qpos(x)[:, 5] ** 2,
        atol=1e-14,
    )


def test_near_limit_fit_and_serialized_path_enforce_absolute_tilt(material):
    source, target, raw, seed = material
    raw[:, 3:7] = Rotation.from_euler("y", 0.45).as_quat()[[3, 0, 1, 2]]
    problem = task_fit.OriginalTaskPath(
        source, target, raw, seed, config=task_fit.OriginalTaskConfig(maximum_iterations=6)
    )
    output, report = task_fit.fit_original_task_path(problem)
    assert report["after"]["weighted_task_squared_error"] < report["before"]["weighted_task_squared_error"]
    assert report["path_constraints"]["passed"]
    assert report["path_constraints"]["maximum_absolute_base_tilt_rad"] <= 0.5 + 2e-7
    assert report["absolute_base_tilt_enforced_separately_from_rotation_change"]
    assert any(row["accepted"] for row in report["iterations"])
    for row in report["iterations"]:
        assert row["qp"]["absolute_base_tilt_linearization_enabled"]
        assert row["qp"]["linearized_base_tilt_limit_with_serialization_margin_rad"] == pytest.approx(0.4975)
        if row["accepted"]:
            assert row["qp"]["independent_original_absolute_row_violation"] <= 1e-8
    assert problem.audit(problem.serialized_variables(problem.serialize(output)))["passed"]
    assert report["deployment_ready"] is False


def test_explicit_feasible_root_seed_does_not_relax_limits_or_change_source(material):
    source, target, raw, seed = material
    raw[:, 3:7] = Rotation.from_euler("y", 0.6).as_quat()[[3, 0, 1, 2]]
    root_seed = np.tile(Rotation.from_euler("y", 0.3).as_quat()[[3, 0, 1, 2]], (len(seed), 1))
    originals = raw.copy(), seed.copy(), root_seed.copy()
    with pytest.raises(ValueError, match="absolute base tilt"):
        task_fit.OriginalTaskPath(source, target, raw, seed)
    problem = task_fit.OriginalTaskPath(source, target, raw, seed, seed_root_quaternion_wxyz=root_seed)
    audit = problem.audit(problem.initial)
    assert audit["passed"]
    assert audit["maximum_absolute_base_tilt_rad"] == pytest.approx(0.3)
    assert audit["maximum_root_rotation_l1_rad"] == pytest.approx(0.3)
    np.testing.assert_allclose(problem.qpos(problem.initial)[:, 3:7], root_seed, atol=1e-14)
    for actual, expected in zip((raw, seed, root_seed), originals, strict=True):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("change", ["short", "nan", "zero", "unnormalized", "excess_tilt", "excess_change"])
def test_invalid_or_out_of_bounds_root_seed_is_rejected(material, change):
    source, target, raw, seed = material
    root_seed = raw[:, 3:7].copy()
    if change == "short":
        root_seed = root_seed[:-1]
    elif change == "nan":
        root_seed[2, 0] = np.nan
    elif change == "zero":
        root_seed[2] = 0
    elif change == "unnormalized":
        root_seed *= 2
    elif change == "excess_tilt":
        raw[:, 3:7] = Rotation.from_euler("y", 0.4).as_quat()[[3, 0, 1, 2]]
        root_seed[:] = Rotation.from_euler("y", 0.6).as_quat()[[3, 0, 1, 2]]
    else:
        root_seed[:] = Rotation.from_euler("z", 0.46).as_quat()[[3, 0, 1, 2]]
    with pytest.raises(ValueError):
        task_fit.OriginalTaskPath(source, target, raw, seed, seed_root_quaternion_wxyz=root_seed)


def test_forged_tilt_violating_qp_rejected_by_original_row_audit(material, monkeypatch):
    source, target, raw, seed = material
    raw[:, 3:7] = Rotation.from_euler("y", 0.49).as_quat()[[3, 0, 1, 2]]
    problem = task_fit.OriginalTaskPath(source, target, raw, seed)
    current = problem.initial.copy()
    residual, jacobian, _ = problem.evaluate(current)
    increment = np.zeros_like(current)
    increment[:, 4] = 0.02

    def forged(*args, **kwargs):
        # Satisfy task equality, trust, temporal, joint and relative-rotation rows.
        # Only the new absolute-tilt row rules out this otherwise plausible step.
        return np.r_[increment.ravel(), residual + jacobian @ increment.ravel()], {
            "status": "Solved",
            "accepted": True,
        }

    monkeypatch.setattr(task_fit, "solve_box_qp", forged)
    output, report = task_fit.solve_task_step(problem, current, residual, jacobian)
    assert output is None
    assert report["status"] == "original_absolute_row_audit_failed"
    assert not report["accepted"]
    assert report["independent_original_absolute_row_violation"] > 1e-3


def test_rejected_qp_preserves_full_seed_and_failure(material, monkeypatch):
    problem = task_fit.OriginalTaskPath(*material)
    monkeypatch.setattr(
        task_fit, "solve_box_qp", lambda *args, **kwargs: (None, {"status": "AlmostSolved", "accepted": False})
    )
    output, report = task_fit.fit_original_task_path(problem)
    np.testing.assert_array_equal(output, problem.initial)
    assert report["failure"] == "task QP failed strict solver/original-row audit"
    assert report["iterations"][0]["accepted"] is False


def test_forged_increment_solution_fails_original_absolute_row_audit(material, monkeypatch):
    problem = task_fit.OriginalTaskPath(*material)

    def forged(diagonal, *args, **kwargs):
        return np.ones_like(diagonal), {"status": "Solved", "accepted": True}

    monkeypatch.setattr(task_fit, "solve_box_qp", forged)
    output, report = task_fit.fit_original_task_path(problem)
    np.testing.assert_array_equal(output, problem.initial)
    assert report["iterations"][0]["qp"]["status"] == "original_absolute_row_audit_failed"
    assert report["iterations"][0]["qp"]["accepted"] is False


@pytest.mark.parametrize(
    "config",
    [
        {"maximum_iterations": True},
        {"maximum_iterations": 1.5},
        {"maximum_root_rotation_l1_rad": 0.5},
        {"maximum_base_tilt_rad": 0.50001},
        {"maximum_base_tilt_rad": 0},
        {"maximum_base_tilt_rad": np.nan},
        {"joint_velocity_rad_s": 5.1},
        {"joint_acceleration_rad_s2": 81},
        {"root_trust_m": np.nan},
        {"serialization_margin_fraction": 1},
    ],
)
def test_invalid_or_expanded_limits_fail_closed(config):
    with pytest.raises(ValueError):
        task_fit.OriginalTaskConfig(**config)


@pytest.mark.parametrize("change", ["short", "nan", "zero_quaternion", "wrong_seed"])
def test_invalid_full_source_rejected(material, change):
    source, target, raw, seed = material
    if change == "short":
        raw = raw[:-1]
    elif change == "nan":
        raw[2, 10] = np.nan
    elif change == "zero_quaternion":
        raw[2, 3:7] = 0
    else:
        seed = seed[:, :-1]
    with pytest.raises(ValueError):
        task_fit.OriginalTaskPath(source, target, raw, seed)
