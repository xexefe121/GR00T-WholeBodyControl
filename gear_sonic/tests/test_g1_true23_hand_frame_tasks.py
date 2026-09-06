import copy
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
import pytest

from gear_sonic.scripts.retarget_g1_true23_hand_frame_trace import select_source_records
from gear_sonic.scripts.retarget_g1_true23_original29_trace import FLAGS, PROFILE
from gear_sonic.utils import g1_23dof_task_space_retarget as retarget
from gear_sonic.utils.g1_23dof_safe_target_transform import SAFE_TARGET_DEFAULT_Q_HARDWARE
from gear_sonic.utils.g1_true23_hand_frame_tasks import (
    HAND_FRAME_CONVENTION,
    NeutralWristHandTaskPath,
    neutral_wrist_hand_tasks,
    task_pose_errors,
)
from gear_sonic.utils.g1_true23_reference_floor import compiled_model_sha256


@pytest.fixture(scope="module")
def models():
    root = Path(__file__).resolve().parents[2]
    return retarget.load_models(root / retarget.DEFAULT_SOURCE_MODEL, root / retarget.DEFAULT_TARGET_MODEL)


def material(models):
    source, target = models
    low, high = retarget.safe_target_joint_bounds(target, native_action_clip=9.5, safe_limit_guard_rad=0.05)
    seed = np.tile(np.clip(SAFE_TARGET_DEFAULT_Q_HARDWARE, low, high), (4, 1))
    poses = np.zeros((4, 36))
    poses[:, 2:4] = [0.8, 1]
    for index, name in enumerate(retarget._model_layout(target).joint_names):
        poses[:, int(source.joint(name).qposadr[0])] = seed[:, index]
    return poses, seed


def test_neutral_wrist_frame_correction_is_derived_and_preserves_models_and_defaults(models):
    source, target = models
    original = retarget.DEFAULT_TASKS
    hashes = compiled_model_sha256(source), compiled_model_sha256(target)
    tasks, report = neutral_wrist_hand_tasks(source, target)
    assert tasks is not original and retarget.DEFAULT_TASKS is original
    assert report["task_point_convention"] == HAND_FRAME_CONVENTION
    for old, new in zip(original, tasks, strict=True):
        assert replace(new, target_point=old.target_point) == old
    for hand in report["hands"]:
        assert hand["neutral_legacy_position_error_m"] == pytest.approx(0.084, abs=1e-12)
        assert hand["neutral_corrected_position_error_m"] < 1e-12
        assert hand["corrected_target_point"][0] == pytest.approx(0.264, abs=1e-12)
    assert hashes == (compiled_model_sha256(source), compiled_model_sha256(target))
    for flag in (
        "default_teleop_and_checkpoint_convention_modified",
        "proxy_is_a_physical_contact_landmark",
        "source_and_target_hand_meshes_identical",
        "teacher_accepted",
        "hardware_authorized",
        "deployment_ready",
    ):
        assert report[flag] is False


def test_shared_pose_alignment_does_not_erase_actual_missing_wrist_motion(models):
    source, target = models
    poses, seed = material(models)
    tasks, _ = neutral_wrist_hand_tasks(source, target)
    native = np.column_stack((poses[:, :7], seed))
    error = task_pose_errors(source, target, poses, native, tasks=tasks)
    for hand in ("left_hand", "right_hand"):
        assert error[hand]["position_max_m"] < 1e-10
    poses[:, int(source.joint("left_wrist_pitch_joint").qposadr[0])] = 0.4
    poses[:, int(source.joint("right_wrist_yaw_joint").qposadr[0])] = -0.3
    original = poses.copy()
    problem = NeutralWristHandTaskPath(source, target, poses, seed)
    error = task_pose_errors(source, target, poses, native, tasks=tasks)
    for hand in ("left_hand", "right_hand"):
        assert error[hand]["position_max_m"] > 0.03
        assert problem.metrics(problem.initial)["per_task"][hand]["position_max_m"] == pytest.approx(
            error[hand]["position_max_m"], abs=1e-12
        )
    np.testing.assert_array_equal(poses, original)
    np.testing.assert_array_equal(problem.source, original)


def test_corrected_hand_task_jacobians_cover_root_attitude_all_23_joints_and_frame_isolation(models):
    source, target = models
    poses, seed = material(models)
    poses[:, int(source.joint("waist_pitch_joint").qposadr[0])] = 0.1
    poses[:, int(source.joint("left_wrist_yaw_joint").qposadr[0])] = 0.3
    problem = NeutralWristHandTaskPath(source, target, poses, seed)
    variables = problem.initial.copy()
    variables[:, :6] = [0.01, -0.01, 0.005, 0.1, -0.07, 0.04]
    variables[:, 6:] += 0.01
    _, jac, _ = problem.evaluate(variables)
    numerical = []
    for column in range(29):
        step = np.zeros_like(variables)
        step[0, column] = 1e-6
        plus = problem.evaluate(variables + step, derivatives=False)[0]
        minus = problem.evaluate(variables - step, derivatives=False)[0]
        numerical.append((plus - minus) / 2e-6)
    np.testing.assert_allclose(jac[:, :29].toarray(), np.asarray(numerical).T, atol=3e-6, rtol=2e-6)
    assert jac.shape[1] == 29 * len(variables)


def test_changed_common_wrist_frame_is_rejected_not_silently_calibrated(models):
    source, _ = models
    root = Path(__file__).resolve().parents[2]
    spec = mujoco.MjSpec.from_file(str(root / retarget.DEFAULT_TARGET_MODEL))
    spec.body("left_wrist_roll_rubber_hand").pos[0] += 0.01
    target = spec.compile()
    with pytest.raises(ValueError, match="neutral shared wrist-roll frames"):
        neutral_wrist_hand_tasks(source, target)


@pytest.mark.parametrize("damage", ["empty", "crop", "nan", "quaternion"])
def test_error_measurement_rejects_incomplete_or_nonfinite_poses(models, damage):
    source, target = models
    poses, seed = material(models)
    native = np.column_stack((poses[:, :7], seed))
    if damage == "empty":
        poses, native = poses[:0], native[:0]
    elif damage == "crop":
        native = native[:-1]
    elif damage == "nan":
        poses[0, 8] = np.nan
    else:
        native[0, 3:7] = 0
    with pytest.raises(ValueError, match="complete matching"):
        task_pose_errors(source, target, poses, native, tasks=retarget.DEFAULT_TASKS)


def source_report():
    return {
        "kind": "g1_sonic_original29_full_hand_collision_recorded_diagnostic_v1",
        "model_variant": {
            "kind": "g1_sonic_original29_hand_collision_variant_v1",
            "original_source_unchanged": True,
            "native23_model_modified": False,
            "variant_compiled_sha256": "example-not-used-to-load-a-model",
            **FLAGS,
        },
        "records": [
            {"name": name, "details": {"profile": PROFILE}}
            for name in ("hand_crawling", "elbow_crawling", "happy_dance")
        ],
        **FLAGS,
    }


def test_collision_source_identity_preserved_without_mutating_or_dropping_rows():
    report = source_report()
    original = copy.deepcopy(report)
    rows, filename, digest = select_source_records(report)
    assert report == original
    assert [row["name"] for row in rows] == [row["name"] for row in original["records"]]
    assert all(row["profile"] == PROFILE for row in rows)
    assert filename == "original29_with_hand_collisions.mjb"
    assert digest == report["model_variant"]["variant_compiled_sha256"]


@pytest.mark.parametrize("damage", ["kind", "model", "profile", "omit", "promote"])
def test_unknown_or_mislabelled_source_experiment_is_rejected(damage):
    report = source_report()
    if damage == "kind":
        report["kind"] = "accepted_teacher"
    elif damage == "model":
        report["model_variant"]["native23_model_modified"] = True
    elif damage == "profile":
        report["records"][0]["profile"] = "legacy_python"
    elif damage == "omit":
        report["records"].pop()
    else:
        report["deployment_ready"] = True
    with pytest.raises(ValueError):
        select_source_records(report)
