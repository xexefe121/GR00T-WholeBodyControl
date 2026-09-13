from pathlib import Path

import mujoco
import numpy as np
import pytest

from gear_sonic.utils.g1_23dof_contract import SOURCE_MJ29_KEEP_INDICES
from gear_sonic.utils.g1_23dof_safe_target_transform import SAFE_TARGET_DEFAULT_Q_HARDWARE
from gear_sonic.utils.g1_true23_continuous_reference_alignment import (
    ContinuousAlignmentConfig,
    ContinuousReferenceAlignment,
    temporal_bounds,
)
from gear_sonic.utils.g1_true23_reference_floor import compiled_model_sha256


@pytest.fixture(scope="module")
def models():
    path = Path(__file__).resolve().parents[2] / "gear_sonic/data/robots/g1"
    return tuple(
        mujoco.MjModel.from_xml_path(str(path / name)) for name in ("g1_29dof.xml", "g1_23dof_rev_1_0.xml")
    )


def initial():
    pose = np.zeros(36)
    pose[2:4] = [0.8, 1]
    pose[7 + np.asarray(SOURCE_MJ29_KEEP_INDICES)] = SAFE_TARGET_DEFAULT_Q_HARDWARE
    return pose


def test_initial_standing_stays_exact_with_no_model_mutation(models):
    pose = initial()
    obj = ContinuousReferenceAlignment(*models, pose)
    before = [compiled_model_sha256(model) for model in models]
    for index in range(12):
        result, report = obj.push(pose, frame_index=index)
        np.testing.assert_array_equal(result, np.r_[pose[:7], pose[7 + obj.keep]])
        assert report["initial_reference_noop"]
    assert before == [compiled_model_sha256(model) for model in models]
    assert np.all(obj.velocity == 0)


@pytest.mark.parametrize("task_name", ["left_foot", "right_foot", "head_proxy"])
def test_complete_root_leg_waist_feature_jacobian(models, task_name):
    pose = initial()
    obj = ContinuousReferenceAlignment(*models, pose)
    obj.current_source = pose.copy()
    obj.source_data.qpos[:] = pose
    mujoco.mj_fwdPosition(models[0], obj.source_data)
    value = obj.previous[:19].copy()
    value[:6] = [0.01, -0.02, 0.005, 0.08, -0.1, 0.03]
    value[6:] += 0.01
    task = next(task for task in obj.tasks if task.name == task_name)
    p, jp, rotation, jr = obj.features(value, [task])[0]
    analytic = np.vstack((jp, jr))
    columns = []
    for column in range(19):
        delta = np.eye(19)[column] * 1e-6
        plus, minus = obj.features(value + delta, [task])[0], obj.features(value - delta, [task])[0]
        columns.append((np.r_[plus[0], plus[2]] - np.r_[minus[0], minus[2]]) / 2e-6)
    np.testing.assert_allclose(analytic, np.asarray(columns).T, atol=2e-8, rtol=2e-6)
    assert p.shape == (3,) and rotation.shape == (9,)


def test_root_envelope_jacobian(models):
    obj = ContinuousReferenceAlignment(*models, initial())
    value = obj.previous[:19].copy()
    value[3:6] = [0.15, -0.08, 0.03]
    _, jacobian = obj.root_constraints(value)
    columns = []
    for column in range(19):
        delta = np.eye(19)[column] * 1e-6
        columns.append((obj.root_constraints(value + delta)[0] - obj.root_constraints(value - delta)[0]) / 2e-6)
    np.testing.assert_allclose(jacobian, np.asarray(columns).T, atol=1e-9, rtol=1e-6)


def test_temporal_braking_remains_feasible_under_adversarial_bound_requests():
    value, velocity = np.array([0.0]), np.array([0.0])
    low, high, vmax, amax = map(np.array, ([-0.4], [0.4], [5.0], [80.0]))
    for frame in range(150):
        lo, hi = temporal_bounds(value, velocity, low, high, vmax, amax)
        target = hi if (frame // 30) % 2 == 0 else lo
        new_velocity = (target - value) / 0.02
        assert np.all(np.abs(new_velocity) <= 5 + 1e-10)
        assert np.all(np.abs(new_velocity - velocity) <= 80 * 0.02 + 1e-10)
        assert np.all(target >= low) and np.all(target <= high)
        value, velocity = target.copy(), new_velocity


def test_empty_temporal_box_does_not_relax_acceleration():
    with pytest.raises(ValueError, match="not relaxed"):
        temporal_bounds(
            np.array([0.399]),
            np.array([5.0]),
            np.array([-0.4]),
            np.array([0.4]),
            np.array([5.0]),
            np.array([80.0]),
        )


@pytest.mark.parametrize("bad_index", [1, -1, True])
def test_index_failure_latches_without_advancing_accepted_state(models, bad_index):
    obj = ContinuousReferenceAlignment(*models, initial())
    before = obj.previous.copy()
    with pytest.raises(ValueError, match="sequential"):
        obj.push(initial(), frame_index=bad_index)
    np.testing.assert_array_equal(obj.previous, before)
    assert obj.index == 0
    with pytest.raises(RuntimeError, match="latched"):
        obj.push(initial(), frame_index=0)


def test_smooth_missing_waist_input_is_bounded_and_causal(models):
    source, native = models
    pose = initial()
    a, b = ContinuousReferenceAlignment(source, native, pose), ContinuousReferenceAlignment(source, native, pose)
    poses = []
    for index in range(7):
        current = pose.copy()
        current[int(source.joint("waist_pitch_joint").qposadr[0])] = 0.0005 * index**2
        out_a, evidence_a = a.push(current, frame_index=index)
        out_b, evidence_b = b.push(current.copy(), frame_index=index)
        np.testing.assert_array_equal(out_a, out_b)
        assert evidence_a == evidence_b
        assert max(evidence_a["foot_position_error_m"]) <= 0.005
        assert np.all(np.abs(a.velocity[6:]) <= 5)
        poses.append(out_a)
    joints = np.asarray(poses)[:, 7:]
    assert np.max(np.abs(np.diff(joints, axis=0))) <= 0.1
    assert not a.contract()["future_source_frames_consumed"]
    assert not a.contract()["deployment_ready"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"joint_velocity_rad_s": 5.01},
        {"root_position_limit_m": 0.1},
        {"base_tilt_limit_rad": 0.51},
        {"foot_position_limit_m": 0.006},
        {"maximum_iterations": 1.5},
    ],
)
def test_expanded_or_invalid_limits_rejected(kwargs):
    with pytest.raises(ValueError):
        ContinuousAlignmentConfig(**kwargs)
