from pathlib import Path

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from gear_sonic.utils.g1_23dof_contract import SOURCE_MJ29_KEEP_INDICES
from gear_sonic.utils.g1_23dof_safe_target_transform import SAFE_TARGET_DEFAULT_Q_HARDWARE
from gear_sonic.utils.g1_true23_reference_floor import compiled_model_sha256
from gear_sonic.utils.g1_true23_torso_reference_alignment import AlignmentConfig, TorsoReferenceAlignment


@pytest.fixture(scope="module")
def models():
    root = Path(__file__).resolve().parents[2] / "gear_sonic/data/robots/g1"
    return tuple(
        mujoco.MjModel.from_xml_path(str(root / name)) for name in ("g1_29dof.xml", "g1_23dof_rev_1_0.xml")
    )


def pose(source):
    value = np.zeros(36)
    value[:7] = [0.2, -0.3, 0.8, 1, 0, 0, 0]
    value[7 + np.asarray(SOURCE_MJ29_KEEP_INDICES)] = SAFE_TARGET_DEFAULT_Q_HARDWARE
    return value


def test_zero_missing_axes_preserve_pose_exactly_and_do_not_mutate_models(models):
    source, native = models
    alignment = TorsoReferenceAlignment(source, native)
    original = pose(source)
    original[3:7] = Rotation.from_euler("xyz", [0.1, -0.2, 0.5]).as_quat()[[3, 0, 1, 2]]
    hashes = tuple(compiled_model_sha256(model) for model in models)
    saved = original.copy()
    result, evidence = alignment.convert(original)
    np.testing.assert_array_equal(result, np.r_[original[:7], original[7 + alignment.keep]])
    np.testing.assert_array_equal(saved, original)
    assert hashes == tuple(compiled_model_sha256(model) for model in models)
    assert evidence["limb_function_evaluations"] == [0, 0, 0, 0]


def test_exact_virtual_torso_transform_does_not_erase_native_structural_offset(models):
    source, native = models
    alignment = TorsoReferenceAlignment(source, native)
    original = pose(source)
    for name, value in (("waist_yaw_joint", 0.2), ("waist_roll_joint", 0.12), ("waist_pitch_joint", -0.15)):
        original[int(source.joint(name).qposadr[0])] = value
    root = alignment.pelvis_pose(original)
    virtual = mujoco.MjData(source)
    virtual.qpos[:7] = root
    virtual.qpos[7 + alignment.keep] = original[7 + alignment.keep]
    mujoco.mj_fwdPosition(source, virtual)
    torso = source.body("torso_link").id
    np.testing.assert_allclose(virtual.xpos[torso], alignment.source_data.xpos[torso], atol=1e-12, rtol=0)
    np.testing.assert_allclose(virtual.xmat[torso], alignment.source_data.xmat[torso], atol=1e-12, rtol=0)
    result, _ = alignment.convert(original)
    data = mujoco.MjData(native)
    data.qpos[:] = result
    mujoco.mj_fwdPosition(native, data)
    native_torso = native.body("torso_link").id
    neutral = mujoco.MjData(native)
    neutral.qpos[:] = 0
    neutral.qpos[3] = 1
    neutral.qpos[7:] = original[7 + alignment.keep]
    mujoco.mj_fwdPosition(native, neutral)
    # Different torso origins include small yaw-dependent structural offsets;
    # derive the full offset, do not assume a constant 1cm norm away from zero yaw.
    expected_offset = Rotation.from_quat(root[[4, 5, 6, 3]]).apply(
        neutral.xpos[native_torso] - alignment.zero_data.xpos[torso]
    )
    np.testing.assert_allclose(data.xpos[native_torso] - virtual.xpos[torso], expected_offset, atol=1e-12, rtol=0)
    assert np.linalg.norm(expected_offset) > 0.009
    for side in ("left", "right"):
        old_id, new_id = source.body(side + "_ankle_roll_link").id, native.body(side + "_ankle_roll_link").id
        assert np.linalg.norm(data.xpos[new_id] - alignment.source_data.xpos[old_id]) < 1e-5


@pytest.mark.parametrize("limb", range(4))
def test_limb_jacobian_matches_finite_differences(models, limb):
    source, native = models
    alignment = TorsoReferenceAlignment(source, native)
    original = pose(source)
    original[int(source.joint("waist_pitch_joint").qposadr[0])] = 0.15
    target = np.r_[alignment.pelvis_pose(original), original[7 + alignment.keep]]
    task, indices = alignment.limbs[limb]
    body = source.body(task.source_body).id
    rotation = alignment.source_data.xmat[body].reshape(3, 3).copy()
    kwargs = dict(
        pose=target,
        task=task,
        indices=indices,
        desired=(alignment.source_data.xpos[body] + rotation @ task.source_point, rotation),
        posture=target[7 + indices],
    )
    q = target[7 + indices] + 0.01
    _, jacobian = alignment.limb_residual_jacobian(q, **kwargs)
    numerical = []
    for index in range(len(indices)):
        delta = np.eye(len(indices))[index] * 1e-6
        plus = alignment.limb_residual_jacobian(q + delta, **kwargs)[0]
        minus = alignment.limb_residual_jacobian(q - delta, **kwargs)[0]
        numerical.append((plus - minus) / 2e-6)
    np.testing.assert_allclose(jacobian, np.asarray(numerical).T, rtol=2e-6, atol=2e-8)


def test_frame_local_conversion_is_order_independent_and_bounded(models):
    source, native = models
    alignment = TorsoReferenceAlignment(source, native)
    first, second = pose(source), pose(source)
    first[int(source.joint("waist_pitch_joint").qposadr[0])] = 0.12
    second[int(source.joint("left_wrist_pitch_joint").qposadr[0])] = 0.5
    before, _ = alignment.convert(first)
    alignment.convert(second)
    after, _ = alignment.convert(first)
    np.testing.assert_array_equal(before, after)
    assert np.all(after[7:] >= alignment.lower) and np.all(after[7:] <= alignment.upper)
    contract = alignment.contract()
    assert not contract["previous_or_future_frames_consumed"]
    assert not contract["hardware_authorized"] and not contract["deployment_ready"]


@pytest.mark.parametrize("invalid", [np.zeros(36), np.full(36, np.nan), np.zeros((1, 36)), np.zeros(30)])
def test_invalid_pose_rejected(models, invalid):
    with pytest.raises(ValueError):
        TorsoReferenceAlignment(*models).convert(invalid)


def test_retained_waist_outside_reference_bounds_is_rejected_not_clipped(models):
    alignment = TorsoReferenceAlignment(*models)
    original = pose(models[0])
    original[7 + alignment.keep[12]] = alignment.upper[12] + 0.01
    with pytest.raises(ValueError, match="waist yaw"):
        alignment.convert(original)


@pytest.mark.parametrize(
    "kwargs",
    [{"arm_orientation_scale_m": 0}, {"leg_posture_scale_m": float("nan")}, {"max_function_evaluations": 1.5}],
)
def test_invalid_settings_rejected(kwargs):
    with pytest.raises(ValueError):
        AlignmentConfig(**kwargs)
