from pathlib import Path

import mujoco
import numpy as np
import pytest

from gear_sonic.scripts.audit_g1_sonic_low_latency29_baseline import reference_features
from gear_sonic.scripts.record_g1_sonic_public29_baseline import stock
from gear_sonic.utils import g1_true23_original29_reference as module
from gear_sonic.utils.g1_true23_reference_floor import compiled_model_sha256


@pytest.fixture
def model():
    path = Path(__file__).resolve().parents[2] / "gear_sonic/data/robots/g1/g1_29dof.xml"
    return mujoco.MjModel.from_xml_path(str(path))


def poses():
    q = np.tile(np.r_[[0, 0, 0.76, 1, 0, 0, 0], stock.DEFAULT_ANGLES], (12, 1))
    q[:, 0] = np.arange(12) * 0.01
    q[:, 20] = np.arange(12) * 0.02  # Missing physical waist roll, retained source intent.
    q[:, 27] = np.arange(12) * 0.01  # Missing physical left wrist pitch.
    return q


def native(q):
    return dict(
        fps=np.array([50.0]),
        joint_pos=q[:, 7:][:, module.SOURCE_MJ29_KEEP_INDICES].copy(),
        body_pos_w=np.repeat(q[:, None, :3], 24, axis=1),
        body_quat_w=np.repeat(q[:, None, 3:7], 24, axis=1),
    )


def test_new_reference_matches_the_executed_original29_vr_boundary(model):
    q = poses()
    before, model_before = q.copy(), compiled_model_sha256(model)
    reference = module.build_original29_reference(model, q)
    expected = reference_features(model, q, np.zeros((len(q), 29)))[:, 240:261]
    np.testing.assert_array_equal(reference.virtual_vr21, expected)
    np.testing.assert_array_equal(q, before)
    assert compiled_model_sha256(model) == model_before
    assert not reference.virtual_vr21.flags.writeable
    exported = reference.arrays()
    exported["source_qpos29"][0, 0] = 20
    assert reference.source_qpos29[0, 0] == 0
    assert module.verify_unmodified_native_pair(reference, native(q))["frames"] == len(q)


def test_world_targets_reconstructed_from_the_same_received_intent(model):
    q = poses()
    q[:, 3:7] = [0.8, 0, 0, 0.6]
    reference = module.build_original29_reference(model, q)
    p, quat = module.task_targets_from_received_vr(
        q[:, :3].astype(np.float32), q[:, 3:7].astype(np.float32), reference.virtual_vr21
    )
    np.testing.assert_allclose(p, reference.source_task_position_w, rtol=0, atol=1e-7)
    np.testing.assert_allclose(quat, reference.source_task_quaternion_wxyz, rtol=0, atol=1e-7)


def test_absent_source_angles_are_not_reconstructed_from_native23(model):
    q = poses()
    neutral = q.copy()
    neutral[:, [20, 27]] = 0
    np.testing.assert_array_equal(native(q)["joint_pos"], native(neutral)["joint_pos"])
    full = module.build_original29_reference(model, q)
    dropped = module.build_original29_reference(model, neutral)
    assert np.max(np.abs(full.virtual_vr21 - dropped.virtual_vr21)) > 0.01
    assert np.max(np.abs(full.source_task_position_w - dropped.source_task_position_w)) > 0.01


def test_frame_locality_without_future_sample_use(model):
    q = poses()
    all_frames = module.build_original29_reference(model, q)
    for i in (0, 5, 11):
        single = module.build_original29_reference(model, q[i : i + 1])
        for name, value in single.arrays().items():
            np.testing.assert_array_equal(value, all_frames.arrays()[name][i : i + 1])


@pytest.mark.parametrize("invalid", ("empty", "shape", "nonfinite", "quaternion", "complex"))
def test_invalid_source_never_silently_repaired(model, invalid):
    q = poses()
    if invalid == "empty":
        q = q[:0]
    elif invalid == "shape":
        q = q[:, :30]
    elif invalid == "nonfinite":
        q[0, 7] = np.nan
    elif invalid == "quaternion":
        q[0, 3] = 0.5
    else:
        q = q.astype(complex)
    with pytest.raises(ValueError):
        module.build_original29_reference(model, q)


@pytest.mark.parametrize("field", ("joint_pos", "body_pos_w", "body_quat_w", "fps"))
def test_native_pair_mismatch_cannot_be_relabelled(model, field):
    q = poses()
    motion = native(q)
    reference = module.build_original29_reference(model, q)
    motion[field].flat[0] += 0.001
    with pytest.raises(ValueError):
        module.verify_unmodified_native_pair(reference, motion)


@pytest.mark.parametrize("invalid", ("shape", "nonfinite", "root_quaternion", "task_quaternion"))
def test_received_task_reconstruction_rejects_bad_packets(invalid):
    p, q = np.zeros((1, 3)), np.array([[1.0, 0, 0, 0]])
    vr = np.zeros((1, 21))
    vr[:, [9, 13, 17]] = 1
    if invalid == "shape":
        vr = vr[:, :20]
    elif invalid == "nonfinite":
        p[0, 0] = np.inf
    elif invalid == "root_quaternion":
        q[0, 0] = 0.5
    else:
        vr[0, 9] = 0.5
    with pytest.raises(ValueError):
        module.task_targets_from_received_vr(p, q, vr)


def test_native_model_cannot_masquerade_as_source():
    path = Path(__file__).resolve().parents[2] / "gear_sonic/data/robots/g1/g1_23dof_rev_1_0.xml"
    model = mujoco.MjModel.from_xml_path(str(path))
    with pytest.raises(ValueError, match="exactly original29"):
        module.build_original29_reference(model, poses())


def test_contract_keeps_physical_and_recorded_dof_separate():
    contract = module.reference_contract("a" * 64)
    assert len(contract["source_joint_names"]) == 29
    assert len(contract["physical_joint_names"]) == contract["physical_joint_count"] == 23
    assert not contract["old_zero_absent_reference_checkpoint_may_be_relabelled"]
    assert all(contract[key] is False for key in module.FLAGS)
    with pytest.raises(ValueError):
        module.reference_contract("invalid")
