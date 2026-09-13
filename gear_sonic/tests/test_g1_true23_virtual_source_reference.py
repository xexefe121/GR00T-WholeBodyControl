from pathlib import Path

import numpy as np
import pytest

from gear_sonic.utils.g1_23dof_safe_target_transform import SAFE_TARGET_DEFAULT_Q_HARDWARE
from gear_sonic.utils.g1_true23_virtual_source_reference import virtual_source_vr_terms


def source_model():
    # This test needs only checked-in robot geometry, not any policy weights.
    path = Path(__file__).resolve().parents[2] / "gear_sonic/data/robots/g1/g1_29dof.xml"
    if not path.exists():
        pytest.skip("original29 model unavailable")
    return path


def test_geometry_embedding_is_world_invariant_and_preserves_reference_joints():
    q = np.tile(np.asarray(SAFE_TARGET_DEFAULT_Q_HARDWARE), (2, 1))
    before = q.copy()
    first = virtual_source_vr_terms(dict(joint_pos=q), source_model())
    second = virtual_source_vr_terms(dict(joint_pos=q, body_pos_w=np.full((2, 24, 3), 999.0)), source_model())
    np.testing.assert_array_equal(q, before)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (2, 21)
    assert first.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(first[:, 9:].reshape(2, 3, 4), axis=-1), 1, atol=1e-7)
    np.testing.assert_allclose(first[0, :3], first[0, 3:6] * [1, -1, 1], atol=2e-5)


def test_native_model_cannot_masquerade_as_source_geometry():
    path = source_model().with_name("g1_23dof_rev_1_0.xml")
    with pytest.raises(ValueError, match="original29 geometry"):
        virtual_source_vr_terms(dict(joint_pos=np.zeros((1, 23))), path)


def test_each_virtual_frame_uses_only_that_frames_joints():
    q = np.tile(np.asarray(SAFE_TARGET_DEFAULT_Q_HARDWARE), (3, 1))
    q[1, 16], q[2, 21] = 1.1, 1.4
    batch = virtual_source_vr_terms(dict(joint_pos=q), source_model())
    for index in range(len(q)):
        single = virtual_source_vr_terms(dict(joint_pos=q[index : index + 1]), source_model())
        np.testing.assert_array_equal(single[0], batch[index])


@pytest.mark.parametrize("q", [np.zeros((0, 23)), np.zeros((1, 29)), np.full((1, 23), np.inf)])
def test_invalid_reference_rejected_before_model_loading(q):
    with pytest.raises(ValueError, match="reference"):
        virtual_source_vr_terms(dict(joint_pos=q), "nonexistent.xml")
