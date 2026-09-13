import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from gear_sonic.utils.g1_sonic_v11_onnx_reference import (
    heading_orientation_6d,
    pack_v11_encoder,
    validate_native_history,
)


def quat(angles):
    return Rotation.from_euler("xyz", angles).as_quat()[[3, 0, 1, 2]]


def test_heading_removes_only_measured_yaw_not_reference_tilt():
    measured = quat([0.4, -0.25, 1.2])
    desired = quat([0.1, 0.3, -0.2])
    expected = (
        (Rotation.from_euler("z", -1.2) * Rotation.from_quat(desired[[1, 2, 3, 0]])).as_matrix()[:, :2].reshape(6)
    )
    np.testing.assert_allclose(heading_orientation_6d(measured, desired), expected, atol=3e-8, rtol=0)


def test_heading_same_yaw_different_measured_tilt_is_same():
    desired = quat([0.1, 0.3, -0.2])
    np.testing.assert_array_equal(
        heading_orientation_6d(quat([0, 0, 1]), desired), heading_orientation_6d(quat([0.7, -0.4, 1]), desired)
    )


@pytest.mark.parametrize("bad", [np.zeros(4), np.array([np.nan, 0, 0, 0]), np.ones(3)])
def test_heading_rejects_invalid_quaternion(bad):
    with pytest.raises(ValueError, match="normalized WXYZ"):
        heading_orientation_6d(bad, quat([0, 0, 0]))


def test_actual_v11_graph_offsets_not_original1762_offsets():
    semantic = np.arange(267, dtype=np.float32)
    packed = pack_v11_encoder(semantic)
    assert packed.shape == (1, 1751) and packed.dtype == np.float32
    assert packed[0, 0] == 1
    np.testing.assert_array_equal(packed[0, 650:911], semantic[:261])
    np.testing.assert_array_equal(packed[0, 644:650], semantic[261:])
    assert not packed[0, 1:644].any() and not packed[0, 911:].any()


def test_old_or_wrong_width_input_rejected():
    with pytest.raises(ValueError, match="267"):
        pack_v11_encoder(np.zeros(1762, dtype=np.float32))


def test_absent_proprioception_slots_remain_zero():
    history = np.zeros(930, np.float32)
    validate_native_history(history)
    history[30 + 5] = 1
    with pytest.raises(ValueError, match="absent joint slots"):
        validate_native_history(history)
