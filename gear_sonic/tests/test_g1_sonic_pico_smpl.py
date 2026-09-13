"""CPU adapter contracts; no policy checkpoint, training, or robot access."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from gear_sonic.teleop.buffered_source_horizon import _relative_orientation_6d
from gear_sonic.utils.g1_sonic_pico_smpl import (
    OUTPUT_JOINTS,
    smpl_input,
    smpl_relative_from_teleop,
)


def test_smpl_points_are_thumbs_not_smpl24_hands():
    assert OUTPUT_JOINTS == (*range(22), 39, 54)
    assert OUTPUT_JOINTS[-2:] != (22, 23)


def test_per_frame_concatenation_not_whole_term_concatenation():
    joints = np.arange(4 * 72, dtype=np.float32).reshape(4, 24, 3)
    root = np.arange(4 * 9, dtype=np.float32).reshape(4, 3, 3)
    wrists = np.arange(4 * 6, dtype=np.float32).reshape(4, 6) + 1000
    value = smpl_input(joints, root, wrists).reshape(4, 84)
    np.testing.assert_array_equal(value[:, :72], joints.reshape(4, 72))
    np.testing.assert_array_equal(value[:, 72:78], root[:, :, :2].reshape(4, 6))
    np.testing.assert_array_equal(value[:, 78:], wrists)


@pytest.mark.parametrize("seed", range(8))
def test_relative_composition_matches_actual_measured_pelvis(seed):
    r = Rotation.random(6, random_state=seed)
    q = r.as_quat()[:, [3, 0, 1, 2]].astype(np.float32)
    encoded = _relative_orientation_6d(q[0], q[1])
    value = smpl_relative_from_teleop(encoded, q[1], q[2:])
    expected = np.stack([_relative_orientation_6d(q[0], x) for x in q[2:]])
    np.testing.assert_allclose(value[:, :, :2].reshape(4, 6), expected, atol=3e-7, rtol=0)


def test_reject_wrong_frame_count():
    with pytest.raises(ValueError, match="four frames"):
        smpl_input(np.zeros((3, 24, 3)), np.zeros((4, 3, 3)), np.zeros((4, 6)))


def test_reject_nonfinite_input():
    joints = np.zeros((4, 24, 3))
    joints[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        smpl_input(joints, np.zeros((4, 3, 3)), np.zeros((4, 6)))
