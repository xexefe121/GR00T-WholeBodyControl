import io
import pickle

import numpy as np
import pytest

from gear_sonic.scripts.prepare_g1_true23_twist2_replay import (
    NumpyOnlyUnpickler,
    load_pinned_recording,
    native23_poses,
)
from gear_sonic.utils.g1_23dof_contract import SOURCE_MJ29_KEEP_INDICES


def test_resampling_preserves_time_all_retained_joints_and_quaternion_order():
    count, fps = 41, 25.0
    times = np.arange(count) / fps
    joints = times[:, None] + np.arange(29)[None, :] / 100
    recording = dict(
        fps=fps,
        root_pos=np.tile([1, 2, 3], (count, 1)),
        root_rot=np.tile([0, 0, 0, 1], (count, 1)),
        dof_pos=joints,
    )
    poses, source_times, target_times = native23_poses(recording)
    assert len(poses) == 81
    assert source_times[-1] == target_times[-1] == 1.6
    np.testing.assert_allclose(
        poses[:, 7:], target_times[:, None] + np.asarray(SOURCE_MJ29_KEEP_INDICES)[None, :] / 100
    )
    np.testing.assert_array_equal(poses[:, 3:7], np.tile([1, 0, 0, 0], (81, 1)))
    np.testing.assert_array_equal(recording["dof_pos"], joints)


def test_quaternion_sign_change_uses_same_rotation():
    recording = dict(
        fps=25.0,
        root_pos=np.zeros((12, 3)),
        root_rot=np.tile([0.0, 0.0, 0.0, 1.0], (12, 1)),
        dof_pos=np.zeros((12, 29)),
    )
    recording["root_rot"][6:] *= -1
    poses, _, _ = native23_poses(recording)
    np.testing.assert_allclose(np.abs(poses[:, 3]), 1)
    np.testing.assert_array_equal(poses[:, 4:7], 0)


def test_pickle_global_is_rejected_without_execution():
    with pytest.raises(pickle.UnpicklingError, match="unsupported pickle global"):
        NumpyOnlyUnpickler(io.BytesIO(b"cos\nsystem\n.")).load()


def test_altered_download_rejected_before_unpickling(tmp_path):
    path = tmp_path / "0807_yanjie_walk_002.pkl"
    path.write_bytes(b"cos\nsystem\n.")
    with pytest.raises(ValueError, match="pinned Git blob"):
        load_pinned_recording(path)
