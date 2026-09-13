"""Received-only bridge, attempt retention, and unchanged-loop restoration."""

from types import SimpleNamespace

import numpy as np
import pytest

from gear_sonic.scripts import record_g1_sonic_public29_full_pose as recorder
from gear_sonic.utils.g1_23dof_contract import HARDWARE_23_JOINT_NAMES, SOURCE_MJ29_KEEP_INDICES
from gear_sonic.utils.g1_sonic_full_pose_reference import full_pose_encoder640


def source():
    poses = np.zeros((31, 36))
    poses[:, 2:4] = [0.78, 1]
    poses[:, 7:] = np.arange(31)[:, None] * np.arange(29)[None] * 0.001
    poses[-10:] = poses[-1]
    return poses


def emit(bridge, index):
    pose = bridge.poses[index]
    return bridge.push(
        source_timestamp_s=index * 0.02,
        arrival_timestamp_s=index * 0.02,
        joint_names=HARDWARE_23_JOINT_NAMES,
        joint_position23=pose[7:][list(SOURCE_MJ29_KEEP_INDICES)],
        root_position_w=pose[:3],
        root_quaternion_wxyz=pose[3:7],
        virtual_source_vr21=np.r_[np.zeros(9), np.tile([1, 0, 0, 0], 3)].astype(np.float32),
    )


def test_only_received_eleven_samples_consumed():
    poses = source()
    teacher = SimpleNamespace(infer=lambda e, h: (np.zeros(29, np.float32), np.zeros(64, np.float32)))
    bridge = recorder.FullPoseBridge(teacher, poses)
    assert bridge.factory() is bridge
    for index in range(19):
        emit(bridge, index)
    # The following frames have not arrived and must not affect the first input.
    bridge.poses[20:, 7:] += 100
    window = emit(bridge, 19)
    measured = np.array([1, 0, 0, 0], np.float32)
    diagnostic = window.encoder267(measured)
    np.testing.assert_array_equal(
        bridge.pending["encoder640"], full_pose_encoder640(poses[9:20].astype(np.float32), measured)
    )
    bridge.infer(diagnostic, np.zeros(930, np.float32))
    arrays = bridge.recorded_arrays(1)
    assert arrays["attempt_anchor_sample_index"].tolist() == [9]
    assert arrays["attempt_output_present"].tolist() == [True]
    assert arrays["encoder640"].shape == (1, 640)
    with pytest.raises(RuntimeError, match="fresh"):
        bridge.infer(diagnostic, np.zeros(930, np.float32))
    with pytest.raises(RuntimeError, match="one fresh"):
        bridge.factory()


def test_failed_inference_input_retained_without_fabricated_output():
    def fail(encoder, history):
        raise ValueError("test output failure")

    bridge = recorder.FullPoseBridge(SimpleNamespace(infer=fail), source())
    for index in range(20):
        window = emit(bridge, index)
    diagnostic = window.encoder267(np.array([1, 0, 0, 0], np.float32))
    with pytest.raises(ValueError, match="test output"):
        bridge.infer(diagnostic, np.zeros(930, np.float32))
    arrays = bridge.recorded_arrays(0)
    assert arrays["encoder640"].shape == (0, 640)
    assert arrays["attempt_encoder640"].shape == (1, 640)
    assert arrays["attempt_output_present"].tolist() == [False]
    assert bridge.attempts[0]["failure"]["message"] == "test output failure"


def test_baseline_buffer_restored_on_failure(monkeypatch):
    before = recorder.baseline.ReceivedSourceHorizon

    def fail(*args):
        assert recorder.baseline.ReceivedSourceHorizon is not before
        raise RuntimeError("sentinel")

    monkeypatch.setattr(recorder.baseline, "run_case", fail)
    with pytest.raises(RuntimeError, match="sentinel"):
        recorder.matched_case(None, None, None, None, source(), [])
    assert recorder.baseline.ReceivedSourceHorizon is before
