"""Packet fault handling and frozen-feature parity without policy or physics."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from gear_sonic.utils.g1_true23_mpc_student import GoalFeatures
from gear_sonic.utils.g1_true23_student_packets import (
    DT,
    HORIZON,
    NATIVE_SHAPES,
    StudentPacket,
    StudentPacketGate,
)


@pytest.fixture
def samples():
    count = 80
    time = np.arange(count) * DT
    native = {name: np.zeros((count, *shape)) for name, shape in NATIVE_SHAPES.items()}
    native['joint_pos'][:] = time[:, None] * np.linspace(-.2, .2, 23)
    native['joint_vel'][:] = np.linspace(-.2, .2, 23)
    native['body_pos_w'][:] = time[:, None, None] * np.array([.3, -.2, .1])
    quats = Rotation.from_euler('z', .4 * time).as_quat()[:, [3, 0, 1, 2]]
    native['body_quat_w'][:] = quats[:, None]
    native['body_lin_vel_w'][:] = [.3, -.2, .1]
    native['body_ang_vel_w'][:] = [0, 0, .4]
    original_native = {key: value.copy() for key, value in native.items()}
    original_native['joint_pos'] += .07
    tasks = dict(source_task_position_w=np.stack([np.c_[time**2, .2 * time, .5 + .1 * time]] * 3, axis=1),
        source_task_quaternion_wxyz=np.stack([quats] * 3, axis=1))
    packets = [StudentPacket(0, i, i * DT,
        {key: value[i] for key, value in original_native.items()},
        {key: value[i] for key, value in native.items()},
        tasks['source_task_position_w'][i], tasks['source_task_quaternion_wxyz'][i], i == count - 1)
        for i in range(count)]
    contract = dict(default_q=np.zeros(23), body_names=['left_ankle_roll_link', 'right_ankle_roll_link'])
    return packets, native, original_native, tasks, contract


def test_all_received_windows_equal_offline_features_including_previous_derivative_and_eof(samples):
    packets, native, original_native, tasks, contract = samples
    offline = GoalFeatures(native, tasks, contract)
    gate = StudentPacketGate()
    qpos = np.r_[.2, -.1, .8, 1., 0., 0., 0., np.zeros(23)]
    qvel = np.linspace(-.1, .1, 29)
    previous_target = np.linspace(-.2, .2, 23)
    windows = []
    for packet in packets:
        assert gate.receive(packet, packet.source_time)
        window = gate.consume(packet.source_time)
        if window is not None:
            windows.append(window)
    final_time = packets[-1].source_time
    while (window := gate.consume(final_time)) is not None:
        windows.append(window)
    assert len(windows) == len(packets) == gate.consumed
    assert gate.final_consumed and gate.fault is None
    assert windows[0].playback_latency == pytest.approx(.74)
    assert len(windows[-1].samples) == 1 and windows[-1].samples[0].packet.final
    for i, window in enumerate(windows):
        bfm, retargeted, original = window.arrays()
        offset = window.feature_frame
        np.testing.assert_array_equal(bfm['joint_pos'][offset], original_native['joint_pos'][i])
        np.testing.assert_array_equal(retargeted['joint_pos'][offset], native['joint_pos'][i])
        actual = GoalFeatures(retargeted, original, contract)(qpos, qvel, previous_target, offset)
        np.testing.assert_array_equal(actual, offline(qpos, qvel, previous_target, i))
        if i:
            assert window.previous.packet.sequence == i - 1


def test_accepted_packets_do_not_alias_sender_and_cannot_regain_write_access(samples):
    packet = samples[0][0]
    gate = StudentPacketGate()
    assert gate.receive(replace(packet, final=True), 0)
    packet.original_native['joint_pos'][0] = 99
    packet.retargeted_native['joint_pos'][0] = 88
    packet.task_position_w[0, 0] = 77
    saved = gate.consume(0).samples[0].packet
    assert saved.original_native['joint_pos'][0] == .07
    assert saved.retargeted_native['joint_pos'][0] == 0
    assert saved.task_position_w[0, 0] == 0
    for array in (saved.task_position_w, saved.task_quaternion_wxyz,
                  *saved.original_native.values(), *saved.retargeted_native.values()):
        with pytest.raises(ValueError):
            array.setflags(write=True)
    with pytest.raises(TypeError):
        saved.original_native['joint_pos'] = np.zeros(23)


@pytest.mark.parametrize('kind', ['gap', 'future', 'stale', 'bad_time', 'bad_shape',
    'nonfinite', 'native_quaternion', 'task_quaternion', 'epoch', 'bool_sequence'])
def test_invalid_packet_latches_and_later_valid_packets_are_ignored(samples, kind):
    packet = samples[0][0]
    now = 0.
    if kind == 'gap':
        packet = replace(packet, sequence=1, source_time=DT)
        now = DT
    elif kind == 'future':
        now = -.001
    elif kind == 'stale':
        now = .101
    elif kind == 'bad_time':
        packet = replace(packet, source_time=float('nan'))
    elif kind == 'bad_shape':
        packet = replace(packet, task_position_w=np.zeros((2, 3)))
    elif kind == 'nonfinite':
        task = packet.task_position_w.copy()
        task[0, 0] = float('inf')
        packet = replace(packet, task_position_w=task)
    elif kind == 'native_quaternion':
        fields = dict(packet.retargeted_native)
        fields['body_quat_w'] = np.zeros((24, 4))
        packet = replace(packet, retargeted_native=fields)
    elif kind == 'task_quaternion':
        packet = replace(packet, task_quaternion_wxyz=np.zeros((3, 4)))
    elif kind == 'epoch':
        packet = replace(packet, epoch=1)
    elif kind == 'bool_sequence':
        packet = replace(packet, sequence=False)
    gate = StudentPacketGate()
    assert not gate.receive(packet, now)
    reason = gate.fault['reason']
    assert not gate.receive(samples[0][0], max(0, now))
    assert gate.ignored == 1 and gate.fault['reason'] == reason
    assert gate.consume(max(0, now)) is None


def test_short_buffer_cannot_repeat_unreceived_samples_and_timeout_latches(samples):
    gate = StudentPacketGate()
    for packet in samples[0][:HORIZON - 1]:
        assert gate.receive(packet, packet.source_time)
        assert gate.consume(packet.source_time) is None
    assert gate.consumed == 0
    assert gate.consume(samples[0][HORIZON - 2].source_time + .101) is None
    assert gate.fault['reason'] == 'packet_timeout'


def test_gap_duplicate_future_clock_and_post_final_are_distinct_faults(samples):
    first, second = samples[0][:2]
    gate = StudentPacketGate()
    assert gate.receive(first, 0)
    assert not gate.receive(first, 0)
    assert gate.fault['reason'] == 'packet_sequence_gap_or_reorder'
    gate = StudentPacketGate()
    assert gate.receive(first, 0)
    assert not gate.receive(second, .01)
    assert gate.fault['reason'] == 'packet_from_future'
    gate = StudentPacketGate()
    assert gate.receive(replace(first, final=True), 0)
    assert not gate.receive(second, DT)
    assert gate.fault['reason'] == 'packet_after_final'


def test_buffer_overflow_is_bounded_and_latched(samples):
    gate = StudentPacketGate(max_buffer=HORIZON)
    for packet in samples[0][:HORIZON]:
        assert gate.receive(packet, packet.source_time)
    packet = samples[0][HORIZON]
    assert not gate.receive(packet, packet.source_time)
    assert gate.fault['reason'] == 'packet_buffer_overflow'
    assert gate.consume(packet.source_time) is None


def test_explicit_rearm_retains_audit_and_rejects_future_epoch(samples):
    gate = StudentPacketGate()
    assert gate.receive(samples[0][0], 0)
    with pytest.raises(ValueError):
        gate.rearm(.1)
    assert gate.consume(.101) is None
    gate.rearm(1.)
    assert gate.epoch == 1 and gate.fault is None and gate.consumed == 0
    assert not gate.receive(samples[0][0], 1.)
    assert gate.ignored == 1 and gate.fault is None
    assert gate.receive(replace(samples[0][0], epoch=1, final=True), 1.)
    window = gate.consume(1.)
    assert window.previous is None and window.feature_frame == 0
    assert [event['kind'] for event in gate.events] == ['latched_input_fault', 'explicit_input_rearm']


def test_consumption_clock_cannot_regress(samples):
    gate = StudentPacketGate()
    assert gate.receive(replace(samples[0][0], final=True), .01)
    assert gate.consume(0.) is None
    assert gate.fault['reason'] == 'receive_clock_invalid_or_regressed'


@pytest.mark.parametrize('bad_clock', [np.array(0.), np.array([0.]), False, np.bool_(False)])
def test_mutable_or_boolean_clock_metadata_is_rejected(samples, bad_clock):
    gate = StudentPacketGate()
    assert not gate.receive(replace(samples[0][0], source_time=bad_clock), 0.)
    assert gate.fault['reason'] == 'packet_source_clock_invalid'
    with pytest.raises(ValueError):
        StudentPacketGate(now=bad_clock)
    with pytest.raises(ValueError):
        StudentPacketGate(stale_seconds=bad_clock)


def test_immutable_numpy_source_clock_is_canonicalized(samples):
    gate = StudentPacketGate()
    assert gate.receive(replace(samples[0][0], source_time=np.float64(0.), final=True), np.float64(0.))
    assert type(gate.consume(0).samples[0].packet.source_time) is float


@pytest.mark.parametrize('source', [
    'original_native', 'retargeted_native', 'task_position_w', 'task_quaternion_wxyz',
])
def test_noncanonical_float32_input_is_rejected_instead_of_changing_feature_math(samples, source):
    packet = samples[0][0]
    value = getattr(packet, source)
    changed = ({key: array.astype(np.float32) for key, array in value.items()}
               if isinstance(value, dict) else value.astype(np.float32))
    gate = StudentPacketGate()
    assert not gate.receive(replace(packet, **{source: changed}), 0.)
    assert gate.fault['reason'].startswith('packet_dtype_requires_float64:')


@pytest.mark.parametrize('operation', ['receive', 'consume'])
def test_oversized_host_clock_latches_instead_of_throwing(samples, operation):
    gate = StudentPacketGate()
    if operation == 'receive':
        assert not gate.receive(samples[0][0], 10**1000)
    else:
        assert gate.consume(10**1000) is None
    assert gate.fault['reason'] == 'receive_clock_invalid_or_regressed'
