"""Protocol/timing tests. Synthetic model tests do not qualify learned balance."""

import math
import os
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET
import time

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import torch

from gear_sonic.scripts.evaluate_g1_true23_bfmzero_stream import delivery_schedule
from gear_sonic.scripts.evaluate_g1_true23_bfmzero import corrected_goal
from gear_sonic.utils.g1_true23_bfmzero_inference import reference_features
from gear_sonic.utils.g1_true23_bfmzero_stream_clock import SimulationPacer
from gear_sonic.utils.g1_23dof_contract import HARDWARE_23_JOINT_NAMES
from gear_sonic.utils.g1_true23_bfmzero_stream import (
    BFMStreamSimulator,
    DT,
    HORIZON,
    Packet,
    PacketGate,
    ReferenceAnchor,
    ReceivedSample,
    SOURCE_BUFFER_SECONDS,
    StandingReference,
    StreamMode,
    rotation_wxyz,
    received_goal,
    stack_samples,
)


@pytest.fixture
def contract():
    root = Path(__file__).resolve().parents[1]
    xml = ET.parse(root / "data/robots/g1/g1_23dof_rev_1_0.xml")
    body_names = tuple(node.attrib["name"] for node in xml.findall(".//worldbody//body"))
    return {
        "default_q": np.zeros(23),
        "kp": np.ones(23),
        "kd": np.ones(23),
        "training_effort": np.ones(23),
        "body_names": body_names,
    }


@pytest.fixture
def fields():
    positions = np.zeros((24, 3))
    positions[:, 2] = 0.8
    return {
        "joint_pos": np.zeros(23),
        "joint_vel": np.zeros(23),
        "body_pos_w": positions,
        "body_quat_w": np.tile([1.0, 0.0, 0.0, 0.0], (24, 1)),
        "body_lin_vel_w": np.zeros((24, 3)),
        "body_ang_vel_w": np.zeros((24, 3)),
    }


def packet(sequence, fields, *, epoch=0, final=False):
    return Packet(epoch, sequence, sequence * DT, {key: value.copy() for key, value in fields.items()}, final)


def test_received_window_requires_140ms_history_and_never_contains_unsent_future(contract, fields):
    gate = PacketGate(contract)
    for sequence in range(HORIZON):
        assert gate.receive(packet(sequence, fields), sequence * DT)
        assert gate.ready() == (sequence == 7)
    window = gate.window()
    assert [sample.packet.sequence for sample in window] == list(range(8))
    assert window[-1].received_at - window[0].received_at == pytest.approx(0.14)
    assert SOURCE_BUFFER_SECONDS == pytest.approx(0.14)
    assert gate.consume().packet.sequence == 0
    assert gate.consumed == 1 and len(gate.queue) == 7


@pytest.mark.parametrize(
    "failure",
    [
        "sequence",
        "timestamp",
        "nan_time",
        "nonfinite",
        "quaternion",
        "shape",
        "future",
        "complex",
        "array_epoch",
        "wrong_type",
    ],
)
def test_bad_packet_latches_and_quarantines_all_source_goals(contract, fields, failure):
    gate = PacketGate(contract)
    assert gate.receive(packet(0, fields), 0.0)
    damaged = packet(1, fields)
    now = DT
    if failure == "sequence":
        damaged = packet(2, fields)
    elif failure == "timestamp":
        damaged = Packet(0, 1, 0.021, fields)
    elif failure == "nan_time":
        damaged = Packet(0, 1, float("nan"), fields)
    elif failure == "nonfinite":
        damaged.fields["joint_pos"][7] = np.nan
    elif failure == "quaternion":
        damaged.fields["body_quat_w"][0] = 0
    elif failure == "shape":
        damaged.fields["joint_vel"] = np.zeros(29)
    elif failure == "future":
        now = 0.0
    elif failure == "complex":
        damaged.fields["joint_pos"] = damaged.fields["joint_pos"].astype(complex) + 1j
    elif failure == "array_epoch":
        damaged = Packet(np.array([0, 1]), 1, DT, fields)
    elif failure == "wrong_type":
        damaged = {"epoch": 0, "sequence": 1}
    assert not gate.receive(damaged, now)
    assert gate.fault is not None and gate.rejected == 1 and not gate.queue
    assert not gate.receive(packet(1, fields), DT)
    assert gate.ignored == 1 and not gate.ready()


def test_packet_buffer_owns_validated_values(contract, fields):
    gate = PacketGate(contract)
    incoming = packet(0, fields)
    assert gate.receive(incoming, 0)
    incoming.fields["joint_pos"][:] = np.nan
    assert np.isfinite(gate.window()[0].packet.fields["joint_pos"]).all()


def test_timeout_requires_explicit_new_epoch_before_packets_can_resume(contract, fields):
    gate = PacketGate(contract)
    gate.receive(packet(0, fields), 0)
    assert not gate.check_freshness(0.1)
    assert gate.check_freshness(0.12)
    assert gate.fault["reason"] == "packet_timeout"
    assert not gate.receive(packet(1, fields), 0.12)
    current = np.r_[[2, -1, 0.8], [1, 0, 0, 0], np.zeros(23)]
    gate.rearm(0.14, current)
    assert not gate.receive(packet(0, fields, epoch=0), 0.14)
    assert gate.receive(packet(0, fields, epoch=1), 0.14)
    assert gate.epochs[0]["full_source_consumed"] is False
    assert gate.anchor is not None
    np.testing.assert_allclose(gate.window()[0].packet.fields["body_pos_w"][0], [2, -1, 0.8])


def test_received_final_can_drain_without_inventing_source_frames(contract, fields):
    gate = PacketGate(contract)
    for sequence in range(3):
        gate.receive(packet(sequence, fields, final=sequence == 2), sequence * DT)
    assert gate.ready() and len(gate.window()) == 3
    assert not gate.check_freshness(10.0)
    assert [gate.consume().packet.sequence for _ in range(3)] == [0, 1, 2]
    assert gate.final_consumed and not gate.queue
    assert gate.epoch_report()["full_source_consumed"]


def test_rearm_calibration_is_rigid_keeps_height_and_preserves_joint_motion(fields):
    fields["body_pos_w"][1] = [1, 0, 1.0]
    fields["body_lin_vel_w"][1] = [1, 0, 0]
    current = np.r_[[2, 3, 0.9], rotation_wxyz(Rotation.from_euler("z", np.pi / 2)), np.zeros(23)]
    transformed = ReferenceAnchor(fields, current).apply(fields)
    np.testing.assert_allclose(transformed["body_pos_w"][0], [2, 3, 0.8])
    np.testing.assert_allclose(transformed["body_pos_w"][1], [2, 4, 1.0])
    np.testing.assert_allclose(transformed["body_lin_vel_w"][1], [0, 1, 0], atol=1e-15)
    np.testing.assert_array_equal(transformed["joint_pos"], fields["joint_pos"])
    np.testing.assert_array_equal(fields["body_pos_w"][1], [1, 0, 1])


@pytest.fixture
def synthetic_simulator(contract):
    children = "".join(
        f'<body name="{body}"><inertial pos="0 0 0" mass=".1" diaginertia=".1 .1 .1"/>'
        f'<joint name="{joint}" axis="0 0 1" range="-2 2"/></body>'
        for body, joint in zip(contract["body_names"][1:], HARDWARE_23_JOINT_NAMES, strict=True)
    )
    motors = "".join(f'<motor joint="{joint}"/>' for joint in HARDWARE_23_JOINT_NAMES)
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><compiler angle="radian"/><option gravity="0 0 0" timestep=".002"/>'
        '<worldbody><body name="pelvis"><freejoint/>'
        '<inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>'
        f"{children}</body></worldbody><actuator>{motors}</actuator></mujoco>"
    )

    class Policy:
        def backward(self, state, privileged):
            result = torch.zeros((len(state), 256))
            result[:, 0] = 16
            return result

        def actor(self, state, action, history, goal):
            return torch.zeros((1, 23))

    qpos = np.r_[[0, 0, 0.8], [1, 0, 0, 0], np.zeros(23)]
    return BFMStreamSimulator(
        model,
        SimpleNamespace(decimation=10, effort=np.ones(23)),
        contract,
        Policy(),
        qpos,
        np.zeros(29),
        np.zeros(23),
        0.8,
        np.ones(23) * 100,
        position_gain=0,
        yaw_gain=0,
        stop_seconds=0.2,
    )


def test_control_and_physics_clocks_continue_during_packet_loss(synthetic_simulator, fields):
    simulator = synthetic_simulator
    for control in range(40):
        if control < 8:
            simulator.receive(packet(control, fields), control * DT)
        simulator.step(control * DT)
    trace = simulator.arrays()
    assert simulator.controls == 40 and len(trace["physics_torque"]) == 400
    assert simulator.data.time == pytest.approx(0.8)
    assert trace["source_sequence"][7] == 0 and trace["source_buffer_age"][7] == pytest.approx(0.14)
    assert trace["source_latest_timestamp_used"][7] == pytest.approx(0.14)
    assert simulator.gate.fault["reason"] == "packet_timeout"
    assert simulator.mode == StreamMode.STOPPING
    # Gravity-free test apparatus has no foot loads, so it must never claim standing qualification.
    assert not simulator.report()["standing_return_verified"]
    assert not simulator.rearm(0.8)
    assert not simulator.receive(packet(8, fields), 0.8)
    assert len(trace["qpos"]) == 41


def test_packet_failure_never_rewrites_measured_simulator_state(synthetic_simulator, fields):
    simulator = synthetic_simulator
    before_qpos, before_qvel = simulator.data.qpos.copy(), simulator.data.qvel.copy()
    broken = packet(0, fields)
    broken.fields["body_pos_w"][0, 0] = np.inf
    assert not simulator.receive(broken, 0)
    np.testing.assert_array_equal(simulator.data.qpos, before_qpos)
    np.testing.assert_array_equal(simulator.data.qvel, before_qvel)
    simulator.step(0)
    assert simulator.controls == 1 and len(simulator.trace["physics_torque"]) == 10
    assert simulator.report()["simulator_pose_writes_after_initialization"] == 0


def test_stop_reference_has_zero_endpoint_velocity_and_measured_xy_yaw(synthetic_simulator, contract):
    simulator = synthetic_simulator
    qpos = simulator.data.qpos.copy()
    qpos[:3] = [3, -2, 0.9]
    qpos[3:7] = rotation_wxyz(Rotation.from_euler("xyz", [0.1, -0.2, 0.7]))
    qpos[7:] = np.linspace(-0.4, 0.4, 23)
    reference = StandingReference(simulator.model, contract, qpos, np.zeros(23), 0.8, duration=2)
    start, initial_velocity = reference.pose(0)
    stop, final_velocity = reference.pose(100)
    np.testing.assert_allclose(start, qpos, atol=1e-15)
    np.testing.assert_array_equal(initial_velocity, np.zeros(29))
    np.testing.assert_array_equal(final_velocity, np.zeros(29))
    np.testing.assert_allclose(stop[:3], [3, -2, 0.8])
    np.testing.assert_allclose(stop[3:7], rotation_wxyz(Rotation.from_euler("z", 0.7)), atol=1e-15)
    np.testing.assert_array_equal(stop[7:], np.zeros(23))
    half, half_velocity = reference.pose(50)
    assert np.all(half[7:] <= np.maximum(qpos[7:], 0)) and np.all(half[7:] >= np.minimum(qpos[7:], 0))
    assert np.linalg.norm(half_velocity) > 0
    for index in range(1, 120):
        reference.window(index)
        assert len(reference.cache) <= HORIZON + 1
    np.testing.assert_array_equal(simulator.data.qpos, np.r_[[0, 0, 0.8], [1, 0, 0, 0], np.zeros(23)])


def test_fault_delivery_schedule_preserves_original_source_time_and_explicit_rearm():
    schedule = delivery_schedule(100, "reorder", 0.2, 0.5, 6, 8)
    arrivals = [(row[0], row[3]) for row in schedule if row[1] == 1]
    assert (0.2, 11) in arrivals and (0.22, 10) in arrivals
    resumed = delivery_schedule(100, "resume", 0.2, 0.5, 6, 0.4)
    rearm = [row for row in resumed if row[1] == 0]
    assert rearm == [(6, 0, 1, -1, False, False)]
    restarted = [row for row in resumed if row[2] == 1 and row[1] == 1]
    assert len(restarted) == math.ceil(0.4 / DT)
    assert restarted[0][0] == 6 and restarted[0][3] == 0
    assert not any(row[-1] for row in restarted)  # A dropped suffix cannot masquerade as complete EOF.


@pytest.mark.parametrize("gains", [(0, 0), (1, 2), (0, 2), (1, 0)])
@pytest.mark.parametrize("horizon", [1, 5, 8])
def test_cached_received_goal_preserves_exact_referee_backward_inputs(contract, fields, gains, horizon):
    # A varied rigid pose/velocity witness catches changes in frame transforms,
    # clipping, casting, and accumulation order before the nonlinear network.
    rng = np.random.default_rng(613)
    samples = []
    for sequence in range(horizon):
        values = {name: rng.normal(size=value.shape) for name, value in fields.items()}
        values["body_quat_w"] = Rotation.random(24, random_state=rng).as_quat()[:, [3, 0, 1, 2]]
        state, privileged = reference_features({name: value[None] for name, value in values.items()}, contract)
        samples.append(
            ReceivedSample(Packet(0, sequence, sequence * DT, values), sequence * DT, state[0], privileged[0])
        )
    motion, states, privileged = stack_samples(samples)
    qpos = np.r_[rng.normal(size=3), rotation_wxyz(Rotation.random(random_state=rng)), rng.normal(size=23)]

    class Capture:
        def backward(self, state, privileged):
            self.state, self.privileged = state.clone(), privileged.clone()
            return torch.cat((state, privileged[:, :204]), dim=-1)

    referee, cached = Capture(), Capture()
    expected = corrected_goal(referee, states, privileged, motion, 0, qpos, horizon, *gains)
    actual, _, _ = received_goal(cached, samples, qpos, *gains)
    torch.testing.assert_close(cached.state, referee.state, atol=0, rtol=0)
    torch.testing.assert_close(cached.privileged, referee.privileged, atol=0, rtol=0)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    np.testing.assert_array_equal(np.stack([sample.state for sample in samples]), states)
    np.testing.assert_array_equal(np.stack([sample.privileged for sample in samples]), privileged)


@pytest.mark.skipif(os.name != "nt", reason="Windows native timer integration")
def test_high_resolution_timer_waits_then_closes_handle_on_exception():
    pacer = SimulationPacer("windows-high-resolution")
    deadline = time.perf_counter() + 0.003
    with pytest.raises(ValueError, match="witness"):
        with pacer:
            assert pacer.handle is not None
            pacer.sleep_until(deadline)
            assert time.perf_counter() >= deadline
            raise ValueError("witness")
    assert pacer.closed and pacer.handle is None
    with pytest.raises(RuntimeError, match="closed"):
        pacer.sleep_until(time.perf_counter())


def test_pacer_rejects_invalid_deadlines_without_entering_wait():
    with SimulationPacer() as pacer:
        with pytest.raises(ValueError, match="nonfinite"):
            pacer.sleep_until(float("nan"))
        with pytest.raises(ValueError, match="short"):
            pacer.sleep_until(time.perf_counter() + 10)
