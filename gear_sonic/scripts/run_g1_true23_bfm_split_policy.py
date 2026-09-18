"""Fix 5 policy side of the split native23 BFM timing qualification.

The native executable owns lowcmd at 500 Hz.  This process only receives
recorded LowState-shaped samples, batches ordered IMU updates at 50 Hz, then
runs the unchanged BFM, PacketGate, and bounded final-target brake.  It has no
DDS imports and cannot publish to a robot topic or network interface.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import struct
import sys
import time

import mujoco
import numpy as np
import torch
import zmq

from gear_sonic.scripts.evaluate_g1_true23_bfmzero import MODEL, PACKAGE, PHYSICS, ROOT, load_motion
from gear_sonic.scripts.run_g1_true23_bfm_teleop_sim import WindowsRealtimeScope, decode_packet, packet_fields, Packet
from gear_sonic.scripts.time_g1_true23_bfm_hardware_loop import BoundedAnkleBrake, summary_ms
from gear_sonic.utils.g1_true23_bfm_imu_odometry import Native23IMUOdometry
from gear_sonic.utils.g1_true23_bfmzero_inference import BFMHistory, BFMZeroInference, load_contract, state_and_terms
from gear_sonic.utils.g1_true23_bfmzero_stream import (
    DT, JOINT_LIMIT_BRAKE_STEP_RAD, GateAdmissionAudit, PacketGate,
    TeleopClockAlignment, received_goal,
)
from gear_sonic.utils.g1_true23_bfmzero_stream_clock import SimulationPacer
from gear_sonic.utils.g1_true23_step1b_mujoco import prepare_true23_model


LOWSTATE_DT = 0.002
CONTROL_TICKS = 10
REPLAY_MAGIC = 0x344D4642
STATE_MAGIC = 0x34535442
TARGET_MAGIC = 0x34544742
REPLAY_HEADER = struct.Struct("<IIQd")
REPLAY_SAMPLE = struct.Struct("<d23f23f4f3f3f")
STATE_WIRE = struct.Struct("<IIQQQd23f23f4f3f3f")
TARGET_WIRE = struct.Struct("<IIQQQ69f")
INITIAL_COMMAND = struct.Struct("<II69f")


def configure_policy_runtime(priority: str):
    """Apply the already-reviewed Windows policy measures, outside timing."""
    gc_enabled = gc.isenabled()
    gc.collect()
    gc.freeze()
    gc.disable()
    return gc_enabled


def restore_gc(enabled: bool):
    if enabled:
        gc.enable()
    gc.unfreeze()
    gc.collect()


def sample_from_values(values):
    return {
        "timestamp_s": float(values[0]),
        "joint_q": np.asarray(values[1:24], dtype=np.float32),
        "joint_dq": np.asarray(values[24:47], dtype=np.float32),
        "imu_quat_wxyz": np.asarray(values[47:51], dtype=np.float32),
        "gyro_body": np.asarray(values[51:54], dtype=np.float32),
        "accel_specific_force_body": np.asarray(values[54:57], dtype=np.float32),
    }


def export_replay(args):
    with np.load(args.sensor_stream, allow_pickle=False) as source:
        required = ("timestamp_s", "joint_q", "joint_dq", "imu_quat_wxyz", "gyro_body", "accel_specific_force_body")
        if set(source.files) != set(required):
            raise ValueError("sensor stream fields do not match the Fix 3 recorded lowstate replay")
        arrays = {key: np.asarray(source[key]) for key in required}
    count = len(arrays["timestamp_s"])
    if count == 0 or any(len(value) != count for value in arrays.values()):
        raise ValueError("sensor stream has inconsistent lengths")
    if not np.allclose(np.diff(arrays["timestamp_s"]), LOWSTATE_DT, rtol=0.0, atol=1e-10):
        raise ValueError("sensor replay is not exact 500 Hz")
    args.replay.parent.mkdir(parents=True, exist_ok=True)
    with args.replay.open("xb") as output:
        output.write(REPLAY_HEADER.pack(REPLAY_MAGIC, 1, count, LOWSTATE_DT))
        for index in range(count):
            output.write(REPLAY_SAMPLE.pack(
                float(arrays["timestamp_s"][index]), *np.asarray(arrays["joint_q"][index], np.float32),
                *np.asarray(arrays["joint_dq"][index], np.float32), *np.asarray(arrays["imu_quat_wxyz"][index], np.float32),
                *np.asarray(arrays["gyro_body"][index], np.float32), *np.asarray(arrays["accel_specific_force_body"][index], np.float32)))
    contract = load_contract(PACKAGE / "bfmzero_inspect_v1/config.yaml")
    with args.initial_command.open("xb") as output:
        output.write(INITIAL_COMMAND.pack(TARGET_MAGIC, 1, *np.asarray(contract["default_q"], np.float32),
                                          *np.asarray(contract["kp"], np.float32), *np.asarray(contract["kd"], np.float32)))
    result = {"kind": "g1_true23_bfm_fix4_flat_replay_v1", "samples": count, "duration_seconds": float(arrays["timestamp_s"][-1] + LOWSTATE_DT),
              "replay": str(args.replay), "initial_command": str(args.initial_command), "sample_bytes": REPLAY_SAMPLE.size}
    args.manifest.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


def load_sensor(path: Path):
    with np.load(path, allow_pickle=False) as archive:
        required = ("timestamp_s", "joint_q", "joint_dq", "imu_quat_wxyz", "gyro_body", "accel_specific_force_body")
        if set(archive.files) != set(required):
            raise ValueError("sensor replay field mismatch")
        return {key: np.asarray(archive[key]).copy() for key in required}


def check_estimator_batching(args):
    sensor = load_sensor(args.sensor_stream)
    _, model, _ = prepare_true23_model(ROOT.parent / "GR00T-WholeBodyControl" / MODEL, ROOT / PHYSICS)
    reference, batched = Native23IMUOdometry(model), Native23IMUOdometry(model)
    maximum = 0.0
    fields = None
    ticks = 0
    pending = []
    for index in range(len(sensor["timestamp_s"])):
        sample = {key: sensor[key][index] for key in sensor}
        reference_result = reference.update(**sample)
        pending.append(sample)
        if index % CONTROL_TICKS != 0:
            continue
        batched_result = None
        for queued in pending:
            batched_result = batched.update(**queued)
        pending.clear()
        fields = sorted(reference_result)
        for field in fields:
            difference = float(np.max(np.abs(np.asarray(reference_result[field]) - np.asarray(batched_result[field]))))
            maximum = max(maximum, difference)
            if difference != 0.0:
                result = {"passed": False, "first_differing_control_tick": ticks, "field": field, "maximum_absolute_difference": maximum}
                args.output.write_text(json.dumps(result, indent=2))
                raise RuntimeError("C1 failed: batched estimator differs from per-sample estimator: " + json.dumps(result))
        ticks += 1
    result = {"kind": "g1_true23_bfm_fix4_estimator_batch_equivalence_v1", "passed": True, "control_ticks": ticks,
              "output_fields": fields, "maximum_absolute_difference": maximum}
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


def policy_step(policy, contract, estimator_result, sample, history, action, brake, gate):
    if not gate.ready():
        return action, None
    samples = gate.window()
    measured_qpos = np.r_[estimator_result["position_start"], sample["imu_quat_wxyz"], sample["joint_q"]]
    sensed, terms = state_and_terms(sample["joint_q"], sample["joint_dq"], sample["imu_quat_wxyz"], sample["gyro_body"], action, contract["default_q"])
    goal, _, _ = received_goal(policy, samples, measured_qpos, 1.0, 2.0)
    raw = policy.actor(
        torch.as_tensor(sensed[None], device=policy.device), torch.as_tensor(action[None], device=policy.device),
        torch.as_tensor(history.before_update(terms)[None], device=policy.device), goal.to(policy.device))[0].cpu().numpy()
    action = raw * 5.0
    requested = contract["default_q"] + action * 0.25 * contract["training_effort"] / contract["kp"]
    target = brake.apply(requested, sample["joint_q"], sample["joint_dq"])
    gate.consume()
    return action, target


def lockstep_targets(sensor, motion, model, contract, policy, batched):
    estimator = Native23IMUOdometry(model)
    brake = BoundedAnkleBrake(model, JOINT_LIMIT_BRAKE_STEP_RAD)
    gate = PacketGate(contract, stale_seconds=0.1)
    history, action = BFMHistory(), np.zeros(23, np.float32)
    target = np.asarray(contract["default_q"], np.float32).copy()
    pending = []
    emitted = []
    for index in range(len(sensor["timestamp_s"])):
        sample = {key: sensor[key][index] for key in sensor}
        if batched:
            pending.append(sample)
            estimator_result = None
        else:
            estimator_result = estimator.update(**sample)
        if index % CONTROL_TICKS:
            continue
        if batched:
            for queued in pending:
                estimator_result = estimator.update(**queued)
            pending.clear()
        sequence = index // CONTROL_TICKS
        if sequence < len(motion["joint_pos"]) - 11:
            packet = Packet(0, sequence, sequence * DT, packet_fields(motion, sequence + 11), False)
            gate.receive(packet, sequence * DT)
        action, updated = policy_step(policy, contract, estimator_result, sample, history, action, brake, gate)
        if updated is not None:
            target = updated
        emitted.append(target.copy())
    return np.asarray(emitted)


def check_lockstep(args):
    from gear_sonic.scripts import evaluate_g1_true23_bfmzero as evaluator
    evaluator.DATA = args.data_root
    sensor = load_sensor(args.sensor_stream)
    motion, _, _ = load_motion("pico")
    contract = load_contract(PACKAGE / "bfmzero_inspect_v1/config.yaml")
    _, model, _ = prepare_true23_model(ROOT.parent / "GR00T-WholeBodyControl" / MODEL, ROOT / PHYSICS)
    weights = PACKAGE / "bfmzero_inference_v1/inference.safetensors"
    torch.set_num_threads(args.torch_threads)
    single = lockstep_targets(sensor, motion, model, contract, BFMZeroInference(weights, "cpu"), batched=False)
    split = lockstep_targets(sensor, motion, model, contract, BFMZeroInference(weights, "cpu"), batched=True)
    differences = np.max(np.abs(single - split), axis=1)
    maximum = float(differences.max())
    first = int(np.flatnonzero(differences != 0.0)[0]) if np.any(differences != 0.0) else None
    result = {"kind": "g1_true23_bfm_fix4_lockstep_parity_v1", "passed": maximum == 0.0, "targets": int(len(single)),
              "maximum_absolute_difference": maximum, "first_differing_control_tick": first}
    args.output.write_text(json.dumps(result, indent=2))
    if not result["passed"]:
        raise RuntimeError("C2 failed: " + json.dumps(result))
    print(json.dumps(result, indent=2))


def policy_process(args):
    if args.output.exists():
        raise FileExistsError("Fix 5 policy evidence refuses overwrite")
    for name, endpoint in (("state", args.state_endpoint), ("target", args.target_endpoint)):
        if not endpoint.startswith("tcp://"):
            raise ValueError(f"Fix 5 {name} endpoint must be TCP, got {endpoint!r}")
    args.output.mkdir(parents=True)
    torch.set_num_threads(args.torch_threads)
    context = zmq.Context.instance()
    states = context.socket(zmq.PULL)
    targets = context.socket(zmq.PUSH)
    teleop = context.socket(zmq.SUB)
    for socket in (states, targets, teleop):
        socket.linger = 0
    states.setsockopt(zmq.RCVHWM, 65536)
    targets.setsockopt(zmq.SNDHWM, 65536)
    teleop.setsockopt(zmq.SUBSCRIBE, b"")
    states.connect(args.state_endpoint)
    targets.connect(args.target_endpoint)
    teleop.connect(args.teleop_endpoint)
    contract = load_contract(PACKAGE / "bfmzero_inspect_v1/config.yaml")
    policy = BFMZeroInference(PACKAGE / "bfmzero_inference_v1/inference.safetensors", "cpu")
    _, model, _ = prepare_true23_model(ROOT.parent / "GR00T-WholeBodyControl" / MODEL, ROOT / PHYSICS)
    estimator = Native23IMUOdometry(model)
    brake = BoundedAnkleBrake(model, args.brake_step_rad)
    gate = PacketGate(contract, stale_seconds=0.1)
    teleop_clock = TeleopClockAlignment()
    teleop_gate_audit = GateAdmissionAudit()
    history, action = BFMHistory(), np.zeros(23, np.float32)
    pending_states, pending_teleop = [], []
    expected_sequence = 0
    state_gaps = state_decode_errors = teleop_rejected = teleop_accepted = target_eagain = 0
    path = {"start_lateness_ms": [], "work_ms": [], "deadline_misses": 0, "policy_updates": 0}
    # The policy must have its ZMQ connects established before the timing source
    # begins.  This wait is deliberately outside measurement.
    if not states.poll(60_000):
        raise TimeoutError("no lowstate sample arrived from native loop")
    first_message = states.recv()
    gc_enabled = configure_policy_runtime(args.priority)
    try:
        with WindowsRealtimeScope("consumer", args.priority, "none"), SimulationPacer("windows-high-resolution" if os.name == "nt" else "sleep") as pacer:
            start = time.perf_counter()
            for control in range(round(args.duration_seconds / DT)):
                due = start + control * DT
                pacer.sleep_until(due)
                begun = time.perf_counter()
                path["start_lateness_ms"].append(max(0.0, (begun - due) * 1000.0))
                messages = [first_message] if control == 0 else []
                first_message = None
                while states.poll(0):
                    messages.append(states.recv())
                for message in messages:
                    try:
                        values = STATE_WIRE.unpack(message)
                        if values[0] != STATE_MAGIC or values[1] != 1:
                            raise ValueError("wire magic")
                        sequence = values[2]
                        if sequence != expected_sequence:
                            state_gaps += abs(sequence - expected_sequence)
                            expected_sequence = sequence
                        expected_sequence += 1
                        pending_states.append((sequence, sample_from_values(values[5:]), values[3], time.perf_counter_ns()))
                    except (ValueError, struct.error):
                        state_decode_errors += 1
                estimator_result = None
                latest_sequence = 0
                sample = None
                latest_state_sent_mono = latest_policy_receive_mono = None
                for latest_sequence, sample, latest_state_sent_mono, latest_policy_receive_mono in pending_states:
                    estimator_result = estimator.update(**sample)
                pending_states.clear()
                now = control * DT
                while teleop.poll(0):
                    try:
                        packet = decode_packet(teleop.recv_json())
                        pending_teleop.append((
                            packet, teleop_clock.observe(packet, now, gate), teleop_clock.gate_source_time_offset(gate)
                        ))
                    except (ValueError, TypeError, KeyError):
                        teleop_rejected += 1
                delivered = 0
                while (
                    delivered < len(pending_teleop)
                    and (pending_teleop[delivered][1] is None or pending_teleop[delivered][1] <= now + 1e-7)
                ):
                    packet, _, source_time_offset = pending_teleop[delivered]
                    teleop_accepted += int(teleop_gate_audit.receive(
                        gate, packet, now, source_time_offset=0.0 if source_time_offset is None else source_time_offset
                    ))
                    delivered += 1
                if delivered:
                    del pending_teleop[:delivered]
                if sample is not None and estimator_result is not None:
                    action, target = policy_step(policy, contract, estimator_result, sample, history, action, brake, gate)
                    if target is not None:
                        output_policy_mono = time.perf_counter_ns()
                        payload = TARGET_WIRE.pack(TARGET_MAGIC, 1, latest_sequence, latest_policy_receive_mono, output_policy_mono, *np.asarray(target, np.float32),
                                                   *np.asarray(contract["kp"], np.float32), *np.asarray(contract["kd"], np.float32))
                        if targets.send(payload, flags=zmq.DONTWAIT) is None:
                            pass
                        path["policy_updates"] += 1
                ended = time.perf_counter()
                path["work_ms"].append((ended - begun) * 1000.0)
                path["deadline_misses"] += int(ended > due + DT)
    except zmq.Again:
        target_eagain += 1
        raise RuntimeError("target IPC queue overflow; native loop must never be blocked")
    finally:
        restore_gc(gc_enabled)
        states.close(); targets.close(); teleop.close()
    for field in ("start_lateness_ms", "work_ms"):
        path[field] = summary_ms(path[field])
    report = {"kind": "g1_true23_bfm_fix5_policy_50hz_v1", "placement": args.placement, "duration_seconds": args.duration_seconds,
              "path_50hz": path, "state_sequence_gaps": state_gaps, "state_decode_errors": state_decode_errors,
              "teleop_packets_accepted": teleop_accepted, "teleop_packets_rejected": teleop_rejected,
              "teleop_packets_gate_rejected": teleop_gate_audit.rejected,
              "teleop_gate_rejection_reasons": teleop_gate_audit.reasons,
              "teleop_clock_alignment": {"method": "first_packet_reception_offset_per_gate_epoch",
                                         "anchors": teleop_clock.anchors},
              "target_send_eagain": target_eagain, "target_age_clock": "NTP four-timestamp mapping between loop and policy monotonic clocks; valid across machines",
              "brake_step_rad": args.brake_step_rad,
              "brake_engagement_controls": brake.engagements, "brake_target_step_max_rad": brake.maximum_step,
              "no_dds": True, "no_robot_network": True}
    report["passed"] = bool(path["deadline_misses"] == 0 and state_gaps == 0 and target_eagain == 0)
    (args.output / "policy_report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(required=True)
    export = sub.add_parser("export-replay")
    export.add_argument("--sensor-stream", type=Path, required=True)
    export.add_argument("--replay", type=Path, required=True)
    export.add_argument("--initial-command", type=Path, required=True)
    export.add_argument("--manifest", type=Path, required=True)
    export.set_defaults(function=export_replay)
    c1 = sub.add_parser("check-batching")
    c1.add_argument("--sensor-stream", type=Path, required=True)
    c1.add_argument("--output", type=Path, required=True)
    c1.set_defaults(function=check_estimator_batching)
    c2 = sub.add_parser("lockstep")
    c2.add_argument("--sensor-stream", type=Path, required=True)
    c2.add_argument("--data-root", type=Path, required=True)
    c2.add_argument("--output", type=Path, required=True)
    c2.add_argument("--torch-threads", type=int, default=4)
    c2.set_defaults(function=check_lockstep)
    run = sub.add_parser("run")
    run.add_argument("--state-endpoint", required=True)
    run.add_argument("--target-endpoint", required=True)
    run.add_argument("--teleop-endpoint", required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--duration-seconds", type=float, default=115.6)
    run.add_argument("--placement", choices=("wsl", "windows"), required=True)
    run.add_argument("--priority", choices=("normal", "high"), default="high")
    run.add_argument("--torch-threads", type=int, default=4)
    run.add_argument("--brake-step-rad", type=float, default=JOINT_LIMIT_BRAKE_STEP_RAD)
    run.set_defaults(function=policy_process)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
