"""Time the no-physics, native23 hardware-shaped BFM control loop.

This program is intentionally a replay and command-sink qualification tool.  It
never subscribes to a robot, never uses a robot lowcmd topic, and never opens a
robot-facing interface.  Lowstate is a recorded 500 Hz sensor-only stream;
teleoperation is received through the established ZMQ publisher and PacketGate.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import platform
import struct
import sys
import time

import mujoco
import numpy as np
import torch
import zmq

from gear_sonic.scripts.evaluate_g1_true23_bfmzero import MODEL, PACKAGE, PHYSICS, ROOT, load_motion
from gear_sonic.scripts.run_g1_true23_bfm_teleop_sim import WindowsRealtimeScope, decode_packet
from gear_sonic.utils.g1_true23_bfm_imu_odometry import Native23IMUOdometry
from gear_sonic.utils.g1_true23_bfmzero_inference import BFMHistory, BFMZeroInference, load_contract, state_and_terms
from gear_sonic.utils.g1_true23_bfmzero_stream import (
    DT, HORIZON, JOINT_LIMIT_BRAKE_LOOKAHEAD_S, JOINT_LIMIT_BRAKE_STEP_RAD,
    JOINT_TARGET_MARGIN_RAD, PacketGate, received_goal,
)
from gear_sonic.utils.g1_true23_bfmzero_stream_clock import SimulationPacer
from gear_sonic.utils.g1_true23_step1b_mujoco import prepare_true23_model


LOWSTATE_DT = 0.002
CONTROL_TICKS = 10
SAFE_DDS_TOPIC = "rt/fix3_timing_no_robot_lowcmd"
SAFE_DDS_DOMAIN = 232


def summary_ms(values):
    values = np.asarray(values, dtype=np.float64)
    return dict(p50=float(np.percentile(values, 50)), p95=float(np.percentile(values, 95)), max=float(values.max()))


class WindowsCommandBuffer:
    """Fixed-size local serialization stand-in; it does no network I/O."""

    kind = "fixed_size_local_command_buffer"

    def __init__(self, contract):
        self.buffer = bytearray(4 + 23 * 3 * 4)
        self.kp = np.asarray(contract["kp"], np.float32)
        self.kd = np.asarray(contract["kd"], np.float32)
        self.sends = 0

    def send(self, tick, target):
        struct.pack_into("<I", self.buffer, 0, tick)
        struct.pack_into("<23f", self.buffer, 4, *np.asarray(target, np.float32))
        struct.pack_into("<23f", self.buffer, 4 + 23 * 4, *self.kp)
        struct.pack_into("<23f", self.buffer, 4 + 46 * 4, *self.kd)
        self.sends += 1

    def close(self):
        pass


class LoopbackDDSCommandSink:
    """Real Unitree HG LowCmd messages, confined to loopback/domain 232."""

    kind = "unitree_hg_lowcmd_dds_loopback_only"

    def __init__(self, contract):
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
        from unitree_sdk2py.utils.crc import CRC

        # Explicit loopback prevents any traffic from leaving WSL via eth0.
        ChannelFactoryInitialize(SAFE_DDS_DOMAIN, "lo")
        self.publisher = ChannelPublisher(SAFE_DDS_TOPIC, LowCmd_)
        self.publisher.Init()
        self.command = unitree_hg_msg_dds__LowCmd_()
        self.command.mode_pr = 0
        self.command.mode_machine = 0
        self.kp, self.kd = contract["kp"], contract["kd"]
        self.crc = CRC()
        self.sends = 0

    def send(self, tick, target):
        self.command.mode_pr = 0
        self.command.mode_machine = 0
        for index in range(23):
            motor = self.command.motor_cmd[index]
            motor.q = float(target[index])
            motor.dq = 0.0
            motor.tau = 0.0
            motor.kp = float(self.kp[index])
            motor.kd = float(self.kd[index])
        self.command.crc = self.crc.Crc(self.command)
        self.publisher.Write(self.command)
        self.sends += 1

    def close(self):
        self.publisher.Close()


class BoundedAnkleBrake:
    """The deployed final-target layer, without a simulator dependency."""

    def __init__(self, model, step_rad):
        if not np.isfinite(step_rad) or step_rad <= 0:
            raise ValueError("brake step must be finite and positive")
        self.limits = np.asarray(model.jnt_range[1:], dtype=np.float64)
        self.margin = np.zeros(23)
        self.margin[model.joint("left_ankle_roll_joint").id - 1] = JOINT_TARGET_MARGIN_RAD
        self.low, self.high = self.limits[:, 0] + self.margin, self.limits[:, 1] - self.margin
        self.step = float(step_rad)
        self.previous = None
        self.engagements = 0
        self.maximum_step = 0.0

    def apply(self, requested, q, dq):
        policy_target = np.clip(np.asarray(requested), self.limits[:, 0], self.limits[:, 1])
        target = policy_target.copy()
        protected = self.margin > 0
        predicted = np.asarray(q) + np.asarray(dq) * JOINT_LIMIT_BRAKE_LOOKAHEAD_S
        low = protected & (dq < 0) & (predicted < self.low)
        high = protected & (dq > 0) & (predicted > self.high)
        engaged = low | high
        if np.any(low):
            value = np.minimum(policy_target[low] + self.step, self.high[low])
            if self.previous is not None:
                value = np.clip(value, self.previous[low] - self.step, self.previous[low] + self.step)
            target[low] = value
        if np.any(high):
            value = np.maximum(policy_target[high] - self.step, self.low[high])
            if self.previous is not None:
                value = np.clip(value, self.previous[high] - self.step, self.previous[high] + self.step)
            target[high] = value
        change = 0.0
        if self.previous is not None and np.any(engaged):
            change = float(np.max(np.abs(target[engaged] - self.previous[engaged])))
            if change > self.step + 1e-12:
                raise RuntimeError("bounded brake target-step invariant failed")
        self.previous = target.copy()
        self.engagements += int(np.any(engaged))
        self.maximum_step = max(self.maximum_step, change)
        return target


def configure_linux_priority():
    result = {"requested": "SCHED_FIFO priority 10, nice -10", "applied": []}
    try:
        os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(10))
        result["applied"].append("SCHED_FIFO:10")
    except (AttributeError, PermissionError, OSError) as error:
        result["sched_fifo_error"] = f"{type(error).__name__}: {error}"
    try:
        os.nice(-10)
        result["applied"].append("nice:-10")
    except (AttributeError, PermissionError, OSError) as error:
        result["nice_error"] = f"{type(error).__name__}: {error}"
    return result


def tensors(policy, array):
    return torch.as_tensor(array, device=policy.device)


def check_cuda_equivalence(weights, trace_path):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    recorded = np.load(trace_path)
    if len(recorded["state"]) < 200:
        raise ValueError("equivalence trace has fewer than 200 recorded inputs")
    cpu, gpu = BFMZeroInference(weights, "cpu"), BFMZeroInference(weights, "cuda")
    maximum = 0.0
    with torch.inference_mode():
        for index in range(200):
            result_cpu = cpu.actor(
                torch.from_numpy(recorded["state"][index:index + 1]),
                torch.from_numpy(recorded["action"][index:index + 1]),
                torch.from_numpy(recorded["history"][index:index + 1]),
                torch.from_numpy(recorded["goal"][index:index + 1]),
            ).cpu().numpy()
            result_gpu = gpu.actor(
                tensors(gpu, recorded["state"][index:index + 1]),
                tensors(gpu, recorded["action"][index:index + 1]),
                tensors(gpu, recorded["history"][index:index + 1]),
                tensors(gpu, recorded["goal"][index:index + 1]),
            ).cpu().numpy()
            maximum = max(maximum, float(np.max(np.abs(result_cpu - result_gpu))))
    return {"recorded_inputs": 200, "maximum_absolute_output_difference": maximum, "passed": maximum <= 1e-5}


def run(args):
    if args.output.exists():
        raise FileExistsError("timing evidence refuses overwrite")
    if args.duration_seconds <= 0 or args.duration_seconds > 130.0:
        raise ValueError("duration must be within 0..130 seconds")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA timing requested but CUDA is unavailable")
    args.output.mkdir(parents=True)
    torch.set_num_threads(args.torch_threads)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")
    contract = load_contract(PACKAGE / "bfmzero_inspect_v1/config.yaml")
    weights = PACKAGE / "bfmzero_inference_v1/inference.safetensors"
    policy = BFMZeroInference(weights, args.device)
    _, model, _ = prepare_true23_model(ROOT.parent / "GR00T-WholeBodyControl" / MODEL, ROOT / PHYSICS)
    loaded_sensor = np.load(args.sensor_stream)
    required = {"timestamp_s", "joint_q", "joint_dq", "imu_quat_wxyz", "gyro_body", "accel_specific_force_body"}
    if set(loaded_sensor.files) != required or len(loaded_sensor["timestamp_s"]) < round(args.duration_seconds / LOWSTATE_DT):
        raise ValueError("sensor stream is not a sufficient native23 500 Hz stream")
    # np.load keeps .npz entries compressed and would otherwise decompress six
    # arrays on every 2 ms tick.  Materialize the immutable replay before the
    # timing start, exactly as a socket receiver would have a decoded LowState.
    sensor = {name: np.asarray(loaded_sensor[name]).copy() for name in required}
    loaded_sensor.close()
    estimator = Native23IMUOdometry(model)
    brake = BoundedAnkleBrake(model, args.brake_step_rad)
    sink = WindowsCommandBuffer(contract) if os.name == "nt" else LoopbackDDSCommandSink(contract)
    context = zmq.Context.instance()
    socket = context.socket(zmq.SUB)
    socket.linger = 0
    socket.setsockopt(zmq.SUBSCRIBE, b"")
    socket.connect(args.endpoint)
    gate = PacketGate(contract, stale_seconds=0.1)
    # Do not start the qualification clock before the external publisher has
    # loaded the PICO capture, bound its PUB socket, and completed the SUB
    # handshake.  This is setup, not part of either timed path.
    if not socket.poll(60_000):
        raise TimeoutError("no teleop packet received before timing start")
    pending = [decode_packet(socket.recv_json())]
    history, action, target = BFMHistory(), np.zeros(23, np.float32), contract["default_q"].copy()
    # Kernel/allocator warm-up is strictly before the measured loop.
    dummy = torch.zeros(1, 52, device=policy.device)
    policy.actor(dummy, torch.zeros(1, 23, device=policy.device), torch.zeros(1, 300, device=policy.device), torch.ones(1, 256, device=policy.device))
    first = dict(timestamp_s=float(sensor["timestamp_s"][0]), joint_q=sensor["joint_q"][0], joint_dq=sensor["joint_dq"][0],
                 imu_quat_wxyz=sensor["imu_quat_wxyz"][0], gyro_body=sensor["gyro_body"][0], accel_specific_force_body=sensor["accel_specific_force_body"][0])
    warm_estimator = Native23IMUOdometry(model)
    warm_estimator.update(**first)
    del warm_estimator
    gc_enabled = gc.isenabled()
    gc.collect(); gc.freeze(); gc.disable()
    priority = {"platform": platform.system(), "requested": args.priority}
    if os.name != "nt":
        priority.update(configure_linux_priority())
    path_500 = {"start_lateness_ms": [], "work_ms": [], "deadline_misses": 0, "send_times": []}
    path_50 = {"start_lateness_ms": [], "work_ms": [], "deadline_misses": 0, "policy_updates": 0}
    accepted = rejected = 0
    ticks = round(args.duration_seconds / LOWSTATE_DT)
    try:
        with WindowsRealtimeScope("consumer", args.priority, "none"), SimulationPacer("windows-high-resolution" if os.name == "nt" else "sleep") as pacer:
            start = time.perf_counter()
            for tick in range(ticks):
                due = start + tick * LOWSTATE_DT
                pacer.sleep_until(due)
                tick_start = time.perf_counter()
                path_500["start_lateness_ms"].append(max(0.0, (tick_start - due) * 1000.0))
                sample = {name: sensor[name][tick] for name in required}
                estimate = estimator.update(**sample)
                if tick % CONTROL_TICKS == 0:
                    control_due = due
                    control_start = time.perf_counter()
                    path_50["start_lateness_ms"].append(max(0.0, (control_start - control_due) * 1000.0))
                    while socket.poll(0):
                        try:
                            pending.append(decode_packet(socket.recv_json()))
                        except (ValueError, TypeError, KeyError):
                            rejected += 1
                    now = tick * LOWSTATE_DT
                    delivered = 0
                    while delivered < len(pending) and pending[delivered].source_time <= now + 1e-7:
                        accepted += int(gate.receive(pending[delivered], now))
                        delivered += 1
                    if delivered:
                        del pending[:delivered]
                    if gate.ready():
                        samples = gate.window()
                        measured_qpos = np.r_[estimate["position_start"], sample["imu_quat_wxyz"], sample["joint_q"]]
                        sensed, terms = state_and_terms(sample["joint_q"], sample["joint_dq"], sample["imu_quat_wxyz"], sample["gyro_body"], action, contract["default_q"])
                        goal, _, _ = received_goal(policy, samples, measured_qpos, 1.0, 2.0)
                        raw = policy.actor(tensors(policy, sensed[None]), tensors(policy, action[None]), tensors(policy, history.before_update(terms)[None]), goal.to(policy.device))[0].cpu().numpy()
                        action = raw * 5.0
                        requested = contract["default_q"] + action * 0.25 * contract["training_effort"] / contract["kp"]
                        target = brake.apply(requested, sample["joint_q"], sample["joint_dq"])
                        gate.consume()
                        path_50["policy_updates"] += 1
                    control_end = time.perf_counter()
                    path_50["work_ms"].append((control_end - control_start) * 1000.0)
                    path_50["deadline_misses"] += int(control_end > control_due + DT)
                sink.send(tick, target)
                sent = time.perf_counter()
                path_500["send_times"].append(sent)
                path_500["work_ms"].append((sent - tick_start) * 1000.0)
                path_500["deadline_misses"] += int(sent > due + LOWSTATE_DT)
    finally:
        socket.close(); sink.close()
        if gc_enabled: gc.enable()
        gc.unfreeze(); gc.collect()
    gaps = np.diff(path_500.pop("send_times")) * 1000.0
    for path in (path_500, path_50):
        path["start_lateness_ms"] = summary_ms(path["start_lateness_ms"])
        path["work_ms"] = summary_ms(path["work_ms"])
    report = {
        "kind": "g1_true23_bfm_hardware_shaped_timing_v1", "platform": platform.platform(), "python": sys.version,
        "duration_seconds": args.duration_seconds, "lowstate_source": str(args.sensor_stream), "lowstate_replay_hz": 500,
        "teleop_source": "existing run_g1_true23_bfm_teleop_sim.py ZMQ publisher through PacketGate", "teleop_packets_accepted": accepted,
        "teleop_packets_rejected": rejected, "device": args.device, "command_sink": sink.kind,
        "dds_safety": "Windows: local fixed-size serialization only; Linux: domain 232, topic rt/fix3_timing_no_robot_lowcmd, interface lo; never rt/lowcmd or eth0",
        "no_mujoco_stepping_inside_timed_loop": True, "automatic_gc_disabled_during_timed_loop": True, "priority": priority,
        "brake_step_rad": args.brake_step_rad, "brake_engagement_controls": brake.engagements,
        "brake_target_step_max_rad": brake.maximum_step, "path_500hz": path_500, "path_50hz": path_50,
        "largest_lowcmd_send_gap_ms": float(gaps.max()) if len(gaps) else 0.0,
    }
    report["passed"] = bool(path_500["deadline_misses"] == path_50["deadline_misses"] == 0 and report["largest_lowcmd_send_gap_ms"] < 4.0)
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5610")
    parser.add_argument("--sensor-stream", type=Path, required=True)
    parser.add_argument("--duration-seconds", type=float, default=115.6)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--priority", choices=("normal", "high"), default="high")
    parser.add_argument("--brake-step-rad", type=float, default=JOINT_LIMIT_BRAKE_STEP_RAD)
    parser.add_argument("--verify-cuda-trace", type=Path)
    args = parser.parse_args()
    if args.verify_cuda_trace is not None:
        result = check_cuda_equivalence(PACKAGE / "bfmzero_inference_v1/inference.safetensors", args.verify_cuda_trace)
        if not result["passed"]:
            raise RuntimeError("CPU/GPU BFM output equivalence exceeds 1e-5: " + json.dumps(result))
        (args.output.parent / "cuda_cpu_equivalence.json").write_text(json.dumps(result, indent=2))
    run(args)


if __name__ == "__main__":
    main()
