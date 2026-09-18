"""Two-process native23 BFM-Zero teleoperation simulation over ZMQ.

The publisher sends the recorded native23 reference at 50 Hz.  The consumer
advances its 500 Hz simulation continuously after the first packet, admits
only packets received from the socket through BFMStreamSimulator's existing
gate, and supplies the policy with joint/IMU-only feedback plus estimated root
XY from Native23IMUOdometry.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import time

import mujoco
import numpy as np
import torch
import zmq

from gear_sonic.scripts.evaluate_g1_true23_bfmzero import MODEL, PACKAGE, PHYSICS, ROOT, load_motion
from gear_sonic.scripts import evaluate_g1_true23_bfmzero as bfm_evaluator
from gear_sonic.utils.g1_true23_bfm_imu_odometry import Native23IMUOdometry
from gear_sonic.utils.g1_true23_bfmzero_inference import BFMZeroInference, load_contract
from gear_sonic.utils.g1_true23_bfmzero_stream import (
    BFMStreamSimulator, DT, FIELD_SHAPES, GateAdmissionAudit, Packet,
    TeleopClockAlignment, packet_fields,
)
from gear_sonic.utils.g1_true23_bfmzero_stream_clock import SimulationPacer
from gear_sonic.utils.g1_true23_generalist_benchmark import task_points
from gear_sonic.utils.g1_true23_step1b_mujoco import _quaternion_matrix, prepare_true23_model


class ControlProfiler:
    """Opt-in control-loop timings, kept entirely out of normal runs."""

    def __init__(self):
        self.samples = {}
        self.active = None
        self.near_or_missed = []
        self.gc_events = []
        self.clock_zero = None

    def begin_control(self, control):
        self.active = {"control": control}

    def finish_control(self, elapsed_ms, missed):
        if self.active is None:
            return
        self.active["full_loop_ms"] = elapsed_ms
        self.active["missed_deadline"] = bool(missed)
        if elapsed_ms > 18.0 or missed:
            self.near_or_missed.append(self.active)
        self.active = None

    def install_gc_callback(self, clock_zero):
        self.clock_zero = clock_zero

        def callback(phase, info):
            self.gc_events.append({
                "timestamp_ms": (time.perf_counter() - self.clock_zero) * 1000.0,
                "phase": phase,
                "generation": int(info["generation"]),
                "collected": int(info.get("collected", 0)),
                "uncollectable": int(info.get("uncollectable", 0)),
            })

        self._gc_callback = callback
        gc.callbacks.append(callback)

    def remove_gc_callback(self):
        callback = getattr(self, "_gc_callback", None)
        if callback in gc.callbacks:
            gc.callbacks.remove(callback)

    def record(self, name, seconds):
        milliseconds = seconds * 1000.0
        self.samples.setdefault(name, []).append(milliseconds)
        if self.active is not None:
            self.active[name] = self.active.get(name, 0.0) + milliseconds

    def report(self):
        result = {
            name: {
                "mean_ms": float(np.mean(values)),
                "p95_ms": float(np.percentile(values, 95)),
                "max_ms": float(np.max(values)),
                "samples": len(values),
            }
            for name, values in self.samples.items()
        }
        result["near_or_missed_controls"] = self.near_or_missed
        result["gc_events"] = self.gc_events
        return result


def working_set_bytes():
    """Current-process working set without a third-party dependency."""
    if os.name != "nt":
        return None
    import ctypes
    from ctypes import wintypes

    class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    counters = PROCESS_MEMORY_COUNTERS_EX()
    counters.cb = ctypes.sizeof(counters)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return int(counters.WorkingSetSize)


class WindowsRealtimeScope:
    """Scoped priority and role-separated affinity; no policy semantics change."""

    def __init__(self, role, priority, affinity):
        self.role, self.priority, self.affinity = role, priority, affinity
        self.kernel = self.process = self.thread = self.old_priority = self.old_thread_priority = self.old_mask = None
        self.applied_affinity_mask = None

    def __enter__(self):
        if os.name != "nt":
            return self
        import ctypes
        from ctypes import wintypes

        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel.GetCurrentProcess.restype = wintypes.HANDLE
        self.kernel.GetPriorityClass.argtypes = [wintypes.HANDLE]
        self.kernel.GetPriorityClass.restype = wintypes.DWORD
        self.kernel.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        self.kernel.SetPriorityClass.restype = wintypes.BOOL
        self.kernel.GetCurrentThread.restype = wintypes.HANDLE
        self.kernel.GetThreadPriority.argtypes = [wintypes.HANDLE]
        self.kernel.GetThreadPriority.restype = ctypes.c_int
        self.kernel.SetThreadPriority.argtypes = [wintypes.HANDLE, ctypes.c_int]
        self.kernel.SetThreadPriority.restype = wintypes.BOOL
        self.kernel.GetProcessAffinityMask.argtypes = [
            wintypes.HANDLE, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t),
        ]
        self.kernel.GetProcessAffinityMask.restype = wintypes.BOOL
        self.kernel.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
        self.kernel.SetProcessAffinityMask.restype = wintypes.BOOL
        self.process = self.kernel.GetCurrentProcess()
        if self.priority == "high":
            self.old_priority = self.kernel.GetPriorityClass(self.process)
            if not self.old_priority or not self.kernel.SetPriorityClass(self.process, 0x00000080):
                raise ctypes.WinError(ctypes.get_last_error())
            self.thread = self.kernel.GetCurrentThread()
            self.old_thread_priority = self.kernel.GetThreadPriority(self.thread)
            if self.old_thread_priority == 0x7FFFFFFF or not self.kernel.SetThreadPriority(self.thread, 2):
                raise ctypes.WinError(ctypes.get_last_error())
        if self.affinity == "role-separated":
            process_mask = ctypes.c_size_t()
            system_mask = ctypes.c_size_t()
            if not self.kernel.GetProcessAffinityMask(self.process, ctypes.byref(process_mask), ctypes.byref(system_mask)):
                raise ctypes.WinError(ctypes.get_last_error())
            bits = [1 << bit for bit in range(process_mask.value.bit_length()) if process_mask.value & (1 << bit)]
            if len(bits) > 1:
                self.old_mask = process_mask.value
                # Reserve one core for the publisher and place the consumer on
                # every other allowed core.  This keeps the four Torch workers
                # separate without artificially constraining the consumer.
                mask = sum(bits[1:]) if self.role == "consumer" else bits[0]
                if not self.kernel.SetProcessAffinityMask(self.process, ctypes.c_size_t(mask)):
                    raise ctypes.WinError(ctypes.get_last_error())
                self.applied_affinity_mask = mask
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.kernel is not None:
            if self.old_mask is not None:
                self.kernel.SetProcessAffinityMask(self.process, self.old_mask)
            if self.old_priority is not None:
                self.kernel.SetPriorityClass(self.process, self.old_priority)
            if self.old_thread_priority is not None:
                self.kernel.SetThreadPriority(self.thread, self.old_thread_priority)
        return False


def json_packet(packet):
    return {
        **{name: value.tolist() for name, value in packet.fields.items()},
        "sequence": packet.sequence,
        "source_time": packet.source_time,
        "epoch": packet.epoch,
        "final": packet.final,
    }


def decode_packet(message):
    required = {*FIELD_SHAPES, "sequence", "source_time", "epoch", "final"}
    if set(message) != required:
        raise ValueError("transport_packet_fields")
    fields = {name: np.asarray(message[name], dtype=np.float64) for name in FIELD_SHAPES}
    return Packet(
        epoch=message["epoch"], sequence=message["sequence"], source_time=message["source_time"],
        fields=fields, final=message["final"],
    )


def publish(args):
    if args.fault_at < 0 or args.fault_duration <= 0 or args.start_delay < 0:
        raise ValueError("invalid publisher timing")
    torch.set_num_threads(1)
    if args.data_root is not None:
        bfm_evaluator.DATA = args.data_root
    motion, _, _ = load_motion(args.clip)
    count = len(motion["joint_pos"]) - 11
    context = zmq.Context.instance()
    socket = context.socket(zmq.PUB)
    socket.linger = 0
    socket.bind(args.endpoint)
    try:
        # Bind before waiting so an already-started subscriber can complete its
        # subscription handshake.  The source clock itself begins after delay.
        if args.start_delay:
            time.sleep(args.start_delay)
        clock = "windows-high-resolution" if os.name == "nt" else "sleep"
        with WindowsRealtimeScope("publisher", args.realtime_priority, args.affinity), SimulationPacer(clock) as pacer:
            start = time.perf_counter()
            sent = 0
            for sequence in range(count):
                source_time = sequence * DT
                if args.fault == "disconnect" and source_time >= args.fault_at:
                    break
                if args.fault == "pause" and args.fault_at <= source_time < args.fault_at + args.fault_duration:
                    continue
                pacer.sleep_until(start + source_time)
                fields = packet_fields(motion, sequence + 11)
                socket.send_json(json_packet(Packet(0, sequence, source_time, fields, sequence == count - 1)))
                sent += 1
        print(json.dumps({"clip": args.clip, "fault": args.fault, "sent_samples": sent}), flush=True)
    finally:
        socket.close()


class SensorOnlyFeedback:
    """Causal native pelvis IMU + encoder feed, mirroring observable evaluator."""

    def __init__(self, model, data, profiler=None):
        gyro_id = model.sensor("imu-pelvis-angular-velocity").id
        accel_id = model.sensor("imu-pelvis-linear-acceleration").id
        self.gyro_slice = slice(model.sensor_adr[gyro_id], model.sensor_adr[gyro_id] + 3)
        self.accel_slice = slice(model.sensor_adr[accel_id], model.sensor_adr[accel_id] + 3)
        if model.sensor_objid[gyro_id] != model.site("imu_in_pelvis").id:
            raise ValueError("pelvis gyro sensor attachment mismatch")
        if model.sensor_objid[accel_id] != model.site("imu_in_pelvis").id:
            raise ValueError("pelvis accelerometer sensor attachment mismatch")
        self.estimator = Native23IMUOdometry(model)
        self.profiler = profiler
        self.packet = {
            "timestamp_s": 0.0, "joint_q": data.qpos[7:].copy(), "joint_dq": data.qvel[6:].copy(),
            "imu_quat_wxyz": data.qpos[3:7].copy(), "gyro_body": data.sensordata[self.gyro_slice].copy(),
            "accel_specific_force_body": data.sensordata[self.accel_slice].copy(),
        }
        started = time.perf_counter() if profiler is not None else None
        self.estimate = self.estimator.update(**self.packet)
        if profiler is not None:
            profiler.record("estimator_updates_500hz", time.perf_counter() - started)

    def feedback(self):
        return {
            "joint_q": self.packet["joint_q"].copy(), "joint_dq": self.packet["joint_dq"].copy(),
            "imu_quat_wxyz": self.packet["imu_quat_wxyz"].copy(), "gyro_body": self.packet["gyro_body"].copy(),
            "root_position": self.estimate["position_start"].copy(),
        }

    def after_step(self, pre, data):
        # MuJoCo's IMU values generated by mj_step correspond to this copied
        # pre-integration state, as in evaluate_g1_true23_bfm_observable.py.
        if pre["timestamp_s"] <= self.estimator.timestamp + 1e-10:
            return
        pre["gyro_body"] = data.sensordata[self.gyro_slice].copy()
        pre["accel_specific_force_body"] = data.sensordata[self.accel_slice].copy()
        self.packet = pre
        started = time.perf_counter() if self.profiler is not None else None
        self.estimate = self.estimator.update(**pre)
        if self.profiler is not None:
            self.profiler.record("estimator_updates_500hz", time.perf_counter() - started)


def make_simulator(first, contract, policy, profiler=None, brake_step_rad=0.100):
    _, model, physics = prepare_true23_model(ROOT.parent / "GR00T-WholeBodyControl" / MODEL, ROOT / PHYSICS)
    physical_config = json.loads((ROOT / PHYSICS).read_text())
    initial_qpos = np.r_[first.fields["body_pos_w"][0], first.fields["body_quat_w"][0], first.fields["joint_pos"]]
    initial_qvel = np.r_[
        first.fields["body_lin_vel_w"][0],
        _quaternion_matrix(initial_qpos[3:7]).T @ first.fields["body_ang_vel_w"][0],
        first.fields["joint_vel"],
    ]
    # The physical initial pose is a one-time simulation initialization.  All
    # policy root feedback thereafter is supplied by the observer below.
    data = mujoco.MjData(model)
    data.qpos[:] = initial_qpos
    data.qvel[:] = initial_qvel
    mujoco.mj_forward(model, data)
    observer = SensorOnlyFeedback(model, data, profiler)
    standing = physical_config["initial_state"]
    simulator = BFMStreamSimulator(
        model, physics, contract, policy, initial_qpos, initial_qvel,
        np.asarray(standing["joint_position_hardware_rad"]), standing["base_position_m"][2],
        np.asarray(physical_config["physics"]["velocity_limit_hardware_radps"]),
        sensor_feedback=observer.feedback, sensor_step=observer.after_step,
        record_trace=False,
        brake_step_rad=brake_step_rad,
    )
    return simulator, observer


def source_metrics(simulator):
    return simulator.source_metrics()


def consume(args):
    if args.max_seconds is not None and args.max_seconds <= 0:
        raise ValueError("max-seconds must be positive")
    if args.torch_threads <= 0:
        raise ValueError("torch-threads must be positive")
    if args.output.exists():
        raise FileExistsError("teleop evidence refuses overwrite")
    args.output.mkdir(parents=True)
    torch.set_num_threads(args.torch_threads)
    # Subscribe before any heavyweight model initialization.  This makes the
    # documented publisher --start-delay a real transport handshake interval.
    context = zmq.Context.instance()
    socket = context.socket(zmq.SUB)
    socket.linger = 0
    socket.setsockopt(zmq.SUBSCRIBE, b"")
    socket.connect(args.endpoint)
    contract = load_contract(PACKAGE / "bfmzero_inspect_v1/config.yaml")
    policy = BFMZeroInference(PACKAGE / "bfmzero_inference_v1/inference.safetensors")
    # CPU warmup only; no physical state, history, gate, or controller update.
    for _ in range(3):
        policy.actor(torch.zeros(1, 52), torch.zeros(1, 23), torch.zeros(1, 300), torch.ones(1, 256))
    pending = []
    transport_rejected = 0
    teleop_clock = TeleopClockAlignment()
    gate_admission_audit = GateAdmissionAudit()
    simulator = observer = None
    start = None
    late, missed, controls = [], [], []
    profiler = ControlProfiler() if args.profile else None
    gc_enabled_before = gc.isenabled()
    gc_manual_active = False
    gc_memory_start = gc_memory_end = None
    with WindowsRealtimeScope("consumer", args.realtime_priority, args.affinity) as realtime, SimulationPacer("windows-high-resolution" if args.paced else "sleep") as pacer:
        try:
            # No simulation clock exists before the first received source
            # packet.  Once it is established, every control tick advances ten
            # 500 Hz physics steps regardless of socket state.
            while simulator is None:
                if socket.poll(1000):
                    try:
                        received_started = time.perf_counter() if profiler is not None else None
                        first = decode_packet(socket.recv_json())
                        if profiler is not None:
                            profiler.record("zmq_receive_decode", time.perf_counter() - received_started)
                        simulator, observer = make_simulator(
                            first, contract, policy, profiler, brake_step_rad=args.brake_step_rad
                        )
                        # Warm an independent estimator so no first-use allocation
                        # or BLAS dispatch is charged to the control loop.
                        warm_estimator = Native23IMUOdometry(observer.estimator.model)
                        for warm_index in range(3):
                            warm_packet = {name: value.copy() if isinstance(value, np.ndarray) else value
                                           for name, value in observer.packet.items()}
                            warm_packet["timestamp_s"] = warm_index * 0.002
                            warm_estimator.update(**warm_packet)
                        del warm_estimator
                        if args.gc_mode == "manual":
                            # These collections are deliberately before/after,
                            # never inside a paced control interval.
                            gc.collect()
                            gc.freeze()
                            gc.disable()
                            gc_manual_active = True
                            gc_memory_start = working_set_bytes()
                        start = time.perf_counter()
                        if profiler is not None:
                            profiler.install_gc_callback(start)
                        pending.append((
                            first, teleop_clock.observe(first, 0.0, simulator.gate),
                            teleop_clock.gate_source_time_offset(simulator.gate)
                        ))
                    except (ValueError, TypeError, KeyError):
                        transport_rejected += 1
            control = 0
            while True:
                scheduled = control * DT
                if args.paced:
                    pacing_started = time.perf_counter() if profiler is not None else None
                    pacer.sleep_until(start + scheduled)
                    if profiler is not None:
                        profiler.record("pacing_wait", time.perf_counter() - pacing_started)
                tick = time.perf_counter()
                if profiler is not None:
                    profiler.begin_control(control)
                now = tick - start if args.paced else scheduled
                while socket.poll(0):
                    try:
                        received_started = time.perf_counter() if profiler is not None else None
                        packet = decode_packet(socket.recv_json())
                        pending.append((
                            packet, teleop_clock.observe(packet, now, simulator.gate),
                            teleop_clock.gate_source_time_offset(simulator.gate)
                        ))
                        if profiler is not None:
                            profiler.record("zmq_receive_decode", time.perf_counter() - received_started)
                    except (ValueError, TypeError, KeyError):
                        transport_rejected += 1
                # Do not give a packet to the strict gate before its source
                # time mapped into this consumer's clock.  PacketGate retains
                # the canonical source timestamp and its unchanged age limit.
                deliver = 0
                admission_started = time.perf_counter() if profiler is not None else None
                while (
                    deliver < len(pending)
                    and (pending[deliver][1] is None or pending[deliver][1] <= now + 1e-7)
                ):
                    packet, _, source_time_offset = pending[deliver]
                    rejected_before = simulator.gate.rejected
                    accepted = simulator.receive(
                        packet, now, source_time_offset=0.0 if source_time_offset is None else source_time_offset
                    )
                    gate_admission_audit.record(simulator.gate, accepted, rejected_before)
                    deliver += 1
                if profiler is not None:
                    profiler.record("stream_admission", time.perf_counter() - admission_started)
                if deliver:
                    del pending[:deliver]
                simulator.step(now, profiler)
                ended = time.perf_counter()
                late.append(max(0.0, (tick - start - scheduled) * 1000) if args.paced else 0.0)
                missed.append(ended > start + (control + 1) * DT if args.paced else ended - tick > DT)
                controls.append((ended - tick) * 1000)
                if profiler is not None:
                    profiler.finish_control(controls[-1], missed[-1])
                control += 1
                done = (
                    simulator.mode.value in ("complete", "latched_standing")
                    and simulator.stable_controls >= simulator.stable_controls_required
                ) and (
                    simulator.gate.fault is not None or simulator.gate.final_consumed
                )
                if done or simulator.physical_failure is not None:
                    break
                if args.max_seconds is not None and control * DT >= args.max_seconds:
                    break
        finally:
            if profiler is not None:
                profiler.remove_gc_callback()
            socket.close()
    if gc_manual_active:
        gc_memory_end = working_set_bytes()
        if gc_enabled_before:
            gc.enable()
        gc.unfreeze()
        gc.collect()
    result = simulator.report()
    metrics = source_metrics(simulator)
    result.update({
        "scenario": "fault" if result["any_fault_latched"] else "normal",
        "paced": args.paced,
        "received_samples": simulator.gate.received,
        "dropped_or_reordered_samples": simulator.gate.rejected + transport_rejected,
        "teleop_packets_gate_rejected": gate_admission_audit.rejected,
        "teleop_gate_rejection_reasons": gate_admission_audit.reasons,
        "teleop_clock_alignment": {
            "method": "first_packet_reception_offset_per_gate_epoch",
            "anchors": teleop_clock.anchors,
        },
        "late_control_deadlines": int(np.count_nonzero(np.asarray(late) > 0.0)),
        "missed_control_deadlines": int(np.count_nonzero(missed)),
        "control_start_late_ms_p95_max": np.percentile(late, [95, 100]).tolist(),
        "full_loop_ms_p50_p95_max": np.percentile(controls, [50, 95, 100]).tolist(),
        "source_phase_metrics": metrics,
        "leg_rmse": None if metrics is None else metrics["leg_rmse"],
        "arm_rmse": None if metrics is None else metrics["arm_rmse"],
        "root_p95": None if metrics is None else metrics["root_p95"],
        "ground_truth_pose_feedback": False,
        "root_xy_feedback_source": "causal Native23IMUOdometry from simulated pelvis IMU and joint states",
        "simulator_pose_writes_after_initialization": 0,
        "root_assistance_forces": 0,
        "controller_fallback": False,
        "runtime_controls": {
            "automatic_gc_disabled_during_control_loop": gc_manual_active,
            "gc_working_set_start_bytes": gc_memory_start,
            "gc_working_set_end_bytes": gc_memory_end,
            "pacing_timer": "high-resolution waitable timer" if args.paced else "sleep",
            "process_priority": args.realtime_priority,
            "affinity": args.affinity,
            "applied_affinity_mask": realtime.applied_affinity_mask,
        },
    })
    if profiler is not None:
        result["control_step_profile_ms"] = profiler.report()
    (args.output / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({key: result[key] for key in (
        "controls", "physics_steps", "any_fault_latched", "physical_failure", "standing_return_verified",
        "full_uninterrupted_source_consumed", "received_samples", "missed_control_deadlines",
    )}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    subcommands = parser.add_subparsers(required=True)
    pub = subcommands.add_parser("publish")
    pub.add_argument("--clip", choices=("walk002", "walk003", "walk008", "pico"), required=True)
    pub.add_argument("--endpoint", default="tcp://127.0.0.1:5591")
    pub.add_argument("--fault", choices=("none", "pause", "disconnect"), default="none")
    pub.add_argument("--fault-at", type=float, default=9.0)
    pub.add_argument("--fault-duration", type=float, default=0.5)
    pub.add_argument("--start-delay", type=float, default=0.0)
    pub.add_argument("--data-root", type=Path, help="explicit reference-data root; required for WSL replay")
    pub.add_argument("--realtime-priority", choices=("normal", "high"), default="high")
    pub.add_argument("--affinity", choices=("none", "role-separated"), default="none")
    pub.set_defaults(function=publish)
    con = subcommands.add_parser("consume")
    con.add_argument("--endpoint", default="tcp://127.0.0.1:5591")
    con.add_argument("--output", type=Path, required=True)
    con.add_argument("--paced", action="store_true")
    con.add_argument("--profile", action="store_true", help="write opt-in per-part timing statistics")
    con.add_argument("--gc-mode", choices=("enabled", "manual"), default="manual")
    con.add_argument("--realtime-priority", choices=("normal", "high"), default="high")
    con.add_argument("--affinity", choices=("none", "role-separated"), default="none")
    con.add_argument("--torch-threads", type=int, default=4)
    con.add_argument("--brake-step-rad", type=float, default=0.100)
    con.add_argument("--max-seconds", type=float)
    con.set_defaults(function=consume)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
