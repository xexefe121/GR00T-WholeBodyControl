"""Fix 5 policy side of the split native23 BFM timing qualification.

The native executable owns lowcmd at 500 Hz.  This process only receives
recorded LowState-shaped samples, batches ordered IMU updates at 50 Hz, then
runs the unchanged BFM, PacketGate, and bounded final-target brake.  It has no
DDS imports and cannot publish to a robot topic or network interface.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import json
import os
from pathlib import Path
import struct
import subprocess
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
STATE_WIRE = struct.Struct("<IIQQQd23f23f4f3f3f22d")
TARGET_WIRE = struct.Struct("<IIQQQ69f")
INITIAL_COMMAND = struct.Struct("<II69f")


class WindowsPolicyTelemetry:
    """Cheap per-control Windows scheduler and virtual-memory counters.

    ``GetProcessMemoryInfo`` supplies cheap total-fault deltas.  The documented
    thread-profiling API supplies the current control thread's context-switch
    delta and wait-reason bitmap.  Hard-fault and process-wide switch counters
    require an expensive system-wide query, so they are captured at run bounds
    rather than inserted into every 20 ms control.
    """

    _SYSTEM_PROCESS_INFORMATION = 5
    _STATUS_INFO_LENGTH_MISMATCH = 0xC0000004
    _PROCESS_HEADER_BYTES = 256  # 64-bit SYSTEM_PROCESS_INFORMATION through I/O transfer counts.
    _SYSTEM_THREAD_INFORMATION_BYTES = 80

    def __init__(self, enabled: bool):
        self.enabled = bool(enabled and os.name == "nt")
        self.available = False
        self.note = None
        self.kernel = self.psapi = self.ntdll = None
        self.profile_handle = None
        if not self.enabled:
            return
        try:
            from ctypes import wintypes

            self.wintypes = wintypes
            self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            self.psapi = ctypes.WinDLL("psapi", use_last_error=True)
            self.ntdll = ctypes.WinDLL("ntdll")
            self.kernel.GetCurrentProcess.restype = wintypes.HANDLE
            self.kernel.GetCurrentProcessId.restype = wintypes.DWORD
            self.kernel.GetCurrentThreadId.restype = wintypes.DWORD
            self.kernel.GetCurrentThread.restype = wintypes.HANDLE
            self.psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
            self.psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
            self.ntdll.NtQuerySystemInformation.argtypes = [ctypes.c_ulong, ctypes.c_void_p, ctypes.c_ulong, ctypes.POINTER(ctypes.c_ulong)]
            self.ntdll.NtQuerySystemInformation.restype = ctypes.c_long
            self.kernel.EnableThreadProfiling.argtypes = [wintypes.HANDLE, wintypes.DWORD, ctypes.c_ulonglong, ctypes.POINTER(wintypes.HANDLE)]
            self.kernel.EnableThreadProfiling.restype = wintypes.DWORD
            self.kernel.ReadThreadProfilingData.argtypes = [wintypes.HANDLE, wintypes.DWORD, ctypes.c_void_p]
            self.kernel.ReadThreadProfilingData.restype = wintypes.DWORD
            self.kernel.DisableThreadProfiling.argtypes = [wintypes.HANDLE]
            self.kernel.DisableThreadProfiling.restype = wintypes.DWORD
            self.pid = int(self.kernel.GetCurrentProcessId())
            self.tid = int(self.kernel.GetCurrentThreadId())
            handle = wintypes.HANDLE()
            error = self.kernel.EnableThreadProfiling(self.kernel.GetCurrentThread(), 1, 0, ctypes.byref(handle))
            if error:
                raise OSError(error, "EnableThreadProfiling failed")
            self.profile_handle = handle
            self.available = True
        except Exception as error:  # Evidence collection must never stop control.
            self.note = f"Windows counter setup failed: {error!r}"

    def _total_faults(self):
        class COUNTERS(ctypes.Structure):
            _fields_ = [("cb", self.wintypes.DWORD), ("PageFaultCount", self.wintypes.DWORD),
                         ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                         ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                         ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                         ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
                         ("PrivateUsage", ctypes.c_size_t)]
        counters = COUNTERS()
        counters.cb = ctypes.sizeof(counters)
        if not self.psapi.GetProcessMemoryInfo(self.kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return int(counters.PageFaultCount)

    def _system_snapshot(self):
        size = 128 * 1024
        while True:
            buffer = ctypes.create_string_buffer(size)
            returned = ctypes.c_ulong()
            status = int(self.ntdll.NtQuerySystemInformation(
                self._SYSTEM_PROCESS_INFORMATION, buffer, size, ctypes.byref(returned))) & 0xFFFFFFFF
            if status != self._STATUS_INFO_LENGTH_MISMATCH:
                if status:
                    raise OSError(f"NtQuerySystemInformation failed: 0x{status:08x}")
                break
            size = max(size * 2, int(returned.value) + 4096)
        data = buffer.raw
        offset = 0
        while offset < len(data):
            next_offset, thread_count = struct.unpack_from("<II", data, offset)
            pid = struct.unpack_from("<Q", data, offset + 80)[0]
            if pid == self.pid:
                hard_faults = struct.unpack_from("<I", data, offset + 16)[0]
                process_switches = 0
                thread_switches = None
                thread_base = offset + self._PROCESS_HEADER_BYTES
                for index in range(thread_count):
                    item = thread_base + index * self._SYSTEM_THREAD_INFORMATION_BYTES
                    if item + self._SYSTEM_THREAD_INFORMATION_BYTES > len(data):
                        break
                    switches = struct.unpack_from("<I", data, item + 64)[0]
                    process_switches += switches
                    if struct.unpack_from("<Q", data, item + 48)[0] == self.tid:
                        thread_switches = switches
                return int(hard_faults), int(process_switches), thread_switches
            if next_offset == 0:
                break
            offset += next_offset
        raise RuntimeError("current process absent from SystemProcessInformation")

    def sample(self):
        if not self.available:
            return None
        try:
            class PERFORMANCE_DATA(ctypes.Structure):
                _fields_ = [("Size", ctypes.c_ushort), ("Version", ctypes.c_ubyte), ("HwCountersCount", ctypes.c_ubyte),
                           ("ContextSwitchCount", ctypes.c_ulong), ("WaitReasonBitMap", ctypes.c_ulonglong),
                           ("CycleTime", ctypes.c_ulonglong), ("RetryCount", ctypes.c_ulong), ("Reserved", ctypes.c_ulong),
                           # MAX_HW_COUNTERS (16) * sizeof(HARDWARE_COUNTER_DATA) (16).
                           ("HwCounters", ctypes.c_byte * 256)]
            performance = PERFORMANCE_DATA()
            performance.Size, performance.Version = ctypes.sizeof(performance), 1
            error = self.kernel.ReadThreadProfilingData(self.profile_handle, 1, ctypes.byref(performance))
            if error:
                raise OSError(error, "ReadThreadProfilingData failed")
            total = self._total_faults()
            return {"total_faults": total, "thread_context_switches": int(performance.ContextSwitchCount),
                    "thread_wait_reason_bitmap": int(performance.WaitReasonBitMap)}
        except Exception as error:
            self.available = False
            self.note = f"Windows counter read failed: {error!r}"
            return None

    @staticmethod
    def delta(before, after):
        if before is None or after is None:
            return None
        result = {name + "_delta": (None if after[name] is None or before[name] is None else int(after[name] - before[name]))
                  for name in before}
        # NT does not expose voluntary/involuntary context-switch categories.
        result["voluntary_context_switches_delta"] = None
        result["involuntary_context_switches_delta"] = None
        return result

    def expensive_run_counters(self):
        """Bounded-only counters: retained as evidence without perturbing ticks."""
        if not self.available:
            return None
        try:
            hard, process_switches, _ = self._system_snapshot()
            total = self._total_faults()
            return {"hard_faults": hard, "soft_faults": max(0, total - hard), "process_context_switches": process_switches}
        except Exception as error:
            self.note = f"Windows bounded counter read failed: {error!r}"
            return None

    def close(self):
        if self.profile_handle:
            self.kernel.DisableThreadProfiling(self.profile_handle)
            self.profile_handle = None


class WindowsTimerResolutionScope:
    """Request 0.5 ms timer resolution only for a measured process lifetime."""

    def __init__(self, enabled: bool):
        self.enabled = bool(enabled and os.name == "nt")
        self.before_100ns = self.requested_100ns = self.achieved_100ns = None
        self.ntdll = None

    def _query(self):
        minimum = ctypes.c_ulong()
        maximum = ctypes.c_ulong()
        current = ctypes.c_ulong()
        status = int(self.ntdll.NtQueryTimerResolution(ctypes.byref(minimum), ctypes.byref(maximum), ctypes.byref(current)))
        if status:
            raise OSError(f"NtQueryTimerResolution failed: 0x{status & 0xffffffff:08x}")
        return int(minimum.value), int(maximum.value), int(current.value)

    def __enter__(self):
        if not self.enabled:
            return self
        self.ntdll = ctypes.WinDLL("ntdll")
        self.ntdll.NtQueryTimerResolution.argtypes = [ctypes.POINTER(ctypes.c_ulong)] * 3
        self.ntdll.NtQueryTimerResolution.restype = ctypes.c_long
        self.ntdll.NtSetTimerResolution.argtypes = [ctypes.c_ulong, ctypes.c_ubyte, ctypes.POINTER(ctypes.c_ulong)]
        self.ntdll.NtSetTimerResolution.restype = ctypes.c_long
        self.before_100ns = self._query()
        current = ctypes.c_ulong()
        status = int(self.ntdll.NtSetTimerResolution(5000, True, ctypes.byref(current)))
        if status:
            raise OSError(f"NtSetTimerResolution failed: 0x{status & 0xffffffff:08x}")
        self.requested_100ns = 5000
        self.achieved_100ns = int(current.value)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.requested_100ns is not None:
            current = ctypes.c_ulong()
            self.ntdll.NtSetTimerResolution(self.requested_100ns, False, ctypes.byref(current))
        return False

    def report(self):
        convert = lambda value: None if value is None else value / 10_000.0
        return {"requested": self.enabled, "before_ms": None if self.before_100ns is None else {
            "minimum": convert(self.before_100ns[0]), "maximum": convert(self.before_100ns[1]), "current": convert(self.before_100ns[2])},
            "requested_ms": convert(self.requested_100ns), "achieved_ms": convert(self.achieved_100ns),
            "undo": "The process releases NtSetTimerResolution(5000, FALSE) automatically on exit; no persistent setting is changed."}


class WindowsMmcssScope:
    """Register just the control thread with MMCSS and always revert it."""

    def __init__(self, enabled: bool):
        self.enabled = bool(enabled and os.name == "nt")
        self.handle = self.avrt = None

    def __enter__(self):
        if not self.enabled:
            return self
        from ctypes import wintypes
        self.avrt = ctypes.WinDLL("avrt", use_last_error=True)
        self.avrt.AvSetMmThreadCharacteristicsW.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(wintypes.DWORD)]
        self.avrt.AvSetMmThreadCharacteristicsW.restype = wintypes.HANDLE
        self.avrt.AvSetMmThreadPriority.argtypes = [wintypes.HANDLE, ctypes.c_int]
        self.avrt.AvSetMmThreadPriority.restype = wintypes.BOOL
        self.avrt.AvRevertMmThreadCharacteristics.argtypes = [wintypes.HANDLE]
        self.avrt.AvRevertMmThreadCharacteristics.restype = wintypes.BOOL
        task_index = wintypes.DWORD()
        self.handle = self.avrt.AvSetMmThreadCharacteristicsW("Pro Audio", ctypes.byref(task_index))
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        if not self.avrt.AvSetMmThreadPriority(self.handle, 2):  # AVRT_PRIORITY_HIGH
            error = ctypes.get_last_error()
            self.avrt.AvRevertMmThreadCharacteristics(self.handle)
            self.handle = None
            raise ctypes.WinError(error)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.handle:
            self.avrt.AvRevertMmThreadCharacteristics(self.handle)
        return False


class WindowsMemoryResidencyScope:
    """Lock the pre-existing static control buffers for one measured process.

    Per-control decoded state and feature arrays are intentionally not included:
    they are allocated after this scope is entered.  The report makes that
    boundary explicit so a successful ``VirtualLock`` is not overstated as a
    claim about every transient allocation in the Python timed path.
    """

    def __init__(self, enabled: bool, policy, contract, history, action):
        self.enabled = bool(enabled and os.name == "nt")
        self.policy, self.contract, self.history, self.action = policy, contract, history, action
        self.kernel = None
        self.regions = []
        self.errors = []

    def _add_numpy(self, name, value):
        array = np.asarray(value)
        if array.nbytes:
            self.regions.append((name, int(array.ctypes.data), int(array.nbytes)))

    def __enter__(self):
        if not self.enabled:
            return self
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel.VirtualLock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        self.kernel.VirtualLock.restype = ctypes.c_int
        self.kernel.VirtualUnlock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        self.kernel.VirtualUnlock.restype = ctypes.c_int
        for index, tensor in enumerate(self.policy.weights.values()):
            if tensor.numel():
                self.regions.append((f"weight_{index}", int(tensor.data_ptr()), int(tensor.numel() * tensor.element_size())))
        for name in ("default_q", "kp", "kd", "training_effort"):
            self._add_numpy("contract_" + name, self.contract[name])
        self._add_numpy("action", self.action)
        for name, value in self.history.data.items():
            self._add_numpy("history_" + name, value)
        locked = []
        for name, address, size in self.regions:
            if self.kernel.VirtualLock(ctypes.c_void_p(address), ctypes.c_size_t(size)):
                locked.append((name, address, size))
            else:
                self.errors.append({"region": name, "bytes": size, "winerror": int(ctypes.get_last_error())})
        self.regions = locked
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.kernel is not None:
            for _, address, size in reversed(self.regions):
                self.kernel.VirtualUnlock(ctypes.c_void_p(address), ctypes.c_size_t(size))
        return False

    def report(self):
        return {
            "requested": self.enabled,
            "scope": "static weights, contract arrays, action and BFM history; not per-control decoded or feature allocations",
            "locked_regions": len(self.regions),
            "locked_bytes": int(sum(size for _, _, size in self.regions)),
            "lock_errors": self.errors,
            "undo": "VirtualUnlock is called automatically for every locked region on process exit; no persistent setting is changed.",
        }


def windows_activity_snapshot():
    """Capture only after an outlier; never run a process scan in every tick."""
    if os.name != "nt":
        return None
    script = (
        "$p=Get-CimInstance Win32_PerfFormattedData_PerfProc_Process | "
        "Sort-Object PercentProcessorTime -Descending | Select-Object -First 12 Name,IDProcess,PercentProcessorTime;"
        "$s=Get-Service -Name WinDefend,WSearch,SysMain,OneSyncSvc -ErrorAction SilentlyContinue | "
        "Select-Object Name,Status;"
        "[pscustomobject]@{top_processes=$p;services=$s}|ConvertTo-Json -Compress -Depth 3"
    )
    try:
        completed = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
                                   capture_output=True, text=True, timeout=4, check=True)
        return json.loads(completed.stdout)
    except Exception as error:
        return {"capture_error": repr(error)}


class WindowsActivityCapture:
    """Asynchronous per-stall-episode activity evidence.

    A synchronous CIM query in the control process turns one late control into
    a series of self-inflicted late controls.  This helper starts its query
    after the first outlier in a contiguous episode and leaves the timed path
    immediately.  Full per-control timing remains in ``controls``.
    """

    def __init__(self, directory: Path):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.processes = []

    def capture(self, control: int):
        output = self.directory / f"control_{control:05d}.json"
        error = self.directory / f"control_{control:05d}.err"
        script = (
            "$top=Get-CimInstance Win32_PerfFormattedData_PerfProc_Process|Sort-Object PercentProcessorTime -Descending|Select-Object -First 12 Name,IDProcess,PercentProcessorTime;"
            "$tracked=Get-CimInstance Win32_PerfFormattedData_PerfProc_Process|Where-Object {$_.Name -match 'MsMpEng|Defender|SearchIndexer|OneDrive|backup|veeam|synology|carbonite'}|Select-Object Name,IDProcess,PercentProcessorTime;"
            "$services=Get-Service -Name WinDefend,WSearch,SysMain -ErrorAction SilentlyContinue|Select-Object Name,Status;"
            "[pscustomobject]@{top_processes=$top;tracked_processes=$tracked;services=$services;captured_utc=[DateTime]::UtcNow.ToString('o')}|ConvertTo-Json -Compress -Depth 4"
        )
        out_handle, err_handle = output.open("w", encoding="utf-8"), error.open("w", encoding="utf-8")
        process = subprocess.Popen(["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
                                   stdout=out_handle, stderr=err_handle, text=True)
        self.processes.append((process, out_handle, err_handle))
        return {"control": control, "activity_json": str(output), "activity_stderr": str(error)}

    def close(self):
        # Do not wait in the control process: captures can complete after the
        # report is written and are keyed by control number for later review.
        for _, out_handle, err_handle in self.processes:
            out_handle.close()
            err_handle.close()


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


def snapshot_from_values(values):
    if len(values) != 22:
        raise ValueError("native estimator snapshot field count mismatch")
    return {
        "position_start": np.asarray(values[0:3], dtype=np.float64),
        "velocity_start": np.asarray(values[3:6], dtype=np.float64),
        "accel_bias_body": np.asarray(values[6:9], dtype=np.float64),
        "quaternion_start": np.asarray(values[9:13], dtype=np.float64),
        "support_weights": np.asarray(values[13:21], dtype=np.float64),
        "timestamp_s": float(values[21]),
    }


def load_native_snapshots(path: Path):
    fields = ("position_start", "velocity_start", "accel_bias_body", "quaternion_start", "support_weights", "timestamp_s")
    table = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64, ndmin=2)
    if table.shape[1] != 23 or not np.isfinite(table).all():
        raise ValueError("native estimator CSV must contain tick plus 22 finite float64 fields")
    return [snapshot_from_values(row[1:]) for row in table], fields


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
    if path.suffix == ".flatbin":
        raw = path.read_bytes()
        magic, version, count, period = REPLAY_HEADER.unpack_from(raw)
        if magic != REPLAY_MAGIC or version != 1 or period != LOWSTATE_DT:
            raise ValueError("invalid flat replay header")
        dtype = np.dtype([
            ("timestamp_s", "<f8"), ("joint_q", "<f4", 23), ("joint_dq", "<f4", 23),
            ("imu_quat_wxyz", "<f4", 4), ("gyro_body", "<f4", 3), ("accel_specific_force_body", "<f4", 3),
        ])
        values = np.frombuffer(raw, dtype=dtype, count=count, offset=REPLAY_HEADER.size)
        if len(values) != count or len(raw) != REPLAY_HEADER.size + count * dtype.itemsize:
            raise ValueError("truncated flat replay")
        return {name: values[name].copy() for name in dtype.names}
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


def check_native_estimator(args):
    """G1: compare the robot's CSV snapshot at every 50 Hz consumption point."""
    sensor = load_sensor(args.sensor_stream)
    if args.samples is not None:
        sensor = {key: value[:args.samples] for key, value in sensor.items()}
    snapshots, fields = load_native_snapshots(args.native_snapshots)
    expected_controls = (len(sensor["timestamp_s"]) - 1) // CONTROL_TICKS + 1
    if len(snapshots) != expected_controls:
        raise ValueError(f"native snapshot count {len(snapshots)} does not match {expected_controls} consumption points")
    _, model, _ = prepare_true23_model(ROOT.parent / "GR00T-WholeBodyControl" / MODEL, ROOT / PHYSICS)
    estimator = Native23IMUOdometry(model)
    maximum = {field: 0.0 for field in fields}
    first = None
    control = 0
    for index in range(len(sensor["timestamp_s"])):
        sample = {key: sensor[key][index] for key in sensor}
        reference = estimator.update(**sample)
        if index % CONTROL_TICKS:
            continue
        native = snapshots[control]
        for field in fields:
            difference = float(np.max(np.abs(np.asarray(reference[field]) - np.asarray(native[field]))))
            maximum[field] = max(maximum[field], difference)
            if first is None and difference > 1e-9:
                first = {"control": control, "field": field, "difference": difference}
        control += 1
    result = {
        "kind": "g1_true23_bfm_fix7_native_estimator_equivalence_v1",
        "passed": first is None,
        "control_points": control,
        "maximum_absolute_difference_by_field": maximum,
        "first_difference_over_1e-9": first,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if not result["passed"]:
        raise RuntimeError("G1 failed: " + json.dumps(result))
    print(json.dumps(result, indent=2))


def policy_step(policy, contract, estimator_result, sample, history, action, brake, gate, phases=None):
    if not gate.ready():
        return action, None
    phase_start = time.perf_counter_ns()
    samples = gate.window()
    measured_qpos = np.r_[estimator_result["position_start"], sample["imu_quat_wxyz"], sample["joint_q"]]
    sensed, terms = state_and_terms(sample["joint_q"], sample["joint_dq"], sample["imu_quat_wxyz"], sample["gyro_body"], action, contract["default_q"])
    goal, _, _ = received_goal(policy, samples, measured_qpos, 1.0, 2.0)
    if phases is not None:
        phases["feature_goal_ms"] = (time.perf_counter_ns() - phase_start) / 1_000_000.0
    phase_start = time.perf_counter_ns()
    raw = policy.actor(
        torch.as_tensor(sensed[None], device=policy.device), torch.as_tensor(action[None], device=policy.device),
        torch.as_tensor(history.before_update(terms)[None], device=policy.device), goal.to(policy.device))[0].cpu().numpy()
    if phases is not None:
        phases["bfm_inference_ms"] = (time.perf_counter_ns() - phase_start) / 1_000_000.0
    phase_start = time.perf_counter_ns()
    action = raw * 5.0
    requested = contract["default_q"] + action * 0.25 * contract["training_effort"] / contract["kp"]
    target = brake.apply(requested, sample["joint_q"], sample["joint_dq"])
    if phases is not None:
        phases["brake_ms"] = (time.perf_counter_ns() - phase_start) / 1_000_000.0
    gate.consume()
    return action, target


def lockstep_targets(sensor, motion, model, contract, policy, native_snapshots=None):
    estimator = Native23IMUOdometry(model)
    brake = BoundedAnkleBrake(model, JOINT_LIMIT_BRAKE_STEP_RAD)
    gate = PacketGate(contract, stale_seconds=0.1)
    history, action = BFMHistory(), np.zeros(23, np.float32)
    target = np.asarray(contract["default_q"], np.float32).copy()
    emitted = []
    for index in range(len(sensor["timestamp_s"])):
        sample = {key: sensor[key][index] for key in sensor}
        estimator_result = None if native_snapshots is not None else estimator.update(**sample)
        if index % CONTROL_TICKS:
            continue
        sequence = index // CONTROL_TICKS
        if native_snapshots is not None:
            estimator_result = native_snapshots[sequence]
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
    if args.samples is not None:
        sensor = {key: value[:args.samples] for key, value in sensor.items()}
    motion, _, _ = load_motion("pico")
    contract = load_contract(PACKAGE / "bfmzero_inspect_v1/config.yaml")
    _, model, _ = prepare_true23_model(ROOT.parent / "GR00T-WholeBodyControl" / MODEL, ROOT / PHYSICS)
    weights = PACKAGE / "bfmzero_inference_v1/inference.safetensors"
    torch.set_num_threads(args.torch_threads)
    native, _ = load_native_snapshots(args.native_snapshots)
    single = lockstep_targets(sensor, motion, model, contract, BFMZeroInference(weights, "cpu"))
    split = lockstep_targets(sensor, motion, model, contract, BFMZeroInference(weights, "cpu"), native_snapshots=native)
    differences = np.max(np.abs(single - split), axis=1)
    maximum = float(differences.max())
    first = int(np.flatnonzero(differences != 0.0)[0]) if np.any(differences != 0.0) else None
    result = {"kind": "g1_true23_bfm_fix7_lockstep_target_parity_v1", "passed": maximum <= 1e-6, "targets": int(len(single)),
              "maximum_absolute_difference": maximum, "first_differing_control_tick": first}
    args.output.write_text(json.dumps(result, indent=2))
    if not result["passed"]:
        raise RuntimeError("C2 failed: " + json.dumps(result))
    print(json.dumps(result, indent=2))


def replay_server(args):
    """Local 500 Hz state source for Windows policy scheduling experiments.

    This deliberately contains no DDS or robot interface.  It replays the
    immutable low-state and Fix 7 native-estimator snapshots over localhost
    TCP, while draining targets solely to avoid artificial ZMQ backpressure.
    """
    if args.output.exists():
        raise FileExistsError("Fix 8 local replay evidence refuses overwrite")
    sensor = load_sensor(args.sensor_stream)
    snapshots, _ = load_native_snapshots(args.native_snapshots)
    ticks = min(len(sensor["timestamp_s"]), len(snapshots) * CONTROL_TICKS, round(args.duration_seconds / LOWSTATE_DT))
    if ticks != round(args.duration_seconds / LOWSTATE_DT) or len(snapshots) < (ticks + CONTROL_TICKS - 1) // CONTROL_TICKS:
        raise ValueError("replay and snapshot inputs do not cover the requested duration")
    args.output.mkdir(parents=True)
    context = zmq.Context.instance()
    states, targets = context.socket(zmq.PUSH), context.socket(zmq.PULL)
    for socket in (states, targets):
        socket.linger = 0
    states.setsockopt(zmq.SNDHWM, 65536)
    targets.setsockopt(zmq.RCVHWM, 65536)
    states.bind(args.state_endpoint)
    targets.bind(args.target_endpoint)
    sent = received = 0
    work_ms = []
    try:
        with SimulationPacer("windows-high-resolution" if os.name == "nt" else "sleep") as pacer:
            start = time.perf_counter()
            for index in range(ticks):
                due = start + index * LOWSTATE_DT
                pacer.sleep_until(due)
                begun = time.perf_counter()
                sample = [float(sensor["timestamp_s"][index]), *np.asarray(sensor["joint_q"][index], np.float32),
                          *np.asarray(sensor["joint_dq"][index], np.float32), *np.asarray(sensor["imu_quat_wxyz"][index], np.float32),
                          *np.asarray(sensor["gyro_body"][index], np.float32), *np.asarray(sensor["accel_specific_force_body"][index], np.float32)]
                snapshot = snapshots[index // CONTROL_TICKS]
                snapshot_values = [*snapshot["position_start"], *snapshot["velocity_start"], *snapshot["accel_bias_body"],
                                   *snapshot["quaternion_start"], *snapshot["support_weights"], snapshot["timestamp_s"]]
                states.send(STATE_WIRE.pack(STATE_MAGIC, 2, index, time.perf_counter_ns(), time.time_ns(), *sample, *snapshot_values))
                sent += 1
                while targets.poll(0):
                    targets.recv()
                    received += 1
                work_ms.append((time.perf_counter() - begun) * 1000.0)
    finally:
        states.close(); targets.close()
    report = {"kind": "g1_true23_bfm_fix8_local_state_replay_v1", "duration_seconds": args.duration_seconds,
              "sent_lowstates": sent, "targets_received": received, "work_ms": summary_ms(work_ms),
              "no_dds": True, "no_robot_network": True}
    (args.output / "replay_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


def policy_process(args):
    if args.output.exists():
        raise FileExistsError("Fix 5 policy evidence refuses overwrite")
    for name, endpoint in (("state", args.state_endpoint), ("target", args.target_endpoint)):
        if not endpoint.startswith("tcp://"):
            raise ValueError(f"Fix 5 {name} endpoint must be TCP, got {endpoint!r}")
    args.output.mkdir(parents=True)
    if args.omp_wait_policy == "passive":
        os.environ["OMP_WAIT_POLICY"] = "PASSIVE"
    elif args.omp_wait_policy == "default":
        os.environ.pop("OMP_WAIT_POLICY", None)
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
    estimator = Native23IMUOdometry(model) if args.estimator_source == "python" else None
    brake = BoundedAnkleBrake(model, args.brake_step_rad)
    gate = PacketGate(contract, stale_seconds=0.1)
    teleop_clock = TeleopClockAlignment()
    teleop_gate_audit = GateAdmissionAudit()
    history, action = BFMHistory(), np.zeros(23, np.float32)
    pending_states, pending_teleop, prefetched_teleop = [], [], []
    expected_sequence = 0
    state_gaps = state_decode_errors = teleop_rejected = teleop_accepted = target_eagain = 0
    path = {"start_lateness_ms": [], "work_ms": [], "deadline_misses": 0, "policy_updates": 0}
    warmup = {"performed": False}
    telemetry = WindowsPolicyTelemetry(args.trace_windows)
    run_counters_before = telemetry.expensive_run_counters() if args.trace_windows else None
    if args.trace_windows:
        path["controls"] = []
    activity_capture = WindowsActivityCapture(args.output / "outlier_activity") if args.trace_windows else None
    outlier_active = False
    # The policy must have its ZMQ connects established before the timing source
    # begins.  This wait is deliberately outside measurement.
    if args.ready_file is not None:
        args.ready_file.parent.mkdir(parents=True, exist_ok=True)
        args.ready_file.write_text(json.dumps({"pid": os.getpid(), "state": "initialized"}) + "\n")
    if not states.poll(60_000):
        raise TimeoutError("no lowstate sample arrived from native loop")
    first_message = states.recv()
    gc_enabled = configure_policy_runtime(args.priority)
    try:
        with WindowsRealtimeScope("consumer", "normal" if args.priority == "mmcss" else args.priority, args.affinity), WindowsMmcssScope(args.priority == "mmcss"), WindowsTimerResolutionScope(args.timer_resolution_0_5ms) as timer_resolution, WindowsMemoryResidencyScope(args.memory_residency, policy, contract, history, action) as memory_residency, SimulationPacer("windows-high-resolution" if os.name == "nt" else "sleep", spin_margin_s=args.pacer_spin_ms / 1000.0) as pacer:
            # The first control through this path pays every lazy initialisation
            # cost at once - torch's first inference allocations, MuJoCo's first
            # kinematics call, and ZMQ's first receive - and measured 23.5 ms
            # against a 20 ms budget, whose overrun then left the following
            # control 14.4 ms late.  Those were the only two controls in a
            # 5,780-control run that missed anything.  Pay the cost here instead,
            # outside the measured region, driving one complete control on
            # throwaway copies so that no observable state is touched: the real
            # history, action, brake, gate and estimator are all left untouched,
            # and nothing is sent.
            # The measurement machinery itself is first-call expensive: at
            # control 0 the phases accounted for only 4.1 ms of a 23.5 ms
            # control, and the gate was not yet ready, so no inference ran at
            # all.  Pay the ctypes and query costs here.
            for _ in range(3):
                telemetry.sample()
            try:
                warmup_values = STATE_WIRE.unpack(first_message)
                if warmup_values[0] == STATE_MAGIC and warmup_values[1] == 2:
                    warmup_sample = sample_from_values(warmup_values[5:62])
                    if args.estimator_source == "native":
                        warmup_estimate = snapshot_from_values(warmup_values[62:])
                    else:
                        warmup_estimate = Native23IMUOdometry(model).update(**warmup_sample)
                    warmup_history = BFMHistory()
                    warmup_history.data = {name: values.copy() for name, values in history.data.items()}
                    # policy_step itself returns immediately until the admission
                    # gate is ready, which is why the first real inference landed
                    # on control 7 and cost 59 ms there.  Drive the actor directly
                    # with correctly shaped inputs so that every kernel it uses is
                    # allocated now.  The goal is the actor's 256-wide z vector.
                    warmup_action = action.copy()
                    warmup_sensed, warmup_terms = state_and_terms(
                        warmup_sample["joint_q"], warmup_sample["joint_dq"], warmup_sample["imu_quat_wxyz"],
                        warmup_sample["gyro_body"], warmup_action, contract["default_q"])
                    warmup_raw = policy.actor(
                        torch.as_tensor(warmup_sensed[None], device=policy.device),
                        torch.as_tensor(warmup_action[None], device=policy.device),
                        torch.as_tensor(warmup_history.before_update(warmup_terms)[None], device=policy.device),
                        torch.zeros((1, 256), dtype=torch.float32, device=policy.device))[0].cpu().numpy()
                    warmup_requested = contract["default_q"] + (warmup_raw * 5.0) * 0.25 * contract["training_effort"] / contract["kp"]
                    warmup_target = BoundedAnkleBrake(model, args.brake_step_rad).apply(
                        warmup_requested, warmup_sample["joint_q"], warmup_sample["joint_dq"])
                    TARGET_WIRE.pack(TARGET_MAGIC, 1, 0, 0, 0, *np.asarray(warmup_target, np.float32),
                                     *np.asarray(contract["kp"], np.float32),
                                     *np.asarray(contract["kd"], np.float32))
                    warmup["performed"] = True
            except (ValueError, struct.error, KeyError, TypeError) as failure:
                warmup["error"] = f"{type(failure).__name__}: {failure}"
            # The warm-up above takes time, and lowstates keep arriving while it
            # runs, so control 0 would otherwise open by decoding the whole
            # backlog that accumulated during it.  Drop to the newest state, which
            # is what a control uses anyway: the receive loop keeps only the last
            # sample of the batch.  With the Python estimator this is not safe,
            # because that estimator integrates every sample, so the backlog is
            # left in place in that mode.
            if args.estimator_source == "native":
                while states.poll(0):
                    first_message = states.recv()
                    warmup["states_discarded"] = warmup.get("states_discarded", 0) + 1
            # Teleop packets queue up the same way, and decoding them was the
            # largest unmeasured cost left in control 0: 50.9 ms of work with
            # every phase at zero.  Decode whatever is already waiting now and
            # hand it to control 0, which admits it at its own clock.
            while teleop.poll(0):
                try:
                    prefetched_teleop.append(decode_packet(teleop.recv_json()))
                except (ValueError, TypeError, KeyError):
                    teleop_rejected += 1
            warmup["teleop_prefetched"] = len(prefetched_teleop)
            start = time.perf_counter()
            for control in range(round(args.duration_seconds / DT)):
                due = start + control * DT
                pacer.sleep_until(due)
                begun = time.perf_counter()
                counters_before = telemetry.sample() if args.trace_windows else None
                phases = {"state_receive_ms": 0.0, "teleop_receive_ms": 0.0, "feature_goal_ms": 0.0,
                          "bfm_inference_ms": 0.0, "brake_ms": 0.0, "target_send_ms": 0.0}
                state_phase_start = time.perf_counter_ns()
                path["start_lateness_ms"].append(max(0.0, (begun - due) * 1000.0))
                messages = [first_message] if control == 0 else []
                first_message = None
                while states.poll(0):
                    messages.append(states.recv())
                for message in messages:
                    try:
                        values = STATE_WIRE.unpack(message)
                        if values[0] != STATE_MAGIC or values[1] != 2:
                            raise ValueError("wire magic")
                        sequence = values[2]
                        if sequence != expected_sequence:
                            state_gaps += abs(sequence - expected_sequence)
                            expected_sequence = sequence
                        expected_sequence += 1
                        pending_states.append((sequence, sample_from_values(values[5:62]), snapshot_from_values(values[62:]), values[3], time.perf_counter_ns()))
                    except (ValueError, struct.error):
                        state_decode_errors += 1
                estimator_result = None
                latest_sequence = 0
                sample = None
                latest_state_sent_mono = latest_policy_receive_mono = None
                for latest_sequence, sample, native_snapshot, latest_state_sent_mono, latest_policy_receive_mono in pending_states:
                    estimator_result = native_snapshot if args.estimator_source == "native" else estimator.update(**sample)
                pending_states.clear()
                phases["state_receive_ms"] = (time.perf_counter_ns() - state_phase_start) / 1_000_000.0
                now = control * DT
                teleop_phase_start = time.perf_counter_ns()
                # Packets decoded during the warm-up are admitted here, at this
                # control's clock, so that the epoch anchoring is exactly what it
                # would have been had they been decoded in this loop.
                for packet in prefetched_teleop:
                    pending_teleop.append((
                        packet, teleop_clock.observe(packet, now, gate), teleop_clock.gate_source_time_offset(gate)
                    ))
                prefetched_teleop.clear()
                while teleop.poll(0):
                    try:
                        packet = decode_packet(teleop.recv_json())
                        pending_teleop.append((
                            packet, teleop_clock.observe(packet, now, gate), teleop_clock.gate_source_time_offset(gate)
                        ))
                    except (ValueError, TypeError, KeyError):
                        teleop_rejected += 1
                phases["teleop_receive_ms"] = (time.perf_counter_ns() - teleop_phase_start) / 1_000_000.0
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
                    action, target = policy_step(policy, contract, estimator_result, sample, history, action, brake, gate, phases)
                    if target is not None:
                        phase_start = time.perf_counter_ns()
                        output_policy_mono = time.perf_counter_ns()
                        payload = TARGET_WIRE.pack(TARGET_MAGIC, 1, latest_sequence, latest_policy_receive_mono, output_policy_mono, *np.asarray(target, np.float32),
                                                   *np.asarray(contract["kp"], np.float32), *np.asarray(contract["kd"], np.float32))
                        if targets.send(payload, flags=zmq.DONTWAIT) is None:
                            pass
                        phases["target_send_ms"] = (time.perf_counter_ns() - phase_start) / 1_000_000.0
                        path["policy_updates"] += 1
                ended = time.perf_counter()
                work_ms = (ended - begun) * 1000.0
                missed = bool(ended > due + DT)
                path["work_ms"].append(work_ms)
                path["deadline_misses"] += int(missed)
                if args.trace_windows:
                    counters_after = telemetry.sample()
                    path["controls"].append({"control": control, "scheduled_offset_ms": control * DT * 1000.0,
                                             "start_lateness_ms": path["start_lateness_ms"][-1], "work_ms": work_ms,
                                             "deadline_missed": missed, "phases": phases,
                                             "windows": WindowsPolicyTelemetry.delta(counters_before, counters_after)})
                    outlier = missed or work_ms >= 18.0
                    if outlier and not outlier_active:
                        episode = activity_capture.capture(control)
                        episode.update({"start_lateness_ms": path["start_lateness_ms"][-1], "work_ms": work_ms,
                                        "deadline_missed": missed})
                        path.setdefault("outlier_activity", []).append(episode)
                    outlier_active = outlier
    except zmq.Again:
        target_eagain += 1
        raise RuntimeError("target IPC queue overflow; native loop must never be blocked")
    finally:
        restore_gc(gc_enabled)
        if activity_capture is not None:
            activity_capture.close()
        run_counters_after = telemetry.expensive_run_counters() if args.trace_windows else None
        telemetry.close()
        states.close(); targets.close(); teleop.close()
    run_counters = None
    if run_counters_before is not None and run_counters_after is not None:
        run_counters = {name + "_delta": int(run_counters_after[name] - run_counters_before[name]) for name in run_counters_before}
        # A zero hard-fault total at the run bounds proves no hard fault occurred
        # in any control, so total-fault deltas are soft-fault deltas per tick.
        if run_counters["hard_faults_delta"] == 0:
            for record in path.get("controls", []):
                if record["windows"] is not None:
                    record["windows"]["hard_faults_delta"] = 0
                    record["windows"]["soft_faults_delta"] = record["windows"].pop("total_faults_delta")
        else:
            for record in path.get("controls", []):
                if record["windows"] is not None:
                    record["windows"]["hard_faults_delta"] = None
                    record["windows"]["soft_faults_delta"] = None
    mean_work_ms = float(np.mean(path["work_ms"])) if path["work_ms"] else 0.0
    for field in ("start_lateness_ms", "work_ms"):
        values = path[field]
        path[field] = summary_ms(values)
        if field == "work_ms":
            path[field]["p99"] = float(np.percentile(values, 99)) if values else float("nan")
    path["duty_cycle_percent"] = 100.0 * mean_work_ms / (DT * 1000.0)
    report = {"kind": "g1_true23_bfm_fix8_policy_50hz_v1", "placement": args.placement, "duration_seconds": args.duration_seconds,
              "path_50hz": path, "first_control_warmup": warmup,
              "state_sequence_gaps": state_gaps, "state_decode_errors": state_decode_errors,
              "teleop_packets_accepted": teleop_accepted, "teleop_packets_rejected": teleop_rejected,
              "teleop_packets_gate_rejected": teleop_gate_audit.rejected,
              "teleop_gate_rejection_reasons": teleop_gate_audit.reasons,
              "teleop_clock_alignment": {"method": "first_packet_reception_offset_per_gate_epoch",
                                         "anchors": teleop_clock.anchors},
              "target_send_eagain": target_eagain, "target_age_clock": "NTP four-timestamp mapping between loop and policy monotonic clocks; valid across machines",
              "brake_step_rad": args.brake_step_rad,
              "estimator_source": args.estimator_source,
              "brake_engagement_controls": brake.engagements, "brake_target_step_max_rad": brake.maximum_step,
              "windows_telemetry": {"enabled": bool(args.trace_windows), "available": telemetry.available,
                                    "note": telemetry.note,
                                    "run_boundary_counters": run_counters,
                                    "context_switch_semantics": "Windows SystemProcessInformation exposes only total switches; voluntary/involuntary values are intentionally null unless ETW is used."},
              "timer_resolution": timer_resolution.report() if 'timer_resolution' in locals() else None,
              "memory_residency": memory_residency.report() if 'memory_residency' in locals() else None,
              "runtime_configuration": {"priority": args.priority, "torch_threads": args.torch_threads,
                                        "omp_wait_policy": args.omp_wait_policy, "affinity": args.affinity},
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
    c2.add_argument("--native-snapshots", type=Path, required=True)
    c2.add_argument("--samples", type=int)
    c2.add_argument("--data-root", type=Path, required=True)
    c2.add_argument("--output", type=Path, required=True)
    c2.add_argument("--torch-threads", type=int, default=4)
    c2.set_defaults(function=check_lockstep)
    g1 = sub.add_parser("check-native-estimator")
    g1.add_argument("--sensor-stream", type=Path, required=True)
    g1.add_argument("--native-snapshots", type=Path, required=True)
    g1.add_argument("--output", type=Path, required=True)
    g1.add_argument("--samples", type=int)
    g1.set_defaults(function=check_native_estimator)
    local = sub.add_parser("local-replay", help="localhost-only 500 Hz state replay for Windows scheduling measurements")
    local.add_argument("--sensor-stream", type=Path, required=True)
    local.add_argument("--native-snapshots", type=Path, required=True)
    local.add_argument("--state-endpoint", required=True)
    local.add_argument("--target-endpoint", required=True)
    local.add_argument("--output", type=Path, required=True)
    local.add_argument("--duration-seconds", type=float, default=115.6)
    local.set_defaults(function=replay_server)
    run = sub.add_parser("run")
    run.add_argument("--state-endpoint", required=True)
    run.add_argument("--target-endpoint", required=True)
    run.add_argument("--teleop-endpoint", required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--duration-seconds", type=float, default=115.6)
    run.add_argument("--placement", choices=("wsl", "windows"), required=True)
    run.add_argument("--priority", choices=("normal", "high", "realtime", "mmcss"), default="high")
    run.add_argument("--torch-threads", type=int, default=4)
    run.add_argument("--omp-wait-policy", choices=("default", "passive"), default="default")
    run.add_argument("--affinity", choices=("none", "role-separated"), default="none",
                     help="keep this process off its first allowed logical CPU; restored on exit")
    run.add_argument("--memory-residency", action="store_true",
                     help="VirtualLock static timed-loop buffers for this process only")
    run.add_argument("--estimator-source", choices=("native", "python"), default="native")
    run.add_argument("--brake-step-rad", type=float, default=JOINT_LIMIT_BRAKE_STEP_RAD)
    run.add_argument("--trace-windows", action="store_true", help="record per-control Windows fault, scheduler, and phase evidence")
    run.add_argument("--timer-resolution-0-5ms", action="store_true", help="request 0.5 ms NT timer resolution for this process only")
    run.add_argument("--pacer-spin-ms", type=float, default=0.0,
                     help="busy-wait over this final part of each control wait, keeping the core awake")
    run.add_argument("--ready-file", type=Path, help="write after the policy subscriber and model are initialized, before the first state")
    run.set_defaults(function=policy_process)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
