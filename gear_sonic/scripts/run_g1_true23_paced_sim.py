"""Measured 50-Hz localhost PICO-reference consumer, SIM ONLY.

The caller supplies the source length for a bounded recorded-input trial. A
40-ms startup buffer changes arrival-to-control latency, not source speed or
motion values. After the requested source length, input is deliberately ended
and the existing balance controller runs for a measured, paced tail. This is
neither normal firmware handback nor a physically deployable robot launcher.
"""

import argparse
from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path
import time

import numpy as np
import zmq

from gear_sonic.scripts.record_g1_true23_saved_teleop_diagnostic import (
    build_recording_controller,
    preserve_calibrated_source_orientation,
)
from gear_sonic.teleop.clocked_sim_session import SubstepObservation
from gear_sonic.teleop.paced_sim_runtime import LoopbackSubInbox, PacedSimSession
from gear_sonic.utils.g1_23dof_contract import HARDWARE_23_JOINT_NAMES
from gear_sonic.utils.g1_true23_clean_mujoco_teleop import sha256_file
from gear_sonic.utils.g1_true23_frozen_lora_live_teleop import (
    LiveTransportFault,
    initialize_live_controller,
    validate_live_packet,
)
from gear_sonic.utils.g1_true23_generalist_benchmark import MODEL, PHYSICS
from gear_sonic.utils.g1_true23_reference_floor import compiled_model_sha256
from gear_sonic.utils.g1_true23_step1b_mujoco import _projected_gravity


@dataclass(frozen=True)
class PreparedPacedSim:
    controller: object
    model_hash: str


def prepare_stream(controller):
    """Complete expensive model serialization BEFORE subscribing to any stream."""
    if controller.completed != 0 or float(controller.data.time) != 0:
        raise ValueError("paced input requires a fresh, explicitly prepared simulation")
    return PreparedPacedSim(controller, compiled_model_sha256(controller.model))


def run_stream(prepared, inbox, *, source_controls, tail_controls, first_control_index=None):
    if not isinstance(prepared, PreparedPacedSim):
        raise ValueError("prepare the simulation before subscribing to source packets")
    controller, model_hash = prepared.controller, prepared.model_hash
    if type(source_controls) is not int or not 2 <= source_controls <= 10000:
        raise ValueError("bounded source requires2..10000controls")
    if type(tail_controls) is not int or not 1 <= tail_controls <= 1000:
        raise ValueError("bounded balance tail requires1..1000controls")
    first, second = inbox.startup_pair()
    first_summary = validate_live_packet(first, previous=None, maximum_age_ns=100_000_000)
    validate_live_packet(second, previous=first_summary, maximum_age_ns=100_000_000)
    if first_control_index is not None and first_summary["control_index"] != first_control_index:
        raise LiveTransportFault("gap", "recorded-input trial lost its required first source frame")
    preserve_calibrated_source_orientation(controller, first)
    initialize_live_controller(controller, first, second)
    observation = SubstepObservation(controller.module)
    controller.module = observation
    initial_packets = deque((first, second))
    runtime = PacedSimSession(controller, start_ns=first_summary["control_monotonic_ns"] + 40_000_000)
    qpos, qvel, sim_time = (
        [controller.data.qpos.copy()],
        [controller.data.qvel.copy()],
        [float(controller.data.time)],
    )
    modes, used_indices, source_times = [], [], []
    failure = None

    def capture():
        # Retain failed attempts, including partial physics, rather than hiding
        # their state behind the last successful full control.
        qpos.append(controller.data.qpos.copy())
        qvel.append(controller.data.qvel.copy())
        sim_time.append(float(controller.data.time))
        modes.append(bool(controller.fallback_active))

    for tick in range(source_controls + tail_controls):
        inbox.wait_until(runtime.next_deadline_ns)
        item = initial_packets.popleft() if initial_packets else inbox.pop()
        if tick >= source_controls:
            # Explicit bounded source end, even if a live publisher sends more.
            # This tail cannot be scored as following those additional frames.
            item = None
        error = item if isinstance(item, LiveTransportFault) else None
        packet = None if error is not None else item
        used_indices.append(packet.get("control_source_frame_index", -1) if isinstance(packet, dict) else -1)
        source_times.append(packet.get("control_monotonic_ns", -1) if isinstance(packet, dict) else -1)
        try:
            runtime.tick(packet, input_fault=error, capture=capture)
        except (RuntimeError, ValueError) as exc:
            failure = f"{type(exc).__name__}: {exc}"
            break
    if failure is None:
        inbox.wait_until(runtime.next_deadline_ns)
    wall_end = time.monotonic_ns()
    arrays = dict(
        qpos=np.asarray(qpos),
        qvel=np.asarray(qvel),
        simulation_time=np.asarray(sim_time),
        fallback_mode=np.asarray(modes, dtype=bool),
        source_index=np.asarray(used_indices, dtype=np.int64),
        source_timestamp_ns=np.asarray(source_times, dtype=np.int64),
        **observation.arrays(),
        **{key: np.asarray([row[key] for row in runtime.rows]) for key in runtime.rows[0]},
    )
    finite = all(np.isfinite(value).all() for value in arrays.values())
    ranges = controller.model.jnt_range[1:]
    joint = arrays["physics_joint_pos"]
    excess = np.maximum(np.maximum(ranges[:, 0] - joint, joint - ranges[:, 1]), 0)
    torque_excess = np.maximum(np.abs(arrays["physics_command_torque"]) - controller.physics.effort, 0)
    tilts = [float(np.arccos(np.clip(-_projected_gravity(q[3:7])[2], -1, 1))) for q in qpos] if finite else []
    complete = controller.completed == source_controls + tail_controls
    uninterrupted = bool(
        complete
        and len(joint) == controller.completed * 10
        and np.allclose(arrays["simulation_time"], np.arange(controller.completed + 1) * 0.02, rtol=0, atol=1e-8)
        and np.allclose(
            arrays["physics_time"], np.arange(controller.completed * 10) * 0.002 + 0.002, rtol=0, atol=1e-8
        )
    )
    same_model = compiled_model_sha256(controller.model) == model_hash
    physical_screen = bool(
        failure is None
        and uninterrupted
        and finite
        and same_model
        and min(q[2] for q in qpos) >= 0.45
        and max(tilts, default=99) <= 1.0
        and np.max(excess, initial=0) <= 1e-6
        and np.max(torque_excess, initial=0) <= 1e-8
    )
    timing = runtime.rows
    age = arrays["started_ns"] - arrays["source_timestamp_ns"]
    admitted = (arrays["source_timestamp_ns"] >= 0) & ~arrays["fallback_mode"]
    report = dict(
        kind="native23_wall_clock_paced_loopback_sim_v1",
        requested_source_controls=source_controls,
        requested_balance_tail_controls=tail_controls,
        completed_controls=controller.completed,
        attempted_controls=len(timing),
        completed_sonic_controls=runtime.session.sonic_controls,
        fallback_controls=int(np.count_nonzero(modes)),
        full_source_sonic_completed=runtime.session.sonic_controls == source_controls,
        input_fault=runtime.session.fault,
        input_fault_tick=runtime.session.fault_tick,
        input_fault_detail=runtime.session.fault_detail,
        runtime_fault=runtime.runtime_fault,
        runtime_fault_tick=runtime.runtime_fault_tick,
        controller_failure=failure,
        later_packets_ignored_after_latch=runtime.session.packets_ignored_after_latch,
        physical_screen_passed=physical_screen,
        uninterrupted_500hz_virtual_physics=uninterrupted,
        measured_compute_deadlines_passed=runtime.runtime_fault is None and complete,
        minimum_height_m=float(min(q[2] for q in qpos)) if finite else None,
        maximum_tilt_rad=max(tilts, default=None),
        maximum_joint_range_excess_rad=float(np.max(excess, initial=0)) if finite else None,
        maximum_command_effort_excess_nm=float(np.max(torque_excess, initial=0)) if finite else None,
        final_maximum_joint_speed_rad_s=float(np.max(np.abs(qvel[-1][6:]))) if finite else None,
        maximum_execution_ms=max(row["execution_ns"] for row in timing) / 1e6,
        execution_p95_ms=float(np.percentile(arrays["execution_ns"], 95)) / 1e6,
        maximum_wake_lateness_ms=max(row["wake_lateness_ns"] for row in timing) / 1e6,
        missed_compute_deadlines=int(np.count_nonzero(arrays["missed_compute_deadline"])),
        maximum_admitted_packet_age_ms=float(age[admitted].max()) / 1e6 if np.any(admitted) else None,
        wall_start_ns=timing[0]["started_ns"],
        wall_end_ns=wall_end,
        measured_wall_duration_s=(wall_end - timing[0]["started_ns"]) / 1e9,
        initial_buffer_ms=40,
        maximum_packet_age_ms=100,
        control_period_ms=20,
        packet_time_domain="producer_and_consumer_same_host_monotonic_ns_required",
        received_messages=len(inbox.receipts),
        maximum_inbox_depth=inbox.maximum_depth,
        model_compiled_sha256=model_hash,
        physical_model_unchanged=same_model,
        physical_dof=23,
        joint_names=list(HARDWARE_23_JOINT_NAMES),
        source_pose_initialization=True,
        ordinary_standing_acquisition_tested=False,
        sonic_reentry_tested=False,
        firmware_handback_tested=False,
        simulated_500hz_substeps_batched_within_each_wall_clock_50hz_control=True,
        operating_system_hard_realtime_claimed=False,
        tracking_qualified=False,
        deployment_ready=False,
        hardware_authorized=False,
        robot_commands_sent=False,
    )
    return arrays, report


def save_result(directory, arrays, report, inbox):
    with (directory / "trace.npz").open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    report["trace_sha256"] = sha256_file(directory / "trace.npz")
    with (directory / "received.json").open("x") as stream:
        json.dump(inbox.receipts, stream, indent=2, allow_nan=False)
    report["received_sha256"] = sha256_file(directory / "received.json")
    with (directory / "report.json").open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repository-root", "encoder-report", "decoder-report", "output-directory"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--source-controls", type=int, required=True)
    parser.add_argument("--tail-controls", type=int, default=250)
    parser.add_argument("--first-control-index", type=int)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    args.legacy_unpaired_diagnostic = False
    args.output_directory.mkdir(parents=True, exist_ok=False)
    controller, identity = build_recording_controller(args)
    prepared = prepare_stream(controller)
    context = zmq.Context(io_threads=1)
    inbox = LoopbackSubInbox(context, args.endpoint)
    try:
        arrays, report = run_stream(
            prepared,
            inbox,
            source_controls=args.source_controls,
            tail_controls=args.tail_controls,
            first_control_index=args.first_control_index,
        )
        report.update(
            policy_identity=identity,
            endpoint=args.endpoint,
            model_sha256=sha256_file(args.repository_root / MODEL),
            physics_sha256=sha256_file(args.repository_root / PHYSICS),
        )
        save_result(args.output_directory, arrays, report, inbox)
        print(json.dumps(report, sort_keys=True, allow_nan=False))
        return 0 if report["physical_screen_passed"] and report["measured_compute_deadlines_passed"] else 1
    except Exception as error:
        with (args.output_directory / "failure.json").open("x") as stream:
            json.dump(
                dict(
                    error=f"{type(error).__name__}: {error}",
                    deployment_ready=False,
                    hardware_authorized=False,
                    completed_controls=controller.completed,
                ),
                stream,
                indent=2,
            )
        raise
    finally:
        inbox.close()
        context.term()


if __name__ == "__main__":
    raise SystemExit(main())
