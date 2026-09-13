"""Replay complete existing PICO-derived clips and fault schedules in native23 SIM.

No network, wall-clock simulation claim, future motion fabrication or physical
robot interface. Missing input advances the existing balance actor every tick,
not a frozen plant followed by a separate unrecorded hold. Fault recovery here
means balance fallback only; automatic SONIC reacquisition is forbidden.
"""

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from gear_sonic.scripts.record_g1_true23_saved_teleop_diagnostic import (
    build_recording_controller,
    preserve_calibrated_source_orientation,
)
from gear_sonic.scripts.replay_g1_true23_pico_packets_zmq import load_reference_packets
from gear_sonic.teleop.clocked_sim_session import ClockedSimSession, SubstepObservation
from gear_sonic.utils.g1_23dof_contract import HARDWARE_23_JOINT_NAMES
from gear_sonic.utils.g1_true23_clean_mujoco_teleop import sha256_file
from gear_sonic.utils.g1_true23_frozen_lora_live_teleop import CONTROL_PERIOD_NS, initialize_live_controller
from gear_sonic.utils.g1_true23_generalist_benchmark import MODEL, PHYSICS
from gear_sonic.utils.g1_true23_reference_floor import compiled_model_sha256
from gear_sonic.utils.g1_true23_step1b_mujoco import _projected_gravity

SCENARIOS = ("nominal", "pause", "gap", "payload", "stale-start", "end-of-stream")


def scheduled_packet(packets, index, *, scenario, fault_control, pause_controls):
    if index >= len(packets) or (scenario == "pause" and fault_control <= index < fault_control + pause_controls):
        return None
    packet = copy.deepcopy(packets[index])
    if index == fault_control:
        if scenario == "gap":
            packet["pico_anchor_source_frame_index"] += 1
            packet["control_source_frame_index"] += 1
        elif scenario == "payload":
            packet["q_ref23_native"] = packet["q_ref23_native"][:-1]
    return packet


def run_case(controller, packets, *, scenario, fault_control, pause_controls, tail_controls):
    if scenario not in SCENARIOS or len(packets) < 2:
        raise ValueError("requires a known scenario and at least two source packets")
    if type(fault_control) is not int or not 0 <= fault_control < len(packets):
        raise ValueError("fault control must identify an original source tick")
    if type(pause_controls) is not int or not 1 <= pause_controls <= len(packets) - fault_control:
        raise ValueError("pause must be positive and cannot extend beyond the original source")
    if type(tail_controls) is not int or not 1 <= tail_controls <= 1000:
        raise ValueError("EOF fallback tail must be bounded to 1..1000 controls")
    physical_hash = compiled_model_sha256(controller.model)
    observe = SubstepObservation(controller.module)
    controller.module = observe
    preserve_calibrated_source_orientation(controller, packets[0])
    initialize_live_controller(controller, packets[0], packets[1])
    # The first two source samples are received before initialization, exactly
    # as in the existing live consumer: first control is one sample (20 ms) old.
    start_ns = packets[1]["control_monotonic_ns"] + (200_000_000 if scenario == "stale-start" else 0)
    session = ClockedSimSession(controller, start_ns=start_ns)
    requested = len(packets) + (tail_controls if scenario == "end-of-stream" else 0)
    qpos, qvel, times = [controller.data.qpos.copy()], [controller.data.qvel.copy()], [float(controller.data.time)]
    modes, deadlines = [], []
    failure = None
    for tick in range(requested):
        packet = scheduled_packet(
            packets, tick, scenario=scenario, fault_control=fault_control, pause_controls=pause_controls
        )
        before = float(controller.data.time)
        deadline = start_ns + tick * CONTROL_PERIOD_NS
        try:
            session.tick(packet, deadline_ns=deadline)
        except RuntimeError as error:
            failure = str(error)
        if float(controller.data.time) > before:
            qpos.append(controller.data.qpos.copy())
            qvel.append(controller.data.qvel.copy())
            times.append(float(controller.data.time))
            modes.append(int(controller.fallback_active))
            deadlines.append(deadline)
        if failure is not None:
            break
    arrays = dict(
        qpos=np.asarray(qpos),
        qvel=np.asarray(qvel),
        simulation_time=np.asarray(times),
        fallback_mode=np.asarray(modes, dtype=bool),
        deadline_ns=np.asarray(deadlines, dtype=np.int64),
        **observe.arrays(),
    )
    count = len(qpos) - 1
    if compiled_model_sha256(controller.model) != physical_hash:
        raise RuntimeError("physical model changed during replay")
    finite = all(np.isfinite(value).all() for value in arrays.values())
    tilts = [float(np.arccos(np.clip(-_projected_gravity(p[3:7])[2], -1, 1))) for p in qpos] if finite else []
    joint = arrays["physics_joint_pos"]
    ranges = controller.model.jnt_range[1:]
    excess = np.maximum(np.maximum(ranges[:, 0] - joint, joint - ranges[:, 1]), 0)
    torque_excess = np.maximum(np.abs(arrays["physics_command_torque"]) - controller.physics.effort, 0)
    uninterrupted = bool(
        count == requested
        and len(joint) == requested * 10
        and np.allclose(arrays["simulation_time"], np.arange(requested + 1) * 0.02, rtol=0, atol=1e-8)
        and np.allclose(arrays["physics_time"], np.arange(requested * 10) * 0.002 + 0.002, rtol=0, atol=1e-8)
    )
    expected_fault = dict(
        pause="timeout", gap="gap", payload="payload", **{"stale-start": "stale", "end-of-stream": "timeout"}
    ).get(scenario)
    expected_tick = (
        len(packets) if scenario == "end-of-stream" else 0 if scenario == "stale-start" else fault_control
    )
    expected_tick = None if scenario == "nominal" else expected_tick
    fault_correct = session.fault == expected_fault and session.fault_tick == expected_tick
    latch_correct = (
        bool(not any(modes))
        if expected_tick is None
        else bool(count > expected_tick and not any(modes[:expected_tick]) and all(modes[expected_tick:]))
    )
    physical_screen = bool(
        finite
        and failure is None
        and uninterrupted
        and max(tilts, default=99) <= 1.0
        and min((p[2] for p in qpos), default=0) >= 0.45
        and np.max(excess, initial=0) <= 1e-6
        and np.max(torque_excess, initial=0) <= 1e-8
    )
    return arrays, dict(
        kind="native23_clocked_saved_input_sim_v1",
        scenario=scenario,
        requested_ticks=requested,
        source_controls=len(packets),
        completed_ticks=count,
        attempted_ticks=session.ticks,
        source_sonic_controls=session.sonic_controls,
        fallback_controls=int(np.count_nonzero(modes)),
        failure=failure,
        observed_input_fault=session.fault,
        first_input_fault_tick=session.fault_tick,
        fault_detail=session.fault_detail,
        expected_fault=expected_fault,
        expected_fault_tick=expected_tick,
        fault_classification_and_timing_passed=fault_correct,
        fallback_latch_passed=latch_correct,
        uninterrupted_500hz_physics=uninterrupted,
        finite=finite,
        physical_screen_passed=physical_screen,
        scenario_screen_passed=bool(physical_screen and fault_correct and latch_correct),
        minimum_height_m=float(min(p[2] for p in qpos)) if finite else None,
        maximum_tilt_rad=max(tilts) if finite else None,
        maximum_measured_joint_range_excess_rad=float(np.max(excess, initial=0)) if finite else None,
        maximum_command_effort_excess_nm=float(np.max(torque_excess, initial=0)) if finite else None,
        joint_range_excess_by_name_rad=dict(
            zip(HARDWARE_23_JOINT_NAMES, np.max(excess, axis=0, initial=0).tolist())
        ),
        final_maximum_joint_speed_rad_s=float(np.max(np.abs(qvel[-1][6:]))) if finite else None,
        later_packets_ignored_after_latch=session.packets_ignored_after_latch,
        source_pose_initialization=True,
        ordinary_standing_acquisition_tested=False,
        sonic_reacquisition_tested=False,
        firmware_handback_tested=False,
        input_pause_controls=pause_controls if scenario == "pause" else 0,
        eof_fallback_tail_controls=tail_controls if scenario == "end-of-stream" else 0,
        missing_input_response="existing_balance_actor_on_first_missing_control_deadline",
        historical_blocking_receive_policy_changed=False,
        wall_clock_schedulability_proven=False,
        prerecorded_inputs_not_live_headset=True,
        source_motion_retimed=False,
        physical_model_compiled_sha256=physical_hash,
        physical_dof=23,
        tracking_qualified=False,
        hardware_authorized=False,
        deployment_ready=False,
        robot_commands_sent=False,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repository-root", "encoder-report", "decoder-report", "packets", "output-directory"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--scenario", choices=SCENARIOS, required=True)
    parser.add_argument("--fault-control", type=int, default=200)
    parser.add_argument("--pause-controls", type=int, default=25)
    parser.add_argument("--tail-controls", type=int, default=250)
    parser.add_argument("--baseline-report", type=Path)
    args = parser.parse_args(argv)
    if args.output_directory.exists():
        raise FileExistsError("refusing overwrite of clocked replay evidence")
    args.legacy_unpaired_diagnostic = False
    packets = load_reference_packets(args.packets)
    if not 2 <= len(packets) <= 10000:
        raise ValueError("bounded replay requires 2..10000 packets")
    controller, identity = build_recording_controller(args)
    arrays, report = run_case(
        controller,
        packets,
        scenario=args.scenario,
        fault_control=args.fault_control,
        pause_controls=args.pause_controls,
        tail_controls=args.tail_controls,
    )
    if args.baseline_report is not None:
        baseline = json.loads(args.baseline_report.read_text())
        if args.scenario != "nominal" or baseline["packets_sha256"] != sha256_file(args.packets):
            raise ValueError("baseline equality requires the same complete nominal source")
        if any(baseline[key] != identity[key] for key in ("encoder_sha256", "decoder_sha256")):
            raise ValueError("baseline policy differs")
        if sha256_file(Path(baseline["trace_path"])) != baseline["trace_sha256"]:
            raise ValueError("baseline trace hash differs")
        with np.load(baseline["trace_path"], allow_pickle=False) as archive:
            for key in ("qpos", "qvel", "simulation_time"):
                np.testing.assert_array_equal(arrays[key], archive[key])
        report.update(nominal_baseline_bit_exact=True, baseline_trace_sha256=baseline["trace_sha256"])
    args.output_directory.mkdir(parents=True, exist_ok=False)
    trace = args.output_directory / "trace.npz"
    with trace.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    report.update(
        policy=identity,
        packets=str(args.packets.resolve()),
        packets_sha256=sha256_file(args.packets),
        trace_path=str(trace.resolve()),
        trace_sha256=sha256_file(trace),
        source_files={
            str(p.resolve()): sha256_file(p)
            for p in (
                Path(__file__),
                Path(__file__).parents[1] / "teleop/clocked_sim_session.py",
                args.repository_root / MODEL,
                args.repository_root / PHYSICS,
            )
        },
    )
    with (args.output_directory / "report.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(
        json.dumps(
            {
                key: value
                for key, value in report.items()
                if key not in ("policy", "source_files", "joint_range_excess_by_name_rad")
            }
        ),
        flush=True,
    )
    return int(not report["scenario_screen_passed"])


if __name__ == "__main__":
    raise SystemExit(main())
