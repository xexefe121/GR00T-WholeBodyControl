"""Run the selected frozen-LoRA true23 policy from live PICO ZMQ in MuJoCo.

This consumer is simulator-only. It never opens DDS or a Unitree robot channel.
Transport faults latch the reviewed zero-velocity balance policy in MuJoCo.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import zmq

from gear_sonic.utils.g1_true23_frozen_lora_live_teleop import (
    EXPECTED_FAULT_TRIGGERS,
    LiveTransportFault,
    build_live_controller,
    hold_transport_fallback,
    initialize_live_controller,
    load_frozen_lora_live_profile,
    step_live_packet,
    validate_live_packet,
)


def _exclusive_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _receive(socket: zmq.Socket[bytes], poller: zmq.Poller, timeout_ms: int) -> Any:
    if not dict(poller.poll(timeout_ms)).get(socket):
        raise LiveTransportFault("timeout", "timed out waiting for a live PICO packet")
    try:
        return socket.recv_json()
    except (TypeError, ValueError, zmq.ZMQError) as error:
        raise LiveTransportFault("payload", f"invalid live PICO ZMQ payload: {error}") from error


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--decoder-report", type=Path, required=True)
    parser.add_argument("--candidate-summary", type=Path, required=True)
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5557")
    parser.add_argument("--steps", type=int, default=684)
    parser.add_argument("--startup-timeout-ms", type=int, default=10_000)
    parser.add_argument("--receive-timeout-ms", type=int, default=500)
    parser.add_argument("--maximum-age-ms", type=int, default=100)
    parser.add_argument("--fallback-hold-steps", type=int, default=100)
    parser.add_argument(
        "--expected-transport-fault",
        choices=("none", *EXPECTED_FAULT_TRIGGERS),
        default="none",
    )
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.endpoint.startswith("tcp://127.0.0.1:"):
        raise ValueError("live frozen-LoRA consumer is restricted to localhost TCP")
    if args.steps < 2:
        raise ValueError("live session requires at least two transitions")
    if args.startup_timeout_ms < args.receive_timeout_ms:
        raise ValueError("startup timeout cannot be shorter than the running receive timeout")
    if args.receive_timeout_ms < 20 or args.maximum_age_ms < 20:
        raise ValueError("transport timeout and maximum age must be at least one 50-Hz period")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite live teleop report: {output}")
    root = args.repository_root.expanduser().resolve(strict=True)
    profile = load_frozen_lora_live_profile(
        decoder_report_path=args.decoder_report,
        candidate_summary_path=args.candidate_summary,
    )
    controller = build_live_controller(repository_root=root, profile=profile)

    context = zmq.Context(io_threads=1)
    socket = context.socket(zmq.SUB)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.RCVHWM, 8)
    socket.setsockopt(zmq.SUBSCRIBE, b"")
    socket.connect(args.endpoint)
    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)
    passive = None
    initialized = False
    observed_fault: str | None = None
    fault_detail: str | None = None
    previous: dict[str, Any] | None = None
    first_index: int | None = None
    last_index: int | None = None
    received_transitions = 0
    maximum_age_ns = 0
    minimum_height = float("inf")
    maximum_tilt = 0.0
    fallback_evidence: dict[str, Any] = {
        "fallback_hold_transitions": 0,
        "fallback_minimum_base_height_m": None,
        "fallback_maximum_base_tilt_rad": None,
        "fallback_maximum_absolute_torque_nm": None,
        "fallback_policy_query_count": 0,
        "fallback_stable": False,
    }
    try:
        first = _receive(socket, poller, args.startup_timeout_ms)
        first_summary = validate_live_packet(
            first, previous=None, maximum_age_ns=args.maximum_age_ms * 1_000_000
        )
        second = _receive(socket, poller, args.startup_timeout_ms)
        second_summary = validate_live_packet(
            second,
            previous=first_summary,
            maximum_age_ns=args.maximum_age_ms * 1_000_000,
        )
        initialize_live_controller(controller, first, second)
        initialized = True
        if args.viewer:
            import mujoco.viewer

            passive = mujoco.viewer.launch_passive(controller.model, controller.data)
        for packet, summary in ((first, first_summary), (second, second_summary)):
            evidence = step_live_packet(controller, packet)
            if passive is not None:
                passive.sync()
            received_transitions += 1
            maximum_age_ns = max(maximum_age_ns, int(summary["age_ns"]))
            minimum_height = min(minimum_height, float(evidence["base_height_m"]))
            maximum_tilt = max(maximum_tilt, float(evidence["base_tilt_rad"]))
            previous = summary
            first_index = summary["control_index"] if first_index is None else first_index
            last_index = summary["control_index"]
        while received_transitions < args.steps:
            packet = _receive(socket, poller, args.receive_timeout_ms)
            summary = validate_live_packet(
                packet,
                previous=previous,
                maximum_age_ns=args.maximum_age_ms * 1_000_000,
            )
            evidence = step_live_packet(controller, packet)
            if passive is not None:
                passive.sync()
            received_transitions += 1
            maximum_age_ns = max(maximum_age_ns, int(summary["age_ns"]))
            minimum_height = min(minimum_height, float(evidence["base_height_m"]))
            maximum_tilt = max(maximum_tilt, float(evidence["base_tilt_rad"]))
            previous = summary
            last_index = summary["control_index"]
    except LiveTransportFault as error:
        observed_fault = error.fault
        fault_detail = str(error)
        if initialized:
            fallback_evidence = hold_transport_fallback(
                controller,
                trigger=error.trigger,
                steps=args.fallback_hold_steps,
            )
            if passive is not None:
                passive.sync()
    finally:
        if passive is not None:
            passive.close()
        socket.close()
        context.term()

    expected_fault = None if args.expected_transport_fault == "none" else args.expected_transport_fault
    nominal_passed = bool(
        expected_fault is None
        and observed_fault is None
        and received_transitions == args.steps
        and controller.fallback_active is False
    )
    fault_passed = bool(
        expected_fault is not None
        and observed_fault == expected_fault
        and initialized
        and controller.fallback_active is True
        and controller.fallback_trigger == EXPECTED_FAULT_TRIGGERS[expected_fault]
        and fallback_evidence["fallback_hold_transitions"] == args.fallback_hold_steps
        and fallback_evidence["fallback_stable"] is True
    )
    passed = nominal_passed or fault_passed
    fallback_evidence["fallback_policy_query_count"] = controller.fallback_policy.query_count
    fallback_transition_count = (
        0
        if controller.fallback_transition is None
        else controller.completed - controller.fallback_transition
    )
    report = {
        "schema_version": 1,
        "kind": "g1_true23_frozen_lora_live_teleop_v1",
        "mode": "live_pico_causal_zmq_to_cpu_mujoco",
        "endpoint": args.endpoint,
        "passed": passed,
        "qualification_mode": "nominal" if expected_fault is None else "transport_fault",
        "expected_transport_fault": expected_fault,
        "observed_transport_fault": observed_fault,
        "transport_fault_detail": fault_detail,
        "completed_live_transitions": received_transitions,
        "completed_total_transitions": controller.completed,
        "requested_live_transitions": args.steps,
        "first_control_source_frame_index": first_index,
        "last_control_source_frame_index": last_index,
        "maximum_reference_age_ns": maximum_age_ns,
        "maximum_reference_age_gate_ns": args.maximum_age_ms * 1_000_000,
        "receive_timeout_ms": args.receive_timeout_ms,
        "startup_timeout_ms": args.startup_timeout_ms,
        "minimum_base_height_m": None if minimum_height == float("inf") else minimum_height,
        "maximum_base_tilt_rad": maximum_tilt,
        "minimum_base_height_gate_m": 0.30,
        "maximum_base_tilt_gate_rad": 1.0,
        "fallback_base_tilt_trigger_rad": controller.fallback_tilt_trigger_rad,
        "fallback_active": controller.fallback_active,
        "fallback_trigger": controller.fallback_trigger,
        "fallback_first_transition": controller.fallback_transition,
        "fallback_transition_count": fallback_transition_count,
        **fallback_evidence,
        "decoder_sha256": profile.decoder_sha256,
        "decoder_report": str(profile.decoder_report_path),
        "decoder_report_sha256": profile.decoder_report_sha256,
        "candidate_summary": str(profile.candidate_summary_path),
        "candidate_summary_sha256": profile.candidate_summary_sha256,
        "checkpoint_update_count": profile.base_update_count,
        "residual_alpha": profile.residual_alpha,
        "physical_dof": 23,
        "decoder_output_dof": 23,
        "source_29dof_physics_used": False,
        "gain_profile": "released_retained",
        "pico_reference_retargeting": "pinned_native23_forward_kinematics",
        "safety_fallback_enabled": True,
        "live_transport_path_exercised": initialized,
        "live_transport_proven": nominal_passed and received_transitions >= 500,
        "live_headset_source_proven": False,
        "authorization": {
            "simulator_only": True,
            "localhost_only": True,
            "dds_opened": False,
            "robot_channel_opened": False,
            "hardware_authorized": False,
            "robot_commands_published": False,
        },
    }
    _exclusive_json(output, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))  # noqa: T201
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
