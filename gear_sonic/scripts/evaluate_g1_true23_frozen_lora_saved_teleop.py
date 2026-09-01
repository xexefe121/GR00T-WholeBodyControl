"""Replay an offline authentic PICO packet clip with one LoRA decoder.

This is a CPU MuJoCo diagnostic.  It opens no DDS/ZMQ channel and never sends
robot commands.  The saved packet timestamps are checked for contiguous 50 Hz
causal semantics before control starts.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Sequence

from gear_sonic.utils.g1_23dof_artifact import sha256_file
from gear_sonic.utils.g1_true23_clean_mujoco_teleop import (
    CleanSonicPolicy,
    CleanTrue23MujocoController,
    pico_reference_policy_history,
    reference_initial_state,
    run_reference_sequence,
    validate_reference_terms,
)


def _load_decoder_report(path: Path) -> tuple[Path, str, int]:
    value = json.loads(path.read_text(encoding="utf-8"))
    decoder = value.get("decoder")
    source = value.get("source")
    kind = value.get("kind")
    if (
        value.get("diagnostic_only") is not True
        or value.get("deployment_ready") is not False
        or value.get("hardware_authorized") is not False
        or value.get("active_motor_control_authorized") is not False
        or not isinstance(decoder, dict)
        or not isinstance(source, dict)
    ):
        raise ValueError("LoRA decoder diagnostic contract mismatch")
    if kind == "g1_true23_frozen_lora_diagnostic_decoder_onnx":
        update = source.get("update_count")
    elif kind == (
        "g1_true23_frozen_lora_happy_residual_diagnostic_decoder_onnx"
    ):
        if (
            value.get("closed_loop_happy_dance_passed") is not True
            or value.get("promotion_eligible") is not False
            or value.get("robot_network_commands") is not False
        ):
            raise ValueError("residual decoder diagnostic contract mismatch")
        update = source.get("base_update_count")
    else:
        raise ValueError("LoRA decoder diagnostic contract mismatch")
    decoder_path = path.with_name(decoder["filename"]).resolve(strict=True)
    expected = decoder["sha256"]
    if (
        not isinstance(expected, str)
        or len(expected) != 64
        or sha256_file(decoder_path) != expected
        or isinstance(update, bool)
        or not isinstance(update, int)
        or update < 0
    ):
        raise ValueError("LoRA decoder diagnostic identity mismatch")
    return decoder_path, expected, update


def _load_packets(path: Path) -> tuple[list[dict[str, Any]], str]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict) or set(value) != {
        "robot_independent_reference_packets",
        "semantic_packets",
    }:
        raise ValueError("saved PICO packet bundle schema mismatch")
    packets = value["robot_independent_reference_packets"]
    if (
        not isinstance(packets, list)
        or len(packets) < 2
        or any(not isinstance(packet, dict) for packet in packets)
    ):
        raise ValueError("saved PICO packet bundle is too short")
    for packet in packets:
        validate_reference_terms(packet)
    return packets, sha256_file(path)


def _exclusive_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--decoder-report", type=Path, required=True)
    parser.add_argument("--packets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.repository_root.expanduser().resolve(strict=True)
    report_path = args.decoder_report.expanduser().resolve(strict=True)
    packet_path = args.packets.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite teleop result: {output}")
    decoder, decoder_hash, update = _load_decoder_report(report_path)
    packets, packet_hash = _load_packets(packet_path)
    first_summary = validate_reference_terms(packets[0])
    second_summary = validate_reference_terms(packets[1])
    if (
        second_summary["anchor_index"] != first_summary["anchor_index"] + 1
        or second_summary["anchor_monotonic_ns"]
        != first_summary["anchor_monotonic_ns"] + 20_000_000
    ):
        raise ValueError("saved PICO startup packets are not contiguous")

    encoder = root / (
        "artifacts/g1_true23/causal_model_250_20260803/"
        "causal_model_250.encoder.onnx"
    )
    controller = CleanTrue23MujocoController(
        model_path=root / "gear_sonic/data/robots/g1/g1_23dof_rev_1_0.xml",
        physics_path=root / "gear_sonic/config/sim_validation/g1_23dof_mujoco_sim2sim.json",
        minimum_base_height_m=0.30,
        maximum_base_tilt_rad=1.0,
        policy=CleanSonicPolicy(
            encoder,
            decoder,
            expected_decoder_sha256=decoder_hash,
        ),
    )
    controller.use_released_retained_gains()
    q10, _ = reference_initial_state(packets[0])
    q11, qd10 = reference_initial_state(packets[1])
    base_height = controller.reference_root_height(q10)
    next_height = controller.reference_root_height(q11)
    controller.reset(
        base_position=[0.0, 0.0, base_height],
        base_quaternion_wxyz=[1.0, 0.0, 0.0, 0.0],
        joint_position_hardware=q10,
        root_velocity=[0.0, 0.0, (next_height - base_height) / 0.02, 0.0, 0.0, 0.0],
        joint_velocity_hardware=qd10,
    )
    controller.history = pico_reference_policy_history(packets[0])
    result = run_reference_sequence(
        controller=controller,
        packets=iter(packets),
        steps=len(packets),
        retarget_native23_fk=True,
    )
    gate_passed = (
        result["minimum_base_height_m"] >= 0.30
        and result["maximum_base_tilt_rad"] <= 1.0
    )
    result.update(
        {
            "mode": "frozen_lora_saved_pico_causal_packet_replay",
            "passed": result["passed"] is True and gate_passed,
            "decoder_report": str(report_path),
            "decoder_sha256": decoder_hash,
            "checkpoint_update_count": update,
            "saved_packet_bundle": str(packet_path),
            "saved_packet_bundle_sha256": packet_hash,
            "offline_saved_capture": True,
            "live_freshness_checked": False,
            "live_transport_proven": False,
            "physical_dof": 23,
            "decoder_output_dof": 23,
            "source_29dof_physics_used": False,
            "gain_profile": "released_retained",
            "minimum_base_height_gate_m": 0.30,
            "maximum_base_tilt_gate_rad": 1.0,
            "safety_fallback_enabled": False,
            "deployment_ready": False,
            "hardware_authorized": False,
        }
    )
    _exclusive_json(output, result)
    print(output)  # noqa: T201
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
