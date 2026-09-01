"""Hash-bound live PICO runtime and readiness contracts for the true23 candidate."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np

from gear_sonic.utils.g1_23dof_artifact import sha256_file
from gear_sonic.utils.g1_true23_clean_mujoco_teleop import (
    CleanSonicPolicy,
    SupervisedCleanTrue23MujocoController,
    UnitreeZeroVelocityFallbackPolicy,
    encoder267_from_reference,
    pico_reference_policy_history,
    reference_initial_state,
    validate_reference_terms,
)

CONTROL_PERIOD_NS = 20_000_000
ENCODER_RELATIVE_PATH = Path(
    "artifacts/g1_true23/causal_model_250_20260803/causal_model_250.encoder.onnx"
)
FALLBACK_RELATIVE_PATH = Path(
    "artifacts/external/unitree_rl_mjlab/deploy/robots/g1/config/"
    "policy/velocity/v0/exported/policy.onnx"
)
EXPECTED_FAULT_TRIGGERS = {
    "timeout": "transport_timeout",
    "gap": "transport_gap",
    "stale": "transport_stale",
    "payload": "transport_payload",
}
EXPECTED_DECODER_SHA256 = "44d1fb2701f1e65460f1c2c23f676bce4f1d4a44b3b112798dc5034af37946b8"
EXPECTED_DECODER_REPORT_SHA256 = "02197e5682a9bddc8f11aa6fa9c32ba909b97ec7d1c316c9a0d660cba2d25b7d"
EXPECTED_CANDIDATE_SUMMARY_SHA256 = "ed3f5513ed7da8625195b37b674163d64948697d311722ad936be3f4668db801"


@dataclass(frozen=True)
class FrozenLoraLiveProfile:
    decoder_path: Path
    decoder_sha256: str
    decoder_report_path: Path
    decoder_report_sha256: str
    candidate_summary_path: Path
    candidate_summary_sha256: str
    base_update_count: int
    residual_alpha: float


class LiveTransportFault(RuntimeError):
    """One fail-closed live transport classification."""

    def __init__(self, fault: str, detail: str) -> None:
        if fault not in EXPECTED_FAULT_TRIGGERS:
            raise ValueError(f"unsupported live transport fault: {fault}")
        super().__init__(detail)
        self.fault = fault
        self.trigger = EXPECTED_FAULT_TRIGGERS[fault]


def _object(path: Path, context: str) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a JSON object")
    return value


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    return value


def load_frozen_lora_live_profile(
    *,
    decoder_report_path: Path,
    candidate_summary_path: Path,
) -> FrozenLoraLiveProfile:
    """Validate the selected decoder and all evidence binding it to parity."""

    report_path = decoder_report_path.expanduser().resolve(strict=True)
    summary_path = candidate_summary_path.expanduser().resolve(strict=True)
    report = _object(report_path, "decoder report")
    summary = _object(summary_path, "candidate summary")
    decoder = _mapping(report.get("decoder"), "decoder")
    source = _mapping(report.get("source"), "decoder source")
    candidate = _mapping(summary.get("candidate"), "candidate")
    evidence = _mapping(summary.get("evidence"), "candidate evidence")
    report_evidence = _mapping(evidence.get("decoder_report"), "decoder-report evidence")
    report_hash = sha256_file(report_path)
    summary_hash = sha256_file(summary_path)
    decoder_hash = decoder.get("sha256")
    decoder_filename = decoder.get("filename")
    if (
        report_hash != EXPECTED_DECODER_REPORT_SHA256
        or summary_hash != EXPECTED_CANDIDATE_SUMMARY_SHA256
        or decoder_hash != EXPECTED_DECODER_SHA256
        or report.get("kind")
        != "g1_true23_frozen_lora_happy_residual_diagnostic_decoder_onnx"
        or report.get("closed_loop_happy_dance_passed") is not True
        or report.get("diagnostic_only") is not True
        or report.get("deployment_ready") is not False
        or report.get("hardware_authorized") is not False
        or report.get("active_motor_control_authorized") is not False
        or report.get("robot_network_commands") is not False
        or not isinstance(decoder_filename, str)
        or not isinstance(decoder_hash, str)
        or len(decoder_hash) != 64
        or decoder.get("input_shape") != [1, 994]
        or decoder.get("output_shape") != [1, 23]
    ):
        raise ValueError("selected decoder safety or shape contract mismatch")
    decoder_path = report_path.with_name(decoder_filename).resolve(strict=True)
    if sha256_file(decoder_path) != decoder_hash:
        raise ValueError("selected decoder hash mismatch")
    if (
        summary.get("kind") != "g1_true23_frozen_lora_parity_candidate_summary_v1"
        or summary.get("simulator_diagnostic_default_candidate") is not True
        or summary.get("original_sonic_happy_dance_parity") is not True
        or summary.get("diagnostic_only") is not True
        or summary.get("deployment_ready") is not False
        or summary.get("hardware_authorized") is not False
        or summary.get("robot_network_commands") is not False
        or candidate.get("decoder_sha256") != decoder_hash
        or report_evidence.get("filename") != report_path.name
        or report_evidence.get("sha256") != report_hash
        or _mapping(summary.get("saved_pico_walk001"), "saved PICO evidence").get("passed") is not True
        or _mapping(summary.get("saved_pico_walk001"), "saved PICO evidence").get(
            "completed_transitions"
        )
        != 684
        or _mapping(summary.get("saved_pico_walk001"), "saved PICO evidence").get(
            "fallback_active"
        )
        is not False
        or _mapping(summary.get("full_suite"), "full suite").get("lost_passing_cases") != []
    ):
        raise ValueError("candidate parity evidence contract mismatch")
    update = source.get("base_update_count")
    alpha = source.get("alpha")
    if type(update) is not int or update < 0 or not isinstance(alpha, (int, float)) or isinstance(alpha, bool):
        raise ValueError("candidate update identity mismatch")
    if candidate.get("base_update_count") != update or float(candidate.get("residual_alpha")) != float(alpha):
        raise ValueError("candidate summary identity mismatch")
    return FrozenLoraLiveProfile(
        decoder_path=decoder_path,
        decoder_sha256=decoder_hash,
        decoder_report_path=report_path,
        decoder_report_sha256=report_hash,
        candidate_summary_path=summary_path,
        candidate_summary_sha256=summary_hash,
        base_update_count=update,
        residual_alpha=float(alpha),
    )


def build_live_controller(
    *, repository_root: Path, profile: FrozenLoraLiveProfile
) -> SupervisedCleanTrue23MujocoController:
    root = repository_root.resolve(strict=True)
    controller = SupervisedCleanTrue23MujocoController(
        model_path=root / "gear_sonic/data/robots/g1/g1_23dof_rev_1_0.xml",
        physics_path=root / "gear_sonic/config/sim_validation/g1_23dof_mujoco_sim2sim.json",
        minimum_base_height_m=0.30,
        maximum_base_tilt_rad=1.0,
        fallback_tilt_trigger_rad=0.50,
        policy=CleanSonicPolicy(
            root / ENCODER_RELATIVE_PATH,
            profile.decoder_path,
            expected_decoder_sha256=profile.decoder_sha256,
        ),
        fallback_policy=UnitreeZeroVelocityFallbackPolicy(root / FALLBACK_RELATIVE_PATH),
    )
    controller.use_released_retained_gains()
    return controller


def initialize_live_controller(
    controller: SupervisedCleanTrue23MujocoController,
    first: Mapping[str, Any],
    second: Mapping[str, Any],
) -> None:
    first_summary = validate_reference_terms(first)
    second_summary = validate_reference_terms(second)
    if (
        second_summary["control_index"] != first_summary["control_index"] + 1
        or second_summary["control_monotonic_ns"]
        != first_summary["control_monotonic_ns"] + CONTROL_PERIOD_NS
    ):
        raise LiveTransportFault("gap", "startup packets are not contiguous at exact 50 Hz")
    q10, _ = reference_initial_state(first)
    q11, qd10 = reference_initial_state(second)
    height = controller.reference_root_height(q10)
    next_height = controller.reference_root_height(q11)
    controller.reset(
        base_position=[0.0, 0.0, height],
        base_quaternion_wxyz=[1.0, 0.0, 0.0, 0.0],
        joint_position_hardware=q10,
        root_velocity=[0.0, 0.0, (next_height - height) / 0.02, 0.0, 0.0, 0.0],
        joint_velocity_hardware=qd10,
    )
    controller.history = pico_reference_policy_history(first)


def validate_live_packet(
    packet: Any,
    *,
    previous: Mapping[str, Any] | None,
    maximum_age_ns: int,
    now_ns: int | None = None,
) -> dict[str, Any]:
    try:
        summary = validate_reference_terms(packet)
    except (KeyError, TypeError, ValueError) as error:
        raise LiveTransportFault("payload", str(error)) from error
    if previous is not None and (
        summary["control_index"] != previous["control_index"] + 1
        or summary["control_monotonic_ns"] != previous["control_monotonic_ns"] + CONTROL_PERIOD_NS
    ):
        raise LiveTransportFault("gap", "live PICO stream lost a contiguous 50-Hz packet")
    age_ns = (time.monotonic_ns() if now_ns is None else now_ns) - summary["control_monotonic_ns"]
    if age_ns < 0 or age_ns > maximum_age_ns:
        raise LiveTransportFault("stale", f"live PICO packet age is {age_ns} ns")
    summary["age_ns"] = age_ns
    return summary


def step_live_packet(
    controller: SupervisedCleanTrue23MujocoController, packet: Mapping[str, Any]
) -> Mapping[str, Any]:
    policy_packet = controller.retarget_pico_reference_packet(packet)
    encoder = encoder267_from_reference(policy_packet, controller.buffered_robot_pelvis_q9)
    return controller.step(encoder)


def hold_transport_fallback(
    controller: SupervisedCleanTrue23MujocoController,
    *,
    trigger: str,
    steps: int,
) -> dict[str, Any]:
    if steps < 1:
        raise ValueError("fallback hold steps must be positive")
    controller.activate_fallback(trigger)
    minimum_height = float("inf")
    maximum_tilt = 0.0
    maximum_torque = 0.0
    for _ in range(steps):
        evidence = controller.step(np.zeros(267, dtype=np.float32))
        minimum_height = min(minimum_height, float(evidence["base_height_m"]))
        maximum_tilt = max(maximum_tilt, float(evidence["base_tilt_rad"]))
        maximum_torque = max(maximum_torque, float(evidence["max_abs_torque_nm"]))
    return {
        "fallback_hold_transitions": steps,
        "fallback_minimum_base_height_m": minimum_height,
        "fallback_maximum_base_tilt_rad": maximum_tilt,
        "fallback_maximum_absolute_torque_nm": maximum_torque,
        "fallback_policy_query_count": controller.fallback_policy.query_count,
        "fallback_stable": minimum_height >= 0.45 and maximum_tilt <= 1.0,
    }


def audit_frozen_lora_live_readiness(
    *,
    profile: FrozenLoraLiveProfile,
    nominal_report_path: Path,
    fault_report_paths: Mapping[str, Path],
    original_report_path: Path,
    health_report_path: Path | None,
) -> dict[str, Any]:
    """Require a sustained session plus timeout/gap/stale fallback drills."""

    nominal_path = nominal_report_path.resolve(strict=True)
    nominal = _object(nominal_path, "nominal live report")
    nominal_auth = _mapping(nominal.get("authorization"), "nominal authorization")
    nominal_age = nominal.get("maximum_reference_age_ns")
    nominal_count = nominal.get("completed_live_transitions")
    nominal_ready = bool(
        nominal.get("kind") == "g1_true23_frozen_lora_live_teleop_v1"
        and nominal.get("passed") is True
        and nominal.get("qualification_mode") == "nominal"
        and nominal.get("expected_transport_fault") is None
        and nominal.get("observed_transport_fault") is None
        and nominal.get("decoder_sha256") == profile.decoder_sha256
        and nominal.get("decoder_report_sha256") == profile.decoder_report_sha256
        and nominal.get("candidate_summary_sha256") == profile.candidate_summary_sha256
        and type(nominal_count) is int
        and nominal_count >= 500
        and type(nominal_age) is int
        and 0 <= nominal_age <= 100_000_000
        and nominal.get("physical_dof") == 23
        and nominal.get("decoder_output_dof") == 23
        and nominal.get("source_29dof_physics_used") is False
        and nominal.get("fallback_active") is False
        and nominal.get("safety_fallback_enabled") is True
        and nominal.get("live_transport_proven") is True
        and nominal_auth.get("simulator_only") is True
        and nominal_auth.get("localhost_only") is True
        and nominal_auth.get("dds_opened") is False
        and nominal_auth.get("robot_channel_opened") is False
        and nominal_auth.get("hardware_authorized") is False
        and nominal_auth.get("robot_commands_published") is False
    )

    fault_results: dict[str, Any] = {}
    for fault in ("timeout", "gap", "stale"):
        path = fault_report_paths.get(fault)
        if path is None:
            raise ValueError(f"missing {fault} qualification report")
        resolved = path.resolve(strict=True)
        value = _object(resolved, f"{fault} qualification report")
        auth = _mapping(value.get("authorization"), f"{fault} authorization")
        passed = bool(
            value.get("kind") == "g1_true23_frozen_lora_live_teleop_v1"
            and value.get("passed") is True
            and value.get("qualification_mode") == "transport_fault"
            and value.get("expected_transport_fault") == fault
            and value.get("observed_transport_fault") == fault
            and value.get("decoder_sha256") == profile.decoder_sha256
            and value.get("decoder_report_sha256") == profile.decoder_report_sha256
            and value.get("candidate_summary_sha256") == profile.candidate_summary_sha256
            and value.get("fallback_active") is True
            and value.get("fallback_trigger") == EXPECTED_FAULT_TRIGGERS[fault]
            and type(value.get("completed_live_transitions")) is int
            and value["completed_live_transitions"] >= 100
            and type(value.get("fallback_hold_transitions")) is int
            and value["fallback_hold_transitions"] >= 100
            and value.get("fallback_stable") is True
            and value.get("live_transport_path_exercised") is True
            and auth.get("simulator_only") is True
            and auth.get("localhost_only") is True
            and auth.get("dds_opened") is False
            and auth.get("robot_channel_opened") is False
            and auth.get("hardware_authorized") is False
            and auth.get("robot_commands_published") is False
        )
        fault_results[fault] = {
            "report": str(resolved),
            "report_sha256": sha256_file(resolved),
            "observed_trigger": value.get("fallback_trigger"),
            "live_transitions_before_fault": value.get("completed_live_transitions"),
            "fallback_hold_transitions": value.get("fallback_hold_transitions"),
            "passed": passed,
        }

    original_path = original_report_path.resolve(strict=True)
    original = _object(original_path, "original true23 live report")
    original_ready = bool(
        original.get("kind") == "g1_true23_clean_mujoco_teleop_session"
        and original.get("passed") is True
        and original.get("physical_dof") == 23
        and original.get("decoder_output_dof") == 23
        and original.get("source_29dof_physics_used") is False
        and original.get("completed_transitions") == 684
        and original.get("fallback_active") is False
        and original.get("live_transport_proven") is True
    )
    transport_parity = bool(
        original_ready
        and nominal_ready
        and nominal_count == original.get("completed_transitions")
        and nominal.get("first_control_source_frame_index")
        == original.get("first_control_source_frame_index")
        and nominal.get("last_control_source_frame_index")
        == original.get("last_control_source_frame_index")
    )

    health_path: Path | None = None
    health_hash: str | None = None
    headset_tracking_ready = False
    health_boundary_valid = False
    if health_report_path is not None:
        health_path = health_report_path.resolve(strict=True)
        health_hash = sha256_file(health_path)
        health = _object(health_path, "PICO health report")
        health_auth = _mapping(health.get("authorization"), "PICO health authorization")
        health_boundary_valid = bool(
            health.get("kind") == "g1_true23_pico_tracking_health_probe_v1"
            and health_auth.get("read_only") is True
            and health_auth.get("dds_opened") is False
            and health_auth.get("robot_channel_opened") is False
            and health_auth.get("hardware_authorized") is False
            and health_auth.get("robot_commands_published") is False
        )
        headset_tracking_ready = health_boundary_valid and health.get("passed") is True

    software_ready = bool(
        nominal_ready
        and all(value["passed"] is True for value in fault_results.values())
        and transport_parity
    )
    return {
        "schema_version": 1,
        "kind": "g1_true23_frozen_lora_live_readiness_v1",
        "candidate_decoder_sha256": profile.decoder_sha256,
        "candidate_summary_sha256": profile.candidate_summary_sha256,
        "physical_dof": 23,
        "decoder_output_dof": 23,
        "source_29dof_physics_used": False,
        "nominal_live_report": str(nominal_path),
        "nominal_live_report_sha256": sha256_file(nominal_path),
        "nominal_sustained_session_ready": nominal_ready,
        "transport_fault_qualifications": fault_results,
        "original_true23_live_report": str(original_path),
        "original_true23_live_report_sha256": sha256_file(original_path),
        "original_true23_live_report_valid": original_ready,
        "original_true23_transport_parity": transport_parity,
        "comparison": {
            "candidate_completed_transitions": nominal_count,
            "original_completed_transitions": original.get("completed_transitions"),
            "candidate_minimum_base_height_m": nominal.get("minimum_base_height_m"),
            "original_minimum_base_height_m": original.get("minimum_base_height_m"),
            "candidate_maximum_base_tilt_rad": nominal.get("maximum_base_tilt_rad"),
            "original_maximum_base_tilt_rad": original.get("maximum_base_tilt_rad"),
            "candidate_maximum_reference_age_ns": nominal_age,
            "original_maximum_reference_age_ns": original.get("maximum_reference_age_ns"),
            "candidate_safety_fallback_enabled": True,
            "original_safety_fallback_enabled": original.get("safety_fallback_enabled"),
        },
        "software_live_teleop_ready": software_ready,
        "live_pico_health_report": None if health_path is None else str(health_path),
        "live_pico_health_report_sha256": health_hash,
        "live_pico_health_boundary_valid": health_boundary_valid,
        "live_pico_tracking_ready": headset_tracking_ready,
        "live_pico_to_mujoco_ready": software_ready and headset_tracking_ready,
        "physical_robot_teleop_ready": False,
        "hardware_authorized": False,
        "dds_opened": False,
        "robot_channel_opened": False,
        "robot_commands_published": False,
        "blocking_gate": (
            "live_pico_tracking_health"
            if software_ready and not headset_tracking_ready
            else (None if software_ready else "software_live_teleop_qualification")
        ),
    }


__all__ = (
    "CONTROL_PERIOD_NS",
    "EXPECTED_FAULT_TRIGGERS",
    "EXPECTED_CANDIDATE_SUMMARY_SHA256",
    "EXPECTED_DECODER_REPORT_SHA256",
    "EXPECTED_DECODER_SHA256",
    "FrozenLoraLiveProfile",
    "LiveTransportFault",
    "build_live_controller",
    "audit_frozen_lora_live_readiness",
    "hold_transport_fallback",
    "initialize_live_controller",
    "load_frozen_lora_live_profile",
    "step_live_packet",
    "validate_live_packet",
)
