"""Qualify true23 active-controller exit behavior without opening DDS.

This entrypoint runs dependency-light core tests, a compiled/source surface
audit, and the actual restore routine with offline RPC/LowState substitutes.
It never launches the active controller binary. A zero-damping LowCmd count
alone cannot qualify a no-damping handoff: high-level FSM requests also count.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

from gear_sonic.scripts.audit_g1_true23_restore_rpc_no_robot import run_audit

HARNESS_MARKER = (
    "robot_free_lifecycle_scenarios=9 recovery_frames=4000 published_damping_frames=0 dds_opened=false"
)
FORBIDDEN_SOURCE = (
    "publisher->Write(ToLowCmd(value.BuildDampingCommand()))",
    "kFaultDampingCycles",
    "fail-safe damping",
)
REQUIRED_SOURCE = (
    "active::IsPositiveGainRuntimeCommand(command)",
    "outgoing non-positive-gain LowCmd rejected before DDS",
    "emergency_motion_mode_restored",
    "writer_emergency_mode_handoff",
    "mode_handoff_interlock.Request()",
    "WaitForWriterQuiescence(mode_handoff_interlock)",
    "writer_quiesced_before_select",
    "publisher.reset()",
    "lowcmd_publisher_closed_before_select",
    "value.BeginSoftwareFaultReturnHold(recovery_ns)",
    "RestoreMotionModeAfterNormalHold(released_motion_mode,",
    "SwitchToInternalCtrl(",
    "InternalFsmMode::WALKRUN",
    "kMotionRestoreStableSamples",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _run(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - every executable path is explicit/pinned.
        command,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )


def restore_rpc_contract_checks(report: dict) -> dict[str, bool]:
    """Screen the explicit normal-standing/no-damp requirement, not firmware."""
    scenarios = report.get("scenarios", {})
    unhealthy = (
        "disabled_801",
        "bit30_801",
        "missing_state_801",
        "stale_state_801",
        "frozen_tick_801",
        "bad_crc_801",
        "nonfinite_801",
        "zero_torque_801",
    )
    standing = scenarios.get("already_801", {})
    return {
        "actual_restore_fault_injection_completed": report.get("experiment_completed") is True,
        "runtime_checks_physical_motor_evidence": report.get("runtime_has_physical_gate") is True,
        "normal_restore_never_requests_damp_fsm": report.get("no_damp_rpc_contract_met_for_all_cases") is True,
        "unhealthy_final_states_rejected": all(
            scenarios.get(name, {}).get("restore_returned_success") is False for name in unhealthy
        ),
        "powered_standing_observed": (
            standing.get("restore_returned_success") is True
            and standing.get("physical_state_verified") is True
            and standing.get("stable_samples") == 100
            and standing.get("final_mock_fsm_id") == 801
        ),
        "crouched_state_not_labelled_normal_standing": (
            scenarios.get("crouched_801", {}).get("restore_returned_success") is False
        ),
        "no_stand_commands_after_mock_motor_disable": (
            scenarios.get("disabled_during_kick", {}).get("stand_commands_while_mock_disabled") == 0
        ),
    }


def qualify(repository_root: Path, *, rpc_audit_directory: Path) -> dict[str, object]:
    repository_root = repository_root.resolve()
    harness = repository_root / "gear_sonic_deploy/target/release/true23_active_gantry_core_harness"
    controller = repository_root / "gear_sonic_deploy/target/release/g1_true23_active_gantry"
    source = repository_root / ("gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_true23_active_gantry.cpp")
    audit = repository_root / (
        "gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/tests/check_true23_active_gantry_binary.cmake"
    )
    required_paths = (harness, controller, source, audit)
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing qualification input(s): " + ", ".join(missing))

    source_text = source.read_text(encoding="utf-8")
    forbidden_hits = [value for value in FORBIDDEN_SOURCE if value in source_text]
    missing_guards = [value for value in REQUIRED_SOURCE if value not in source_text]

    harness_run = _run([str(harness)], cwd=repository_root)
    harness_passed = harness_run.returncode == 0 and HARNESS_MARKER in harness_run.stdout
    audit_run = _run(
        [
            "cmake",
            f"-DBINARY={controller}",
            f"-DSOURCE={source}",
            "-P",
            str(audit),
        ],
        cwd=repository_root,
    )
    surface_audit_passed = (
        audit_run.returncode == 0 and "true23 active gantry surface audit passed" in audit_run.stdout
    )
    rpc_audit = run_audit(repository_root, rpc_audit_directory)
    rpc_checks = restore_rpc_contract_checks(rpc_audit)
    passed = (
        harness_passed
        and surface_audit_passed
        and not forbidden_hits
        and not missing_guards
        and all(rpc_checks.values())
    )
    return {
        "schema_version": 2,
        "kind": "g1_true23_active_lifecycle_no_robot_qualification",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "robot_commands_published": False,
        "dds_opened": False,
        "lowcmd_channel_opened": False,
        "active_controller_launched": False,
        "scope": (
            "compiled safety-core lifecycle, binary/source surface and offline actual restore RPC "
            "fault injection; "
            "does not prove physical Unitree RPC, network, balance, or actuator behavior"
        ),
        "lifecycle": {
            "scenarios": 9,
            "positive_gain_recovery_frames": 4000,
            "published_damping_frames": 0,
            "injected_stalls_ms": [101, 290, 500],
            "normal_completion_tested": True,
            "operator_stop_tested": True,
            "deadman_release_tested": True,
            "writer_failure_boundary_tested": True,
            "walkrun_standing_restore_match_tested": True,
            "internal_control_handoff_audited": True,
            "ten_second_restore_stability_gate_tested": True,
            "writer_quiescence_before_mode_rpc_tested": True,
            "lowcmd_publisher_close_before_mode_rpc_audited": True,
        },
        "checks": {
            "core_harness_passed": harness_passed,
            "surface_audit_passed": surface_audit_passed,
            "forbidden_source_hits": forbidden_hits,
            "missing_runtime_guards": missing_guards,
            "restore_rpc_contract": rpc_checks,
        },
        "restore_rpc_audit": {
            "path": str(rpc_audit_directory.resolve() / "report.json"),
            "sha256": _sha256(rpc_audit_directory / "report.json"),
        },
        "deployment_ready": False,
        "hardware_authorized": False,
        "inputs": {
            "core_harness_sha256": _sha256(harness),
            "active_controller_sha256": _sha256(controller),
            "active_controller_source_sha256": _sha256(source),
            "surface_audit_sha256": _sha256(audit),
        },
        "stdout": {
            "core_harness": harness_run.stdout.strip(),
            "surface_audit": audit_run.stdout.strip(),
        },
        "stderr": {
            "core_harness": harness_run.stderr.strip(),
            "surface_audit": audit_run.stderr.strip(),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite qualification report: {output}")
    rpc_audit_directory = output.with_name(output.name + ".restore_rpc")
    if rpc_audit_directory.exists():
        raise FileExistsError(f"refusing to overwrite restore RPC evidence: {rpc_audit_directory}")
    report = qualify(args.repository_root.expanduser(), rpc_audit_directory=rpc_audit_directory)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "passed": report["passed"]}))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
