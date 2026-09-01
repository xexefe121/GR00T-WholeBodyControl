"""Qualify true23 active-controller exit behavior without opening DDS.

This entrypoint runs only dependency-light core tests and a compiled/source
surface audit. It never launches the active controller binary.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

HARNESS_MARKER = (
    "robot_free_lifecycle_scenarios=9 recovery_frames=4000 "
    "published_damping_frames=0 dds_opened=false"
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
    "RestoreMotionModeAfterNormalHold(released_motion_mode)",
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


def qualify(repository_root: Path) -> dict[str, object]:
    repository_root = repository_root.resolve()
    harness = (
        repository_root
        / "gear_sonic_deploy/target/release/true23_active_gantry_core_harness"
    )
    controller = (
        repository_root
        / "gear_sonic_deploy/target/release/g1_true23_active_gantry"
    )
    source = repository_root / (
        "gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/"
        "g1_true23_active_gantry.cpp"
    )
    audit = repository_root / (
        "gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/tests/"
        "check_true23_active_gantry_binary.cmake"
    )
    required_paths = (harness, controller, source, audit)
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing qualification input(s): " + ", ".join(missing))

    source_text = source.read_text(encoding="utf-8")
    forbidden_hits = [value for value in FORBIDDEN_SOURCE if value in source_text]
    missing_guards = [value for value in REQUIRED_SOURCE if value not in source_text]

    harness_run = _run([str(harness)], cwd=repository_root)
    harness_passed = (
        harness_run.returncode == 0 and HARNESS_MARKER in harness_run.stdout
    )
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
        audit_run.returncode == 0
        and "true23 active gantry surface audit passed" in audit_run.stdout
    )
    passed = (
        harness_passed
        and surface_audit_passed
        and not forbidden_hits
        and not missing_guards
    )
    return {
        "schema_version": 1,
        "kind": "g1_true23_active_lifecycle_no_robot_qualification",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "robot_commands_published": False,
        "dds_opened": False,
        "lowcmd_channel_opened": False,
        "active_controller_launched": False,
        "scope": (
            "compiled safety-core lifecycle and binary/source surface only; "
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
            "exact_mode_restore_match_tested": True,
            "writer_quiescence_before_mode_rpc_tested": True,
            "lowcmd_publisher_close_before_mode_rpc_audited": True,
        },
        "checks": {
            "core_harness_passed": harness_passed,
            "surface_audit_passed": surface_audit_passed,
            "forbidden_source_hits": forbidden_hits,
            "missing_runtime_guards": missing_guards,
        },
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
    report = qualify(args.repository_root.expanduser())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "passed": report["passed"]}))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
