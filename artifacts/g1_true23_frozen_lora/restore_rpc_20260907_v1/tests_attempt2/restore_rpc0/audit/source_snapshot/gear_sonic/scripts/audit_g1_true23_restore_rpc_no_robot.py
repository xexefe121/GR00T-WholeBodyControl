"""Execute current restore C++ with offline RPC/clock/LowState substitutes.

This is fault injection, not firmware emulation or standing qualification.
Never runs a robot binary, links the SDK, opens DDS or publishes commands.
The output directory must be new; source and generated evidence are hashed.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


RUNTIME = Path("gear_sonic_deploy/src/g1/g1_deploy_onnx_ref")
SOURCE = RUNTIME / "src/g1_true23_active_gantry.cpp"
FIXTURE = RUNTIME / "tests/true23_restore_rpc_fixture.hpp"
DRIVER = RUNTIME / "tests/true23_restore_rpc_driver.cpp"
HEADERS = (
    "true23_active_gantry_core.hpp",
    "true23_live_shadow_core.hpp",
    "true23_shadow_gate.hpp",
)
# Exact bounded source spans. Fail rather than silently test an old copy when
# the runtime changes structure. Constant and function bodies stay unmodified.
SPANS = (
    ("inline constexpr int kMotionRestoreStableSamples =", "inline constexpr std::int64_t kMaximumFirstHoldWriteDelayNs ="),
    ("json PhysicalStateEvidence(", "\nclass StateMonitor {"),
    ("struct ReleasedMotionMode {", "\nvoid RestoreReleasedMotionModeAfterFailure("),
    ("struct MotionRestoreResult {", "\nvoid RestoreReleasedMotionModeAfterFailure("),
)

SCENARIOS = {
    "already_801": {"fsm_id": 801},
    "zero_fsm": {"fsm_id": 0},
    "damped_fsm": {"fsm_id": 1},
    "completed_4": {"fsm_id": 4},
    "walkready_500": {"fsm_id": 500},
    "disabled_801": {"fsm_id": 801, "disabled": True},
    "bit30_801": {"fsm_id": 801, "bit30": True},
    "missing_state_801": {"fsm_id": 801, "missing_state": True},
    "stale_state_801": {"fsm_id": 801, "stale_state": True},
    "frozen_tick_801": {"fsm_id": 801, "frozen_tick": True},
    "bad_crc_801": {"fsm_id": 801, "bad_crc": True},
    "nonfinite_801": {"fsm_id": 801, "nonfinite_state": True},
    "zero_torque_801": {"fsm_id": 801, "zero_torque": True},
    "crouched_801": {"fsm_id": 801, "knee_q_rad": 1.5},
    "checkmode_unavailable": {"fsm_id": 0, "check_failures": 20},
    "checkmode_retry": {"fsm_id": 801, "check_failures": 1},
    "service_unavailable": {"service": "", "select_result": 7002},
    "select_error_but_applied": {"service": "", "fsm_id": 801, "select_result": 7002, "select_applies_despite_error": True},
    "internal_rpc_rejected": {"fsm_id": 0, "internal_result": 3104},
    "fsm_unreadable": {"fsm_id": 0, "fsm_read_result": 3104},
    "stand_rpc_rejected": {"fsm_id": 0, "set_fsm_fail_id": 4},
    "stand_transition_incomplete": {"fsm_id": 0, "stand_never_completes": True},
    "upright_rpc_rejected": {"fsm_id": 0, "set_fsm_fail_id": 801},
    "upright_rpc_accepted_but_ignored": {"fsm_id": 0, "set_fsm_ignored_id": 801},
    "disabled_during_kick": {"fsm_id": 0, "disabled_at_ns": 1_100_000_000},
    "transient_disabled_during_kick": {"fsm_id": 0, "disabled_at_ns": 1_100_000_000, "disabled_until_ns": 1_300_000_000},
}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def extract_runtime(source: str) -> tuple[str, list[dict]]:
    pieces = []
    bindings = []
    for start, end in SPANS:
        if source.count(start) != 1:
            raise ValueError(f"expected one runtime marker: {start}")
        begin = source.index(start)
        finish = source.find(end, begin + len(start))
        if finish < 0:
            raise ValueError(f"missing runtime end marker: {end}")
        original = source[begin:finish]
        transformed = original.replace("std::this_thread::sleep_for(", "fixture::SleepFor(")
        pieces.append(transformed)
        bindings.append({
            "start_marker": start,
            "start_line": source.count("\n", 0, begin) + 1,
            "end_line": source.count("\n", 0, finish) + 1,
            "source_span_sha256": hashlib.sha256(original.encode()).hexdigest(),
            "sleep_substitutions": original.count("std::this_thread::sleep_for("),
        })
    return "\n".join(pieces), bindings


def summarize(result: dict) -> dict:
    calls = result["calls"]
    fsm_calls = [row for row in calls if row["method"] == "SetFsmId"]
    reads = [row["elapsed_ns"] for row in calls if row["method"] == "LatestPhysicalState"]
    # Include start/end: failure before any read is an explicit observation gap.
    boundaries = [0, *reads, result["elapsed_virtual_ns"]]
    return {
        "restore_returned_success": result["restore_returned_success"],
        "error": result["error"],
        "fsm_commands": [row["args"][0] for row in fsm_calls],
        "damp_rpc_requests": sum(row["args"] == [1] for row in fsm_calls),
        "damp_rpc_accepted": sum(row["args"] == [1] and row["result"] == 0 for row in fsm_calls),
        "stand_commands_while_mock_disabled": sum(row["args"][0] in (4, 801) and row["mock_disabled"] for row in fsm_calls),
        "health_reads_observing_mock_disabled": sum(row["method"] == "LatestPhysicalState" and row["mock_disabled"] for row in calls),
        "maximum_restore_health_read_gap_seconds": max(b - a for a, b in zip(boundaries, boundaries[1:])) / 1e9,
        "elapsed_virtual_seconds": result["elapsed_virtual_ns"] / 1e9,
        "final_mock_fsm_id": result["final_mock_fsm_id"],
        "stable_samples": result["stable_samples"],
        "physical_state_verified": result["physical_state"].get("verified", False),
    }


def run_audit(repository: Path, output: Path, compiler: str = "g++") -> dict:
    repository = repository.resolve()
    output = output.resolve()
    executable = shutil.which(compiler)
    if executable is None:
        raise FileNotFoundError(compiler)
    paths = [repository / path for path in (SOURCE, FIXTURE, DRIVER)]
    paths += [repository / RUNTIME / "include" / name for name in HEADERS]
    paths += [Path(__file__).resolve()]
    source_hashes = {str(path): sha256(path) for path in paths}
    generated, spans = extract_runtime((repository / SOURCE).read_text(encoding="utf-8"))
    # Exclusive directory preserves failed builds and previous observations.
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "source_snapshot"
    for path in paths:
        target = snapshot / path.relative_to(repository)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    (output / "source_bindings.json").write_text(json.dumps(source_hashes, indent=2), encoding="utf-8")
    (output / "generated_runtime.inc").write_text(generated, encoding="utf-8")
    binary = output / "restore_rpc_offline"
    # Existing core header has a range-loop copy warning unrelated to this
    # experiment. Retain that warning without editing the hardware worktree.
    command = [executable, "-std=c++20", "-O0", "-Wall", "-Wextra", "-Werror",
               "-Wno-error=range-loop-construct",
               "-I", str(snapshot / RUNTIME / "include"),
               "-I", str(output), str(snapshot / DRIVER), "-o", str(binary)]
    build = subprocess.run(command, capture_output=True, text=True, timeout=90, check=False)
    (output / "build.json").write_text(json.dumps(dict(command=command, returncode=build.returncode, stdout=build.stdout, stderr=build.stderr), indent=2), encoding="utf-8")
    if build.returncode:
        raise RuntimeError(f"offline harness build failed; see {output / 'build.json'}")
    summaries = {}
    for name, config in SCENARIOS.items():
        run = subprocess.run([str(binary), json.dumps(config)], capture_output=True, text=True, timeout=10, check=False)
        if run.returncode:
            raise RuntimeError(f"offline scenario {name} failed: {run.stderr}")
        result = json.loads(run.stdout)
        result["scenario"] = name
        result["configuration"] = config
        (output / f"{name}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        summaries[name] = summarize(result)
    if any(sha256(Path(path)) != digest for path, digest in source_hashes.items()):
        raise RuntimeError("source changed during offline audit")
    report = {
        "kind": "g1_true23_actual_restore_rpc_offline_audit_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_completed": True,
        "source_files": source_hashes,
        "extracted_spans": spans,
        "artifacts": {str(path.relative_to(output)): sha256(path) for path in sorted(output.rglob("*")) if path.is_file()},
        "scenarios": summaries,
        "scope": "Actual runtime restore algorithm with fake RPC responses, fake LowState and virtual time; not firmware or plant simulation.",
        "writer_and_monitor_threads_executed": False,
        "rpc_network_latency_modeled": False,
        "health_gap_means_restore_routine_read_gap_not_DDS_callback_gap": True,
        "no_damp_rpc_contract_met_for_all_cases": all(row["damp_rpc_requests"] == 0 for row in summaries.values()),
        "physical_damping_cause_proven": False,
        "hardware_authorized": False,
        "deployment_ready": False,
        "robot_commands_published": False,
        "dds_opened": False,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--compiler", default="g++")
    args = parser.parse_args()
    report = run_audit(args.repository_root, args.output_directory, args.compiler)
    print(json.dumps({key: report[key] for key in ("experiment_completed", "no_damp_rpc_contract_met_for_all_cases", "deployment_ready", "scenarios")}))
    return 0  # Audit execution success is not a handoff qualification pass.


if __name__ == "__main__":
    raise SystemExit(main())
