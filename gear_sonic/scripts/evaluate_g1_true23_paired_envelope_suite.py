"""Run every manifest clip through the paired, constrained native23 simulator.

Unlike the legacy decoder-only suite, this requires both checkpoint exports
and uses current-observation timing and stateful V2 actuation. No hardware
imports, network commands, clip shortening or policy promotion are performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

from gear_sonic.utils.g1_true23_diagnostic_pair import load_diagnostic_pair


def file_identity(path: Path) -> dict:
    path = path.resolve(strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "sha256": digest.hexdigest()}


def manifest_cases(path: Path, asset_root: Path, repository_root: Path) -> tuple[dict, list[dict]]:
    manifest = json.loads(path.read_text())
    if manifest.get("kind") not in {
        "g1_true23_pico_fullbody_multimotion_manifest_v1",
        "g1_true23_stance_candidate_manifest_v1",
    }:
        raise ValueError("unsupported full-motion manifest")
    if manifest.get("kind") == "g1_true23_stance_candidate_manifest_v1" and (
        manifest.get("teacher_accepted") is not False
        or manifest.get("hardware_authorized") is not False
        or manifest.get("deployment_ready") is not False
    ):
        raise ValueError("stance candidates must retain their unaccepted diagnostic status")
    motions = manifest.get("motions")
    if not isinstance(motions, list) or not motions:
        raise ValueError("manifest must contain motions")
    result, names, paths = [], set(), set()
    for item in motions:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise ValueError("motion entries require a name and source path")
        name = item.get("name")
        if not isinstance(name, str) or re.fullmatch(r"[a-z0-9_]+", name) is None or name in names:
            raise ValueError("motion names must be unique safe labels")
        motion = (asset_root / item["path"]).resolve(strict=True)
        if not any(motion.is_relative_to(root) for root in (asset_root, repository_root)) or motion in paths:
            raise ValueError("motion must be unique and inside the source/worktree roots")
        with np.load(motion, allow_pickle=False) as archive:
            joints = archive["joint_pos"]
            if joints.ndim != 2 or joints.shape[1] != 23 or len(joints) < 12:
                raise ValueError("motion must contain a complete native23 causal transition")
            frames = len(joints)
        result.append({"name": name, "motion": file_identity(motion), "frames": frames})
        names.add(name)
        paths.add(motion)
    return manifest, result


def case_command(args: argparse.Namespace, pair: dict, case: dict, output: Path, *, measured: bool) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "gear_sonic.scripts.evaluate_g1_true23_deployment_envelope",
        "--repository-root",
        str(args.repository_root),
        "--asset-root",
        str(args.asset_root),
        "--motion",
        case["motion"]["path"],
        "--output-dir",
        str(output),
        "--encoder",
        pair["encoder"]["path"],
        "--expected-encoder-sha256",
        pair["encoder"]["sha256"],
        "--decoder",
        pair["decoder"]["path"],
        "--expected-decoder-sha256",
        pair["decoder"]["sha256"],
        "--encoder-report",
        str(args.encoder_report),
        "--decoder-report",
        str(args.decoder_report),
        "--gain-profiles",
        "configured_sim",
        "--fractions",
        "1",
        "--ankle-efforts",
        "35",
        "--slew-rates",
        "5",
        "--project-active-effort",
        "--stateful-native-controller",
        "--trace-active-actuation",
        "--initial-states",
        "measured" if measured else "reference",
    ]
    if measured:
        command.extend(
            [
                "--motor-health-snapshot",
                str(args.motor_health_snapshot),
                "--transition-balance-model",
                str(args.transition_balance_model),
                "--startup-hold-s",
                "5",
                "--return-hold-s",
                "5",
                "--align-reference-start",
                "--project-transition-effort",
            ]
        )
    return command


def verify_case(summary: dict, expected_pair: dict, case: dict) -> dict:
    if summary.get("diagnostic_pair") != expected_pair or summary.get("unpaired_diagnostic_only") is not False:
        raise ValueError("case did not use the required checkpoint pair")
    if len(summary.get("cases", [])) != 1:
        raise ValueError("suite requires exactly one fixed-envelope result per case")
    result = summary["cases"][0]
    required = {
        "requested_transitions": case["frames"] - 11,
        "gain_profile": "configured_sim",
        "action_fraction": 1.0,
        "ankle_effort_nm": 35.0,
        "target_slew_rad_s": 5.0,
        "stateful_native_controller": True,
        "encoder_decoder_pair_validated": True,
        "active_effort_target_projection": True,
        "actuation_trace_recorded": True,
        "observation_timing": "current_post_integration_pose_and_velocity_v2",
    }
    if any(result.get(key) != value for key, value in required.items()):
        raise ValueError("case shortened or changed the fixed paired actuation contract")
    if summary.get("sources", {}).get(case["motion"]["path"]) != case["motion"]["sha256"]:
        raise ValueError("case reference bytes differ from manifest input")
    if summary.get("authorization", {}).get("deployment_ready") is not False:
        raise ValueError("simulator result must remain unauthorized")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--encoder-report", type=Path, required=True)
    parser.add_argument("--decoder-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--measured-motion", action="append", default=[])
    parser.add_argument("--motor-health-snapshot", type=Path)
    parser.add_argument("--transition-balance-model", type=Path)
    args = parser.parse_args(argv)
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    measured_options = [
        bool(args.measured_motion),
        bool(args.motor_health_snapshot),
        bool(args.transition_balance_model),
    ]
    if any(measured_options) and not all(measured_options):
        parser.error("measured cases require both recorded health and standing model inputs")
    manifest, cases = manifest_cases(args.manifest, args.asset_root, args.repository_root)
    if len(set(args.measured_motion)) != len(args.measured_motion) or set(args.measured_motion) - {
        case["name"] for case in cases
    }:
        parser.error("measured motion names must be unique members of the manifest")
    pair = load_diagnostic_pair(args.encoder_report, args.decoder_report)
    manifest_identity, runner_identity = file_identity(args.manifest), file_identity(Path(__file__))
    optional_inputs = [
        file_identity(path) for path in (args.motor_health_snapshot, args.transition_balance_model) if path
    ]
    args.output_dir.mkdir(parents=True)
    results = []
    for case in cases:
        for measured in [False, True] if case["name"] in args.measured_motion else [False]:
            label = case["name"] + ("_measured" if measured else "_reference")
            destination = args.output_dir / label
            command = case_command(args, pair, case, destination, measured=measured)
            print("START", label, flush=True)
            subprocess.run(command, cwd=args.repository_root, check=True)
            summary_path = destination / "summary.json"
            summary = json.loads(summary_path.read_text())
            result = verify_case(summary, pair, case)
            results.append(
                {
                    "name": case["name"],
                    "measured_start": measured,
                    "command": command,
                    "summary": file_identity(summary_path),
                    "result": result,
                    "trajectories": [file_identity(path) for path in sorted(destination.glob("*.npz"))],
                }
            )
    inputs = [manifest_identity, runner_identity, *optional_inputs, *(case["motion"] for case in cases)]
    inputs.extend(
        {"path": pair[part]["report_path"], "sha256": pair[part]["report_sha256"]}
        for part in ("encoder", "decoder")
    )
    inputs.extend({"path": pair[part]["path"], "sha256": pair[part]["sha256"]} for part in ("encoder", "decoder"))
    if any(file_identity(Path(item["path"])) != item for item in inputs):
        raise ValueError("suite input bytes changed during evaluation")
    output = {
        "kind": "g1_true23_paired_envelope_suite_v1",
        "inputs": inputs,
        "diagnostic_pair": pair,
        "manifest_kind": manifest["kind"],
        "reference_clip_count": len(cases),
        "reference_frame_count": sum(case["frames"] for case in cases),
        "case_count": len(results),
        "cases": results,
        "full_clip_motion_fidelity_passed_count": sum(
            row["result"]["motion_fidelity"]["passed"] for row in results
        ),
        "paired_lifecycle_screen_passed_count": sum(
            row["result"]["paired_lifecycle_simulator_screen_passed"] for row in results
        ),
        "diagnostic_only": True,
        "deployment_ready": False,
        "hardware_authorized": False,
        "robot_commands_published": False,
        "dds_opened": False,
        "limitations": [
            "Reference resets are synthetic; full reference replay is not physical acquisition.",
            "Recorded posture is historical, not fresh hardware state; gantry forces are absent.",
            "Standing compatibility policy is not a native Unitree FSM handoff.",
            "All source clips remain represented; stance candidates remain unaccepted references.",
        ],
    }
    with (args.output_dir / "suite_summary.json").open("x") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(args.output_dir / "suite_summary.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
