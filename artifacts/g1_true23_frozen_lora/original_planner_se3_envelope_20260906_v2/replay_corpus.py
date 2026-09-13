"""Full paired-controller diagnostic on all eight new original-source references.

The existing evaluator's case commands and strict result checks are reused.
This new diagnostic manifest kind is validated explicitly, not relabelled as
an accepted teacher or stance certificate. There is no robot or DDS path.
"""

import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

from gear_sonic.scripts import evaluate_g1_true23_paired_envelope_suite as suite
from gear_sonic.scripts.refine_g1_true23_reference_forces import dump
from gear_sonic.scripts.refine_g1_true23_stance_contacts import load_motion
from gear_sonic.utils.g1_true23_diagnostic_pair import load_diagnostic_pair
from gear_sonic.utils.g1_true23_sonic_library_replay import validate_library_motion

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ASSETS = ROOT.parent / "GR00T-WholeBodyControl"
FITTED = HERE.parent / "original_planner_se3_fit_20260906_v2"
BASELINE = HERE.parent / "projection_cost_20260906_v1/baseline100/eval"
FLAGS = dict(
    diagnostic_only=True,
    deployment_ready=False,
    hardware_authorized=False,
    teacher_accepted=False,
    robot_commands_published=False,
    dds_opened=False,
)


def main():
    if (HERE / "started.json").exists():
        raise FileExistsError("use a new immutable replay directory")
    manifest_path = FITTED / "motions.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["kind"] != "g1_true23_original_planner_se3_task_fit_diagnostic_manifest_v1" or any(
        manifest.get(flag) is not False for flag in ("teacher_accepted", "hardware_authorized", "deployment_ready")
    ):
        raise ValueError("expected an explicitly unaccepted original-source fitted manifest")
    original_path = (
        ROOT / "gear_sonic/config/sim_validation/g1_true23_frozen_lora_original_sonic_rehearsal_v1.json"
    )
    original = json.loads(original_path.read_text())["motions"]
    if [(e["name"], e["weight"]) for e in manifest["motions"]] != [
        (e["name"], e["weight"]) for e in original
    ] or len(original) != 8:
        raise ValueError("replay must retain every original clip, order and weight")
    args = SimpleNamespace(
        repository_root=ROOT,
        asset_root=ASSETS,
        encoder_report=BASELINE / "model_100.diagnostic.encoder.json",
        decoder_report=BASELINE / "model_100.diagnostic.decoder.json",
        motor_health_snapshot=HERE.parent / "readiness_audit_20260905_v1/motor_health.json",
        transition_balance_model=ASSETS
        / "artifacts/external/unitree_rl_mjlab/deploy/robots/g1/config/policy/velocity/v0/exported/policy.onnx",
    )
    pair = load_diagnostic_pair(args.encoder_report, args.decoder_report)
    inputs = [
        suite.file_identity(path)
        for path in (
            Path(__file__),
            Path(suite.__file__),
            manifest_path,
            FITTED / "report.json",
            original_path,
            args.encoder_report,
            args.decoder_report,
            args.motor_health_snapshot,
            args.transition_balance_model,
            Path(pair["encoder"]["path"]),
            Path(pair["decoder"]["path"]),
        )
    ]
    cases = []
    paths = set()
    for entry in manifest["motions"]:
        path = Path(entry["path"]).resolve(strict=True)
        if path in paths or not any(path.is_relative_to(root) for root in (ROOT, ASSETS)):
            raise ValueError("reference must be unique and within the two explicit repository roots")
        paths.add(path)
        identity = suite.file_identity(path)
        if identity["sha256"] != entry["sha256"]:
            raise ValueError("manifest reference hash mismatch")
        frames = validate_library_motion(load_motion(path))
        inputs.append(identity)
        cases.append({"name": entry["name"], "frames": frames, "motion": identity})
    dump(HERE / "started.json", {"inputs": inputs, "diagnostic_pair": pair, "complete": False, **FLAGS})
    records = []
    for case in cases:
        for measured in (False, True) if case["name"] == "original_sonic_happy_dance" else (False,):
            label = case["name"] + ("_measured" if measured else "_reference")
            destination = HERE / label
            command = suite.case_command(args, pair, case, destination, measured=measured)
            print(json.dumps({"starting_case": label, "requested_transitions": case["frames"] - 11}), flush=True)
            subprocess.run(command, cwd=ROOT, check=True)
            summary_path = destination / "summary.json"
            summary = json.loads(summary_path.read_text())
            result = suite.verify_case(summary, pair, case)
            records.append(
                {
                    "name": case["name"],
                    "measured_start": measured,
                    "command": command,
                    "summary": suite.file_identity(summary_path),
                    "result": result,
                    "trajectories": [suite.file_identity(path) for path in sorted(destination.glob("*.npz"))],
                }
            )
            dump(HERE / f"{label}.case.json", records[-1])
            print(
                json.dumps(
                    {
                        "case": label,
                        "completed": result.get("completed_transitions"),
                        "requested": result["requested_transitions"],
                        "fidelity": result["motion_fidelity"],
                        "lifecycle": result["paired_lifecycle_simulator_screen_passed"],
                    }
                ),
                flush=True,
            )
    if any(suite.file_identity(Path(item["path"])) != item for item in inputs):
        raise ValueError("replay source inputs changed")
    dump(
        HERE / "suite_summary.json",
        {
            "kind": "g1_true23_original_source_se3_paired_envelope_suite_v1",
            "inputs": inputs,
            "diagnostic_pair": pair,
            "reference_clip_count": len(cases),
            "reference_frame_count": sum(case["frames"] for case in cases),
            "case_count": len(records),
            "cases": records,
            "full_clip_motion_fidelity_passed_count": sum(
                row["result"]["motion_fidelity"]["passed"] for row in records
            ),
            "paired_lifecycle_screen_passed_count": sum(
                row["result"]["paired_lifecycle_simulator_screen_passed"] for row in records
            ),
            "historical_measured_pose_is_not_fresh_hardware_state": True,
            "standing_compatibility_model_is_not_native_unitree_fsm_handoff": True,
            **FLAGS,
        },
    )


if __name__ == "__main__":
    main()
