"""Diagnostic replay of every complete newly fitted source clip.

Incomplete elbow remains an explicit unavailable case. This is not a complete
eight-clip qualification and does not replace the unchanged PICO evidence.
"""

import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

from gear_sonic.scripts import evaluate_g1_true23_paired_envelope_suite as suite
from gear_sonic.scripts.record_g1_sonic_original29_baseline import dump
from gear_sonic.scripts.refine_g1_true23_stance_contacts import load_motion
from gear_sonic.utils.g1_sonic_cpp_parameters import file_sha256
from gear_sonic.utils.g1_true23_diagnostic_pair import load_diagnostic_pair
from gear_sonic.utils.g1_true23_sonic_library_replay import validate_library_motion

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ASSETS = ROOT.parent / "GR00T-WholeBodyControl"
FITTED = HERE.parent / "original29_recorded_native23_fit_20260906_v1"
BASELINE = HERE.parent / "projection_cost_20260906_v1/baseline100/eval"
FLAGS = dict(teacher_accepted=False, hardware_authorized=False, deployment_ready=False)


def main():
    if (HERE / "started.json").exists():
        raise FileExistsError("use a new diagnostic directory")
    fitted = json.loads((FITTED / "report.json").read_text())
    if fitted["kind"] != "g1_true23_original29_recorded_task_fit_diagnostic_v1" or any(
        fitted.get(key) is not False for key in FLAGS
    ):
        raise ValueError("expected an unaccepted recorded-policy fit")
    inputs = dict(fitted["inputs"])
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
    for path in (
        Path(__file__),
        Path(suite.__file__),
        FITTED / "report.json",
        args.encoder_report,
        args.decoder_report,
        args.motor_health_snapshot,
        args.transition_balance_model,
        Path(pair["encoder"]["path"]),
        Path(pair["decoder"]["path"]),
    ):
        inputs[str(path)] = file_sha256(path)
    if any(file_sha256(path) != digest for path, digest in inputs.items()):
        raise ValueError("recorded-fit source changed")
    dump(HERE / "started.json", {"inputs": inputs, "pair": pair, **FLAGS})
    records = []
    for row in fitted["records"]:
        if row["failure"] is not None or row["output"] is None:
            records.append({"name": row["name"], "not_executed": True, "reason": row["failure"]})
            continue
        path = Path(row["output"])
        if file_sha256(path) != row["output_sha256"]:
            raise ValueError("fitted reference changed")
        case = {
            "name": row["name"],
            "frames": validate_library_motion(load_motion(path)),
            "motion": suite.file_identity(path),
        }
        for measured in (False, True) if row["name"] == "happy_dance" else (False,):
            label = row["name"] + ("_historical_measured" if measured else "_reference")
            destination = HERE / label
            command = suite.case_command(args, pair, case, destination, measured=measured)
            print(json.dumps({"starting_case": label}), flush=True)
            subprocess.run(command, cwd=ROOT, check=True)
            summary_path = destination / "summary.json"
            summary = json.loads(summary_path.read_text())
            result = suite.verify_case(summary, pair, case)
            for source_path, digest in summary["sources"].items():
                if source_path in inputs and inputs[source_path] != digest:
                    raise ValueError("replay disagrees on an existing input")
                inputs[source_path] = digest
            inputs[str(summary_path)] = file_sha256(summary_path)
            for trace in destination.glob("*.npz"):
                inputs[str(trace)] = file_sha256(trace)
            records.append({"name": row["name"], "measured": measured, "command": command, "result": result})
            dump(HERE / f"{label}.case.json", records[-1])
            print(
                json.dumps(
                    {
                        "case": label,
                        "completed": result["completed_transitions"],
                        "requested": result["requested_transitions"],
                    }
                ),
                flush=True,
            )
    if any(file_sha256(path) != digest for path, digest in inputs.items()):
        raise ValueError("replay evidence changed")
    dump(
        HERE / "suite_summary.json",
        {
            "kind": "g1_true23_original29_recorded_fit_paired_diagnostic_v1",
            "inputs": inputs,
            "pair": pair,
            "records": records,
            "all_original_clips_qualified": False,
            "pico_corpus_rerun": False,
            "historical_posture_is_not_fresh_robot_state": True,
            "standing_policy_is_not_native_unitree_fsm_handoff": True,
            **FLAGS,
        },
    )


if __name__ == "__main__":
    main()
