"""Fixed paired-controller diagnostics for both explicit hand-frame fits.

Preserve incomplete elbow in each source experiment. Byte-identical references
can reuse one measured result, explicitly labelled and hash-bound; they do not
count as another execution. This is not the full SONIC/PICO qualification.
"""

import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

from gear_sonic.scripts import evaluate_g1_true23_paired_envelope_suite as suite
from gear_sonic.scripts.record_g1_sonic_original29_baseline import CLIPS, dump
from gear_sonic.scripts.refine_g1_true23_stance_contacts import load_motion
from gear_sonic.utils.g1_sonic_cpp_parameters import file_sha256
from gear_sonic.utils.g1_true23_diagnostic_pair import load_diagnostic_pair
from gear_sonic.utils.g1_true23_hand_frame_tasks import HAND_FRAME_CONVENTION
from gear_sonic.utils.g1_true23_sonic_library_replay import validate_library_motion

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ASSETS = ROOT.parent / "GR00T-WholeBodyControl"
BASELINE = HERE.parent / "projection_cost_20260906_v1/baseline100/eval"
SOURCES = {
    "original_masks": HERE.parent / "original29_neutral_hand_frame_fit_20260906_v1",
    "hand_collisions": HERE.parent / "original29_hand_collision_neutral_fit_20260906_v1",
}
FLAGS = dict(teacher_accepted=False, hardware_authorized=False, deployment_ready=False)


def main():
    if (HERE / "started.json").exists():
        raise FileExistsError("preserve prior diagnostic; use a new directory")
    inputs, reports = {}, {}

    def bind(path, expected=None):
        path = Path(path).resolve(strict=True)
        digest = file_sha256(path)
        if digest != inputs.get(str(path), digest) or (expected is not None and digest != expected):
            raise ValueError(f"replay dependency changed: {path}")
        inputs[str(path)] = digest
        return path

    for label, folder in SOURCES.items():
        report = json.loads(bind(folder / "report.json").read_text())
        if (
            report.get("kind") != "g1_true23_neutral_wrist_hand_frame_fit_diagnostic_v1"
            or report.get("task_point_convention") != HAND_FRAME_CONVENTION
            or any(report.get(key) is not False for key in FLAGS)
            or [row["name"] for row in report["records"]] != list(CLIPS)
        ):
            raise ValueError("expected a complete-request, explicitly unaccepted hand-frame fit report")
        for path, digest in report["inputs"].items():
            bind(path, digest)
        reports[label] = report
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
        args.encoder_report,
        args.decoder_report,
        args.motor_health_snapshot,
        args.transition_balance_model,
        Path(pair["encoder"]["path"]),
        Path(pair["decoder"]["path"]),
    ):
        bind(path)
    dump(HERE / "started.json", {"inputs": dict(inputs), "pair": pair, **FLAGS})
    records, cache, executions = [], {}, 0
    for source_name, fitted in reports.items():
        for row in fitted["records"]:
            common = {"source": source_name, "source_kind": fitted["source_kind"], "name": row["name"]}
            if row["failure"] is not None or row["output"] is None:
                records.append({**common, "not_executed": True, "reason": row["failure"]})
                continue
            path = bind(row["output"], row["output_sha256"])
            case = {
                "name": row["name"],
                "frames": validate_library_motion(load_motion(path)),
                "motion": suite.file_identity(path),
            }
            for measured in (False, True) if row["name"] == "happy_dance" else (False,):
                label = source_name + "_" + row["name"] + ("_historical_measured" if measured else "_reference")
                key = (case["motion"]["sha256"], case["frames"], measured)
                if key in cache:
                    records.append(
                        {
                            **common,
                            "measured": measured,
                            "executed_again": False,
                            "reused_case_label": cache[key]["label"],
                            "identical_reference_sha256": case["motion"]["sha256"],
                            "same_fixed_pair_contract_and_historical_inputs": True,
                            "result": cache[key]["result"],
                        }
                    )
                    print(json.dumps({"case": label, "reused_identical_case": cache[key]["label"]}), flush=True)
                    continue
                destination = HERE / label
                command = suite.case_command(args, pair, case, destination, measured=measured)
                print(json.dumps({"starting_case": label}), flush=True)
                subprocess.run(command, cwd=ROOT, check=True)
                summary_path = destination / "summary.json"
                summary = json.loads(bind(summary_path).read_text())
                result = suite.verify_case(summary, pair, case)
                for source_path, digest in summary["sources"].items():
                    bind(source_path, digest)
                for trace in destination.glob("*.npz"):
                    bind(trace)
                records.append(
                    {**common, "label": label, "measured": measured, "command": command, "result": result}
                )
                cache[key] = {"label": label, "result": result}
                executions += 1
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
    for path in list(inputs):
        bind(path)
    dump(
        HERE / "suite_summary.json",
        {
            "kind": "g1_true23_hand_frame_fit_paired_diagnostic_v1",
            "inputs": inputs,
            "pair": pair,
            "records": records,
            "actual_controller_case_executions": executions,
            "all_original_clips_qualified": False,
            "full_eight_clip_sonic_pico_qualification_performed": False,
            "pico_corpus_rerun": False,
            "hand_task_convention_not_promoted_to_policy_inputs": True,
            "historical_posture_is_not_fresh_robot_state": True,
            "standing_policy_is_not_native_unitree_fsm_handoff": True,
            **FLAGS,
        },
    )


if __name__ == "__main__":
    main()
