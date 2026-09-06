"""Offline fixed-pair native23 effort-headroom comparison with full requests.

Preserve both complete recorded-source fits and all five contact-restored PICO
clips. The incomplete original elbow remains unavailable, never replaced by
the historical mislabelled controller rollout. Test all selected fractions,
including the exact outer-envelope baseline and historical standing lifecycle.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys

import mujoco
import numpy as np
import onnxruntime as ort

from gear_sonic.scripts import evaluate_g1_true23_deployment_envelope as envelope
from gear_sonic.scripts.evaluate_g1_true23_paired_envelope_suite import manifest_cases
from gear_sonic.scripts.record_g1_sonic_original29_baseline import CLIPS, dump
from gear_sonic.utils.g1_sonic_cpp_parameters import file_sha256
from gear_sonic.utils.g1_true23_actuation_profile import NativeSupportActuationProfile, read_joint_amplitude_scale
from gear_sonic.utils.g1_true23_clean_mujoco_teleop import UnitreeZeroVelocityFallbackPolicy
from gear_sonic.utils.g1_true23_diagnostic_pair import load_diagnostic_pair
from gear_sonic.utils.g1_true23_hand_frame_tasks import HAND_FRAME_CONVENTION
from gear_sonic.utils.g1_true23_interior_target_filter import run_interior_case
from gear_sonic.utils.g1_true23_sonic_library_replay import validate_library_motion

PICO_NAMES = (
    "pico_upright_anchor",
    "pico_standing_anchor",
    "pico_crouch_anchor",
    "pico_twist2_walk_001",
    "pico_twist2_walk_010",
)
FLAGS = dict(teacher_accepted=False, hardware_authorized=False, deployment_ready=False)


def select_fit_rows(report):
    if (
        report.get("kind") != "g1_true23_neutral_wrist_hand_frame_fit_diagnostic_v1"
        or report.get("task_point_convention") != HAND_FRAME_CONVENTION
        or any(report.get(key) is not False for key in FLAGS)
        or [row["name"] for row in report.get("records", [])] != list(CLIPS)
    ):
        raise ValueError("requires all three explicitly unaccepted corrected-hand source fit attempts")
    return report["records"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--fit-report", type=Path, required=True)
    parser.add_argument("--pico-manifest", type=Path, required=True)
    parser.add_argument("--encoder-report", type=Path, required=True)
    parser.add_argument("--decoder-report", type=Path, required=True)
    parser.add_argument("--baseline-envelope-dir", type=Path, required=True)
    parser.add_argument("--motor-health-snapshot", type=Path, required=True)
    parser.add_argument("--transition-balance-model", type=Path, required=True)
    parser.add_argument("--fractions", nargs="+", type=float, default=[1.0, 0.95, 0.85, 0.7])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.fractions[0] != 1.0
        or len(set(args.fractions)) != len(args.fractions)
        or any(not np.isfinite(f) or not 0 <= f <= 1 for f in args.fractions)
    ):
        parser.error("unique finite inner fractions in 0..1 must start with the exact 1.0 baseline")
    root, assets = Path(__file__).resolve().parents[2], args.asset_root.resolve(strict=True)
    output, old_folder = args.output_dir.resolve(), args.baseline_envelope_dir.resolve(strict=True)
    inputs = {}

    def bind(path, expected=None):
        path = Path(path).resolve(strict=True)
        digest = file_sha256(path)
        if digest != inputs.get(str(path), digest) or (expected is not None and digest != expected):
            raise ValueError(f"interior experiment source changed: {path}")
        inputs[str(path)] = digest
        return path

    fit = json.loads(bind(args.fit_report).read_text())
    fits = select_fit_rows(fit)
    for path, digest in fit["inputs"].items():
        bind(path, digest)
    _, candidates = manifest_cases(bind(args.pico_manifest), assets, root)
    pico = [row for row in candidates if row["name"] in PICO_NAMES]
    if [row["name"] for row in pico] != list(PICO_NAMES):
        raise ValueError("all five exact PICO requests must remain in order")
    cases = []
    for row in fits:
        if row["output"] is None or row["failure"] is not None:
            cases.append({"name": row["name"], "motion": None, "unavailable": row["failure"]})
        else:
            path = bind(row["output"], row["output_sha256"])
            cases.append({"name": row["name"], "motion": path, "unavailable": None})
    for row in pico:
        path = bind(row["motion"]["path"], row["motion"]["sha256"])
        cases.append({"name": row["name"], "motion": path, "unavailable": None})
    pair = load_diagnostic_pair(bind(args.encoder_report), bind(args.decoder_report))
    for part in ("encoder", "decoder"):
        bind(pair[part]["path"], pair[part]["sha256"])
    policy = envelope._policy(
        Path(pair["encoder"]["path"]),
        Path(pair["decoder"]["path"]),
        pair["decoder"]["sha256"],
        encoder_hash=pair["encoder"]["sha256"],
    )
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    balance = UnitreeZeroVelocityFallbackPolicy(bind(args.transition_balance_model), session_options=options)
    measured = envelope.load_measured_initial_state(bind(args.motor_health_snapshot))
    profile = NativeSupportActuationProfile.from_sim_config(bind(root / envelope.PHYSICS))
    scale = np.asarray(read_joint_amplitude_scale(bind(root / envelope.HEADER).read_text()))
    if not np.array_equal(scale, np.ones(23)):
        raise ValueError("this full-target comparison requires identity per-joint amplitude scaling")
    bind(assets / envelope.MODEL)
    old_summary = json.loads(bind(old_folder / "suite_summary.json").read_text())
    if old_summary.get("pair") != pair:
        raise ValueError("historical comparison uses a different encoder/decoder pair")
    for path, digest in old_summary["inputs"].items():
        bind(path, digest)
    bind(Path(__file__))
    for name, module in list(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if name.startswith("gear_sonic.") and path and Path(path).suffix == ".py":
            bind(path)
    output.mkdir(parents=True, exist_ok=False)
    started = {
        "kind": "g1_true23_interior_effort_full_request_comparison_v1",
        "inputs": dict(inputs),
        "pair": pair,
        "requested_clips": [row["name"] for row in cases],
        "inner_fractions": args.fractions,
        "outer_profile": profile.contract(),
        "action_fraction": 1.0,
        "outer_ankle_effort_nm": 35.0,
        "outer_slew_rad_s": 5.0,
        "history_and_applied_target_feedback_unchanged": True,
        "old_manifest_sonic_entries_replaced_by_recorded_source_fit_attempts": True,
        "pico_contact_restored_lineage_preserved": True,
        "new_cpp_observation_source_fit_not_silently_substituted": True,
        "historical_posture_is_not_fresh_robot_state": True,
        "standing_actor_is_not_native_unitree_fsm_handoff": True,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "mujoco": mujoco.__version__,
            "onnxruntime": ort.__version__,
        },
        **FLAGS,
    }
    dump(output / "started.json", started)
    records = []
    old_function = envelope.run_case
    for fraction in args.fractions:
        for case in cases:
            for historical in (False, True) if case["name"] == "happy_dance" else (False,):
                label = f"inner{fraction:g}.{case['name']}." + ("historical" if historical else "reference")
                record = {
                    "label": label,
                    "name": case["name"],
                    "inner_fraction": fraction,
                    "historical_start": historical,
                    "failure": None,
                    **FLAGS,
                }
                if case["motion"] is None:
                    record.update(not_executed=True, reason=case["unavailable"])
                    records.append(record)
                    continue
                print(json.dumps({"starting_native23_case": label}), flush=True)
                with np.load(case["motion"], allow_pickle=False) as archive:
                    motion = {key: np.ascontiguousarray(archive[key]) for key in archive.files}
                count = validate_library_motion(motion)
                report, arrays = run_interior_case(
                    fraction=fraction,
                    root=root,
                    asset_root=assets,
                    policy=policy,
                    motion=motion,
                    kp=np.asarray(profile.kp),
                    kd=np.asarray(profile.kd),
                    joint_scale=scale,
                    ankle_effort=35.0,
                    slew_rate=5.0,
                    initial_state="measured" if historical else "reference",
                    maximum_steps=None,
                    measured_state=measured if historical else None,
                    startup_hold_s=5.0 if historical else 0.0,
                    return_hold_s=5.0 if historical else 0.0,
                    transition_policy=balance if historical else None,
                    align_reference_start=historical,
                    project_transition_effort=historical,
                    project_active_effort=True,
                    stateful_native_controller=True,
                    trace_active_actuation=True,
                )
                if report["requested_transitions"] != count - 11 or envelope.run_case is not old_function:
                    raise ValueError("source frame count or shared evaluator changed")
                path = output / f"{label}.npz"
                with path.open("xb") as stream:
                    np.savez_compressed(stream, **arrays)
                bind(path)
                record.update(
                    result=report,
                    trace_path=str(path),
                    trace_sha256=inputs[str(path)],
                    source_motion_path=str(case["motion"]),
                    source_motion_sha256=inputs[str(case["motion"])],
                )
                if fraction == 1.0 and case["name"] in ("hand_crawling", "happy_dance"):
                    previous = old_folder / (
                        f"original_masks_{case['name']}_" + ("historical_measured" if historical else "reference")
                    )
                    previous /= (
                        "configured_sim_fraction1_ankle35_slew5_"
                        + ("measured" if historical else "reference")
                        + ".npz"
                    )
                    with np.load(bind(previous), allow_pickle=False) as archive:
                        checks = {key: np.array_equal(arrays[key], archive[key]) for key in archive.files}
                    record["baseline_array_equivalence"] = checks
                    if not all(checks.values()):
                        raise ValueError(f"fraction-one baseline changed: {label}")
                records.append(record)
                dump(output / f"{label}.json", record)
                bind(output / f"{label}.json")
                print(
                    json.dumps(
                        {
                            "case": label,
                            "completed": report["completed_transitions"],
                            "requested": report["requested_transitions"],
                            "failure": report["failure"],
                            "motion_fidelity": report["motion_fidelity"]["passed"],
                            "return_passed": report["return_hold"].get("existing_guard_screen_passed"),
                            "engine_passed": report["actual_engine_audit"]["passed"],
                        }
                    ),
                    flush=True,
                )
    for path in list(inputs):
        bind(path)
    dump(
        output / "report.json",
        {
            **started,
            "inputs": inputs,
            "records": records,
            "all_requested_variants_represented": len(records) == len(args.fractions) * 9,
            "full_eight_clip_qualification": False,
            "default_candidate_selected": False,
            "training_or_deployed_controller_modified": False,
        },
    )


if __name__ == "__main__":
    main()
