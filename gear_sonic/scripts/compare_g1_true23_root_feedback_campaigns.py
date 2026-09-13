"""Matched full-lifecycle SIM comparison, recomputed from saved state traces.

Refuse changed references, physics, cases, initial state or acceptance thresholds.
Retain failures and all five landmarks; an improving average cannot hide a worse
foot or return phase. Neither geometric screens nor improvement authorizes motors.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from gear_sonic.utils.g1_true23_contact_step_transition import PROFILE as STEP_PROFILE
from gear_sonic.utils.g1_true23_generalist_benchmark import LANDMARKS
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file
from gear_sonic.utils.g1_true23_generalist_curriculum import MOTION_KEYS, array_digest
from gear_sonic.utils.g1_true23_generalist_lifecycle import assess_lifecycle_diagnostic

CASES = ("nominal", "standing_push_x", "standing_push_y")
PHYSICS_KEYS = (
    "compiled_model_sha256",
    "physics_config_sha256",
    "initial_state_and_history_sha256",
    "physical_dof",
    "actuator_count",
    "physics_hz",
    "policy_hz",
    "kp_hardware",
    "kd_hardware",
    "effort_limit_hardware_nm",
    "requested_controls",
    "available_controls",
)


def validate_reference_profile(report, expected_profile):
    if expected_profile not in ("none", STEP_PROFILE):
        raise ValueError("unsupported comparison reference profile")
    expected_kind = (
        "g1_true23_root_feedback_single_policy_lifecycle_campaign_v1"
        if expected_profile == "none"
        else "g1_true23_contact_step_lifecycle_policy_diagnostic_v1"
    )
    if report.get("kind") != expected_kind:
        raise ValueError("comparison report differs from explicitly requested reference profile")
    if expected_profile != "none" and (
        report.get("contact_step_reference_diagnostic") is not True
        or report["timeline"].get("generated_transition_profile") != expected_profile
        or report["timeline"]
        .get("contact_step_independent_geometry_audit", {})
        .get("provisional_geometry_screen_passed")
        is not True
        or "root_input_counterfactual" in report
    ):
        raise ValueError("step comparison requires independently audited unmodified-policy replays")


def phase_tracking(timeline, arrays, completed):
    """Expose entry/return failures without changing source acceptance criteria."""
    landmarks = np.asarray(arrays["landmark_error_m"])
    root = np.asarray(arrays["desired_root_position_w"]) - np.asarray(arrays["measured_root_position_w"])
    if (
        landmarks.shape != (completed, 5)
        or root.shape != (completed, 3)
        or not np.isfinite(landmarks).all()
        or not np.isfinite(root).all()
        or np.any(landmarks < 0)
    ):
        raise ValueError("phase tracking requires every finite completed-control error")
    names = [row[0] for row in LANDMARKS]
    result = []
    for phase in timeline["phases"]:
        start, stop = phase["control_start"], min(phase["control_stop"], completed)
        count = max(0, stop - start)
        result.append(
            dict(
                name=phase["name"],
                completed_controls=count,
                requested_controls=phase["requested_controls"],
                complete=count == phase["requested_controls"],
                landmark_position_p95_m=dict(
                    zip(names, np.percentile(landmarks[start:stop], 95, axis=0).tolist(), strict=True)
                )
                if count
                else None,
                last_completed_control_landmark_error_m=dict(zip(names, landmarks[stop - 1].tolist(), strict=True))
                if count
                else None,
                root_position_p95_m=float(np.percentile(np.linalg.norm(root[start:stop], axis=1), 95))
                if count
                else None,
                contact_or_lifecycle_qualification=False,
            )
        )
    return result


def read_verified_campaign(path, *, expected_profile="none"):
    inputs = {}

    def bind(path, expected=None):
        path = Path(path).resolve(strict=True)
        digest = sha256_file(path)
        if expected is not None and digest != expected:
            raise ValueError("comparison material bytes differ")
        inputs[str(path)] = digest
        return path

    path = bind(path)
    report = json.loads(path.read_text())
    validate_reference_profile(report, expected_profile)
    if (
        [r["case"] for r in report["records"]] != list(CASES)
        or report.get("only_one_hash_bound_policy_per_case") is not True
        or report.get("no_postinitial_robot_pose_rewrites") is not True
        or report.get("no_fallback_controller") is not True
        or report.get("hardware_authorized") is not False
        or report.get("deployment_ready") is not False
    ):
        raise ValueError("comparison requires complete scheduled single-policy SIM reports")
    timeline = report["timeline"]
    motion_path = bind(timeline["timeline_path"], timeline["timeline_sha256"])
    with np.load(motion_path, allow_pickle=False) as archive:
        reference_digest = array_digest({k: archive[k] for k in MOTION_KEYS})
        if not np.array_equal(archive["fps"], np.array([50.0])):
            raise ValueError("comparison requires the unchanged50Hz timeline")
    summaries, shared_policy = [], None
    for row in report["records"]:
        result = row["result"]
        identity = row["policy_identity"]
        bind(identity["manifest_path"], identity["manifest_sha256"])
        for component in ("encoder", "decoder"):
            bind(identity["component_paths"][component], identity[component + "_sha256"])
        if shared_policy is not None and identity != shared_policy:
            raise ValueError("comparison cannot mix policies between cases")
        shared_policy = identity
        if result["state_pose_writes_after_reset"] != 0 or result["history_resets_during_motion"] != 0:
            raise ValueError("comparison cannot conceal robot or history resets")
        trace_path = bind(row["trace_path"], row["trace_sha256"])
        with np.load(trace_path, allow_pickle=False) as archive:
            arrays = {
                key: archive[key].copy()
                for key in (
                    "qpos",
                    "qvel",
                    "landmark_error_m",
                    "desired_root_position_w",
                    "measured_root_position_w",
                )
            }
        lifecycle = assess_lifecycle_diagnostic(timeline, result, arrays)
        if lifecycle != row["lifecycle"]:
            raise ValueError("stored lifecycle metrics differ from independent state-trace recomputation")
        count = result["completed_controls"]
        root_error = np.linalg.norm(
            arrays["desired_root_position_w"][:count] - arrays["measured_root_position_w"][:count], axis=1
        )
        root_p95 = float(np.percentile(root_error, 95)) if count else None
        if root_p95 != result["root_response"]["current_root_position_error_p95_m"]:
            raise ValueError("stored root metric differs from independent state-trace recomputation")
        screen = lifecycle["source_motion_tracking"]
        summaries.append(
            dict(
                case=row["case"],
                complete=lifecycle["single_policy_full_lifecycle_integrated"],
                completed_controls=count,
                failure=result["failure"],
                root_p95_m=root_p95,
                landmarks_p95_m=screen["landmark_position_p95_m"],
                thresholds_m=screen["reference_landmark_thresholds_m"],
                source_landmark_screen_passed=screen["provisional_reference_landmark_screen_passed"],
                final_standing={k: v for k, v in lifecycle.items() if k.startswith("final_proof_")},
                phase_tracking=phase_tracking(timeline, arrays, count),
                physics={k: result[k] for k in PHYSICS_KEYS},
            )
        )
    for path, digest in inputs.items():
        if sha256_file(Path(path)) != digest:
            raise ValueError("comparison inputs changed while reading")
    return dict(
        generated_transition_profile=expected_profile,
        reference_arrays_sha256=reference_digest,
        phases=timeline["phases"],
        release_compatibility=report["release_compatibility"],
        policy=shared_policy["source"],
        cases=summaries,
        inputs=inputs,
    )


def compare_verified(before, after):
    if any(before[k] != after[k] for k in ("reference_arrays_sha256", "phases", "release_compatibility")) or (
        before.get("generated_transition_profile", "none") != after.get("generated_transition_profile", "none")
    ):
        raise ValueError("matched comparison cannot change reference, timing or runtime semantics")
    if [r["case"] for r in before["cases"]] != list(CASES) or [r["case"] for r in after["cases"]] != list(CASES):
        raise ValueError("matched comparison requires all three scheduled cases")
    results = []
    for old, new in zip(before["cases"], after["cases"], strict=True):
        if old["physics"] != new["physics"] or old["thresholds_m"] != new["thresholds_m"]:
            raise ValueError("matched comparison cannot change physics, initial state or tracking thresholds")
        comparable = bool(old["complete"] and new["complete"])
        changes = (
            {key: new["landmarks_p95_m"][key] - value for key, value in old["landmarks_p95_m"].items()}
            if comparable
            else None
        )
        results.append(
            dict(
                case=old["case"],
                before=old,
                after=new,
                both_full_lifecycles_completed=comparable,
                root_p95_change_m=new["root_p95_m"] - old["root_p95_m"] if comparable else None,
                landmark_p95_change_m=changes,
                final_standing_changes={k: new["final_standing"][k] - v for k, v in old["final_standing"].items()}
                if comparable
                else None,
                all_five_landmarks_nondegrading=bool(comparable and all(v <= 0 for v in changes.values())),
            )
        )
    return dict(
        kind="g1_true23_exact_reference_paired_policy_trace_comparison_v1",
        before_policy=before["policy"],
        after_policy=after["policy"],
        reference_arrays_sha256=before["reference_arrays_sha256"],
        generated_transition_profile=before.get("generated_transition_profile", "none"),
        same_complete_reference_physics_and_thresholds=True,
        trace_metrics_independently_recomputed=True,
        cases=results,
        before_source_landmark_screen_pass_count=sum(r["source_landmark_screen_passed"] for r in before["cases"]),
        after_source_landmark_screen_pass_count=sum(r["source_landmark_screen_passed"] for r in after["cases"]),
        all_six_full_lifecycles_completed=all(r["both_full_lifecycles_completed"] for r in results),
        input_bindings={**before["inputs"], **after["inputs"]},
        original29_fidelity_or_generalization_proven=False,
        dynamics_or_contact_qualification=False,
        simulator_qualified=False,
        hardware_authorized=False,
        deployment_ready=False,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generated-transition-profile", choices=("none", STEP_PROFILE), default="none")
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("policy comparison refuses overwrite")
    result = compare_verified(
        read_verified_campaign(args.before, expected_profile=args.generated_transition_profile),
        read_verified_campaign(args.after, expected_profile=args.generated_transition_profile),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({k: v for k, v in result.items() if k not in ("cases", "input_bindings")}), flush=True)
    for row in result["cases"]:
        print(json.dumps({k: v for k, v in row.items() if k not in ("before", "after")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
