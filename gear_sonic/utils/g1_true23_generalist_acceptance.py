"""Fail-closed, whole-suite acceptance accounting; no hardware authorization.

This consumes measured referee receipts, never interprets a prefix, a retarget
IK verdict, or a survival-only replay as full-lifecycle policy competence.
"""

from __future__ import annotations

import math

from gear_sonic.utils.g1_true23_generalist_corpus import canonical_digest


SEEDS = (1729, 2718, 3141)
PHASES = ("standing_acquisition", "motion", "standing_return")
ZERO_COUNTS = (
    "falls",
    "nonfinite_states",
    "forbidden_contacts",
    "actuator_violations",
    "pose_writes_after_reset",
    "history_resets_during_motion",
    "fallback_controls",
)
ERROR_LIMITS = {"left_foot": 0.05, "right_foot": 0.05, "left_hand": 0.1, "right_hand": 0.1, "head": 0.1}


def _finite(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def lifecycle_case_failures(case, planned=None):
    failures = []
    if case.get("status") != "executed":
        return ["motion rejected, missing or not executed"]
    if case.get("kind") != "g1_true23_generalist_measured_lifecycle_v1":
        failures.append("not a measured lifecycle receipt")
    if case.get("actuation_profile") != "g1_true23_nominal_native_model_actuation_v1":
        failures.append("not the shared nominal native-model profile")
    if case.get("metric_frame") != "world_source_aligned_no_per_frame_registration":
        failures.append("tracking frame missing or hides drift")
    if case.get("foot_metric") != "sole_contact_points":
        failures.append("sole/contact tracking not measured")
    for key in ZERO_COUNTS:
        if type(case.get(key)) is not int or case[key] != 0:
            failures.append(f"{key} must be measured zero")
    phases = case.get("phases", {})
    for name in PHASES:
        row = phases.get(name, {})
        expected, completed = row.get("expected_controls"), row.get("completed_controls")
        if type(expected) is not int or expected <= 0 or type(completed) is not int or completed != expected:
            failures.append(f"{name} not complete")
        if planned is not None and (
            expected != planned["phase_controls"][name] or completed != planned["phase_controls"][name]
        ):
            failures.append(f"{name} differs from immutable full-source evaluation plan")
    errors = case.get("adapted_reference_p95_m", {})
    for name, limit in ERROR_LIMITS.items():
        value = errors.get(name)
        if not _finite(value) or not 0 <= value <= limit:
            failures.append(f"{name} adapted-reference error exceeds {limit} m or missing")
    # Adaptation is reported separately; a small adapted-reference error may
    # not erase large changes made to the requested source choreography.
    source = case.get("source_reference_p95_m", {})
    if any(not _finite(source.get(name)) or source[name] < 0 for name in ERROR_LIMITS):
        failures.append("original-source distortion not measured")
    scale = case.get("duration_scale")
    if not _finite(scale) or not 1 <= scale <= 2:
        failures.append("tempo adaptation outside 1x..2x duration")
    if planned is not None:
        expected_scale = ((planned["phase_controls"]["motion"] - 1) / 50) / (
            (planned["source_frame_count"] - 1) / planned["source_fps"]
        )
        if not _finite(scale) or abs(scale - expected_scale) > 1e-6:
            failures.append("reported tempo differs from complete planned source timing")
        if case.get("source_coverage_frames") != [0, planned["source_frame_count"] - 1]:
            failures.append("source coverage does not include both original endpoints")
    reduction = case.get("maximum_task_space_excursion_reduction_fraction")
    if not _finite(reduction) or not 0 <= reduction <= 0.2:
        failures.append("task-space excursion reduction outside 0..20 percent")
    if case.get("source_time_map_verified") is not True:
        failures.append("source-to-adapted time map not verified")
    if case.get("perturbation_state_verified") is not True:
        failures.append("mass/friction/gains/noise/timing perturbation state not verified")
    if case.get("return_to_standing_stable") is not True:
        failures.append("standing return not stable")
    return failures


def validate_evaluation_plan(plan, corpus_audit, recording_ids):
    if (
        not isinstance(plan, dict)
        or plan.get("kind") != "g1_true23_generalist_full_source_plan_v1"
        or plan.get("split_sha256") != corpus_audit["split_sha256"]
        or plan.get("corpus_manifest_sha256") != corpus_audit["manifest_sha256"]
        or plan.get("seeds") != list(SEEDS)
        or set(plan.get("recordings", {})) != set(recording_ids)
    ):
        raise ValueError("immutable evaluation plan identity mismatch")
    for identifier, entry in plan["recordings"].items():
        asset = entry.get("source_asset_id")
        metadata = corpus_audit.get("asset_metadata", {}).get(asset, {})
        binding = corpus_audit.get("asset_bindings", {}).get(asset, {})
        timing = metadata.get("timing", {})
        if (
            metadata.get("recording_id") != identifier
            or metadata.get("transform") != "original"
            or binding.get("sha256") != entry.get("source_sha256")
            or timing.get("frame_count") != entry.get("source_frame_count")
            or timing.get("fps") != entry.get("source_fps")
        ):
            raise ValueError("evaluation plan source differs from complete original recording")
        frames, fps = entry["source_frame_count"], entry["source_fps"]
        if type(frames) is not int or frames < 2 or not _finite(fps) or fps <= 0:
            raise ValueError("invalid original source timing")
        controls = entry.get("phase_controls", {})
        if set(controls) != set(PHASES) or any(type(v) is not int or v <= 0 for v in controls.values()):
            raise ValueError("evaluation plan requires all three phase counts")
        # N samples span N-1 intervals; every source endpoint remains included.
        source_duration = (frames - 1) / fps
        motion_duration = (controls["motion"] - 1) / 50
        if (
            controls["standing_acquisition"] < 250
            or controls["standing_return"] < 250
            or motion_duration < source_duration - 1e-9
            or motion_duration > 2 * source_duration + 1e-9
        ):
            raise ValueError("evaluation plan truncates source or standing phases, or exceeds tempo bound")
        for key in ("adapted_reference_sha256", "time_map_sha256"):
            value = entry.get(key)
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("evaluation plan must bind adapted reference and full source time map")
    return canonical_digest(plan)


def assess_generalist_suite(
    *,
    corpus_audit,
    recording_ids,
    cases,
    policy_sha256,
    evaluation_plan,
    teleop_receipt=None,
    export_receipt=None,
    regression_receipt=None,
):
    """Every requested original is denominator; all three seeds must pass it."""
    if len(recording_ids) != len(set(recording_ids)) or not recording_ids:
        raise ValueError("evaluation plan must contain unique original recording IDs")
    if (
        not isinstance(policy_sha256, str)
        or len(policy_sha256) != 64
        or any(c not in "0123456789abcdef" for c in policy_sha256)
    ):
        raise ValueError("one checkpoint SHA256 required")
    if corpus_audit.get("manifest_valid") is not True:
        raise ValueError("validated original-recording corpus required")
    if any(corpus_audit.get("recording_splits", {}).get(identifier) != "test" for identifier in recording_ids):
        raise ValueError("all evaluated originals must belong to the pinned held-out test split")
    if any(corpus_audit.get("recording_families", {}).get(identifier) != "dance" for identifier in recording_ids):
        raise ValueError("the 100-dance campaign must identify original dance recordings")
    plan_sha256 = validate_evaluation_plan(evaluation_plan, corpus_audit, recording_ids)
    by_case = {}
    for case in cases:
        key = (case.get("recording_id"), case.get("seed"))
        if key in by_case or key[0] not in recording_ids or type(key[1]) is not int or key[1] not in SEEDS:
            raise ValueError("duplicate or unplanned evaluation case")
        if case.get("policy_sha256") != policy_sha256 or case.get("split_sha256") != corpus_audit["split_sha256"]:
            raise ValueError("mixed checkpoints or corpus splits cannot qualify one policy")
        planned = evaluation_plan["recordings"][key[0]]
        if (
            case.get("evaluation_plan_sha256") != plan_sha256
            or case.get("adapted_reference_sha256") != planned["adapted_reference_sha256"]
            or case.get("time_map_sha256") != planned["time_map_sha256"]
        ):
            raise ValueError("case reference or time map differs from immutable evaluation plan")
        by_case[key] = case
    outcomes = []
    for identifier in recording_ids:
        seed_reports = []
        for seed in SEEDS:
            case = by_case.get((identifier, seed))
            problems = (
                ["missing planned case"]
                if case is None
                else lifecycle_case_failures(case, evaluation_plan["recordings"][identifier])
            )
            seed_reports.append({"seed": seed, "passed": not problems, "failures": problems})
        outcomes.append(
            {
                "recording_id": identifier,
                "passed_all_seeds": all(r["passed"] for r in seed_reports),
                "seeds": seed_reports,
            }
        )
    passed = sum(row["passed_all_seeds"] for row in outcomes)
    fraction = passed / len(recording_ids)
    reasons = []
    if len(recording_ids) < 100:
        reasons.append("fewer than 100 independent held-out recordings")
    if corpus_audit.get("corpus_quantity_and_coverage_sufficient") is not True:
        reasons.append("corpus quantity/family coverage insufficient")
    if fraction < 0.95:
        reasons.append("fewer than 95 percent of original recordings pass all three seeds")
    extras = (
        (
            teleop_receipt,
            "teleop",
            (
                "saved_full_body_passed",
                "paced_stream_passed",
                "unexpected_changes_passed",
                "stale_input_passed",
                "standing_return_passed",
            ),
        ),
        (
            export_receipt,
            "export",
            ("encoder_parity_passed", "decoder_parity_passed", "causal_runtime_profile_verified"),
        ),
        (regression_receipt, "regression", ("all_full_requested_motions_passed", "full_duration_videos_bound")),
    )
    for receipt, label, required in extras:
        if (
            not isinstance(receipt, dict)
            or receipt.get("policy_sha256") != policy_sha256
            or any(receipt.get(key) is not True for key in required)
        ):
            reasons.append(f"{label} evidence incomplete or not bound to this checkpoint")
    return {
        "kind": "g1_true23_generalist_suite_assessment_v1",
        "policy_sha256": policy_sha256,
        "split_sha256": corpus_audit["split_sha256"],
        "evaluation_plan_sha256": plan_sha256,
        "planned_original_recordings": len(recording_ids),
        "passed_all_seeds_original_recordings": passed,
        "success_fraction": fraction,
        "rejections_and_missing_cases_count_as_failures": True,
        "outcomes": outcomes,
        "simulator_qualification_complete": not reasons,
        "remaining_requirements": reasons,
        "hardware_authorized": False,
        "deployment_ready": False,
    }


def progress_decision(evaluations):
    """Require diagnosis after three successive unchanged 100-update evaluations.

    Higher completion and lower task-space error are the only progress axes.
    This is a stop/review decision, not automatic permission to relax a gate.
    """
    if len(evaluations) < 4:
        return {"diagnosis_required": False, "reason": "not yet three comparison intervals"}
    window = evaluations[-4:]
    for row in window:
        if type(row.get("update_count")) is not int or row["update_count"] < 0:
            raise ValueError("evaluation update count invalid")
        for key in ("full_lifecycle_success_fraction", "mean_task_space_p95_m"):
            if not _finite(row.get(key)) or row[key] < 0:
                raise ValueError("progress metrics must be finite nonnegative measured values")
    if len({row.get("evaluation_plan_sha256") for row in window}) != 1 or not window[0].get(
        "evaluation_plan_sha256"
    ):
        raise ValueError("progress comparison must use identical evaluation plan")
    improved = []
    for previous, current in zip(window, window[1:]):
        if current["update_count"] - previous["update_count"] != 100:
            raise ValueError("progress receipts must be spaced by 100 updates")
        improved.append(
            current["full_lifecycle_success_fraction"] > previous["full_lifecycle_success_fraction"] + 1e-6
            or current["mean_task_space_p95_m"] < previous["mean_task_space_p95_m"] - 1e-6
        )
    stop = not any(improved)
    return {
        "diagnosis_required": stop,
        "reason": "three unchanged/worse evaluations" if stop else "measured progress",
        "threshold_relaxation_authorized": False,
    }
