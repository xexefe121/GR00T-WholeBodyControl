"""Explicit same-source excursion restoration for evaluated simulator training.

Not a general data-change waiver: original planner identity, full frame counts,
retiming, models and final fitting limits must match. No hardware operations.
"""

from __future__ import annotations

import json
from pathlib import Path

from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file


def validate_full_excursion_transition(previous, current, old_fit, new_fit, audit):
    for key in ("stage", "native_reference_model_sha256", "native_reference_sim_config_sha256"):
        if previous.get(key) != current.get(key):
            raise ValueError(f"reference transition changed fixed curriculum material: {key}")
    if previous["stage"] != "lifecycle" or previous["derived_arrays_sha256"] == current["derived_arrays_sha256"]:
        raise ValueError("reference transition requires distinct complete lifecycle arrays")
    for fit, curriculum in ((old_fit, previous), (new_fit, current)):
        if (
            fit.get("accepted") is not True
            or fit.get("source_field") != "planned_qpos50"
            or fit.get("recorded_policy_pose_used_as_choreography") is not False
            or fit.get("adapted_motion_sha256") != curriculum["original_training_inputs"]["motion_sha256"]
        ):
            raise ValueError("reference transition requires both accepted original-planner motion bindings")
        selected = fit.get("selected_attempt")
        if type(selected) is not int or not 0 <= selected < len(fit["attempts"]):
            raise ValueError("reference transition selected attempt is invalid")
        attempt = fit["attempts"][selected]
        spans = curriculum["derived_spans"]
        if (
            attempt.get("accepted") is not True
            or attempt.get("failures") != []
            or spans["clip_count"] != 1
            or len(spans["spans"]) != 1
            or spans["spans"][0]["original_source_frames"] != attempt["output_frames"]
        ):
            raise ValueError("reference transition cannot omit frames or add other recordings")
    for key in ("named_source_sha256", "source_frame_count", "source_fps", "compiled_models", "ik_config"):
        if old_fit[key] != new_fit[key]:
            raise ValueError(f"reference transition changed original source/model/configuration: {key}")
    old, new = (fit["attempts"][fit["selected_attempt"]] for fit in (old_fit, new_fit))
    for key in ("requested_duration_scale", "actual_duration_scale", "output_frames"):
        if old[key] != new[key]:
            raise ValueError(f"reference transition changed full source timing: {key}")
    if not 0.8 <= old["requested_excursion_scale"] < 1.0 or new["requested_excursion_scale"] != 1.0:
        raise ValueError("reference transition only restores reduced excursion to exactly full excursion")
    if {k: v for k, v in old_fit["limits"].items() if k != "excursion_scales"} != {
        k: v for k, v in new_fit["limits"].items() if k != "excursion_scales"
    }:
        raise ValueError("reference transition cannot change final fitting limits")
    if previous["derived_spans"]["total_frames"] != current["derived_spans"]["total_frames"]:
        raise ValueError("reference transition changed complete lifecycle frame count")
    if (
        audit.get("kind") != "g1_true23_planned29_all_original_timestamps_fk_audit_v1"
        or audit.get("original_timestamp_fidelity_passed") is not True
        or audit.get("failures") != []
        or audit.get("all_original_timestamps_and_endpoints_evaluated") is not True
        or audit.get("original_frames_evaluated") != new_fit["source_frame_count"]
        or audit.get("control_frames") != new["output_frames"]
        or audit.get("compiled_models") != new_fit["compiled_models"]
    ):
        raise ValueError("reference transition requires the passed complete original-time audit")
    return {
        "kind": "g1_true23_same_planner_full_excursion_reference_transition_v1",
        "old_motion_sha256": old_fit["adapted_motion_sha256"],
        "new_motion_sha256": new_fit["adapted_motion_sha256"],
        "old_derived_arrays_sha256": previous["derived_arrays_sha256"],
        "new_derived_arrays_sha256": current["derived_arrays_sha256"],
        "named_source_sha256": new_fit["named_source_sha256"],
        "old_excursion_scale": old["requested_excursion_scale"],
        "new_excursion_scale": 1.0,
        "actual_duration_scale": new["actual_duration_scale"],
        "original_source_frames": new_fit["source_frame_count"],
        "retimed_source_frames": new["output_frames"],
        "reference_arrays_changed": True,
        "same_reference_resume_claimed": False,
        "parent_evaluation_must_use_new_complete_reference": True,
        "actor_critic_optimizer_and_counters_preserved": True,
        "old_tracking_results_qualify_new_reference": False,
        "final_fitting_or_physical_limits_changed": False,
        "dynamics_teleop_or_generalization_qualified": False,
        "hardware_authorized": False,
        "deployment_ready": False,
    }


def bound_reference_transition(previous, current, *, previous_report, new_report, original_time_audit):
    paths = [Path(value).resolve(strict=True) for value in (previous_report, new_report, original_time_audit)]
    inputs = {str(path): sha256_file(path) for path in paths}
    old_fit, new_fit, audit = [json.loads(path.read_text()) for path in paths]
    report_path, motion_path = paths[1], paths[1].parent / "adapted.true23.npz"
    audit_inputs = audit.get("input_bindings", {})
    if (
        audit_inputs.get(str(report_path)) != inputs[str(report_path)]
        or audit_inputs.get(str(motion_path)) != new_fit["adapted_motion_sha256"]
    ):
        raise ValueError("original-time audit is not bound to the selected new reference")
    for raw_path, expected in audit_inputs.items():
        path = Path(raw_path).resolve(strict=True)
        if sha256_file(path) != expected:
            raise ValueError(f"reference transition audit material changed: {path}")
        inputs[str(path)] = expected
    result = validate_full_excursion_transition(previous, current, old_fit, new_fit, audit)
    for path, expected in inputs.items():
        if sha256_file(Path(path)) != expected:
            raise ValueError(f"reference transition input changed: {path}")
    result["inputs"] = inputs
    return result
