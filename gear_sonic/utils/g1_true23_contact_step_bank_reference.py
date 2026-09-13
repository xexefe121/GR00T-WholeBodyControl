"""Explicit changed-transition training branch on an unchanged audited source bank."""

import json
from pathlib import Path

from gear_sonic.utils.g1_true23_contact_bank_reference import CONTACT_PROFILE
from gear_sonic.utils.g1_true23_contact_step_transition import PROFILE
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file

INDEX_KIND = "g1_true23_contact_step_reference_bank_index_v1"
PHASES = (
    "initial_standing",
    "acquisition_ramp",
    "source_motion",
    "return_ramp",
    "returned_standing",
    "standing_proof_margin",
)


def _validate_layout(row, expected_start):
    timeline = row["timeline"]
    phases = timeline.get("phases", [])
    if (
        [phase.get("name") for phase in phases] != list(PHASES)
        or timeline.get("prehistory_frames") != 11
        or row.get("start") != expected_start
        or timeline.get("source_frame_indices") != list(range(row["original_source_frames"]))
    ):
        raise ValueError("contact-step lifecycle must retain contiguous full recording coverage")
    cursor = 0
    for phase in phases:
        count = phase.get("requested_controls")
        if (
            type(count) is not int
            or count < 1
            or phase.get("control_start") != cursor
            or phase.get("control_stop") != cursor + count
            or phase.get("frame_start") != cursor + 11
            or phase.get("frame_stop") != cursor + count + 11
            or (phase["name"] in ("acquisition_ramp", "return_ramp") and not 100 <= count <= 1000)
            or (phase["name"] == "source_motion" and count != row["original_source_frames"])
        ):
            raise ValueError("contact-step phase offsets, duration or source coverage changed")
        cursor += count
    if (
        timeline.get("total_requested_controls") != cursor
        or timeline.get("total_frames") != cursor + 11
        or row.get("length") != cursor + 11
    ):
        raise ValueError("contact-step total span must cover every lifecycle frame")
    return expected_start + cursor + 11


def load_step_reference_index(path, bank_path):
    path, bank_path = Path(path).resolve(strict=True), Path(bank_path).resolve(strict=True)
    index, bank = json.loads(path.read_text()), json.loads(bank_path.read_text())
    if (
        index.get("kind") != INDEX_KIND
        or index.get("bank_report_sha256") != sha256_file(bank_path)
        or index.get("profile") != PROFILE
        or index.get("hardware_authorized") is not False
        or index.get("deployment_ready") is not False
        or [row["name"] for row in index.get("references", [])] != [row["name"] for row in bank["clips"]]
    ):
        raise ValueError("contact-step index must explicitly bind every unchanged bank member")
    inputs, result = {str(path): sha256_file(path), str(bank_path): sha256_file(bank_path)}, {}
    for row in index["references"]:
        reference = Path(row["report_path"]).resolve(strict=True)
        if sha256_file(reference) != row["report_sha256"] or row["name"] in result:
            raise ValueError("contact-step member bytes or unique identity changed")
        inputs[str(reference)] = row["report_sha256"]
        result[row["name"]] = {**row, "report_path": str(reference)}
    return result, inputs


def validate_step_bank_transition(previous, current, *, explicitly_allowed, allow_contact_conditioning=False):
    """Permit only declared generated entry/return changes and optional contact source branch."""
    contact_changed = previous.get("source_reference_conditioning", "none") != current.get(
        "source_reference_conditioning", "none"
    )
    if (
        not explicitly_allowed
        or previous.get("generated_transition_profile", "none") != "none"
        or current.get("generated_transition_profile") != PROFILE
        or current.get("source_reference_conditioning") != CONTACT_PROFILE
        or previous.get("source_start_registration") != current.get("source_start_registration")
        or previous["original_training_inputs"]["motion_sha256"]
        != current["original_training_inputs"]["motion_sha256"]
        or previous["derived_arrays_sha256"] == current["derived_arrays_sha256"]
        or contact_changed != allow_contact_conditioning
        or (contact_changed and previous.get("source_reference_conditioning", "none") != "none")
    ):
        raise ValueError("contact-step training needs its own explicit changed-reference transition")
    old_rows, new_rows = previous["derived_spans"]["spans"], current["derived_spans"]["spans"]
    if [row["name"] for row in old_rows] != [row["name"] for row in new_rows]:
        raise ValueError("contact-step transition cannot add, remove or reorder recordings")
    old_start = new_start = 0
    for old, new in zip(old_rows, new_rows, strict=True):
        old_start, new_start = _validate_layout(old, old_start), _validate_layout(new, new_start)
        for key in (
            "asset_id",
            "ownership",
            "original_source_frames",
            "source_arrays_sha256",
            "original_source_indices_requested",
            "every_original_source_frame_requested",
        ):
            if old.get(key) != new.get(key):
                raise ValueError(f"contact-step changed original recording identity/coverage: {key}")
        if not contact_changed and any(
            old.get(key) != new.get(key)
            for key in (
                "registered_source_arrays_sha256",
                "source_start_registration",
                "source_reference_conditioning",
            )
        ):
            raise ValueError("contact-step-only transition changed source/reference calibration")
        proof = new.get("source_reference_conditioning", {})
        if contact_changed and (
            proof.get("profile") != CONTACT_PROFILE
            or proof.get("original_source_arrays_sha256") != old["source_arrays_sha256"]
            or proof.get("original_registered_source_arrays_sha256") != old.get("registered_source_arrays_sha256")
            or proof.get("final_registered_source_arrays_sha256") != new.get("registered_source_arrays_sha256")
            or proof.get("final_source_start_registration") != new.get("source_start_registration")
            or proof.get("geometry", {}).get("independent_serialized_geometry_passed") is not True
            or proof.get("all_original_frames_retained_without_retiming") is not True
            or any(
                type(proof.get(key)) is not int or not 0 <= proof[key] <= new["original_source_frames"]
                for key in (
                    "remaining_conditional_force_support_failures",
                    "remaining_conditional_effort_failures",
                )
            )
            or any(
                proof.get(key) is not False
                for key in (
                    "raw_original_fidelity_acceptance_inherited",
                    "same_reference_benchmark_claimed",
                    "measured_robot_state_used",
                    "dynamic_feasibility_proven",
                    "hardware_authorized",
                    "deployment_ready",
                )
            )
        ):
            raise ValueError("combined contact/step branch lacks independently verified source conditioning")
        before, after = old["timeline"], new["timeline"]
        if [p["name"] for p in before["phases"]] != [p["name"] for p in after["phases"]]:
            raise ValueError("contact-step cannot omit lifecycle phases")
        for key in ("configured_standing_qpos", "returned_standing_qpos", "source_frames", "source_timing_scale"):
            if not contact_changed and before.get(key) != after.get(key):
                raise ValueError("contact-step-only transition changed source or standing state")
            if key in ("configured_standing_qpos", "source_frames", "source_timing_scale") and before.get(
                key
            ) != after.get(key):
                raise ValueError("contact-step changed initial state, source length or source speed")
        for old_phase, new_phase in zip(before["phases"], after["phases"], strict=True):
            if (
                old_phase["name"] not in ("acquisition_ramp", "return_ramp")
                and old_phase["requested_controls"] != new_phase["requested_controls"]
            ):
                raise ValueError("contact-step altered source or standalone standing duration")
        receipt = after.get("contact_step_reference", {})
        failures = after.get("contact_step_conditional_support_failures", {})
        transition_counts = {p["name"]: p["requested_controls"] for p in after["phases"]}
        if (
            receipt.get("profile") != PROFILE
            or not receipt.get("input_bindings")
            or after.get("contact_step_independent_geometry_audit", {}).get("provisional_geometry_screen_passed")
            is not True
            or after.get("contact_step_dynamic_feasibility_qualified") is not False
            or set(failures) != {"acquisition_ramp", "return_ramp"}
            or any(
                type(value) is not int or not 0 <= value <= transition_counts[name] + 2
                for name, value in failures.items()
            )
        ):
            raise ValueError("contact-step training lacks independent full-reference geometry evidence")
    return dict(
        kind="g1_true23_explicit_contact_step_bank_training_transition_v1",
        reference_arrays_changed=True,
        generated_transition_timing_changed=True,
        source_contact_conditioning_also_changed=contact_changed,
        same_reference_resume_claimed=False,
        original_audited_source_arrays_preserved=True,
        every_original_source_frame_retained=True,
        original_source_timing_preserved=True,
        actor_critic_optimizer_and_counters_preserved=True,
        every_bank_clip_requires_separate_complete_scheduled_cpu_comparison=True,
        step_support_failures={
            row["name"]: row["timeline"]["contact_step_conditional_support_failures"] for row in new_rows
        },
        source_support_failures={
            row["name"]: {
                key: row["source_reference_conditioning"][key]
                for key in (
                    "remaining_conditional_force_support_failures",
                    "remaining_conditional_effort_failures",
                )
            }
            for row in new_rows
        },
        dynamic_feasibility_proven=False,
        held_out_generalization_verified=False,
        hardware_authorized=False,
        deployment_ready=False,
    )
