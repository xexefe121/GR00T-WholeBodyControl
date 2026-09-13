"""Explicit small-bank expansion or repair with complete simulator comparisons.

Not a corpus, runtime or safety waiver. Previously trained source and lifecycle
arrays must remain intact; newly added clips retain their existing audit/split
proofs. Every selected clip must have all three scheduled CPU outcomes from the
same actual parent checkpoint, including failures. A separate repair plan may
replace collision-invalid references and quarantine omitted recordings; that is
never labelled an unchanged expansion or evidence of generalization. No robot control.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file
from gear_sonic.utils.g1_true23_generalist_curriculum import MOTION_KEYS, array_digest


def load_bank_lifecycle_repairs(
    index_path, bank_path, *, source_start_registration="none", source_reference_conditioning="none"
):
    from gear_sonic.utils.g1_true23_contact_bank_reference import (
        CONTACT_PROFILE,
        CONTACT_REPAIR_INDEX_KIND,
        load_contact_repair_member,
    )
    from gear_sonic.utils.g1_true23_registered_bank_reference import (
        REGISTERED_REPAIR_INDEX_KIND,
        START_REGISTRATION_PROFILE,
        load_registered_repair_member,
    )

    if source_start_registration not in {"none", START_REGISTRATION_PROFILE}:
        raise ValueError("unsupported bank source-start registration profile")
    registered = source_start_registration != "none"
    conditioned = source_reference_conditioning != "none"
    if source_reference_conditioning not in {"none", CONTACT_PROFILE} or (conditioned and not registered):
        raise ValueError("contact bank requires explicit registered source conditioning")
    expected_kind = (
        CONTACT_REPAIR_INDEX_KIND
        if conditioned
        else (REGISTERED_REPAIR_INDEX_KIND if registered else "g1_true23_evaluated_bank_lifecycle_repairs_v1")
    )
    index_path, bank_path = Path(index_path).resolve(strict=True), Path(bank_path).resolve(strict=True)
    inputs = {str(path): sha256_file(path) for path in (index_path, bank_path)}
    index, bank = json.loads(index_path.read_text()), json.loads(bank_path.read_text())
    if (
        index.get("kind") != expected_kind
        or index.get("bank_report_sha256") != inputs[str(bank_path)]
        or bank.get("kind") != "g1_true23_small_collision_audited_local_reference_bank_v2"
        or bank.get("accepted_reference_bank") is not True
        or [row["name"] for row in index.get("repairs", [])] != [row["name"] for row in bank["clips"]]
    ):
        raise ValueError("lifecycle repair index must bind every member of the accepted collision-audited bank")
    repairs = {}
    for row, clip in zip(index["repairs"], bank["clips"], strict=True):
        path = Path(row["report_path"]).resolve(strict=True)
        if (
            row["name"] in repairs
            or (not registered and row["source_motion_sha256"] != clip["source_sha256"])
            or sha256_file(path) != row["report_sha256"]
        ):
            raise ValueError("lifecycle repair member identity or report bytes differ")
        if conditioned:
            row, contact_inputs = load_contact_repair_member(row, clip)
            inputs.update(contact_inputs)
        elif registered:
            row, registration_inputs = load_registered_repair_member(row, clip)
            inputs.update(registration_inputs)
        repairs[row["name"]] = {**row, "report_path": str(path)}
        inputs[str(path)] = row["report_sha256"]
    return repairs, inputs


def validate_bank_layout(
    previous,
    current,
    bank,
    *,
    allow_expansion,
    repair_plan=None,
    previous_bank=None,
    allow_start_registration=False,
    allow_contact_conditioning=False,
    allow_contact_steps=False,
):
    if (
        bank.get("kind") != "g1_true23_small_collision_audited_local_reference_bank_v2"
        or bank.get("accepted_reference_bank") is not True
        or bank.get("all_member_control_and_original_timestamp_gates_passed") is not True
        or bank.get("all_member_control_and_original_timestamp_self_collision_gates_passed") is not True
        or bank.get("all_six_serialized_training_arrays_exactly_equal_original_members") is not True
        or bank.get("bones_members_train_only") is not True
        or bank.get("source_resampling_root_alignment_joint_deletion_or_transition_blending") is not False
        or bank.get("hardware_authorized") is not False
        or bank.get("deployment_ready") is not False
    ):
        raise ValueError("bank continuation requires accepted unchanged collision-audited offline references")
    for key in ("stage", "native_reference_model_sha256", "native_reference_sim_config_sha256"):
        if previous.get(key) != current.get(key):
            raise ValueError("bank continuation may not change stage, model or physics")
    if current["stage"] != "lifecycle":
        raise ValueError("bank requires complete lifecycle curriculum")
    old_rows, new_rows = previous["derived_spans"]["spans"], current["derived_spans"]["spans"]
    clips = bank["clips"]
    if (
        not 2 <= len(clips) <= 16
        or bank["clip_count"] != len(clips)
        or len(new_rows) != len(clips)
        or current["derived_spans"]["clip_count"] != len(clips)
        or current["original_training_inputs"]["motion_sha256"] != bank["output"]["sha256"]
        or [row["name"] for row in new_rows] != [row["name"] for row in clips]
        or len({row["name"] for row in clips}) != len(clips)
    ):
        raise ValueError("bank membership must exactly match derived training spans")
    for clip, row in zip(clips, new_rows, strict=True):
        if (
            clip["split"] not in {"train", "local_regression"}
            or row["original_source_frames"] != clip["length"]
            or row.get("every_original_source_frame_requested") is not True
            or row.get("original_source_indices_requested") != list(range(clip["length"]))
        ):
            raise ValueError("bank cannot admit held-out or cropped source motion")
    new_by_name = {row["name"]: row for row in new_rows}
    changed = previous["derived_arrays_sha256"] != current["derived_arrays_sha256"]
    registration_changed = previous.get("source_start_registration", "none") != current.get(
        "source_start_registration", "none"
    )
    contact_changed = previous.get("source_reference_conditioning", "none") != current.get(
        "source_reference_conditioning", "none"
    )
    steps_changed = previous.get("generated_transition_profile", "none") != current.get(
        "generated_transition_profile", "none"
    )
    if steps_changed or allow_contact_steps:
        from gear_sonic.utils.g1_true23_contact_step_bank_reference import validate_step_bank_transition

        if allow_expansion or repair_plan is not None or registration_changed or allow_start_registration:
            raise ValueError("contact-step transition cannot expand, quarantine or recalibrate the source bank")
        return validate_step_bank_transition(
            previous,
            current,
            explicitly_allowed=allow_contact_steps,
            allow_contact_conditioning=allow_contact_conditioning,
        )
    if contact_changed or allow_contact_conditioning:
        from gear_sonic.utils.g1_true23_contact_bank_reference import validate_contact_transition

        if allow_expansion or repair_plan is not None or registration_changed or allow_start_registration:
            raise ValueError("contact conditioning cannot also expand, replace or recalibrate the original bank")
        return validate_contact_transition(previous, current, explicitly_allowed=allow_contact_conditioning)
    if registration_changed or allow_start_registration:
        from gear_sonic.utils.g1_true23_registered_bank_reference import validate_start_registration_transition

        if allow_expansion or repair_plan is not None:
            raise ValueError("source-start registration cannot also expand or replace the original bank")
        return validate_start_registration_transition(
            previous, current, explicitly_allowed=allow_start_registration
        )
    if repair_plan is not None:
        if allow_expansion or not changed or previous_bank is None:
            raise ValueError(
                "reference repair is an explicit changed-reference branch, not expansion or same-reference resume"
            )
        old_clips = previous_bank["clips"]
        if (
            repair_plan.get("kind") != "g1_true23_collision_repaired_reference_bank_transition_v1"
            or previous_bank.get("accepted_reference_bank") is not True
            or previous_bank["output"]["sha256"] != previous["original_training_inputs"]["motion_sha256"]
            or [r["name"] for r in old_clips] != [r["name"] for r in old_rows]
            or not set(new_by_name) <= {r["name"] for r in old_clips}
            or repair_plan.get("all_omitted_references_remain_unqualified") is not True
            or repair_plan.get("hardware_authorized") is not False
            or repair_plan.get("deployment_ready") is not False
        ):
            raise ValueError(
                "repair must preserve the parent bank's recording identities and disclose every quarantine"
            )
        old_by_name = {row["name"]: row for row in old_clips}
        replacements = repair_plan.get("replacements", [])
        quarantined = repair_plan.get("quarantined", [])
        if [r["name"] for r in replacements] != [r["name"] for r in clips] or [r["name"] for r in quarantined] != [
            r["name"] for r in old_clips if r["name"] not in new_by_name
        ]:
            raise ValueError("repair plan must enumerate each retained and quarantined reference exactly once")
        for row, clip in zip(replacements, clips, strict=True):
            old = old_by_name[clip["name"]]
            if (
                row.get("previous_source_sha256") != old["source_sha256"]
                or row.get("new_source_sha256") != clip["source_sha256"]
                or any(
                    clip[key] != old[key]
                    for key in ("recording_id", "original_frames", "length", "actual_duration_scale", "split")
                )
            ):
                raise ValueError("reference repair cannot crop, retime, relabel or substitute a recording")
        for row in quarantined:
            if (
                row.get("previous_source_sha256") != old_by_name[row["name"]]["source_sha256"]
                or not isinstance(row.get("reason"), str)
                or not 5 <= len(row["reason"]) <= 500
            ):
                raise ValueError("quarantine must bind the prior reference and an explicit failure reason")
        return dict(
            kind="g1_true23_collision_repaired_reference_bank_continuation_v1",
            previous_clip_count=len(old_rows),
            new_clip_count=len(new_rows),
            reference_arrays_changed=True,
            previous_source_and_lifecycle_arrays_preserved=False,
            same_reference_resume_claimed=False,
            replacements=replacements,
            quarantined=quarantined,
            all_omitted_references_remain_unqualified=True,
            every_bank_clip_requires_separate_complete_scheduled_cpu_comparison=True,
            old_single_clip_scores_qualify_bank=False,
            held_out_generalization_verified=False,
            actor_critic_optimizer_and_counters_preserved=True,
            hardware_authorized=False,
            deployment_ready=False,
        )
    for old in old_rows:
        matching = new_by_name.get(old["name"])
        if matching is None or any(
            matching[key] != old[key]
            for key in (
                "length",
                "original_source_frames",
                "source_arrays_sha256",
                "derived_arrays_sha256",
                "timeline",
            )
        ):
            raise ValueError("bank expansion must retain every previous reference/lifecycle exactly")
    if changed and (not allow_expansion or len(new_rows) <= len(old_rows)):
        raise ValueError("adding bank references requires explicit bounded expansion")
    if not changed and allow_expansion:
        raise ValueError("bank expansion flag cannot relabel unchanged reference arrays")
    return dict(
        kind="g1_true23_exact_reference_bank_expansion_v1",
        previous_clip_count=len(old_rows),
        new_clip_count=len(new_rows),
        reference_arrays_changed=changed,
        previous_source_and_lifecycle_arrays_preserved=True,
        same_reference_resume_claimed=not changed,
        every_bank_clip_requires_separate_complete_scheduled_cpu_comparison=True,
        old_single_clip_scores_qualify_bank=False,
        held_out_generalization_verified=False,
        actor_critic_optimizer_and_counters_preserved=True,
        hardware_authorized=False,
        deployment_ready=False,
    )


def bound_bank_campaign(
    previous, current, *, args, evaluation_index_path, checkpoint, checkpoint_sha256, compatibility
):
    from gear_sonic.scripts.audit_g1_true23_reference_bank_self_contacts import measure_self_contacts
    from gear_sonic.scripts.build_g1_true23_audited_reference_bank import validate_collision_evidence
    from gear_sonic.utils.g1_true23_reference_floor import compiled_model_sha256, motion_qpos
    from gear_sonic.utils.g1_true23_root_feedback_campaign import bind_campaign_evaluation_files
    from gear_sonic.utils.g1_true23_step1b_mujoco import prepare_true23_model

    inputs = {}

    def bind(path, expected=None):
        path = Path(path).resolve(strict=True)
        digest = sha256_file(path)
        if expected is not None and digest != expected:
            raise ValueError(f"bank campaign material changed: {path}")
        inputs[str(path)] = digest
        return path

    bank_path, index_path = bind(args.motion_metadata), bind(evaluation_index_path)
    bank, index = json.loads(bank_path.read_text()), json.loads(index_path.read_text())
    for path, expected in current["original_training_inputs"].get("generated_lifecycle_repair_inputs", {}).items():
        bind(path, expected)
    for path, expected in current["original_training_inputs"].get("generated_contact_step_inputs", {}).items():
        bind(path, expected)
    for row in current["derived_spans"]["spans"]:
        for path, expected in (
            row["timeline"].get("generated_reference_repair", {}).get("input_bindings", {}).items()
        ):
            bind(path, expected)
        for path, expected in row["timeline"].get("contact_step_reference", {}).get("input_bindings", {}).items():
            bind(path, expected)
    repair_plan, previous_bank = None, None
    if getattr(args, "reference_bank_repair_plan", None) is not None:
        plan_path = bind(args.reference_bank_repair_plan)
        repair_plan = json.loads(plan_path.read_text())
        if repair_plan.get("new_bank_report_sha256") != inputs[str(bank_path)]:
            raise ValueError("reference repair plan does not bind the selected new bank")
        previous_bank_path = bind(
            repair_plan["previous_bank_report_path"], repair_plan["previous_bank_report_sha256"]
        )
        previous_bank = json.loads(previous_bank_path.read_text())
        previous_motion_path = bind(
            previous_bank_path.parent / previous_bank["output"]["filename"],
            previous["original_training_inputs"]["motion_sha256"],
        )
        with np.load(previous_motion_path, allow_pickle=False) as archive:
            for clip, old in zip(previous_bank["clips"], previous["derived_spans"]["spans"], strict=True):
                original = {
                    key: archive[key][clip["start"] : clip["start"] + clip["length"]] for key in MOTION_KEYS
                }
                if clip["name"] != old["name"] or array_digest(original) != old["source_arrays_sha256"]:
                    raise ValueError(
                        "reference repair parent bank differs from the checkpoint's exact source arrays"
                    )
    transition = validate_bank_layout(
        previous,
        current,
        bank,
        allow_expansion=getattr(args, "allow_reference_bank_transition", False),
        repair_plan=repair_plan,
        previous_bank=previous_bank,
        allow_start_registration=getattr(args, "allow_source_start_registration_transition", False),
        allow_contact_conditioning=getattr(args, "allow_contact_conditioning_transition", False),
        allow_contact_steps=getattr(args, "allow_contact_step_transition", False),
    )
    clips = bank["clips"]
    if (
        index.get("kind") != "g1_true23_reference_bank_per_clip_evaluation_index_v1"
        or index.get("bank_report_sha256") != inputs[str(bank_path)]
        or [row["name"] for row in index.get("evaluations", [])] != [row["name"] for row in clips]
    ):
        raise ValueError("bank must include every selected clip's own CPU evaluation")
    for path, expected in bank["input_bindings"].items():
        bind(path, expected)
    bind(args.native_model)
    bind(args.sim_config)
    _, collision_model, _ = prepare_true23_model(args.native_model, args.sim_config)
    collision_hash = compiled_model_sha256(collision_model)
    if collision_hash != bank["compiled_collision_physics_model_sha256"]:
        raise ValueError("bank collision physics differs from actual configured CPU referee")
    motion_path = bind(args.motion_file, bank["output"]["sha256"])
    with np.load(motion_path, allow_pickle=False) as archive:
        merged = {key: archive[key].copy() for key in MOTION_KEYS}
    outcomes, buffered, lifecycle_clearance, shared_physics = [], [], [], None
    for clip, curriculum_row, evaluation in zip(
        clips, current["derived_spans"]["spans"], index["evaluations"], strict=True
    ):
        member_path = bind(clip["source_path"], clip["source_sha256"])
        collision_path = bind(clip["collision_audit_path"], clip["collision_audit_sha256"])
        collision = json.loads(collision_path.read_text())
        if (
            validate_collision_evidence(
                collision,
                motion_sha256=clip["source_sha256"],
                original_frames=clip["original_frames"],
                control_frames=clip["length"],
                target_model_sha256=clip["target_model_sha256"],
            )
            != collision_hash
        ):
            raise ValueError("bank member collision proof belongs to different physical geometry")
        with np.load(member_path, allow_pickle=False) as member:
            source = {key: member[key].copy() for key in MOTION_KEYS}
        if array_digest(source) != curriculum_row["source_arrays_sha256"]:
            raise ValueError("bank derived source is not the unchanged audited member")
        for key in MOTION_KEYS:
            if not np.array_equal(merged[key][clip["start"] : clip["start"] + clip["length"]], source[key]):
                raise ValueError("bank concatenation differs from audited source member")
        expected_source = source
        if current.get("source_reference_conditioning", "none") != "none":
            from gear_sonic.utils.g1_true23_contact_bank_reference import derive_contact_source

            expected_source, conditioning = derive_contact_source(
                {**source, "fps": np.array([50.0])},
                curriculum_row["source_reference_conditioning"],
                collision_model,
            )
            if conditioning != curriculum_row["source_reference_conditioning"]:
                raise ValueError("bank contact correction differs from independently verified complete source")
        elif current.get("source_start_registration", "none") != "none":
            from gear_sonic.utils.g1_true23_start_registration import register_motion_start

            expected_source, registration = register_motion_start(source)
            if curriculum_row.get("source_start_registration") != registration or curriculum_row.get(
                "registered_source_arrays_sha256"
            ) != array_digest(expected_source):
                raise ValueError("bank source-start transform differs from independent full-source recomputation")
        report_path = bind(evaluation["report_path"], evaluation["report_sha256"])
        report = json.loads(report_path.read_text())
        proof = bind_campaign_evaluation_files(
            report,
            checkpoint=checkpoint,
            checkpoint_sha256=checkpoint_sha256,
            expected_release_compatibility=compatibility,
            expected_reference_arrays_sha256=curriculum_row["derived_arrays_sha256"],
            release_source_geometry=args.release_source_geometry,
            expected_generated_transition_profile=current.get("generated_transition_profile", "none"),
        )
        timeline_path = bind(report["timeline"]["timeline_path"], report["timeline"]["timeline_sha256"])
        with np.load(timeline_path, allow_pickle=False) as reference:
            lifecycle = {key: reference[key].copy() for key in (*MOTION_KEYS, "fps")}
        phase = next(p for p in report["timeline"]["phases"] if p["name"] == "source_motion")
        source_slice = slice(phase["frame_start"], phase["frame_stop"])
        if any(not np.array_equal(lifecycle[key][source_slice], expected_source[key]) for key in MOTION_KEYS):
            raise ValueError("CPU comparison source differs from the exact declared source-frame derivation")
        poses = motion_qpos(collision_model, lifecycle)
        contacts = measure_self_contacts(collision_model, poses)
        if len(poses) != curriculum_row["length"] or contacts["frames_with_robot_robot_penetration"] != 0:
            raise ValueError(
                "bank generated lifecycle, including acquisition/return ramps, contains self-collision"
            )
        lifecycle_clearance.append(
            {"clip": clip["name"], "reference_sha256": inputs[str(timeline_path)], **contacts}
        )
        result = report["records"][0]["result"]
        physics = {
            key: result[key]
            for key in (
                "compiled_model_sha256",
                "physics_config_sha256",
                "kp_hardware",
                "kd_hardware",
                "effort_limit_hardware_nm",
            )
        }
        # Retarget FK and the configured dynamics model have different compiled
        # identities. Bind the unchanged raw model/config, then compare dynamics
        # identities across clips; never equate those two compiled-model hashes.
        if (
            result["model_sha256"] != current["native_reference_model_sha256"]
            or result["physics_config_sha256"] != current["native_reference_sim_config_sha256"]
            or (shared_physics is not None and physics != shared_physics)
            or result["compiled_model_sha256"] != bank["compiled_collision_physics_model_sha256"]
        ):
            raise ValueError("bank comparisons may not change physical model or actuators between clips")
        shared_physics = physics
        inputs.update(proof["inputs"])
        outcomes.extend({"clip": clip["name"], **outcome} for outcome in proof["outcomes"])
        buffered.extend({"clip": clip["name"], **row} for row in proof["buffered_reverification"])
    for path, expected in inputs.items():
        if sha256_file(Path(path)) != expected:
            raise ValueError(f"bank material changed during continuation validation: {path}")
    if compiled_model_sha256(collision_model) != collision_hash:
        raise ValueError("bank collision geometry was modified during offline validation")
    transition["inputs"] = dict(inputs)
    transition["all_generated_lifecycle_control_poses_collision_clear"] = True
    transition["lifecycle_collision_audits"] = lifecycle_clearance
    transition["continuous_between_sample_clearance_proven"] = False
    return dict(transition=transition, inputs=inputs, outcomes=outcomes, buffered_reverification=buffered)
