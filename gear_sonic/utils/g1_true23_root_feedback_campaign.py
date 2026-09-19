"""Evaluated local-training continuation, never a promotion or corpus waiver."""

from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np

from gear_sonic.utils.g1_true23_buffered_reference import BUFFERED_TIMING, reference_profile_contract
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file
from gear_sonic.utils.g1_true23_generalist_curriculum import MOTION_KEYS, array_digest


def validate_buffered_replay_arrays(source, trace):
    """Re-feed saved received samples; no policy inference or physics changes."""
    from gear_sonic.teleop.buffered_source_horizon import ReceivedSourceHorizon
    from gear_sonic.utils.g1_23dof_contract import HARDWARE_23_JOINT_NAMES

    count = len(trace["root_feedback9"])
    if (
        count < 1
        or trace["qpos"].shape != (count + 1, 30)
        or trace["qvel"].shape != (count + 1, 29)
        or trace["actual_policy_encoder267"].shape != (count, 267)
        or trace["source_emission_anchor_setpoint_timestamps_s"].shape != (count, 3)
    ):
        raise ValueError("buffered continuation lacks complete state/input timing traces")
    maximum = 0.0
    buffer = ReceivedSourceHorizon()
    for i in range(19 + count):
        window = buffer.push(
            source_timestamp_s=i * 0.02,
            arrival_timestamp_s=i * 0.02,
            joint_names=HARDWARE_23_JOINT_NAMES,
            joint_position23=source["joint_pos"][i],
            root_position_w=source["root_position_w"][i],
            root_quaternion_wxyz=source["root_quaternion_wxyz"][i],
            virtual_source_vr21=source["virtual_vr21"][i],
        )
        if i < 19:
            continue
        j = i - 19
        qpos, qvel = trace["qpos"][j].astype(np.float32), trace["qvel"][j].astype(np.float32)
        expected = window.encoder267(qpos[3:7])
        actual = trace["actual_policy_encoder267"][j]
        if not np.isfinite(actual).all():
            raise ValueError("buffered input trace is not finite")
        maximum = max(maximum, float(np.abs(actual - expected).max()))
        timestamps = np.array(
            [
                window.emission_source_timestamp_s,
                window.encoder_anchor_timestamp_s,
                window.root_setpoint_timestamp_s,
            ]
        )
        feedback = window.root_feedback9(qpos[:3], qvel[:3], qpos[3:7])
        if (
            not np.array_equal(actual[:261], expected[:261])
            or maximum > 2e-6
            or not np.array_equal(timestamps, trace["source_emission_anchor_setpoint_timestamps_s"][j])
            or not np.array_equal(feedback, trace["root_feedback9"][j])
        ):
            raise ValueError("buffered replay differs from actual received-only encoder/root inputs")
    return dict(
        kind="g1_true23_received_source_cpu_trace_reverification_v1",
        controls_verified=count,
        received_samples_consumed=19 + count,
        lower_and_vr_bit_exact=True,
        root_feedback_bit_exact=True,
        maximum_encoder_abs_error=maximum,
        reference_anchor_latency_s=0.2,
        all_inputs_available_by_declared_emission_time=True,
        hardware_authorized=False,
        deployment_ready=False,
    )


def validate_campaign_evaluation(
    report,
    *,
    checkpoint_sha256,
    actor_sha256,
    lineage_sha256,
    updates,
    expected_release_compatibility=None,
    expected_generated_transition_profile="none",
):
    """Require the complete scheduled comparison, including any failed outcomes."""
    from gear_sonic.utils.g1_true23_contact_step_transition import PROFILE

    if expected_generated_transition_profile not in ("none", PROFILE):
        raise ValueError("unsupported explicit training transition reference profile")
    expected_kind = (
        "g1_true23_root_feedback_single_policy_lifecycle_campaign_v1"
        if expected_generated_transition_profile == "none"
        else "g1_true23_contact_step_lifecycle_policy_diagnostic_v1"
    )
    if report.get("kind") != expected_kind:
        raise ValueError("continuation requires a root-feedback CPU lifecycle campaign")
    if expected_generated_transition_profile != "none" and (
        report.get("contact_step_reference_diagnostic") is not True
        or report["timeline"].get("generated_transition_profile") != PROFILE
        or report["timeline"]
        .get("contact_step_independent_geometry_audit", {})
        .get("provisional_geometry_screen_passed")
        is not True
        or "root_input_counterfactual" in report
    ):
        raise ValueError("explicit contact-step training requires unmodified-policy, independently audited replay")
    if report.get("release_compatibility") != expected_release_compatibility:
        raise ValueError("continuation evaluation release compatibility mismatch")
    for flag in (
        "only_one_hash_bound_policy_per_case",
        "no_postinitial_robot_pose_rewrites",
        "no_fallback_controller",
    ):
        if report.get(flag) is not True:
            raise ValueError("continuation evaluation lacks a single-policy/no-reset contract")
    if any(report.get(flag) is not False for flag in ("deployment_ready", "hardware_authorized")):
        raise ValueError("continuation evaluation cannot authorize hardware")
    rows = report.get("records", [])
    if [row.get("case") for row in rows] != ["nominal", "standing_push_x", "standing_push_y"]:
        raise ValueError("continuation needs nominal and both scheduled push cases")
    count = report["timeline"]["total_requested_controls"]
    shared = None
    outcomes = []
    for row in rows:
        source = row["policy_identity"]["source"]
        expected = {
            "checkpoint_sha256": checkpoint_sha256,
            "actor_state_sha256": actor_sha256,
            "lineage_sha256": lineage_sha256,
            "completed_update_count": updates,
        }
        if any(source.get(key) != value for key, value in expected.items()):
            raise ValueError("continuation evaluation belongs to a different checkpoint")
        result = row["result"]
        if (
            result.get("training_model_counterfactual") is not None
            or result.get("original_cpp_gain_counterfactual_not_nominal_qualification") is True
        ):
            raise ValueError("counterfactual physics cannot replace the original-model campaign evaluation")
        if expected_release_compatibility is not None:
            from gear_sonic.utils.g1_true23_release_compatibility import validate_release_compatibility

            validate_release_compatibility(expected_release_compatibility)
            runtime = result.get("runtime_adapter", {})
            buffered = expected_release_compatibility.get("reference_timing") == BUFFERED_TIMING
            if buffered:
                from gear_sonic.utils.g1_true23_root_feedback import root_feedback_contract

                expected_forces = (
                    []
                    if row["case"] == "nominal"
                    else [
                        dict(
                            start_substep=500,
                            duration_substeps=50,
                            force_world_n=[40.0, 0.0, 0.0]
                            if row["case"] == "standing_push_x"
                            else [0.0, 40.0, 0.0],
                        )
                    ]
                )
                reference_runtime_valid = (
                    runtime.get("kind") == "g1_true23_received_source_horizon_cpu_runtime_v1"
                    and runtime.get("reference_timing") == BUFFERED_TIMING
                    and runtime.get("semantic_profile") == reference_profile_contract(BUFFERED_TIMING)
                    and runtime.get("root_feedback", {}).get("root_feedback_contract")
                    == root_feedback_contract(BUFFERED_TIMING)
                    and runtime.get("root_feedback", {}).get("forces") == expected_forces
                    and runtime.get("source_clock_latency_s") == 0.2
                    and runtime.get("scored_reference_timing_or_physics_changed") is False
                )
            else:
                reference_runtime_valid = (
                    runtime.get("mode") == "causal_past_virtual_source" and runtime.get("case") == row["case"]
                )
            if (
                row["policy_identity"].get("release_compatibility") != expected_release_compatibility
                or result.get("release_compatibility") != expected_release_compatibility
                or not reference_runtime_valid
                or runtime.get("action_convention") != expected_release_compatibility["action_convention"]
                or runtime.get("source_action_codec") != expected_release_compatibility.get("source_action_codec")
                or runtime.get("future_reference_consumed") is not False
            ):
                raise ValueError("continuation evaluated different reference/action runtime")
        if (
            result["requested_controls"] != count
            or result["available_controls"] != count
            or result["diagnostic_prefix_requested"]
            or result["state_pose_writes_after_reset"] != 0
            or result["history_resets_during_motion"] != 0
            or not 0 <= result["completed_controls"] <= count
        ):
            raise ValueError("continuation cannot use shortened or state-reset evaluation")
        contract = {
            key: result[key]
            for key in (
                "initial_state_and_history_sha256",
                "compiled_model_sha256",
                "physics_config_sha256",
                "motion_sha256",
                "kp_hardware",
                "kd_hardware",
                "effort_limit_hardware_nm",
            )
        }
        if shared is not None and contract != shared:
            raise ValueError("continuation evaluation changed physics or reference between cases")
        shared = contract
        outcomes.append(
            dict(
                case=row["case"],
                completed_controls=result["completed_controls"],
                requested_controls=count,
                failure=result["failure"],
                source_motion_tracking=row["lifecycle"]["source_motion_tracking"],
            )
        )
    return outcomes


def bind_campaign_evaluation_files(
    report,
    *,
    checkpoint,
    checkpoint_sha256,
    expected_release_compatibility,
    expected_reference_arrays_sha256,
    release_source_geometry,
    expected_generated_transition_profile="none",
):
    """Bind one complete scheduled comparison and replay its actual received inputs."""
    outcomes = validate_campaign_evaluation(
        report,
        checkpoint_sha256=checkpoint_sha256,
        actor_sha256=checkpoint["actor"]["state_sha256"],
        lineage_sha256=checkpoint["lineage_sha256"],
        updates=checkpoint["trainer_state"]["completed_update_count"],
        expected_release_compatibility=expected_release_compatibility,
        expected_generated_transition_profile=expected_generated_transition_profile,
    )
    reference_path = Path(report["timeline"]["timeline_path"])
    inputs = {str(reference_path): report["timeline"]["timeline_sha256"]}
    with np.load(reference_path, allow_pickle=False) as reference:
        reference_arrays = {key: reference[key].copy() for key in MOTION_KEYS}
        if array_digest(reference_arrays) != expected_reference_arrays_sha256:
            raise ValueError("campaign CPU evaluation does not use the complete training reference")
    buffered_reverification = []
    if (expected_release_compatibility or {}).get("reference_timing") == BUFFERED_TIMING:
        from gear_sonic.utils.g1_true23_buffered_reference import continued_standing_source
        from gear_sonic.utils.g1_true23_virtual_source_reference import virtual_source_vr_terms

        expected_source = continued_standing_source(
            reference_arrays, virtual_source_vr_terms(reference_arrays, release_source_geometry)
        )
        for row in report["records"]:
            path = Path(row["trace_path"])
            received = path.with_name(row["case"] + ".received_source.npz")
            inputs[str(received)] = sha256_file(received)
            with np.load(received, allow_pickle=False) as archive:
                source = {key: archive[key].copy() for key in archive.files}
            if set(source) != set(expected_source) or any(
                not np.array_equal(source[key], expected_source[key]) for key in source
            ):
                raise ValueError(
                    "saved buffered source differs from the complete lifecycle/declared generated tail"
                )
            with np.load(path, allow_pickle=False) as archive:
                trace = {
                    key: archive[key].copy()
                    for key in (
                        "qpos",
                        "qvel",
                        "root_feedback9",
                        "actual_policy_encoder267",
                        "source_emission_anchor_setpoint_timestamps_s",
                    )
                }
            verified = validate_buffered_replay_arrays(source, trace)
            if verified["controls_verified"] != row["result"]["completed_controls"]:
                raise ValueError("buffered trace control count differs from evaluated result")
            buffered_reverification.append(
                dict(
                    case=row["case"],
                    source_path=str(received),
                    source_sha256=inputs[str(received)],
                    verification=verified,
                )
            )
    for row in report["records"]:
        inputs[row["trace_path"]] = row["trace_sha256"]
        identity = row["policy_identity"]
        inputs[identity["manifest_path"]] = identity["manifest_sha256"]
        for component in ("encoder", "decoder"):
            inputs[identity["component_paths"][component]] = identity[component + "_sha256"]
    for path, digest in inputs.items():
        if sha256_file(Path(path)) != digest:
            raise ValueError(f"campaign evaluation bytes changed: {path}")
    return dict(outcomes=outcomes, inputs=inputs, buffered_reverification=buffered_reverification)


def campaign_parent_contract(checkpoint_path, evaluation_path, *, args, curriculum):
    """Bind actual evaluated bytes; old training history is retained, not relabelled."""
    import torch

    from gear_sonic.scripts.export_g1_true23_root_feedback import validate_export_semantics
    from gear_sonic.trl.mjlab.native23_projected_target_ppo import projection_objective_transition
    from gear_sonic.utils.g1_true23_root_feedback_objectives import objective_transition_contract
    from gear_sonic.utils.g1_true23_start_schedule import start_schedule_transition_contract

    paths = [Path(path).expanduser() for path in (checkpoint_path, evaluation_path)]
    if any(path.is_symlink() for path in paths):
        raise ValueError("campaign parent and evaluation must not be symlinks")
    checkpoint_path, evaluation_path = [path.resolve(strict=True) for path in paths]
    inputs = {str(path): sha256_file(path) for path in (checkpoint_path, evaluation_path)}
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    semantics = validate_export_semantics(checkpoint)
    feedback = semantics["root_feedback_training_configuration"]
    if feedback.get("training_physics") != getattr(args, "training_physics_contract", None):
        raise ValueError("training physics change requires fresh initialization, not evaluated continuation")
    compatibility = semantics.get("release_compatibility")
    if compatibility != getattr(args, "release_compatibility_contract", None):
        raise ValueError("campaign continuation may not change release compatibility")
    if feedback.get("reference_timing", "causal_history") != getattr(args, "reference_timing", "causal_history"):
        raise ValueError("campaign continuation may not change reference timing")
    objective_transition = objective_transition_contract(feedback, args)
    start_schedule_transition = start_schedule_transition_contract(feedback, args)
    ppo_objective_transition = projection_objective_transition(feedback, args)
    previous = feedback["curriculum"]
    for key in (
        "stage",
        "native_reference_model_sha256",
        "native_reference_sim_config_sha256",
    ):
        if previous.get(key) != curriculum.get(key):
            raise ValueError(f"campaign continuation changed curriculum material: {key}")
    reference_transition = None
    bank_mode = getattr(args, "reference_bank", False)
    if bank_mode:
        # Exact source retention and each added reference are validated together
        # with all per-clip evaluations below, before any runner is constructed.
        pass
    elif getattr(args, "allow_reference_transition", False):
        from gear_sonic.utils.g1_true23_reference_transition import bound_reference_transition

        reference_transition = bound_reference_transition(
            previous,
            curriculum,
            previous_report=args.previous_reference_report,
            new_report=args.motion_metadata,
            original_time_audit=args.reference_original_time_audit,
        )
        inputs.update(reference_transition["inputs"])
        start_schedule_transition = {
            **start_schedule_transition,
            "reference_arrays_and_evaluation_unchanged": False,
            "reference_change_separately_validated": True,
        }
    elif previous.get("derived_arrays_sha256") != curriculum.get("derived_arrays_sha256"):
        raise ValueError("campaign continuation changed curriculum material: derived_arrays_sha256")
    for key in ("optimizer_profile", "reset_position_range_m", "reset_velocity_range_m_s"):
        if feedback.get(key) != getattr(args, key):
            raise ValueError(f"campaign continuation changed training setting: {key}")
    if checkpoint["trainer_state"]["algorithm_learning_rate"] != args.learning_rate:
        raise ValueError("campaign continuation must preserve optimizer learning rates")
    updates = checkpoint["trainer_state"]["completed_update_count"]
    if not 0 < updates < args.iterations:
        raise ValueError("campaign parent must precede planned total updates")
    if bank_mode:
        from gear_sonic.scripts.g1_true23_reference_bank_campaign import bound_bank_campaign

        proof = bound_bank_campaign(
            previous,
            curriculum,
            args=args,
            evaluation_index_path=evaluation_path,
            checkpoint=checkpoint,
            checkpoint_sha256=inputs[str(checkpoint_path)],
            compatibility=compatibility,
        )
        reference_transition = proof["transition"]
        start_schedule_transition = {
            **start_schedule_transition,
            "reference_arrays_and_evaluation_unchanged": not reference_transition["reference_arrays_changed"],
            "reference_change_separately_validated": True,
        }
    else:
        proof = bind_campaign_evaluation_files(
            json.loads(evaluation_path.read_text()),
            checkpoint=checkpoint,
            checkpoint_sha256=inputs[str(checkpoint_path)],
            expected_release_compatibility=compatibility,
            expected_reference_arrays_sha256=curriculum["derived_arrays_sha256"],
            release_source_geometry=args.release_source_geometry,
        )
    outcomes, buffered_reverification = proof["outcomes"], proof["buffered_reverification"]
    inputs.update(proof["inputs"])
    for path, digest in inputs.items():
        if sha256_file(Path(path)) != digest:
            raise ValueError(f"campaign parent/evaluation bytes changed: {path}")
    result = {
        **({"reference_transition": reference_transition} if reference_transition is not None else {}),
        **(
            {"buffered_reference_evaluation_reverified": buffered_reverification}
            if buffered_reverification
            else {}
        ),
        "kind": "g1_root_feedback_evaluated_training_continuation_v1",
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": inputs[str(checkpoint_path)],
        "actor_state_sha256": checkpoint["actor"]["state_sha256"],
        "critic_state_sha256": checkpoint["critic_state_sha256"],
        "lineage_sha256": checkpoint["lineage_sha256"],
        "completed_update_count": updates,
        "evaluation_path": str(evaluation_path),
        "evaluation_sha256": inputs[str(evaluation_path)],
        "evaluation_outcomes": outcomes,
        "objective_transition": objective_transition,
        "start_schedule_transition": start_schedule_transition,
        "ppo_objective_transition": ppo_objective_transition,
        "actor_critic_optimizer_and_counters_preserved": True,
        "simulator_rollout_and_rng_state_preserved": False,
        "fresh_environment_reset_before_new_rollouts": True,
        "maximum_updates_before_next_cpu_evaluation": 100,
        "original_base_parent": previous["parent_actor_initialization"],
        "audited_corpus_ownership_established": False,
        "held_out_generalization_verified": False,
        "deployment_ready": False,
        "hardware_authorized": False,
    }
    del checkpoint
    gc.collect()
    return result
