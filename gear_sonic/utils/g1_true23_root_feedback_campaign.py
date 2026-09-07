"""Evaluated local-training continuation, never a promotion or corpus waiver."""

from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np

from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file
from gear_sonic.utils.g1_true23_generalist_curriculum import MOTION_KEYS, array_digest


def validate_campaign_evaluation(report, *, checkpoint_sha256, actor_sha256, lineage_sha256, updates):
    """Require the complete scheduled comparison, including any failed outcomes."""
    if report.get("kind") != "g1_true23_root_feedback_single_policy_lifecycle_campaign_v1":
        raise ValueError("continuation requires a root-feedback CPU lifecycle campaign")
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


def campaign_parent_contract(checkpoint_path, evaluation_path, *, args, curriculum):
    """Bind actual evaluated bytes; old training history is retained, not relabelled."""
    import torch
    from gear_sonic.scripts.export_g1_true23_root_feedback import validate_export_semantics
    from gear_sonic.utils.g1_true23_root_feedback_objectives import objective_transition_contract

    paths = [Path(path).expanduser() for path in (checkpoint_path, evaluation_path)]
    if any(path.is_symlink() for path in paths):
        raise ValueError("campaign parent and evaluation must not be symlinks")
    checkpoint_path, evaluation_path = [path.resolve(strict=True) for path in paths]
    inputs = {str(path): sha256_file(path) for path in (checkpoint_path, evaluation_path)}
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    semantics = validate_export_semantics(checkpoint)
    feedback = semantics["root_feedback_training_configuration"]
    objective_transition = objective_transition_contract(feedback, args)
    previous = feedback["curriculum"]
    for key in (
        "stage",
        "derived_arrays_sha256",
        "native_reference_model_sha256",
        "native_reference_sim_config_sha256",
    ):
        if previous.get(key) != curriculum.get(key):
            raise ValueError(f"campaign continuation changed curriculum material: {key}")
    for key in ("optimizer_profile", "reset_position_range_m", "reset_velocity_range_m_s"):
        if feedback.get(key) != getattr(args, key):
            raise ValueError(f"campaign continuation changed training setting: {key}")
    if checkpoint["trainer_state"]["algorithm_learning_rate"] != args.learning_rate:
        raise ValueError("campaign continuation must preserve optimizer learning rates")
    updates = checkpoint["trainer_state"]["completed_update_count"]
    if not 0 < updates < args.iterations:
        raise ValueError("campaign parent must precede planned total updates")
    report = json.loads(evaluation_path.read_text())
    outcomes = validate_campaign_evaluation(
        report,
        checkpoint_sha256=inputs[str(checkpoint_path)],
        actor_sha256=checkpoint["actor"]["state_sha256"],
        lineage_sha256=checkpoint["lineage_sha256"],
        updates=updates,
    )
    reference_path = Path(report["timeline"]["timeline_path"])
    with np.load(reference_path, allow_pickle=False) as reference:
        if array_digest({key: reference[key] for key in MOTION_KEYS}) != curriculum["derived_arrays_sha256"]:
            raise ValueError("campaign CPU evaluation does not use the complete training reference")
    inputs[str(reference_path)] = report["timeline"]["timeline_sha256"]
    for row in report["records"]:
        inputs[row["trace_path"]] = row["trace_sha256"]
        identity = row["policy_identity"]
        inputs[identity["manifest_path"]] = identity["manifest_sha256"]
        for component in ("encoder", "decoder"):
            inputs[identity["component_paths"][component]] = identity[component + "_sha256"]
    for path, digest in inputs.items():
        if sha256_file(Path(path)) != digest:
            raise ValueError(f"campaign parent/evaluation bytes changed: {path}")
    result = {
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
