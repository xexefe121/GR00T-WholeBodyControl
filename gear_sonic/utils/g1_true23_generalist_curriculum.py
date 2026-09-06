"""Derived simulation references; source ownership stays with the input audit.

These are nominal training stages, not feasibility or generalization evidence.
Production inputs are validated before derivation; unaudited local regression
inputs may enter only explicitly bounded smoke runs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from gear_sonic.utils.g1_true23_generalist_lifecycle import build_lifecycle_timeline
from gear_sonic.utils.g1_true23_generalist_corpus import canonical_digest, sha256_file

STAGES = ("acquisition", "lifecycle")
MOTION_KEYS = ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")


def array_digest(motion):
    digest = hashlib.sha256()
    for key in MOTION_KEYS:
        value = np.ascontiguousarray(motion[key])
        digest.update(canonical_digest([key, value.dtype.str, list(value.shape)]).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def derive_curriculum(
    motion, spans, input_contract, *, stage, model, simulation_config, return_target="configured_origin"
):
    """Derive references only from the exact already-validated source arrays."""
    if stage not in STAGES:
        raise ValueError("unsupported generalist curriculum stage")
    audit = input_contract.get("corpus_audit")
    if audit is None and input_contract.get("smoke_only") is not True:
        raise ValueError("curriculum training requires audited train-split inputs")
    sections, rows, cursor = [], [], 0
    for original in spans["spans"]:
        asset_id = original.get("asset_id")
        start, length = original["start"], original["length"]
        source = {key: motion[key][start : start + length].copy() for key in MOTION_KEYS}
        source["fps"] = np.array([50.0])
        ownership = {"asset_id": asset_id, "recording_id": None, "split": None}
        if audit is not None:
            if audit["asset_splits"].get(asset_id) != "train":
                raise ValueError("curriculum derivative must inherit a train split")
            metadata = audit["asset_metadata"][asset_id]
            if metadata["timing"]["fps"] != 50 or metadata["timing"]["frame_count"] != length:
                raise ValueError("audited asset timing differs from source span")
            ownership.update(
                recording_id=metadata["recording_id"],
                split="train",
                source_asset_sha256=audit["asset_bindings"][asset_id]["sha256"],
            )
        requested = source
        if stage == "acquisition":
            # Acquisition learns the first source pose, not a shortened dance.
            requested = {key: np.repeat(source[key][:1], 100, axis=0) for key in MOTION_KEYS}
            for key in ("joint_vel", "body_lin_vel_w", "body_ang_vel_w"):
                requested[key][:] = 0
            requested["fps"] = np.array([50.0])
        derived, timeline = build_lifecycle_timeline(
            requested, model=model, simulation_config=simulation_config, return_target=return_target
        )
        timeline["source_input_kind"] = (
            "complete_original_source" if stage == "lifecycle" else "first_source_pose_repeated_zero_velocity"
        )
        timeline["original_recording_source_frames"] = length
        if stage == "acquisition":
            for phase in timeline["phases"]:
                if phase["name"] == "source_motion":
                    phase["name"] = "acquisition_pose_hold"
        count = len(derived["joint_pos"])
        row = {
            "name": original.get("name", asset_id or f"regression_{len(rows)}"),
            "start": cursor,
            "length": count,
            "asset_id": asset_id,
            "ownership": ownership,
            "original_source_frames": length,
            "source_arrays_sha256": array_digest(source),
            "derived_arrays_sha256": array_digest(derived),
            "original_source_indices_requested": list(range(length)) if stage == "lifecycle" else [0],
            "every_original_source_frame_requested": stage == "lifecycle",
            "timeline": timeline,
        }
        sections.append(derived)
        rows.append(row)
        cursor += count
    result = {key: np.concatenate([part[key] for part in sections]) for key in MOTION_KEYS}
    result["fps"] = np.array([50.0])
    sidecar = dict(
        kind="g1_true23_motion_corpus_spans_v1", fps=50, spans=rows, clip_count=len(rows), total_frames=cursor
    )
    contract = {
        "kind": "g1_native23_generalist_nominal_curriculum_v1",
        "stage": stage,
        "original_training_inputs": input_contract,
        "derived_spans": sidecar,
        "derived_arrays_sha256": array_digest(result),
        "stage_switch": "explicit_parent_actor_transfer_new_optimizer_critic_and_lineage",
        "single_actor": True,
        "reset_start": "configured_standing_zero_velocity",
        "sample_clip_only_at_environment_reset": True,
        "command_updates_write_robot_state": False,
        "random_initial_episode_lengths": False,
        "full_source_lifecycle_reference_enabled": stage == "lifecycle",
        "full_lifecycle_training_completed": False,
        "generated_ramp_contact_or_force_feasibility_qualified": False,
        "domain_randomization_curriculum_implemented": False,
        "teleop_qualification_complete": False,
        "deployment_ready": False,
        "hardware_authorized": False,
    }
    return result, sidecar, contract


def write_curriculum_bundle(directory: Path, motion, spans, contract):
    """Publish once; metadata binds generated arrays and original input audit."""
    directory.mkdir(parents=True, exist_ok=False)
    destination = directory / "curriculum.npz"
    with destination.open("xb") as stream:
        np.savez_compressed(stream, **motion)
    metadata = {
        "schema": "g1_true23_low_latency_recovery_motion_v1",
        "metadata_compatibility_schema_only": True,
        "output": {"filename": destination.name, "sha256": sha256_file(destination)},
        "curriculum": contract,
        "deployment_ready": False,
    }
    for name, value in (("curriculum.json", metadata), ("curriculum.spans.json", spans)):
        with (directory / name).open("x") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
    return destination, directory / "curriculum.json", directory / "curriculum.spans.json"
