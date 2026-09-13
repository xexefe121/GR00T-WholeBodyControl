"""Explicit source-frame registration for evaluated, unchanged-source banks.

Original audited members remain immutable. Only a separately derived lifecycle
uses a single first-sample SE(2) transform. This never follows measured state,
changes choreography, or converts geometric evidence into deployment approval.
"""

import json
from pathlib import Path

import numpy as np

from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file
from gear_sonic.utils.g1_true23_start_registration import register_motion_start

START_REGISTRATION_PROFILE = "once_only_source_start_se2_v1"
REGISTERED_REPAIR_INDEX_KIND = "g1_true23_once_registered_bank_lifecycle_repairs_v1"


def load_registered_repair_member(row, clip):
    """Recompute every registered array from the exact original bank member."""
    from gear_sonic.utils.g1_true23_generalist_curriculum import MOTION_KEYS, array_digest

    inputs = {}

    def bind(path, expected):
        path = Path(path).resolve(strict=True)
        if sha256_file(path) != expected:
            raise ValueError("registered repair source or receipt bytes differ")
        inputs[str(path)] = expected
        return path

    if row.get("original_source_motion_sha256") != clip["source_sha256"]:
        raise ValueError("registration must bind the original unchanged audited bank member")
    original_path = bind(clip["source_path"], clip["source_sha256"])
    registered_path = bind(row["registered_source_path"], row["source_motion_sha256"])
    diagnostic_path = bind(row["registration_report_path"], row["registration_report_sha256"])
    diagnostic = json.loads(diagnostic_path.read_text())
    original_receipts = [
        (path, digest)
        for path, digest in diagnostic.get("input_bindings", {}).items()
        if Path(path).suffix == ".npz" and digest == clip["source_sha256"]
    ]
    if (
        diagnostic.get("kind") != "g1_true23_registered_source_lifecycle_geometry_diagnostic_v1"
        or diagnostic.get("policy_evaluation_performed") is not False
        or diagnostic.get("training_reference_accepted") is not False
        or len(original_receipts) != 1
        or diagnostic.get("timeline", {}).get("source_motion_sha256") != row["source_motion_sha256"]
        or diagnostic.get("hardware_authorized") is not False
        or diagnostic.get("deployment_ready") is not False
    ):
        raise ValueError("registration receipt must disclose separate unevaluated geometric derivation")
    # A fresh independent audit may retain a byte-exact copy at a new path.
    # Bind both actual files; never require a historical path to become current.
    bind(*original_receipts[0])
    with np.load(original_path, allow_pickle=False) as archive:
        original = {key: archive[key].copy() for key in archive.files}
    registered, proof = register_motion_start(original)
    with np.load(registered_path, allow_pickle=False) as archive:
        if set(archive.files) != set(registered) or any(
            archive[key].dtype != registered[key].dtype or not np.array_equal(archive[key], registered[key])
            for key in registered
        ):
            raise ValueError("registered source differs from independently recomputed fixed SE(2) transform")
    if diagnostic.get("registration") != proof or len(original["joint_pos"]) != clip["length"]:
        raise ValueError("registration receipt changes the fixed transform or complete source count")
    # The historical diagnostic is not trusted as current geometric evidence.
    # Exact source equality is rederived here; the repaired lifecycle loader and
    # bank campaign independently recheck actual FK, full-path bounds and contacts.
    result = {
        **row,
        "registered_source_path": str(registered_path),
        "source_start_registration": proof,
        "original_source_arrays_sha256": array_digest({key: original[key] for key in MOTION_KEYS}),
        "registered_source_arrays_sha256": array_digest(registered),
    }
    if any(sha256_file(Path(path)) != expected for path, expected in inputs.items()):
        raise ValueError("registration material changed during independent recomputation")
    return result, inputs


def validate_start_registration_transition(previous, current, *, explicitly_allowed):
    """Allow one disclosed reference-frame change, never an implicit reset."""
    if (
        not explicitly_allowed
        or previous.get("source_start_registration", "none") != "none"
        or current.get("source_start_registration") != START_REGISTRATION_PROFILE
        or previous["derived_arrays_sha256"] == current["derived_arrays_sha256"]
        or previous["original_training_inputs"]["motion_sha256"]
        != current["original_training_inputs"]["motion_sha256"]
    ):
        raise ValueError("source-start registration requires an explicit once-only unchanged-source transition")
    old_rows, new_rows = previous["derived_spans"]["spans"], current["derived_spans"]["spans"]
    if [row["name"] for row in old_rows] != [row["name"] for row in new_rows]:
        raise ValueError("source-start registration cannot add, omit or reorder bank members")
    for old, new in zip(old_rows, new_rows, strict=True):
        for key in (
            "start",
            "length",
            "asset_id",
            "ownership",
            "original_source_frames",
            "source_arrays_sha256",
            "original_source_indices_requested",
            "every_original_source_frame_requested",
        ):
            if old.get(key) != new.get(key):
                raise ValueError("source-start registration cannot change ownership, source arrays or coverage")
        for key in (
            "phases",
            "source_frames",
            "total_frames",
            "total_requested_controls",
            "configured_standing_qpos",
        ):
            if old["timeline"].get(key) != new["timeline"].get(key):
                raise ValueError("source-start registration cannot change phase timing or initial robot state")
        proof = new.get("source_start_registration", {})
        if (
            proof.get("kind") != "g1_true23_once_only_source_start_se2_registration_v1"
            or proof.get("target_root_xy") != [0.0, 0.0]
            or proof.get("target_heading_rad") != 0.0
            or proof.get("frame_count") != new["original_source_frames"]
            or proof.get("joint_positions_and_velocities_bit_exact") is not True
            or proof.get("world_height_channels_bit_exact") is not True
            or proof.get("full_source_frames_and_timing_preserved") is not True
            or proof.get("relative_motion_excursion_scaled") is not False
            or proof.get("measured_robot_state_used") is not False
            or proof.get("midrun_reregistration_or_pose_reset") is not False
            or proof.get("same_unregistered_benchmark_claimed") is not False
            or proof.get("hardware_authorized") is not False
            or proof.get("deployment_ready") is not False
            or not new.get("registered_source_arrays_sha256")
            or not new["timeline"].get("generated_reference_repair")
        ):
            raise ValueError(
                "registered lifecycle must disclose fixed calibration and independently repaired ramps"
            )
    return dict(
        kind="g1_true23_once_registered_reference_bank_continuation_v1",
        previous_clip_count=len(old_rows),
        new_clip_count=len(new_rows),
        reference_arrays_changed=True,
        original_audited_source_arrays_preserved=True,
        previous_source_and_lifecycle_arrays_preserved=False,
        same_reference_resume_claimed=False,
        same_unregistered_world_benchmark_claimed=False,
        first_source_sample_only_calibration=True,
        measured_robot_state_used_for_registration=False,
        every_bank_clip_requires_separate_complete_scheduled_cpu_comparison=True,
        old_single_clip_scores_qualify_bank=False,
        held_out_generalization_verified=False,
        actor_critic_optimizer_and_counters_preserved=True,
        hardware_authorized=False,
        deployment_ready=False,
    )
