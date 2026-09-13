"""Assemble unchanged accepted references into a small local simulation bank.

Requires accepted control-grid fits, complete independent original-time audits,
collision-clearance evidence on both grids, and frozen BONES-SEED train membership.
No retargeting, blending, resampling,
license acceptance, downloading, training, publication, or robot operations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gear_sonic.utils.g1_23dof_contract import HARDWARE_23_JOINT_NAMES
from gear_sonic.utils.g1_true23_generalist_corpus import canonical_digest, sha256_file

MOTION_KEYS = ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")


def validate_collision_evidence(audit, *, motion_sha256, original_frames, control_frames, target_model_sha256):
    required = (
        "complete_control_grid_fidelity_passed",
        "all_original_timestamp_fidelity_passed",
        "serialized_declared_path_bounds_passed",
        "no_control_grid_robot_robot_penetration",
        "no_original_grid_robot_robot_penetration",
    )
    if (
        audit.get("kind") != "g1_true23_full_collision_repair_audit_v1"
        or audit.get("acceptance", {}).get("passed") is not True
        or any(audit.get("acceptance", {}).get("checks", {}).get(key) is not True for key in required)
        or audit.get("accepted_candidate", {}).get("sha256") != motion_sha256
        or audit.get("compiled_models", {}).get("target") != target_model_sha256
        or audit.get("control_frames") != control_frames
        or audit.get("original_frames") != original_frames
        or audit.get("hardware_authorized") is not False
        or audit.get("deployment_ready") is not False
    ):
        raise ValueError("bank requires bound accepted collision-clear motion on both complete grids")
    physics = audit.get("compiled_physics_model_sha256")
    if not isinstance(physics, str) or len(physics) != 64 or any(c not in "0123456789abcdef" for c in physics):
        raise ValueError("bank collision evidence must bind the exact compiled physical model")
    for key, count in (
        ("after_control_self_contacts", control_frames),
        ("after_original_self_contacts", original_frames),
    ):
        row = audit.get(key, {})
        if (
            row.get("frames") != count
            or row.get("frames_with_robot_robot_penetration") != 0
            or row.get("penetration_frame_indices") != []
            or row.get("maximum_penetration_m") != 0
            or row.get("physics_integration_steps") != 0
        ):
            raise ValueError("bank collision evidence contains omitted samples or remaining penetration")
    return physics


def validate_member_evidence(fit, audit, *, motion_sha256, original_frames, control_frames):
    selected = fit.get("selected_attempt")
    if fit.get("accepted") is not True or type(selected) is not int or not 0 <= selected < len(fit["attempts"]):
        raise ValueError("bank requires an accepted selected control-grid fit")
    attempt = fit["attempts"][selected]
    if (
        attempt.get("accepted") is not True
        or attempt.get("failures") != []
        or attempt.get("requested_excursion_scale") != 1.0
        or attempt.get("output_frames") != control_frames
        or fit.get("source_frame_count") != original_frames
        or fit.get("adapted_motion_sha256", fit.get("output", {}).get("sha256")) != motion_sha256
    ):
        raise ValueError("bank fit must bind every full-excursion source/control frame")
    if (
        audit.get("kind")
        not in {
            "g1_true23_bones_seed_all_original_timestamps_fk_audit_v1",
            "g1_true23_planned29_all_original_timestamps_fk_audit_v1",
        }
        or audit.get("original_timestamp_fidelity_passed") is not True
        or audit.get("failures") != []
        or audit.get("all_original_timestamps_and_endpoints_evaluated") is not True
        or audit.get("original_frames_evaluated") != original_frames
        or audit.get("control_frames") != control_frames
        or audit.get("control_fps") != 50
        or audit.get("original_time_sample_indices") != list(range(original_frames))
    ):
        raise ValueError("bank requires every original timestamp, without failed or omitted frames")
    models = fit.get("compiled_models", fit.get("compiled_model_sha256"))
    audit_models = audit.get("compiled_models", audit.get("compiled_model_sha256"))
    if not models or models != audit_models:
        raise ValueError("bank fit/audit model identities differ")
    return {"target_model_sha256": models["target"], "actual_duration_scale": attempt["actual_duration_scale"]}


def concatenate_exact_members(members):
    if not 2 <= len(members) <= 16:
        raise ValueError("local bank requires 2..16 explicitly selected complete references")
    for member in members:
        count = len(member["joint_pos"])
        if (
            count <= 15
            or member["joint_pos"].shape != (count, 23)
            or member["joint_vel"].shape != (count, 23)
            or tuple(member["joint_names"].tolist()) != tuple(HARDWARE_23_JOINT_NAMES)
            or np.asarray(member["fps"]).shape != (1,)
            or float(member["fps"][0]) != 50
        ):
            raise ValueError("bank member must preserve exact native23 order and 50-Hz timing")
        for key in MOTION_KEYS:
            if member[key].shape[0] != count or not np.isfinite(member[key]).all():
                raise ValueError("bank member has missing, nonfinite or mismatched frames")
            if member[key].shape[1:] != members[0][key].shape[1:] or member[key].dtype != members[0][key].dtype:
                raise ValueError("bank cannot change member array shape or precision")
    merged = {key: np.concatenate([member[key] for member in members]) for key in MOTION_KEYS}
    merged["fps"] = members[0]["fps"].copy()
    merged["joint_names"] = members[0]["joint_names"].copy()
    return merged


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output_directory.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite {output}")
    inputs = {}

    def bind(path, expected=None):
        path = Path(path).resolve(strict=True)
        actual = sha256_file(path)
        if expected is not None and actual != expected:
            raise ValueError(f"bank input hash changed: {path}")
        inputs[str(path)] = actual
        return path

    def read(path):
        return json.loads(bind(path).read_text())

    selection = read(args.selection)
    entries = selection["clips"]
    if selection.get("kind") != "g1_true23_local_reference_bank_selection_v1" or not 2 <= len(entries) <= 16:
        raise ValueError("bank requires explicit bounded selection")
    if any(not entry.get("collision_audit") for entry in entries):
        raise ValueError("bank now requires every member's full control/original-time collision audit")
    index_dir = Path(selection["bones_index_directory"]).resolve(strict=True)
    index_path = bind(index_dir / "report.json")
    index = read(index_path)
    splits = read(index_dir / "recording_splits.json")
    if (
        index.get("kind") != "g1_true23_bones_seed_local_source_index_v1"
        or index.get("split_before_mirror_crop_tempo_or_retarget") is not True
        or canonical_digest(splits) != index["split_sha256"]
    ):
        raise ValueError("bank requires intact previously frozen capture-group splits")
    source_index = bind(index_dir / "source_index.jsonl", index["index"]["sha256"])
    names = {entry["name"] for entry in entries}
    if len(names) != len(entries):
        raise ValueError("bank names must be unique")
    indexed = {}
    with source_index.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["motion_id"] in names:
                if row["motion_id"] in indexed:
                    raise ValueError("bank source identity is ambiguous")
                indexed[row["motion_id"]] = row
    members, rows, cursor, targets, physics_models, recording_ids = [], [], 0, set(), set(), set()
    for entry in entries:
        fit_dir = Path(entry["retarget_directory"]).resolve(strict=True)
        fit_path, motion_path = bind(fit_dir / "report.json"), bind(fit_dir / "adapted.true23.npz")
        fit, audit = read(fit_path), read(entry["original_time_audit"])
        with np.load(motion_path, allow_pickle=False) as archive:
            member = {key: archive[key].copy() for key in (*MOTION_KEYS, "fps", "joint_names")}
        length = len(member["joint_pos"])
        detail = validate_member_evidence(
            fit,
            audit,
            motion_sha256=inputs[str(motion_path)],
            original_frames=fit["source_frame_count"],
            control_frames=length,
        )
        targets.add(detail["target_model_sha256"])
        collision_path = bind(entry["collision_audit"])
        collision = read(collision_path)
        physics_models.add(
            validate_collision_evidence(
                collision,
                motion_sha256=inputs[str(motion_path)],
                original_frames=fit["source_frame_count"],
                control_frames=length,
                target_model_sha256=detail["target_model_sha256"],
            )
        )
        for path, expected in collision["input_bindings"].items():
            bind(path, expected)
        audit_inputs = audit.get("input_bindings", {})
        for path in (fit_path, motion_path):
            if audit_inputs.get(str(path)) != inputs[str(path)]:
                raise ValueError("original-time audit is not bound to bank member bytes")
        for path, expected in audit_inputs.items():
            bind(path, expected)
        receipt_path = entry.get("source_receipt")
        if receipt_path is not None:
            receipt_path = bind(receipt_path, fit["bones_seed_source_receipt_sha256"])
            receipt = read(receipt_path)
            source = receipt["indexed_source"]
            if (
                receipt.get("kind") != "g1_true23_bones_seed_named29_source_v1"
                or receipt["index_report_sha256"] != inputs[str(index_path)]
                or receipt["split_sha256"] != index["split_sha256"]
                or source != indexed.get(entry["name"])
                or source["split"] != "train"
                or splits.get(source["recording_id"]) != "train"
                or audit.get("source_recording_id") != source["recording_id"]
                or audit.get("source_split") != "train"
                or fit.get("source_split") != "train"
                or fit.get("source_recording_id") != source["recording_id"]
            ):
                raise ValueError("bank member is not the exact indexed train recording")
            recording_id, family, split = source["recording_id"], source["family"], "train"
        else:
            if (
                fit.get("source_field") != "planned_qpos50"
                or fit.get("recorded_policy_pose_used_as_choreography") is not False
            ):
                raise ValueError("non-BONES member must be original planned choreography")
            recording_id, family, split = (
                "local-planner:" + fit["named_source_sha256"],
                "dance",
                "local_regression",
            )
        if recording_id in recording_ids:
            raise ValueError("bank cannot count one recording twice")
        recording_ids.add(recording_id)
        rows.append(
            dict(
                name=entry["name"],
                start=cursor,
                length=length,
                recording_id=recording_id,
                family=family,
                split=split,
                source_path=str(motion_path),
                source_sha256=inputs[str(motion_path)],
                original_frames=fit["source_frame_count"],
                collision_audit_path=str(collision_path),
                collision_audit_sha256=inputs[str(collision_path)],
                **detail,
            )
        )
        members.append(member)
        cursor += length
    if len(targets) != 1 or len(physics_models) != 1:
        raise ValueError("bank members must share identical compiled native23 FK and physical models")
    merged = concatenate_exact_members(members)
    from gear_sonic.utils.g1_true23_generalist_source_closure import collect_local_source_closure

    for path in collect_local_source_closure(Path(__file__).resolve().parents[2], [Path(__file__)]).files:
        bind(path)
    for path, expected in inputs.items():
        if sha256_file(Path(path)) != expected:
            raise ValueError(f"bank input changed during assembly: {path}")
    output.mkdir(parents=True, exist_ok=False)
    motion_out = output / "references.native23.npz"
    with motion_out.open("xb") as stream:
        np.savez_compressed(stream, **merged)
    with np.load(motion_out, allow_pickle=False) as serialized:
        for row, member in zip(rows, members, strict=True):
            for key in MOTION_KEYS:
                if not np.array_equal(serialized[key][row["start"] : row["start"] + row["length"]], member[key]):
                    raise ValueError("serialized bank changed a member array")
    spans = dict(
        kind="g1_true23_motion_corpus_spans_v1", fps=50, clip_count=len(rows), total_frames=cursor, spans=rows
    )
    report = dict(
        kind="g1_true23_small_collision_audited_local_reference_bank_v2",
        accepted_reference_bank=True,
        output={"filename": motion_out.name, "sha256": sha256_file(motion_out)},
        input_bindings=inputs,
        clips=rows,
        clip_count=len(rows),
        total_frames=cursor,
        all_member_control_and_original_timestamp_gates_passed=True,
        all_member_control_and_original_timestamp_self_collision_gates_passed=True,
        compiled_collision_physics_model_sha256=next(iter(physics_models)),
        generated_lifecycle_ramps_collision_qualified=False,
        continuous_between_sample_clearance_proven=False,
        all_six_serialized_training_arrays_exactly_equal_original_members=True,
        bones_capture_split_sha256=index["split_sha256"],
        bones_members_train_only=True,
        source_resampling_root_alignment_joint_deletion_or_transition_blending=False,
        local_regression_present=any(row["split"] == "local_regression" for row in rows),
        broad_corpus_or_held_out_generalization_qualified=False,
        training_corpus_ready=False,
        automatic_training_continuation_authorized=False,
        license_evidence_imported=False,
        publication_authorized=False,
        dynamic_feasibility_verified=False,
        hardware_authorized=False,
        deployment_ready=False,
    )
    for name, value in (("source.spans.json", spans), ("report.json", report)):
        with (output / name).open("x") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
    print(json.dumps({"output": str(output), "clips": len(rows), "frames": cursor, "hardware_authorized": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
