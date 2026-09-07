"""Local BONES-SEED discovery and lossless named-29 source conversion.

The source metadata, not file stems or converted 23-joint clips, defines
recording groups. All representations and mirrors of a capture take share a
split. This is preparation only: neither file presence nor a source split
certifies motion feasibility, dataset licensing, or controller readiness.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import json
from pathlib import Path, PurePosixPath

from gear_sonic.utils.g1_true23_generalist_corpus import (
    canonical_digest,
    sha256_file,
    split_original_recordings,
)


SOURCE_REVISION = "2f59b2077b9da34dd4e43618e705c7cb962c9a66"
SOURCE_REPOSITORY = "https://huggingface.co/datasets/bones-studio/seed"
CAPTURE_FIELDS = ("take_date", "take_day_part", "take_actor", "take_org_name")
REQUIRED_COLUMNS = {
    "move_name",
    "filename",
    "move_duration_frames",
    "package",
    "category",
    "is_mirror",
    "content_type_of_movement",
    *CAPTURE_FIELDS,
}


def _required(row, key):
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"missing metadata field {key}")
    return value.strip()


def capture_identity(row):
    """Conservative take-level identity, independent of mirror/file naming."""
    capture = {key: _required(row, key) for key in CAPTURE_FIELDS}
    return "bones-seed:take:" + canonical_digest(capture), capture


def motion_family(row):
    package = _required(row, "package").casefold()
    category = _required(row, "category").casefold()
    movement = _required(row, "content_type_of_movement").casefold()
    if package == "dances" or category == "dancing":
        return "dance"
    if movement in {"transition", "transitions"}:
        return "transition"
    if category == "baseline" and movement in {"standing", "idle"}:
        return "standing"
    if package == "locomotion":
        return "locomotion"
    if package == "communication":
        return "gesture"
    if package == "sport":
        return "sport"
    return "other"


def resolve_csv_path(root, row):
    """Resolve a metadata path without following a symlink or path escape."""
    value = row.get("move_g1_path") or row.get("move_g1_mujoco_path")
    if not isinstance(value, str) or "\\" in value:
        raise ValueError("missing or invalid G1 metadata path")
    relative = PurePosixPath(value)
    if (
        len(relative.parts) != 4
        or relative.parts[:2] != ("g1", "csv")
        or any(part in {".", ".."} for part in relative.parts)
        or relative.parts[2] != _required(row, "take_date")
        or relative.name != _required(row, "filename") + ".csv"
    ):
        raise ValueError("G1 metadata path disagrees with capture or filename")
    root = Path(root).resolve(strict=True)
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("G1 source path contains a symlink")
    if not current.resolve().is_relative_to(root):
        raise ValueError("G1 source path escapes dataset root")
    return current, relative.as_posix()


def build_source_index(metadata, extracted_root, *, split_seed):
    """Inspect every metadata row, then split complete capture groups once.

    Nonempty files are not called complete: earlier extraction may have left
    zero-length placeholders or partial CSVs. Payload validation happens only
    during explicitly selected conversion. No pickle/large corpus is loaded.
    """
    metadata = Path(metadata).resolve(strict=True)
    extracted_root = Path(extracted_root).resolve(strict=True)
    before = metadata.stat()
    metadata_hash = sha256_file(metadata)
    rows, groups = [], defaultdict(list)
    seen_names, seen_paths = set(), set()
    with metadata.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        headers = set(reader.fieldnames or ())
        if not REQUIRED_COLUMNS <= headers or not headers & {"move_g1_path", "move_g1_mujoco_path"}:
            raise ValueError("BONES-SEED metadata schema is missing required fields")
        for ordinal, row in enumerate(reader, 1):
            if ordinal > 200000:
                raise ValueError("metadata exceeds bounded 200000-row source index")
            if None in row:
                raise ValueError("metadata CSV row has excess columns")
            name = _required(row, "move_name")
            path, relative = resolve_csv_path(extracted_root, row)
            if name in seen_names or relative in seen_paths:
                raise ValueError("duplicate motion identifier or source path")
            seen_names.add(name)
            seen_paths.add(relative)
            mirror = _required(row, "is_mirror").casefold()
            if mirror not in {"true", "false"}:
                raise ValueError("is_mirror must be explicitly True or False")
            frames = int(_required(row, "move_duration_frames"))
            if frames < 2:
                raise ValueError("source frame count must be at least two")
            identity, capture = capture_identity(row)
            size = path.stat().st_size if path.is_file() else None
            item = {
                "motion_id": name,
                "metadata_row": ordinal,
                "recording_id": identity,
                "capture": capture,
                "family": motion_family(row),
                "package": row["package"],
                "category": row["category"],
                "is_mirror": mirror == "true",
                "source_frames": frames,
                "source_fps": 120,
                "source_relative_path": relative,
                "source_path": str(path),
                "source_size_bytes": size,
                "disk_status": "missing" if size is None else "empty" if size == 0 else "nonempty",
            }
            rows.append(item)
            groups[identity].append(item)
    after = metadata.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("metadata changed during source indexing")
    if not rows:
        raise ValueError("empty BONES-SEED metadata")
    recordings, mixed_families, mirror_only = [], [], []
    for identity, members in sorted(groups.items()):
        families = {member["family"] for member in members}
        if len(families) != 1:
            mixed_families.append(identity)
        if all(member["is_mirror"] for member in members):
            mirror_only.append(identity)
        recordings.append(
            {"recording_id": identity, "family": next(iter(families)) if len(families) == 1 else "other"}
        )
    splits = split_original_recordings(recordings, split_seed)
    group_family = {record["recording_id"]: record["family"] for record in recordings}
    for item in rows:
        item["split"] = splits[item["recording_id"]]
        item["recording_family"] = group_family[item["recording_id"]]
    available_groups = {
        item["recording_id"] for item in rows if item["disk_status"] == "nonempty" and not item["is_mirror"]
    }
    family_counts = defaultdict(Counter)
    for identity, family in group_family.items():
        family_counts[splits[identity]][family] += 1
    report = {
        "kind": "g1_true23_bones_seed_local_source_index_v1",
        "source_repository": SOURCE_REPOSITORY,
        "archive_revision_hint": SOURCE_REVISION,
        "metadata": {"path": str(metadata), "sha256": metadata_hash, "size_bytes": after.st_size},
        "extracted_root": str(extracted_root),
        "metadata_rows": len(rows),
        "disk_status_counts": dict(Counter(item["disk_status"] for item in rows)),
        "nonempty_file_counts_by_family": dict(
            Counter(item["family"] for item in rows if item["disk_status"] == "nonempty")
        ),
        "original_rows": sum(not item["is_mirror"] for item in rows),
        "mirror_rows": sum(item["is_mirror"] for item in rows),
        "capture_groups": len(groups),
        "capture_identity_fields": list(CAPTURE_FIELDS),
        "mixed_family_capture_groups": mixed_families,
        "mirror_only_capture_groups": mirror_only,
        "split_seed": split_seed,
        "split_sha256": canonical_digest(splits),
        "split_before_mirror_crop_tempo_or_retarget": True,
        "split_family_capture_counts": {name: dict(counts) for name, counts in family_counts.items()},
        "test_dance_groups_with_nonempty_original": sum(
            splits[key] == "test" and group_family[key] == "dance" for key in available_groups
        ),
        "archive_membership_verified": False,
        "csv_payloads_verified": False,
        "license_evidence_imported": False,
        "independent_motion_semantics_verified": False,
        "kinematic_feasibility_verified": False,
        "training_corpus_ready": False,
        "simulator_qualified": False,
        "hardware_authorized": False,
        "deployment_ready": False,
    }
    return rows, splits, report


def load_named_source_csv(path, *, expected_frames, joint_names):
    """Preserve all 120-Hz samples and all 29 joints; never clamp or project.

    BONES G1 CSV root XYZ is centimetres, angles are degrees. Euler convention
    intentionally matches the upstream loader (SciPy lowercase xyz = extrinsic
    XYZ); the old loader's prose calling that intrinsic is inaccurate.
    """
    import numpy as np
    from scipy.spatial.transform import Rotation

    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("source CSV must be a regular nonsymlink file")
    if len(joint_names) != 29 or len(set(joint_names)) != 29:
        raise ValueError("source model must name exactly 29 unique joints")
    if type(expected_frames) is not int or not 2 <= expected_frames <= 60000:
        raise ValueError("source frame count exceeds bounded 2..60000 range")
    root_columns = [f"root_{kind}{axis}" for kind in ("translate", "rotate") for axis in "XYZ"]
    expected_columns = ["Frame", *root_columns, *(name + "_dof" for name in joint_names)]
    before = path.stat()
    source_hash = sha256_file(path)
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if len(reader.fieldnames or ()) != len(expected_columns) or set(reader.fieldnames or ()) != set(
            expected_columns
        ):
            raise ValueError("source CSV must contain Frame, six root and exactly 29 named joint columns")
        values = []
        for row in reader:
            if None in row or len(values) >= expected_frames:
                raise ValueError("source CSV exceeds expected shape or frame count")
            values.append([float(row[name]) for name in expected_columns])
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("source CSV changed during conversion")
    data = np.asarray(values, dtype=np.float64)
    if data.shape != (expected_frames, 36) or not np.isfinite(data).all():
        raise ValueError("source CSV is truncated, nonfinite or has wrong frame count")
    if not np.array_equal(data[:, 0], np.arange(expected_frames) + data[0, 0]) or data[0, 0] not in (0, 1):
        raise ValueError("source frame IDs must be consecutive integers starting at zero or one")
    arrays = {
        "joint_names": np.asarray(joint_names),
        "joint_pos": np.deg2rad(data[:, 7:]),
        "root_pos_w": data[:, 1:4] * 0.01,
        "root_quat_wxyz": Rotation.from_euler("xyz", data[:, 4:7], degrees=True).as_quat()[:, [3, 0, 1, 2]],
        "fps": np.asarray([120.0]),
        "timestamps_s": np.arange(expected_frames) / 120.0,
    }
    return arrays, {
        "path": str(path.resolve()),
        "sha256": source_hash,
        "size_bytes": after.st_size,
        "frames": expected_frames,
        "frame_id_start": int(data[0, 0]),
        "fps": 120,
        "all_source_samples_preserved": True,
        "all_29_joints_preserved": True,
        "joint_clipping_applied": False,
        "root_reanchored": False,
        "source_role": "requested_choreography_not_measured_robot_rollout",
    }


def select_training_sources(rows, *, families, per_family, min_frames, max_frames, seed):
    """Freeze a diagnostic cohort, never select by retarget or rollout outcome."""
    if len(set(families)) != len(families) or not families:
        raise ValueError("selection requires distinct motion families")
    if type(per_family) is not int or not 1 <= per_family <= 100:
        raise ValueError("per-family selection must be in 1..100")
    if not 2 <= min_frames <= max_frames <= 60000:
        raise ValueError("source duration bounds must be within 2..60000 frames")
    selected, counts, chosen_groups = [], {}, set()
    for family in families:
        candidates = [
            row
            for row in rows
            if row["recording_family"] == family
            and row["family"] == family
            and row["split"] == "train"
            and not row["is_mirror"]
            and row["disk_status"] == "nonempty"
            and min_frames <= row["source_frames"] <= max_frames
        ]
        candidates.sort(
            key=lambda row: (canonical_digest([seed, row["recording_id"], row["motion_id"]]), row["motion_id"])
        )
        counts[family] = len(candidates)
        family_rows = []
        for row in candidates:
            if row["recording_id"] not in chosen_groups:
                family_rows.append(row)
                chosen_groups.add(row["recording_id"])
            if len(family_rows) == per_family:
                break
        if len(family_rows) != per_family:
            raise ValueError(f"not enough distinct train-split originals for {family}")
        selected.extend(family_rows)
    return {
        "kind": "g1_true23_bones_seed_frozen_training_diagnostic_cohort_v1",
        "selection_seed": seed,
        "split": "train",
        "originals_only": True,
        "min_source_frames": min_frames,
        "max_source_frames": max_frames,
        "selection_before_payload_validation_retarget_or_rollout": True,
        "candidates_by_family": counts,
        "selected": selected,
        "held_out_recordings_included": False,
        "training_corpus_ready": False,
        "kinematic_feasibility_verified": False,
        "hardware_authorized": False,
        "deployment_ready": False,
    }


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
