"""Read-only corpus provenance and original-recording split contracts.

Passing this audit establishes neither kinematic feasibility nor policy quality.
No pickle/NPZ payload is deserialized, robot transport opened, or file downloaded.
License/lineage declarations must point to hash-bound evidence; this validator
checks their integrity, not the legal accuracy of an operator's declarations.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
SPLIT_NAMES = ("train", "validation", "test")
SPLIT_PERCENTAGES = (80, 10, 10)
FAMILIES = frozenset({"dance", "locomotion", "standing", "transition", "gesture", "sport", "other"})
TRANSFORMS = frozenset({"original", "mirror", "crop", "tempo", "retarget", "representation"})
MOTION_SUFFIXES = frozenset({".csv", ".npz", ".npy", ".pkl", ".pickle", ".bvh", ".json"})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{label} must be a nonempty trimmed string")
    return value


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _positive_number(value: Any, label: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    if not math.isfinite(value) or value < 0 or (not allow_zero and value == 0):
        raise ValueError(f"{label} must be finite and {'nonnegative' if allow_zero else 'positive'}")
    return float(value)


def _binding(value: Any, base_dir: Path, label: str) -> dict[str, Any]:
    binding = _mapping(value, label)
    supplied_path = Path(_text(binding.get("path"), f"{label}.path"))
    expected = _text(binding.get("sha256"), f"{label}.sha256")
    if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
        raise ValueError(f"{label}.sha256 must be lowercase SHA256")
    path = (base_dir / supplied_path).resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"{label}.path must identify a regular file")
    before = path.stat()
    actual = sha256_file(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"{label} changed while hashing")
    if actual != expected:
        raise ValueError(f"{label} SHA256 mismatch")
    return {"path": str(path), "sha256": actual, "size_bytes": after.st_size}


def split_original_recordings(recordings: list[dict[str, Any]], seed: str) -> dict[str, str]:
    """Exact largest-remainder 80/10/10 per family, before asset augmentation.

    Dataset membership changes may change assignments. Consumers must pin the
    returned split SHA256 and complete manifest, never rebuild a split per run.
    No filename-derived or asset-count-derived identity is used.
    """
    _text(seed, "split_seed")
    grouped: dict[str, list[str]] = defaultdict(list)
    seen: set[str] = set()
    for record in recordings:
        identifier = _text(record.get("recording_id"), "recording_id")
        family = _text(record.get("family"), "family")
        if identifier in seen or family not in FAMILIES:
            raise ValueError("duplicate recording ID or unsupported motion family")
        seen.add(identifier)
        grouped[family].append(identifier)
    assigned: dict[str, str] = {}
    for family, identifiers in sorted(grouped.items()):
        ordered = sorted(
            identifiers, key=lambda identifier: (canonical_digest([seed, family, identifier]), identifier)
        )
        numerators = [len(ordered) * percentage for percentage in SPLIT_PERCENTAGES]
        counts = [numerator // 100 for numerator in numerators]
        remainder_order = sorted(range(3), key=lambda index: (-(numerators[index] % 100), index))
        for index in remainder_order[: len(ordered) - sum(counts)]:
            counts[index] += 1
        cursor = 0
        for name, count in zip(SPLIT_NAMES, counts, strict=True):
            for identifier in ordered[cursor : cursor + count]:
                assigned[identifier] = name
            cursor += count
    return dict(sorted(assigned.items()))


def audit_manifest(manifest: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    """Validate hash-bound inputs and lineage; return a non-authorizing report.

    Each recording declares one original asset, a globally meaningful source
    URI/recording ID, motion family, license evidence and lineage evidence.
    Every derivative names its parent asset and inherits that original's split.
    """
    _mapping(manifest, "manifest")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported corpus manifest schema_version")
    seed = _text(manifest.get("split_seed"), "split_seed")
    records = manifest.get("recordings")
    assets = manifest.get("assets")
    if not isinstance(records, list) or not isinstance(assets, list):
        raise ValueError("recordings and assets must be arrays")
    if not records or not assets:
        raise ValueError("corpus requires original recordings and assets")
    by_recording: dict[str, dict[str, Any]] = {}
    source_keys: set[tuple[str, str]] = set()
    evidence: list[dict[str, Any]] = []
    for record in records:
        _mapping(record, "recording")
        identifier = _text(record.get("recording_id"), "recording_id")
        if identifier in by_recording:
            raise ValueError("duplicate recording_id")
        source_uri = _text(record.get("source_uri"), "source_uri")
        source_id = _text(record.get("source_recording_id"), "source_recording_id")
        _text(record.get("source_revision"), "source_revision")
        _text(record.get("original_asset_id"), "original_asset_id")
        if record.get("family") not in FAMILIES:
            raise ValueError("unsupported motion family")
        key = (source_uri, source_id)
        if key in source_keys:
            raise ValueError("same source recording declared as multiple originals")
        source_keys.add(key)
        if record.get("lineage_reviewed") is not True:
            raise ValueError("original recording lineage must be explicitly reviewed")
        evidence.append(_binding(record.get("lineage_evidence"), base_dir, "lineage_evidence"))
        license_info = _mapping(record.get("license"), "license")
        _text(license_info.get("identifier"), "license.identifier")
        _text(license_info.get("source_uri"), "license.source_uri")
        if license_info.get("training_and_evaluation_permitted") is not True:
            raise ValueError("license must explicitly permit training and evaluation")
        evidence.append(_binding(license_info.get("evidence"), base_dir, "license.evidence"))
        by_recording[identifier] = record

    split = split_original_recordings(records, seed)
    for record in records:
        if record.get("split") is not None and record["split"] != split[record["recording_id"]]:
            raise ValueError("recording split conflicts with deterministic original split")
    by_asset: dict[str, dict[str, Any]] = {}
    asset_bindings: dict[str, dict[str, Any]] = {}
    paths_to_recording: dict[str, str] = {}
    hashes_to_recording: dict[str, str] = {}
    for asset in assets:
        _mapping(asset, "asset")
        identifier = _text(asset.get("asset_id"), "asset_id")
        if identifier in by_asset:
            raise ValueError("duplicate asset_id")
        recording_id = _text(asset.get("recording_id"), "asset.recording_id")
        if recording_id not in by_recording:
            raise ValueError("asset references unknown recording")
        if asset.get("transform") not in TRANSFORMS:
            raise ValueError("unsupported or ambiguous asset transform")
        _text(asset.get("format"), "asset.format")
        names = asset.get("joint_names")
        if (
            not isinstance(names, list)
            or not names
            or any(not isinstance(name, str) or not name.strip() for name in names)
        ):
            raise ValueError("joint_names must explicitly describe ordered named joints")
        if len(set(names)) != len(names):
            raise ValueError("joint_names cannot contain duplicates")
        timing = _mapping(asset.get("timing"), "asset.timing")
        _positive_number(timing.get("fps"), "timing.fps")
        frames = timing.get("frame_count")
        if isinstance(frames, bool) or not isinstance(frames, int) or frames < 2:
            raise ValueError("timing.frame_count must be an integer >= 2")
        _positive_number(timing.get("start_time_s"), "timing.start_time_s", allow_zero=True)
        if timing.get("sampling") not in {"uniform", "explicit_timestamps"}:
            raise ValueError("timing.sampling must be uniform or explicit_timestamps")
        if timing["sampling"] == "explicit_timestamps":
            evidence.append(_binding(timing.get("timestamps"), base_dir, "timing.timestamps"))
        binding = _binding(asset.get("file"), base_dir, f"asset[{identifier}].file")
        for value, registry in ((binding["path"], paths_to_recording), (binding["sha256"], hashes_to_recording)):
            if value in registry and registry[value] != recording_id:
                raise ValueError("identical motion content declared as independent recordings")
            registry[value] = recording_id
        assigned_split = asset.get("split")
        if assigned_split is not None and assigned_split != split[recording_id]:
            raise ValueError("asset split conflicts with original-recording split")
        by_asset[identifier] = asset
        asset_bindings[identifier] = binding

    for record in records:
        original = by_asset.get(record["original_asset_id"])
        if original is None or original["recording_id"] != record["recording_id"]:
            raise ValueError("original_asset_id must reference its own recording")
        if original["transform"] != "original" or original.get("parent_asset_id") is not None:
            raise ValueError("original asset must have original transform and no parent")
    for identifier, asset in by_asset.items():
        visited: set[str] = set()
        cursor = identifier
        while True:
            if cursor in visited:
                raise ValueError("cyclic asset lineage")
            visited.add(cursor)
            current = by_asset[cursor]
            parent = current.get("parent_asset_id")
            if current["transform"] == "original":
                if parent is not None or cursor != by_recording[asset["recording_id"]]["original_asset_id"]:
                    raise ValueError("unregistered or ambiguous original asset")
                break
            _text(parent, "derived asset parent_asset_id")
            if parent not in by_asset:
                raise ValueError("derived asset requires known parent_asset_id")
            if by_asset[parent]["recording_id"] != asset["recording_id"]:
                raise ValueError("derivative lineage crosses original recordings")
            cursor = parent

    for asset in assets:
        if asset["transform"] in {"crop", "tempo"}:
            parameters = _mapping(asset.get("transform_parameters"), "transform_parameters")
            if asset["transform"] == "tempo":
                _positive_number(parameters.get("duration_scale"), "tempo.duration_scale")
            else:
                start = parameters.get("parent_start_frame")
                stop = parameters.get("parent_stop_frame_exclusive")
                parent_frames = by_asset[asset["parent_asset_id"]]["timing"]["frame_count"]
                if any(isinstance(value, bool) or not isinstance(value, int) for value in (start, stop)):
                    raise ValueError("crop parent frame bounds must be integers")
                if not 0 <= start < stop <= parent_frames:
                    raise ValueError("crop parent frame bounds outside parent recording")

    split_counts = {name: Counter() for name in SPLIT_NAMES}
    for record in records:
        split_counts[split[record["recording_id"]]][record["family"]] += 1
    held_out_dances = split_counts["test"]["dance"]
    gaps: list[str] = []
    if held_out_dances < 100:
        gaps.append(f"test split has {held_out_dances} independent dances; at least 100 required")
    if not split_counts["train"]["dance"]:
        gaps.append("training split has no independent dance recordings")
    if not split_counts["validation"]["dance"]:
        gaps.append("validation split has no independent dance recordings")
    training_families = {family for family, count in split_counts["train"].items() if count}
    if not {"dance", "locomotion", "transition"}.issubset(training_families):
        gaps.append("training split lacks dance, locomotion or transition coverage")
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "g1_true23_generalist_corpus_audit",
        "manifest_valid": True,
        "manifest_sha256": canonical_digest(manifest),
        "split_algorithm": "family_stratified_sha256_rank_largest_remainder_80_10_10_v1",
        "split_seed": seed,
        "split_sha256": canonical_digest(split),
        "recording_splits": split,
        "recording_families": {key: by_recording[key]["family"] for key in sorted(by_recording)},
        "asset_splits": {key: split[value["recording_id"]] for key, value in sorted(by_asset.items())},
        "unique_original_recordings": len(records),
        "asset_count": len(assets),
        "derivative_asset_count": sum(asset["transform"] != "original" for asset in assets),
        "family_recording_counts": dict(sorted(Counter(record["family"] for record in records).items())),
        "split_family_recording_counts": {
            name: dict(sorted(counts.items())) for name, counts in split_counts.items()
        },
        "independent_held_out_dances": held_out_dances,
        "corpus_quantity_and_coverage_sufficient": not gaps,
        "corpus_gaps": gaps,
        "asset_bindings": asset_bindings,
        "asset_metadata": {
            key: {
                "recording_id": value["recording_id"],
                "transform": value["transform"],
                "timing": dict(value["timing"]),
                "joint_names": list(value["joint_names"]),
            }
            for key, value in sorted(by_asset.items())
        },
        "evidence_bindings": sorted(
            {item["path"]: item for item in evidence}.values(), key=lambda item: item["path"]
        ),
        "license_and_lineage_semantics_independently_verified": False,
        "payload_shapes_and_timestamps_verified": False,
        "kinematic_feasibility_verified": False,
        "policy_generalization_verified": False,
        "simulator_qualification_complete": False,
        "hardware_authorized": False,
    }


def inventory_roots(roots: list[Path], *, maximum_files: int = 10000, hash_files: bool = False) -> dict[str, Any]:
    """Bounded filesystem inventory, deliberately not an independent-motion census.

    Never follows directory symlinks or opens pickle/array payloads. Candidate
    filenames are evidence of files only, not license, motion family or lineage.
    """
    if not roots:
        raise ValueError("at least one explicit inventory root is required")
    if isinstance(maximum_files, bool) or not isinstance(maximum_files, int) or not 1 <= maximum_files <= 100000:
        raise ValueError("maximum_files must be in [1, 100000]")
    resolved_roots: list[Path] = []
    for root in roots:
        resolved = root.resolve(strict=True)
        if not resolved.is_dir():
            raise ValueError("inventory root must be a directory")
        if any(
            resolved == existing or resolved in existing.parents or existing in resolved.parents
            for existing in resolved_roots
        ):
            raise ValueError("inventory roots cannot overlap")
        resolved_roots.append(resolved)
    records: list[dict[str, Any]] = []
    skipped_symlinks = 0
    examined_files = 0
    examined_entries = 0
    truncated = False
    for root in sorted(resolved_roots):
        stack = [root]
        while stack:
            directory = stack.pop()
            with os.scandir(directory) as entries:
                ordered = sorted(entries, key=lambda entry: entry.name)
            for entry in ordered:
                if examined_entries >= maximum_files * 20:
                    truncated = True
                    break
                examined_entries += 1
                if entry.is_symlink():
                    skipped_symlinks += 1
                elif entry.is_dir(follow_symlinks=False):
                    if entry.name not in {".git", ".cache", "__pycache__"}:
                        stack.append(Path(entry.path))
                elif entry.is_file(follow_symlinks=False):
                    if examined_files >= maximum_files:
                        truncated = True
                        break
                    examined_files += 1
                    path = Path(entry.path)
                    if path.suffix.lower() in MOTION_SUFFIXES:
                        info: dict[str, Any] = {
                            "root": str(root),
                            "path": str(path.relative_to(root)),
                            "size_bytes": entry.stat().st_size,
                        }
                        if hash_files:
                            info["sha256"] = sha256_file(path)
                        records.append(info)
            if truncated:
                break
        if truncated:
            break
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "g1_true23_motion_file_inventory_not_corpus_qualification",
        "roots": [str(root) for root in resolved_roots],
        "maximum_files": maximum_files,
        "examined_files": examined_files,
        "examined_entries": examined_entries,
        "truncated": truncated,
        "skipped_symlinks": skipped_symlinks,
        "candidate_file_count": len(records),
        "suffix_counts": dict(sorted(Counter(Path(record["path"]).suffix.lower() for record in records).items())),
        "files": sorted(records, key=lambda item: (item["root"], item["path"])),
        "unique_original_recordings": None,
        "independent_held_out_dances": None,
        "license_and_lineage_verified": False,
        "payload_deserialized": False,
        "corpus_quantity_and_coverage_sufficient": False,
        "simulator_qualification_complete": False,
        "hardware_authorized": False,
    }
