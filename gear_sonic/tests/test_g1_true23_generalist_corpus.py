"""Deterministic splits and provenance checks, without loading motion payloads."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from gear_sonic.scripts.audit_g1_true23_generalist_corpus import main
from gear_sonic.utils.g1_true23_generalist_corpus import (
    audit_manifest,
    canonical_digest,
    inventory_roots,
    sha256_file,
    split_original_recordings,
)


def _file(directory: Path, name: str, content: bytes) -> dict[str, str]:
    path = directory / name
    path.write_bytes(content)
    return {"path": name, "sha256": sha256_file(path)}


def _manifest(directory: Path, count: int = 10) -> dict:
    evidence = _file(directory, "provenance.txt", b"Test-only explicit source and license fixture.")
    recordings = []
    assets = []
    for index in range(count):
        identifier = f"recording-{index:05d}"
        asset_id = f"original-{index:05d}"
        recordings.append(
            {
                "recording_id": identifier,
                "source_uri": "https://example.invalid/test-only-fixture",
                "source_revision": "test-immutable-revision",
                "source_recording_id": identifier,
                "original_asset_id": asset_id,
                "family": "dance",
                "lineage_reviewed": True,
                "lineage_evidence": evidence,
                "license": {
                    "identifier": "TEST-ONLY",
                    "source_uri": "https://example.invalid/license",
                    "training_and_evaluation_permitted": True,
                    "evidence": evidence,
                },
            }
        )
        assets.append(
            {
                "asset_id": asset_id,
                "recording_id": identifier,
                "parent_asset_id": None,
                "transform": "original",
                "format": "fixture-not-real-motion",
                "joint_names": ["hip_pitch", "knee"],
                "timing": {"fps": 50.0, "frame_count": 100, "start_time_s": 0.0, "sampling": "uniform"},
                "file": _file(directory, f"{asset_id}.csv", f"unique-test-fixture-{index}".encode()),
            }
        )
    return {"schema_version": 1, "split_seed": "sonic-generalist-v1", "recordings": recordings, "assets": assets}


def _variant(manifest: dict, directory: Path, *, transform: str = "mirror") -> dict:
    derived = copy.deepcopy(manifest["assets"][0])
    derived.update(asset_id=f"derived-{transform}", parent_asset_id=derived["asset_id"], transform=transform)
    derived["file"] = _file(directory, f"{transform}.csv", f"different-{transform}".encode())
    if transform == "crop":
        derived["transform_parameters"] = {"parent_start_frame": 2, "parent_stop_frame_exclusive": 42}
        derived["timing"]["frame_count"] = 40
    if transform == "tempo":
        derived["transform_parameters"] = {"duration_scale": 1.4}
    manifest["assets"].append(derived)
    return derived


def test_split_exact_deterministic_per_family_before_derivatives():
    records = [
        {"recording_id": f"{family}-{index}", "family": family}
        for family in ("dance", "locomotion")
        for index in range(1000)
    ]
    first = split_original_recordings(records, "fixed")
    assert first == split_original_recordings(list(reversed(records)), "fixed")
    assert first != split_original_recordings(records, "another")
    for family in ("dance", "locomotion"):
        values = [value for key, value in first.items() if key.startswith(family)]
        assert [values.count(name) for name in ("train", "validation", "test")] == [800, 100, 100]


@pytest.mark.parametrize("size,counts", [(1, [1, 0, 0]), (5, [4, 1, 0]), (10, [8, 1, 1]), (11, [9, 1, 1])])
def test_small_splits_are_not_fabricated(size, counts):
    result = split_original_recordings([{"recording_id": str(i), "family": "dance"} for i in range(size)], "test")
    assert [list(result.values()).count(name) for name in ("train", "validation", "test")] == counts


@pytest.mark.parametrize("transform", ["mirror", "crop", "tempo", "retarget", "representation"])
def test_variants_inherit_split_and_never_increase_unique_recordings(tmp_path, transform):
    manifest = _manifest(tmp_path)
    baseline = audit_manifest(manifest, tmp_path)
    derivative = _variant(manifest, tmp_path, transform=transform)
    result = audit_manifest(manifest, tmp_path)
    assert result["unique_original_recordings"] == 10
    assert result["derivative_asset_count"] == 1
    assert result["split_sha256"] == baseline["split_sha256"]
    assert result["asset_splits"][derivative["asset_id"]] == result["asset_splits"][derivative["parent_asset_id"]]
    assert result["independent_held_out_dances"] == 1
    assert result["manifest_valid"]
    assert not result["corpus_quantity_and_coverage_sufficient"]
    assert not result["policy_generalization_verified"]
    assert not result["hardware_authorized"]


def test_integrity_does_not_claim_payload_validation(tmp_path):
    result = audit_manifest(_manifest(tmp_path), tmp_path)
    assert not result["payload_shapes_and_timestamps_verified"]
    assert not result["license_and_lineage_semantics_independently_verified"]
    assert not result["simulator_qualification_complete"]


@pytest.mark.parametrize(
    "mutation,error",
    [
        (lambda m: m.update(schema_version=2), "schema_version"),
        (lambda m: m.update(recordings=[]), "requires original"),
        (lambda m: m["recordings"][0].update(lineage_reviewed=False), "lineage"),
        (lambda m: m["recordings"][0]["license"].update(training_and_evaluation_permitted=False), "license"),
        (lambda m: m["recordings"][0].update(family="dance-ish"), "family"),
        (lambda m: m["recordings"][0].update(source_recording_id=""), "source_recording_id"),
        (lambda m: m["recordings"][0].update(source_revision=""), "source_revision"),
        (
            lambda m: m["recordings"][1].update(source_recording_id=m["recordings"][0]["source_recording_id"]),
            "same source",
        ),
        (
            lambda m: m["recordings"][1].update(recording_id=m["recordings"][0]["recording_id"]),
            "duplicate recording",
        ),
        (lambda m: m["assets"][1].update(asset_id=m["assets"][0]["asset_id"]), "duplicate asset"),
        (lambda m: m["assets"][1].update(file=m["assets"][0]["file"]), "identical motion content"),
        (lambda m: m["assets"][0].update(recording_id="unknown"), "unknown recording"),
        (lambda m: m["assets"][0].update(transform="mirrored_and_cropped_unknown"), "transform"),
        (lambda m: m["assets"][0].update(joint_names=[]), "joint_names"),
        (lambda m: m["assets"][0].update(joint_names=["knee", "knee"]), "duplicates"),
        (lambda m: m["assets"][0]["timing"].update(fps=float("nan")), "finite"),
        (lambda m: m["assets"][0]["timing"].update(fps=0), "positive"),
        (lambda m: m["assets"][0]["timing"].update(frame_count=True), "integer"),
        (lambda m: m["assets"][0]["timing"].update(start_time_s=-1), "nonnegative"),
        (lambda m: m["assets"][0]["timing"].update(sampling="unknown"), "sampling"),
        (lambda m: m["assets"][0]["file"].update(sha256="bad"), "SHA256"),
        (lambda m: m["recordings"][0].update(original_asset_id="unknown"), "original_asset_id"),
    ],
)
def test_rejects_invalid_provenance_and_contracts(tmp_path, mutation, error):
    manifest = _manifest(tmp_path)
    mutation(manifest)
    with pytest.raises(ValueError, match=error):
        audit_manifest(manifest, tmp_path)


def test_changed_motion_and_evidence_rejected(tmp_path):
    manifest = _manifest(tmp_path)
    path = tmp_path / manifest["assets"][0]["file"]["path"]
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        audit_manifest(manifest, tmp_path)
    manifest["assets"][0]["file"]["sha256"] = sha256_file(path)
    (tmp_path / "provenance.txt").write_bytes(b"changed license declaration")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        audit_manifest(manifest, tmp_path)


def test_duplicate_bytes_with_distinct_paths_rejected(tmp_path):
    manifest = _manifest(tmp_path)
    first = manifest["assets"][0]["file"]
    duplicate = _file(tmp_path, "different-name.csv", (tmp_path / first["path"]).read_bytes())
    manifest["assets"][1]["file"] = duplicate
    with pytest.raises(ValueError, match="identical motion content"):
        audit_manifest(manifest, tmp_path)


@pytest.mark.parametrize("where", ["recordings", "assets"])
def test_declared_split_conflict_rejected(tmp_path, where):
    manifest = _manifest(tmp_path)
    manifest[where][0]["split"] = "other"
    with pytest.raises(ValueError, match="split conflicts"):
        audit_manifest(manifest, tmp_path)


@pytest.mark.parametrize(
    "parent,error",
    [
        ("unknown", "known parent"),
        ("original-00001", "crosses original"),
        ("derived-mirror", "cyclic"),
        ([], "parent_asset_id"),
    ],
)
def test_invalid_derivative_lineage_rejected(tmp_path, parent, error):
    manifest = _manifest(tmp_path)
    derivative = _variant(manifest, tmp_path)
    derivative["parent_asset_id"] = parent
    with pytest.raises(ValueError, match=error):
        audit_manifest(manifest, tmp_path)


def test_second_original_cannot_masquerade_as_independent(tmp_path):
    manifest = _manifest(tmp_path)
    derivative = _variant(manifest, tmp_path)
    derivative.update(transform="original", parent_asset_id=None)
    with pytest.raises(ValueError, match="unregistered or ambiguous original"):
        audit_manifest(manifest, tmp_path)


@pytest.mark.parametrize(
    "parameters",
    [
        {},
        {"parent_start_frame": -1, "parent_stop_frame_exclusive": 42},
        {"parent_start_frame": 2, "parent_stop_frame_exclusive": 101},
        {"parent_start_frame": 2, "parent_stop_frame_exclusive": 2},
    ],
)
def test_crop_requires_precise_parent_bounds(tmp_path, parameters):
    manifest = _manifest(tmp_path)
    derivative = _variant(manifest, tmp_path, transform="crop")
    derivative["transform_parameters"] = parameters
    with pytest.raises(ValueError, match="crop parent frame"):
        audit_manifest(manifest, tmp_path)


def test_explicit_timestamps_hash_bound_without_false_shape_claim(tmp_path):
    manifest = _manifest(tmp_path)
    timing = manifest["assets"][0]["timing"]
    timing.update(sampling="explicit_timestamps", timestamps=_file(tmp_path, "times.csv", b"0.0\n0.02\n"))
    result = audit_manifest(manifest, tmp_path)
    assert not result["payload_shapes_and_timestamps_verified"]
    assert any(binding["path"].endswith("times.csv") for binding in result["evidence_bindings"])


def test_quantity_gate_needs_100_independent_dances_and_other_families(tmp_path):
    manifest = _manifest(tmp_path, 1020)
    for index, record in enumerate(manifest["recordings"][1000:]):
        record["family"] = "locomotion" if index < 10 else "transition"
    result = audit_manifest(manifest, tmp_path)
    assert result["independent_held_out_dances"] == 100
    assert result["recording_families"] == {
        record["recording_id"]: record["family"] for record in manifest["recordings"]
    }
    assert result["corpus_quantity_and_coverage_sufficient"]
    assert not result["policy_generalization_verified"]
    assert not result["simulator_qualification_complete"]


def test_inventory_never_deserializes_pickle_or_claims_motion_count(tmp_path):
    _file(tmp_path, "evil.pkl", b"not pickle; merely hashed as bytes")
    _file(tmp_path, "dance_mirror.csv", b"same derivative not independent")
    result = inventory_roots([tmp_path], hash_files=True)
    assert result["candidate_file_count"] == 2
    assert result["unique_original_recordings"] is None
    assert not result["payload_deserialized"]
    assert not result["corpus_quantity_and_coverage_sufficient"]
    assert all(len(record["sha256"]) == 64 for record in result["files"])


def test_inventory_bounded_and_rejects_overlapping_roots(tmp_path):
    for index in range(3):
        _file(tmp_path, f"{index}.csv", b"sample")
    report = inventory_roots([tmp_path], maximum_files=2)
    assert report["truncated"] and report["candidate_file_count"] == 2
    nested = tmp_path / "nested"
    nested.mkdir()
    with pytest.raises(ValueError, match="overlap"):
        inventory_roots([tmp_path, nested])


def test_inventory_does_not_follow_symlink(tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()
    _file(nested, "one.csv", b"sample")
    try:
        (tmp_path / "loop").symlink_to(tmp_path, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks unavailable")
    report = inventory_roots([tmp_path])
    assert report["candidate_file_count"] == 1
    assert report["skipped_symlinks"] == 1


def test_cli_reports_missing_corpus_and_preserves_outputs(tmp_path):
    manifest = _manifest(tmp_path)
    source = tmp_path / "manifest.json"
    source.write_text(json.dumps(manifest))
    output = tmp_path / "audit.json"
    args = ["audit", "--manifest", str(source), "--output", str(output), "--require-generalization-corpus"]
    assert main(args) == 2
    report = json.loads(output.read_text())
    assert report["manifest_valid"] and not report["corpus_quantity_and_coverage_sufficient"]
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        main(args)
    assert output.read_bytes() == before


def test_cli_rejects_invalid_json_without_loading_motion(tmp_path):
    source = tmp_path / "bad.json"
    source.write_text("not JSON")
    output = tmp_path / "audit.json"
    assert main(["audit", "--manifest", str(source), "--output", str(output)]) == 2
    assert not json.loads(output.read_text())["manifest_valid"]


def test_canonical_digest_ignores_mapping_order_but_not_contents():
    assert canonical_digest({"a": 1, "b": 2}) == canonical_digest({"b": 2, "a": 1})
    assert canonical_digest({"a": 1}) != canonical_digest({"a": 2})
