from __future__ import annotations

import csv

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from gear_sonic.utils.g1_true23_bones_seed import (
    build_source_index,
    capture_identity,
    load_named_source_csv,
    motion_family,
    resolve_csv_path,
    select_training_sources,
)


def metadata_row(number=1, *, mirror=False, **overrides):
    name = f"dance_{number:03d}__A001" + ("_M" if mirror else "")
    return {
        "move_name": name,
        "filename": name,
        "move_duration_frames": "3",
        "package": "Dances",
        "category": "Dancing",
        "is_mirror": str(mirror),
        "move_g1_path": f"g1/csv/220101/{name}.csv",
        "take_name": f"dance_{number:03d}",
        "take_date": "220101",
        "take_day_part": "_1",
        "take_actor": "A001",
        "take_org_name": f"dance_{number:03d}",
        "content_type_of_movement": "dancing",
        **overrides,
    }


def write_metadata(tmp_path, rows):
    path = tmp_path / "metadata.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_original_and_mirror_use_metadata_capture_not_filename():
    original, mirror = metadata_row(), metadata_row(mirror=True)
    assert capture_identity(original) == capture_identity(mirror)
    renamed = {**original, "filename": "unrelated", "move_name": "unrelated"}
    assert capture_identity(renamed) == capture_identity(original)
    assert capture_identity(metadata_row(take_actor="A002")) != capture_identity(original)


def test_index_splits_whole_takes_before_disk_availability(tmp_path):
    rows = [metadata_row(i, mirror=mirror) for i in range(10) for mirror in (False, True)]
    csv_path = tmp_path / rows[0]["move_g1_path"]
    csv_path.parent.mkdir(parents=True)
    csv_path.write_text("not a qualified payload")
    (tmp_path / rows[1]["move_g1_path"]).touch()
    metadata = write_metadata(tmp_path, rows)
    indexed, split, report = build_source_index(metadata, tmp_path, split_seed="test")
    assert report["capture_groups"] == 10
    assert report["original_rows"] == report["mirror_rows"] == 10
    assert report["disk_status_counts"] == {"nonempty": 1, "empty": 1, "missing": 18}
    assert report["split_family_capture_counts"] == {
        "train": {"dance": 8},
        "validation": {"dance": 1},
        "test": {"dance": 1},
    }
    for i in range(0, len(indexed), 2):
        assert indexed[i]["split"] == indexed[i + 1]["split"]
    csv_path.unlink()
    _, second_split, second_report = build_source_index(metadata, tmp_path, split_seed="test")
    assert second_split == split
    assert report["split_sha256"] == second_report["split_sha256"]
    for field in (
        "archive_membership_verified",
        "csv_payloads_verified",
        "training_corpus_ready",
        "deployment_ready",
        "hardware_authorized",
    ):
        assert report[field] is False


def test_duplicate_metadata_rejected(tmp_path):
    metadata = write_metadata(tmp_path, [metadata_row(), metadata_row()])
    with pytest.raises(ValueError, match="duplicate"):
        build_source_index(metadata, tmp_path, split_seed="test")


def test_missing_capture_identity_rejected():
    with pytest.raises(ValueError, match="take_org_name"):
        capture_identity(metadata_row(take_org_name=""))


@pytest.mark.parametrize(
    "bad",
    [
        "../escape.csv",
        "g1/csv/../escape.csv",
        "g1/csv/220102/dance_001__A001.csv",
        "g1\\csv\\220101\\dance_001__A001.csv",
    ],
)
def test_path_rejects_escape_or_metadata_disagreement(tmp_path, bad):
    with pytest.raises(ValueError, match="path"):
        resolve_csv_path(tmp_path, metadata_row(move_g1_path=bad))


def test_symlink_source_rejected(tmp_path):
    source = tmp_path / "outside.csv"
    source.write_text("placeholder")
    link = tmp_path / metadata_row()["move_g1_path"]
    link.parent.mkdir(parents=True)
    try:
        link.symlink_to(source)
    except OSError:
        pytest.skip("platform cannot create symlinks")
    with pytest.raises(ValueError, match="symlink"):
        resolve_csv_path(tmp_path, metadata_row())


def test_mixed_family_and_mirror_only_groups_are_not_hidden(tmp_path):
    rows = [
        metadata_row(),
        metadata_row(mirror=True, package="Locomotion", category="Baseline", content_type_of_movement="standing"),
        metadata_row(2, mirror=True),
    ]
    _, _, report = build_source_index(write_metadata(tmp_path, rows), tmp_path, split_seed="test")
    assert len(report["mixed_family_capture_groups"]) == 1
    assert len(report["mirror_only_capture_groups"]) == 1


def test_family_comes_from_structured_annotations():
    assert motion_family(metadata_row()) == "dance"
    assert (
        motion_family(metadata_row(package="Locomotion", category="Baseline", content_type_of_movement="standing"))
        == "standing"
    )
    assert (
        motion_family(
            metadata_row(
                package="Locomotion", category="Basic Locomotion Neutral", content_type_of_movement="transition"
            )
        )
        == "transition"
    )


def write_motion(path, *, mutate=None, column_names=None):
    names = [f"joint_{i}" for i in range(29)]
    fields = [
        "Frame",
        *(f"root_{kind}{axis}" for kind in ("translate", "rotate") for axis in "XYZ"),
        *(name + "_dof" for name in names),
    ]
    rows = []
    for index in range(3):
        row = dict(zip(fields, [index + 1, index * 10, 20, 76, 20, 30, 40, *range(29)], strict=True))
        if mutate:
            mutate(index, row)
        rows.append(row)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=column_names or fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return names, fields


def test_source_conversion_preserves_all_joints_samples_units_and_root(tmp_path):
    path = tmp_path / "source.csv"
    names, _ = write_motion(path)
    order = list(reversed(names))
    arrays, receipt = load_named_source_csv(path, expected_frames=3, joint_names=order)
    assert arrays["joint_names"].tolist() == order
    np.testing.assert_allclose(arrays["joint_pos"][0], np.deg2rad(list(reversed(range(29)))))
    np.testing.assert_allclose(arrays["root_pos_w"], [[0, 0.2, 0.76], [0.1, 0.2, 0.76], [0.2, 0.2, 0.76]])
    np.testing.assert_allclose(arrays["timestamps_s"], np.arange(3) / 120)
    expected = Rotation.from_euler("xyz", [20, 30, 40], degrees=True).as_quat()[[3, 0, 1, 2]]
    np.testing.assert_allclose(arrays["root_quat_wxyz"], np.tile(expected, (3, 1)))
    assert receipt["all_source_samples_preserved"] and receipt["all_29_joints_preserved"]
    assert not receipt["joint_clipping_applied"] and not receipt["root_reanchored"]


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda index, row: row.update(Frame=4) if index == 1 else None, "frame IDs"),
        (lambda index, row: row.update(root_translateX="nan"), "nonfinite"),
    ],
)
def test_source_rejects_corrupt_payload(tmp_path, mutation, match):
    path = tmp_path / "source.csv"
    names, _ = write_motion(path, mutate=mutation)
    with pytest.raises(ValueError, match=match):
        load_named_source_csv(path, expected_frames=3, joint_names=names)


def test_source_rejects_partial_or_extra_frames(tmp_path):
    path = tmp_path / "source.csv"
    names, _ = write_motion(path)
    with pytest.raises(ValueError, match="truncated"):
        load_named_source_csv(path, expected_frames=4, joint_names=names)
    with pytest.raises(ValueError, match="frame count"):
        load_named_source_csv(path, expected_frames=2, joint_names=names)


def test_source_rejects_missing_or_duplicate_joints(tmp_path):
    path = tmp_path / "source.csv"
    names, fields = write_motion(path)
    write_motion(path, column_names=fields[:-1])
    with pytest.raises(ValueError, match="exactly 29"):
        load_named_source_csv(path, expected_frames=3, joint_names=names)
    with pytest.raises(ValueError, match="29 unique"):
        load_named_source_csv(path, expected_frames=3, joint_names=names[:-1] + [names[0]])


def test_training_selection_never_uses_mirror_or_heldout(tmp_path):
    rows = [metadata_row(i, mirror=mirror) for i in range(20) for mirror in (False, True)]
    for row in rows:
        target = tmp_path / row["move_g1_path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("unvalidated source")
    indexed, _, _ = build_source_index(write_metadata(tmp_path, rows), tmp_path, split_seed="test")
    cohort = select_training_sources(
        indexed, families=["dance"], per_family=3, min_frames=2, max_frames=5, seed="test"
    )
    assert len(cohort["selected"]) == 3
    assert all(row["split"] == "train" and not row["is_mirror"] for row in cohort["selected"])
    assert not cohort["training_corpus_ready"]
    reversed_cohort = select_training_sources(
        list(reversed(indexed)), families=["dance"], per_family=3, min_frames=2, max_frames=5, seed="test"
    )
    assert cohort == reversed_cohort
    with pytest.raises(ValueError, match="not enough"):
        select_training_sources(
            indexed, families=["standing"], per_family=3, min_frames=2, max_frames=5, seed="test"
        )
