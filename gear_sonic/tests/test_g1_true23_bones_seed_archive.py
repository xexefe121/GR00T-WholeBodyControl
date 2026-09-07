from __future__ import annotations

import hashlib
import io
import tarfile

import pytest

from gear_sonic.scripts.audit_g1_true23_bones_seed_archive import audit_members


def fixture_archive(tmp_path, *, duplicate=False, suffix=b""):
    path = tmp_path / "g1.tar.gz"
    contents = b"synthetic CSV bytes, not licensed motion\n"
    name = "g1/csv/220101/test.csv"
    with tarfile.open(path, mode="w:gz") as archive:
        for _ in range(2 if duplicate else 1):
            info = tarfile.TarInfo(name)
            info.size = len(contents)
            archive.addfile(info, io.BytesIO(contents))
    if suffix:
        with path.open("ab") as stream:
            stream.write(suffix)
    data = path.read_bytes()
    return (
        path,
        [
            {
                "source_relative_path": name,
                "sha256": hashlib.sha256(contents).hexdigest(),
                "size_bytes": len(contents),
            }
        ],
        hashlib.sha256(data).hexdigest(),
        len(data),
    )


def test_archive_audit_hashes_complete_compressed_bytes_and_selected_member(tmp_path):
    path, selected, digest, size = fixture_archive(tmp_path, suffix=b"suffix included in digest")
    report = audit_members(path, selected, expected_archive_sha256=digest, expected_archive_size=size)
    assert report["archive"]["sha256"] == digest
    assert report["selected_member_count"] == 1
    assert report["archive_membership_verified"]
    assert not report["archive_extracted_or_modified"]
    assert not report["training_corpus_ready"]
    assert not (tmp_path / "g1").exists()


@pytest.mark.parametrize("fault", ["archive_hash", "archive_size", "member_hash", "member_size", "missing"])
def test_archive_audit_rejects_wrong_evidence(tmp_path, fault):
    path, selected, digest, size = fixture_archive(tmp_path)
    if fault == "archive_hash":
        digest = "0" * 64
    elif fault == "archive_size":
        size += 1
    elif fault == "member_hash":
        selected[0]["sha256"] = "0" * 64
    elif fault == "member_size":
        selected[0]["size_bytes"] += 1
    else:
        selected[0]["source_relative_path"] = "g1/csv/220101/missing.csv"
    with pytest.raises(ValueError):
        audit_members(path, selected, expected_archive_sha256=digest, expected_archive_size=size)


def test_archive_audit_rejects_duplicate_members(tmp_path):
    path, selected, digest, size = fixture_archive(tmp_path, duplicate=True)
    with pytest.raises(ValueError, match="duplicate"):
        audit_members(path, selected, expected_archive_sha256=digest, expected_archive_size=size)


def test_archive_audit_rejects_traversal(tmp_path):
    path, selected, digest, size = fixture_archive(tmp_path)
    selected[0]["source_relative_path"] = "g1/csv/../../escape.csv"
    with pytest.raises(ValueError, match="contained"):
        audit_members(path, selected, expected_archive_sha256=digest, expected_archive_size=size)
