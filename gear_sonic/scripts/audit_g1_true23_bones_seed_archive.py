"""Read-only, single-pass compressed archive and selected source-byte audit.

No tar member is extracted onto disk. The complete compressed archive must
match the pinned release, and every selected CSV must match both its archive
member and the source-conversion receipt. This never accepts license terms.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile

from gear_sonic.utils.g1_true23_bones_seed import write_json
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file


class HashingReader:
    def __init__(self, stream):
        self.stream = stream
        self.digest = hashlib.sha256()
        self.bytes_read = 0

    def read(self, size=-1):
        value = self.stream.read(size)
        self.digest.update(value)
        self.bytes_read += len(value)
        return value


def audit_members(archive, selected, *, expected_archive_sha256, expected_archive_size):
    archive = Path(archive)
    if archive.is_symlink() or not archive.is_file():
        raise ValueError("archive must be a regular nonsymlink file")
    before = archive.stat()
    if before.st_size != expected_archive_size:
        raise ValueError("archive size differs from pinned release")
    expected = {}
    for row in selected:
        name = row["source_relative_path"]
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or path.parts[:2] != ("g1", "csv") or path.suffix != ".csv":
            raise ValueError("selected source is not a contained G1 CSV member")
        if name in expected:
            raise ValueError("duplicate selected source member")
        expected[name] = row
    if not 1 <= len(expected) <= 1000:
        raise ValueError("archive audit requires 1..1000 selected members")
    found, examined = {}, 0
    with archive.open("rb") as raw:
        reader = HashingReader(raw)
        with tarfile.open(fileobj=reader, mode="r|gz") as members:
            for member in members:
                examined += 1
                name = member.name.removeprefix("./")
                if name not in expected:
                    continue
                if name in found or not member.isfile() or not 0 < member.size <= 128 * 1024 * 1024:
                    raise ValueError("selected archive member is duplicate, nonregular or oversized")
                row = expected[name]
                if member.size != row["size_bytes"]:
                    raise ValueError("selected CSV size differs from archive member")
                digest = hashlib.sha256()
                with members.extractfile(member) as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
                if digest.hexdigest() != row["sha256"]:
                    raise ValueError("selected CSV hash differs from archive member")
                found[name] = {"sha256": digest.hexdigest(), "size_bytes": member.size}
        # Include compressed bytes buffered beyond tar EOF and any suffix.
        for _ in iter(lambda: reader.read(1024 * 1024), b""):
            pass
    after = archive.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("archive changed during audit")
    if reader.bytes_read != expected_archive_size or reader.digest.hexdigest() != expected_archive_sha256:
        raise ValueError("complete compressed archive hash differs from pinned release")
    if set(found) != set(expected):
        raise ValueError("archive lacks selected source members")
    return {
        "kind": "g1_true23_bones_seed_selected_archive_members_v1",
        "archive": {
            "path": str(archive.resolve()),
            "sha256": reader.digest.hexdigest(),
            "size_bytes": reader.bytes_read,
        },
        "archive_entries_examined": examined,
        "selected_member_count": len(found),
        "members": found,
        "archive_membership_verified": True,
        "archive_extracted_or_modified": False,
        "license_evidence_imported": False,
        "training_corpus_ready": False,
        "hardware_authorized": False,
        "deployment_ready": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--sources-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    from gear_sonic.utils.g1_23dof_artifact import MOTION_DATASET_SOURCE_ARCHIVE

    cohort_hash = sha256_file(args.cohort)
    cohort = json.loads(args.cohort.read_text())
    if cohort.get("kind") != "g1_true23_bones_seed_frozen_training_diagnostic_cohort_v1":
        raise ValueError("archive audit requires frozen source cohort")
    selected, receipt_bindings, csv_bindings = [], {}, {}
    for row in cohort["selected"]:
        identifier = row["motion_id"]
        if Path(identifier).name != identifier or identifier in {".", ".."}:
            raise ValueError("invalid source identifier")
        receipt_path = args.sources_directory / identifier / "report.json"
        receipt_bindings[str(receipt_path.resolve())] = sha256_file(receipt_path)
        receipt = json.loads(receipt_path.read_text())
        if receipt["indexed_source"] != row:
            raise ValueError("converted source differs from frozen cohort")
        source = receipt["source"]
        csv_bindings[source["path"]] = sha256_file(Path(source["path"]))
        if csv_bindings[source["path"]] != source["sha256"]:
            raise ValueError("local CSV changed after source conversion")
        selected.append(
            {
                "source_relative_path": row["source_relative_path"],
                "sha256": source["sha256"],
                "size_bytes": source["size_bytes"],
            }
        )
    report = audit_members(
        args.archive,
        selected,
        expected_archive_sha256=MOTION_DATASET_SOURCE_ARCHIVE["sha256"],
        expected_archive_size=MOTION_DATASET_SOURCE_ARCHIVE["size_bytes"],
    )
    for path, expected in {str(args.cohort): cohort_hash, **receipt_bindings, **csv_bindings}.items():
        if sha256_file(Path(path)) != expected:
            raise ValueError(f"source evidence changed during archive audit: {path}")
    report.update(
        cohort_sha256=cohort_hash,
        receipt_bindings=receipt_bindings,
        source_revision=MOTION_DATASET_SOURCE_ARCHIVE["revision"],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
