"""Verify old implementation evidence explicitly without treating it as current code."""

import json
from pathlib import Path
import re

from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file


def archive_local_path(name):
    value = str(name).replace("\\", "/")
    if re.match(r"^[A-Za-z]:/", value):
        value = "/mnt/" + value[0].lower() + value[2:]
    return Path(value).resolve()


def verify_historical_report(report_path, *, repository_root, source_archive=None):
    """Data must still match in place. Only archived Python code can differ.

    This authenticates historical source bytes, not the current implementation
    or a candidate trajectory. The caller must independently replay imported
    commands and bind its current implementation separately.
    """
    report_path = Path(report_path).resolve(strict=True)
    repository_root = Path(repository_root).resolve(strict=True)
    report = json.loads(report_path.read_text())
    manifest, archive = None, None
    if source_archive is not None:
        archive = Path(source_archive).resolve(strict=True)
        manifest = json.loads(archive.read_text())
        if manifest.get("kind") != "g1_true23_pd_shooting_source_archive_v1":
            raise ValueError("unsupported historical Python source archive")
    used = []
    for name, expected in report["inputs"].items():
        current = Path(name).resolve()
        if current.is_file() and sha256_file(current) == expected:
            continue
        if manifest is None or current.suffix != ".py" or not current.is_relative_to(repository_root):
            raise ValueError("historical data or unarchived implementation changed: " + name)
        historical_reports = {archive_local_path(row["path"]): row["sha256"] for row in manifest["reports"]}
        if historical_reports.get(report_path) != sha256_file(report_path):
            raise ValueError("historical source archive does not bind this exact report")
        rows = [row for row in manifest["sources"] if row["original_path"] == name and row["sha256"] == expected]
        if len(rows) != 1:
            raise ValueError("missing or ambiguous archived Python implementation: " + name)
        snapshot = archive_local_path(rows[0]["archived_path"])
        required = (archive.parent / "source_snapshots" / (expected + ".py")).resolve()
        if snapshot != required or sha256_file(snapshot) != expected:
            raise ValueError("historical Python snapshot location or contents changed")
        used.append(dict(original_path=name, historical_sha256=expected, archived_path=str(snapshot)))
    return dict(
        historical_report_path=str(report_path),
        historical_report_sha256=sha256_file(report_path),
        archived_python_sources_used=used,
        archive_path=None if archive is None else str(archive),
        archive_sha256=None if archive is None else sha256_file(archive),
        does_not_validate_current_code_or_physical_trajectory=True,
    )
