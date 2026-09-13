import json

import pytest

from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file
from gear_sonic.utils.g1_true23_pd_source_history import verify_historical_report


@pytest.fixture
def history(tmp_path):
    repo, archive = tmp_path / "repo", tmp_path / "archive"
    repo.mkdir()
    (archive / "source_snapshots").mkdir(parents=True)
    source, data, report = repo / "optimizer.py", tmp_path / "trajectory.npz", tmp_path / "report.json"
    source.write_text("historical_implementation = 1\n")
    data.write_bytes(b"immutable trajectory fixture")
    old_hash = sha256_file(source)
    snapshot = archive / "source_snapshots" / (old_hash + ".py")
    snapshot.write_bytes(source.read_bytes())
    report.write_text(json.dumps(dict(inputs={str(source): old_hash, str(data): sha256_file(data)})))
    manifest = archive / "source_archive.json"
    manifest.write_text(
        json.dumps(
            dict(
                kind="g1_true23_pd_shooting_source_archive_v1",
                reports=[dict(path=str(report), sha256=sha256_file(report))],
                sources=[dict(original_path=str(source), sha256=old_hash, archived_path=str(snapshot))],
            )
        )
    )
    source.write_text("current_implementation = 2\n")
    return repo, manifest, source, data, report, snapshot


def test_changed_python_source_requires_explicit_verified_history(history):
    repo, manifest, _, _, report, snapshot = history
    with pytest.raises(ValueError, match="unarchived implementation"):
        verify_historical_report(report, repository_root=repo)
    result = verify_historical_report(report, repository_root=repo, source_archive=manifest)
    assert len(result["archived_python_sources_used"]) == 1
    assert result["archived_python_sources_used"][0]["archived_path"] == str(snapshot)
    assert result["does_not_validate_current_code_or_physical_trajectory"]


def test_changed_trajectory_data_cannot_use_source_archive_as_a_bypass(history):
    repo, manifest, _, data, report, _ = history
    data.write_bytes(b"changed trajectory")
    with pytest.raises(ValueError, match="historical data"):
        verify_historical_report(report, repository_root=repo, source_archive=manifest)


def test_modified_snapshot_is_rejected(history):
    repo, manifest, _, _, report, snapshot = history
    snapshot.write_text("tampered = True\n")
    with pytest.raises(ValueError, match="contents changed"):
        verify_historical_report(report, repository_root=repo, source_archive=manifest)


def test_modified_report_cannot_claim_archived_source_provenance(history):
    repo, manifest, _, _, report, _ = history
    payload = json.loads(report.read_text())
    payload["changed_report"] = True
    report.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="exact report"):
        verify_historical_report(report, repository_root=repo, source_archive=manifest)
