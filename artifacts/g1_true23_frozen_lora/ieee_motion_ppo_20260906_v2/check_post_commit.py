"""Read-only post-publication rehash; the host supplies verified Git identity."""

import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


commit = sys.argv[1]
if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
    raise ValueError("expected verified host Git commit")
audit_path = HERE / "integrity_report.json"
audit = json.loads(audit_path.read_text())
expected_audit = "2b983c9b42dc20f2e441748dc98517e131f2cd15f450af0fc3fa2f302a03314a"
if digest(audit_path) != expected_audit:
    raise ValueError("independent integrity report changed")
mismatches = [path for path, expected in audit["inputs"].items() if digest(path) != expected]
if mismatches:
    raise ValueError(f"post-commit source or evidence changed: {mismatches}")
result = dict(
    kind="g1_true23_ieee_training_post_commit_rehash_v1",
    host_verified_local_and_remote_commit=commit,
    remote="https://github.com/xexefe121/GR00T-WholeBodyControl.git",
    branch="codex/sonic-transfer-g1-23dof",
    checked_files=len(audit["inputs"]),
    integrity_report_sha256=expected_audit,
    mismatches=mismatches,
    hardware_authorized=False,
    deployment_ready=False,
)
with (HERE / "post_commit_report.json").open("x") as stream:
    json.dump(result, stream, indent=2, sort_keys=True)
    stream.write("\n")
print(json.dumps(result), flush=True)
