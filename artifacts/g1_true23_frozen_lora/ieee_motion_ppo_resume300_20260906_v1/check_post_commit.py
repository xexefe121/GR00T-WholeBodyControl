"""Post-publication source/evidence check; Git identity supplied by host Git."""

import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
commit, expected_integrity = sys.argv[1:]
for value, length in ((commit, 40), (expected_integrity, 64)):
    if len(value) != length or any(char not in "0123456789abcdef" for char in value):
        raise ValueError("expected host-verified Git commit and integrity digest")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


path = HERE / "integrity_report.json"
assert digest(path) == expected_integrity
audit = json.loads(path.read_text())
inputs = dict(audit["inputs"])
for path, expected in inputs.items():
    if digest(path) != expected:
        raise ValueError(f"post-commit source/evidence mismatch: {path}")
post = json.loads((HERE / "post_training_report.json").read_text())
for row in post["stages"]:
    assert row["return_code"] == 0 and digest(row["log"]) == row["log_sha256"]
    inputs[row["log"]] = row["log_sha256"]
for name in ("post_training_report.json", "check_integrity.stage.json"):
    inputs[str(HERE / name)] = digest(HERE / name)
result = dict(
    kind="g1_true23_recorded_boundary_ppo300_post_commit_v1",
    host_verified_local_and_remote_commit=commit,
    branch="codex/sonic-transfer-g1-23dof",
    remote="https://github.com/xexefe121/GR00T-WholeBodyControl.git",
    checked_files=len(inputs),
    integrity_report_sha256=expected_integrity,
    inputs=inputs,
    mismatches=[],
    hardware_authorized=False,
    deployment_ready=False,
)
with (HERE / "post_commit_report.json").open("x") as stream:
    json.dump(result, stream, indent=2, sort_keys=True)
    stream.write("\n")
print(json.dumps({key: value for key, value in result.items() if key != "inputs"}), flush=True)
