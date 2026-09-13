"""Rehash the completed experiment after the host verified the pushed commit."""

import hashlib
import json
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
commit = sys.argv[1]
if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
    raise ValueError("expected exact host-verified pushed commit")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


source = HERE / "integrity_report.json"
report = json.loads(source.read_text())
checks = {}
for index, (path, expected) in enumerate(report["inputs"].items(), 1):
    actual = digest(path)
    if actual != expected:
        raise ValueError(f"post-commit evidence changed: {path}")
    checks[path] = actual
    if index % 100 == 0:
        print(json.dumps(dict(files_checked=index)), flush=True)
assert len(checks) == 457
checks[str(source)] = digest(source)
checks[str(Path(__file__).resolve())] = digest(Path(__file__))
with (HERE / "post_commit_integrity.json").open("x") as stream:
    json.dump(
        dict(
            kind="g1_true23_motion_ppo_post_commit_integrity_v1",
            host_verified_pushed_commit=commit,
            inputs=checks,
            mismatches=[],
            audited_experiment_files=457,
            robot_connected=False,
            hardware_authorized=False,
            deployment_ready=False,
            goal_complete=False,
        ),
        stream,
        indent=2,
    )
print(json.dumps(dict(pushed_commit=commit, files_unchanged=457, mismatches=0)), flush=True)
