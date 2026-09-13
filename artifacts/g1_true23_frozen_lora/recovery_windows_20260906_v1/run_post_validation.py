"""Run regressions then the independent recovery audit, without overlap."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


experiment = json.loads((HERE / "experiment_report.json").read_text())
assert all(row["return_code"] == 0 for row in experiment["stages"])
stages = []
for name, script in (("regressions", "run_regressions.py"), ("check_integrity", "check_integrity.py")):
    command = [sys.executable, str(HERE / script)]
    log = HERE / f"{name}.log"
    print(json.dumps(dict(starting_stage=name)), flush=True)
    start = time.monotonic()
    with log.open("x") as stream:
        code = subprocess.call(command, stdout=stream, stderr=subprocess.STDOUT, cwd=ROOT)
    row = dict(name=name, command=command, return_code=code, elapsed_s=time.monotonic() - start,
               log=str(log), log_sha256=digest(log))
    stages.append(row)
    with (HERE / f"{name}.stage.json").open("x") as stream:
        json.dump(row, stream, indent=2)
        stream.write("\n")
    print(json.dumps(dict(completed_stage=name, return_code=code)), flush=True)
    if code:
        raise RuntimeError(f"recovery validation failed: {name}; inspect retained {log}")
with (HERE / "post_validation_report.json").open("x") as stream:
    json.dump(dict(stages=stages, hardware_authorized=False, deployment_ready=False), stream, indent=2)
    stream.write("\n")
