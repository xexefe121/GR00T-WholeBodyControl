"""Run regression, export/replay and independent integrity strictly serially."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
experiment = json.loads((HERE / "experiment_report.json").read_text())
assert all(row["return_code"] == 0 for row in experiment["stages"])
stages = []
for name in ("run_regressions", "export_evaluate", "check_integrity"):
    command = [sys.executable, str(HERE / (name + ".py"))]
    log = HERE / (name + ".log")
    print(json.dumps(dict(starting_stage=name)), flush=True)
    start = time.monotonic()
    with log.open("x") as stream:
        code = subprocess.call(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    record = dict(
        name=name,
        command=command,
        return_code=code,
        elapsed_s=time.monotonic() - start,
        log=str(log),
        log_sha256=hashlib.sha256(log.read_bytes()).hexdigest(),
    )
    stages.append(record)
    with (HERE / (name + ".stage.json")).open("x") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
    print(json.dumps(dict(completed_stage=name, return_code=code)), flush=True)
    if code:
        raise RuntimeError(f"post-training stage failed: {name}; inspect {log}")
with (HERE / "post_training_report.json").open("x") as stream:
    json.dump(dict(stages=stages, hardware_authorized=False, deployment_ready=False), stream, indent=2)
    stream.write("\n")
