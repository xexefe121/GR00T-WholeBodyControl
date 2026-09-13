"""Serial regressions, exports, full-request recording and backend comparisons."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
inputs, stages = {}, []


def bind(path):
    path = Path(path).resolve(strict=True)
    with path.open("rb") as stream:
        value = hashlib.file_digest(stream, "sha256").hexdigest()
    if value != inputs.get(str(path), value):
        raise ValueError(f"post-training input changed: {path}")
    inputs[str(path)] = value
    return path


bind(Path(__file__))
experiment = json.loads(bind(HERE / "experiment_report.json").read_text())
assert len(experiment["stages"]) == 3 and all(row["return_code"] == 0 for row in experiment["stages"])
for script in ("run_regressions.py", "export_evaluate.py", "compare_recorded_streams.py"):
    command = [sys.executable, str(bind(HERE / script))]
    print(json.dumps(dict(starting=script)), flush=True)
    start = time.monotonic()
    log = HERE / f"post_{Path(script).stem}.log"
    with log.open("x") as stream:
        code = subprocess.call(command, stdout=stream, stderr=subprocess.STDOUT, cwd=ROOT)
    row = dict(command=command, log=str(bind(log)), return_code=code, elapsed_s=time.monotonic() - start)
    stages.append(row)
    with (HERE / f"post_{Path(script).stem}.stage.json").open("x") as stream:
        json.dump(row, stream, indent=2, sort_keys=True)
    print(json.dumps(dict(completed=script, return_code=code)), flush=True)
    if code:
        raise RuntimeError(f"post-training stage failed: {script}; inspect {log}")
for path in list(inputs):
    bind(path)
with (HERE / "post_training_report.json").open("x") as stream:
    json.dump(
        dict(inputs=inputs, stages=stages, hardware_authorized=False, deployment_ready=False), stream, indent=2
    )
