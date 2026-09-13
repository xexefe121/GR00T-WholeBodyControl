"""Serial post-training checks; never overlap large checkpoint loaders."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
records = []
for name in ("export_evaluate", "measure_retained_drift", "check_integrity"):
    script = HERE / f"{name}.py"
    digest = hashlib.sha256(script.read_bytes()).hexdigest()
    print(json.dumps(dict(starting_stage=name)), flush=True)
    with (HERE / f"post_{name}.log").open("x") as stream:
        code = subprocess.call([sys.executable, str(script)], stdout=stream, stderr=subprocess.STDOUT)
    records.append(dict(stage=name, script_sha256=digest, return_code=code))
    print(json.dumps(dict(completed_stage=name, return_code=code)), flush=True)
    if code or hashlib.sha256(script.read_bytes()).hexdigest() != digest:
        raise RuntimeError(f"post-training stage failed or changed: {name}; inspect its log")
with (HERE / "post_training_report.json").open("x") as stream:
    json.dump(dict(stages=records, hardware_authorized=False, deployment_ready=False), stream, indent=2)
