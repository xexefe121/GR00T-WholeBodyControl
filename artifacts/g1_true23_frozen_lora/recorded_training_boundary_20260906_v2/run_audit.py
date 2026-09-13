"""Serial non-actuating recorded-boundary witness with durable exit/log evidence."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PRIOR = HERE.parent / "ieee_motion_ppo_20260906_v2"
command = [
    sys.executable,
    str(ROOT / "gear_sonic/scripts/audit_g1_true23_recorded_training_boundary.py"),
    "--evaluation-report",
    str(PRIOR / "model_100/recorded_evaluation/report.json"),
    "--evaluation-report",
    str(PRIOR / "tf32_recorded_evaluation/report.json"),
    "--model-path",
    str(HERE.parent / "training_replay_parity_20260906_v3/replay.mjb"),
    "--output-dir",
    str(HERE / "audit"),
]
start = time.monotonic()
with (HERE / "audit.log").open("x") as stream:
    code = subprocess.call(command, stdout=stream, stderr=subprocess.STDOUT, cwd=ROOT)
result = dict(
    command=command,
    exit_code=code,
    elapsed_s=time.monotonic() - start,
    driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    log_sha256=hashlib.sha256((HERE / "audit.log").read_bytes()).hexdigest(),
    hardware_authorized=False,
    deployment_ready=False,
)
with (HERE / "audit.stage.json").open("x") as stream:
    json.dump(result, stream, indent=2)
    stream.write("\n")
print(json.dumps(result), flush=True)
raise SystemExit(code)

