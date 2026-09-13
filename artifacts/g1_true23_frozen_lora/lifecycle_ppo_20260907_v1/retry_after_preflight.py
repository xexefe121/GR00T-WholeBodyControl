"""Retry the corrected launcher; preserve the zero-step failed attempt."""

import json
from pathlib import Path
import subprocess
import sys
import time

from gear_sonic.scripts.record_g1_sonic_original29_baseline import dump
from gear_sonic.utils.g1_sonic_cpp_parameters import file_sha256

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
old = json.loads((HERE / "started.json").read_text())
failed = json.loads((HERE / "smoke.stage.json").read_text())
assert failed["return_code"] == 1
assert "missing 1 required keyword-only argument: 'expected_contract'" in (HERE / "smoke.log").read_text()
assert not (HERE / "smoke").exists()
inputs = {str(Path(__file__)): file_sha256(Path(__file__))}
snapshot_map = {}
for path, expected in old["inputs"].items():
    path = Path(path)
    observed = HERE / "source_at_failed_start" / path.name if path.is_relative_to(ROOT / "gear_sonic") else path
    assert file_sha256(observed) == expected
    inputs[str(observed)] = expected
    if observed != path:
        snapshot_map[str(path)] = str(observed)
        inputs[str(path)] = file_sha256(path)
for path in (HERE / "started.json", HERE / "smoke.stage.json", HERE / "smoke.log"):
    inputs[str(path)] = file_sha256(path)
command = list(old["command"])
command[0] = sys.executable
command[command.index("--output-dir") + 1] = str(HERE / "smoke_retry")
dump(HERE / "retry_started.json", dict(command=command, inputs=inputs, failed_source_snapshots=snapshot_map,
    prior_attempt_failed_before_simulation=True, requested_updates=2,
    hardware_authorized=False, deployment_ready=False))
print(json.dumps(dict(starting="corrected_full_lifecycle_smoke", requested_updates=2)), flush=True)
start = time.monotonic()
with (HERE / "smoke_retry.log").open("x") as stream:
    result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
stage = dict(command=command, return_code=result.returncode, elapsed_s=time.monotonic()-start,
             log=str(HERE / "smoke_retry.log"), log_sha256=file_sha256(HERE / "smoke_retry.log"))
dump(HERE / "smoke_retry.stage.json", stage)
print(json.dumps(dict(completed="corrected_full_lifecycle_smoke", return_code=result.returncode)), flush=True)
if result.returncode:
    raise SystemExit(result.returncode)
report_path = HERE / "smoke_retry/report.json"
report = json.loads(report_path.read_text())
assert report["actual_new_updates"] == 2 and report["actual_active_policy_actions"] > 0
assert report["frozen_platform_and_std_unchanged"] is True and report["deployment_artifacts_emitted"] is False
assert len(report["learning"]) == 2 and all(len(row["episodes"]) == 8 for row in report["learning"])
assert report["initial_actor_sha256"] != report["final_actor_sha256"]
inputs[str(report_path)] = file_sha256(report_path)
for path, expected in inputs.items():
    assert file_sha256(path) == expected
dump(HERE / "retry_experiment_report.json", dict(inputs=inputs, stage=stage, actual_new_updates=2,
    actual_active_policy_actions=report["actual_active_policy_actions"],
    failed_source_snapshots=snapshot_map, first_attempt_failed_before_simulation=True,
    full_motion_qualified=False, hardware_authorized=False, deployment_ready=False))
