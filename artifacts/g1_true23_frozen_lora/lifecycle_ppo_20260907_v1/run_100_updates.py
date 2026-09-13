"""One planned 100-update experiment, gated on the completed smoke and tests."""

import json
from pathlib import Path
import subprocess
import sys
import time

from gear_sonic.scripts.record_g1_sonic_original29_baseline import dump
from gear_sonic.utils.g1_sonic_cpp_parameters import file_sha256

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
inputs = {str(Path(__file__)): file_sha256(Path(__file__))}
for name in ("validated_experiment_report.json", "smoke_comparison.json", "regression_report.json"):
    path = HERE / name
    report = json.loads(path.read_text())
    inputs[str(path)] = file_sha256(path)
    for source, expected in report["inputs"].items():
        assert file_sha256(source) == expected, source
        inputs[source] = expected
    if name == "regression_report.json":
        assert report["exit_code"] == 0 and len(report["modules"]) == 52
        assert sum(int(row["tests"]) for row in report["junit_suites"]) == 704
        assert not any(int(row[key]) for row in report["junit_suites"] for key in ("failures", "errors", "skipped"))
    if name == "smoke_comparison.json":
        assert report["new_updates"] == 2 and report["actual_active_policy_actions"] == 656
        assert report["continuous_traces"] == 36 and report["physics_steps"] == 93247

manifest = json.loads((HERE / "validated_started.json").read_text())
inputs[str(HERE / "validated_started.json")] = file_sha256(HERE / "validated_started.json")
command = list(manifest["command"])
command[0] = sys.executable
command[command.index("--updates") + 1] = "100"
command[command.index("--output-dir") + 1] = str(HERE / "train100")
assert not (HERE / "train100").exists() and not (HERE / "train100_started.json").exists()
dump(HERE / "train100_started.json", dict(command=command, inputs=inputs,
    requested_new_updates=100, available_full_episode_attempts_per_update=8,
    initial_actor="same checked prior LoRA100, fresh critic and Adam; not a resume of the two-update smoke",
    complete_evaluation_before_and_after=True, no_motion_trimming_or_motor_limit_changes=True,
    hardware_authorized=False, deployment_ready=False))
print(json.dumps(dict(starting="planned_full_lifecycle_100", requested_updates=100)), flush=True)
start = time.monotonic()
with (HERE / "train100.log").open("x") as stream:
    result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
stage = dict(return_code=result.returncode, elapsed_s=time.monotonic()-start,
             log_sha256=file_sha256(HERE / "train100.log"))
dump(HERE / "train100.stage.json", stage)
print(json.dumps(dict(completed="planned_full_lifecycle_100", **stage)), flush=True)
if result.returncode:
    raise SystemExit(result.returncode)
report_path = HERE / "train100/report.json"
report = json.loads(report_path.read_text())
assert report["actual_new_updates"] == len(report["learning"]) == 100
assert all(len(row["episodes"]) == 8 for row in report["learning"])
assert report["frozen_platform_and_std_unchanged"] is True
assert report["deployment_artifacts_emitted"] is False
inputs[str(report_path)] = file_sha256(report_path)
for path, expected in inputs.items():
    assert file_sha256(path) == expected, path
dump(HERE / "train100_experiment_report.json", dict(inputs=inputs, stage=stage, actual_new_updates=100,
    actual_active_policy_actions=report["actual_active_policy_actions"],
    complete_full_request_evaluation_executed=True, not_a_hardware_readiness_certificate=True,
    hardware_authorized=False, deployment_ready=False))
