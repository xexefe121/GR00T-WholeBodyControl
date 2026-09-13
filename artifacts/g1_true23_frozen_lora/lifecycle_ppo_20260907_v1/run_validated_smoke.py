"""Run after two terminal pre-simulation failures; preserve their exact sources."""

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
preserved = []
for manifest_name, stage_name, snapshot_name, output_name, expected_error in (
    ("started.json", "smoke.stage.json", "source_at_failed_start", "smoke", "expected_contract"),
    ("retry_started.json", "smoke_retry.stage.json", "source_at_failed_retry", "smoke_retry", "directory: 'kind'"),
):
    manifest = json.loads((HERE / manifest_name).read_text())
    stage = json.loads((HERE / stage_name).read_text())
    assert stage["return_code"] == 1 and not (HERE / output_name).exists()
    log = HERE / f"{output_name}.log"
    assert expected_error in log.read_text()
    mapping = {}
    for original, expected in manifest["inputs"].items():
        original = Path(original)
        path = HERE / snapshot_name / original.name if original.is_relative_to(ROOT / "gear_sonic") else original
        assert file_sha256(path) == expected, path
        inputs[str(path)] = expected
        if path != original:
            mapping[str(original)] = str(path)
            inputs[str(original)] = file_sha256(original)
    for path in (HERE / manifest_name, HERE / stage_name, log):
        inputs[str(path)] = file_sha256(path)
    preserved.append(dict(manifest=manifest_name, failed_before_simulation=True, source_mapping=mapping))

command = list(manifest["command"])
command[0] = sys.executable
command[command.index("--output-dir") + 1] = str(HERE / "smoke_validated")
assert not (HERE / "smoke_validated").exists()
dump(HERE / "validated_started.json", dict(command=command, inputs=inputs, preserved_failures=preserved,
    requested_new_updates=2, hardware_authorized=False, deployment_ready=False))
start = time.monotonic()
print(json.dumps(dict(starting="full_lifecycle_smoke", new_updates=2)), flush=True)
with (HERE / "smoke_validated.log").open("x") as stream:
    result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
stage = dict(command=command, return_code=result.returncode, elapsed_s=time.monotonic()-start,
             log_sha256=file_sha256(HERE / "smoke_validated.log"))
dump(HERE / "smoke_validated.stage.json", stage)
print(json.dumps(dict(completed="full_lifecycle_smoke", return_code=result.returncode)), flush=True)
if result.returncode:
    raise SystemExit(result.returncode)
report_path = HERE / "smoke_validated/report.json"
report = json.loads(report_path.read_text())
assert report["actual_new_updates"] == 2 and report["actual_active_policy_actions"] > 0
assert len(report["learning"]) == 2 and all(len(row["episodes"]) == 8 for row in report["learning"])
assert report["frozen_platform_and_std_unchanged"] and not report["deployment_artifacts_emitted"]
assert report["initial_actor_sha256"] != report["final_actor_sha256"]
inputs[str(report_path)] = file_sha256(report_path)
for path, expected in inputs.items():
    assert file_sha256(path) == expected, path
dump(HERE / "validated_experiment_report.json", dict(stage=stage, inputs=inputs, preserved_failures=preserved,
    actual_new_updates=2, actual_active_actions=report["actual_active_policy_actions"],
    full_motion_qualified=False, hardware_authorized=False, deployment_ready=False))
