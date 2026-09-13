"""Run one real two-update full-lifecycle smoke, never a robot operation."""

import json
from pathlib import Path
import subprocess
import sys
import time

from gear_sonic.scripts.record_g1_sonic_original29_baseline import dump
from gear_sonic.utils.g1_sonic_cpp_parameters import file_sha256

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ASSETS = ROOT.parent / "GR00T-WholeBodyControl"
BASE = HERE.parent / "interior_effort_20260906_v1"
checkpoint = HERE.parent / "ieee_motion_ppo_20260906_v2/breadth_serial/checkpoints/frozen_lora_model_100.pt"
checkpoint_hash = "f20f82385dd7c652a7a74b6103a4753f0317b6fb10055452862d647e7dc14de5"
assert file_sha256(checkpoint) == checkpoint_hash
sources = [Path(__file__), *(ROOT / "gear_sonic" / sub / name for sub, name in (
    ("scripts", "train_g1_true23_lifecycle_ppo.py"),
    ("utils", "g1_true23_lifecycle_ppo.py"),
    ("utils", "g1_true23_lifecycle_rollout.py"),
    ("tests", "test_g1_true23_lifecycle_ppo.py"),
))]
inputs = {str(path): file_sha256(path) for path in sources}
inputs[str(checkpoint)] = checkpoint_hash
command = [sys.executable, str(ROOT / "gear_sonic/scripts/train_g1_true23_lifecycle_ppo.py"),
    "--asset-root", str(ASSETS), "--full-request-report", str(BASE / "report.json"),
    "--stationary-report", str(BASE / "stationary/report.json"),
    "--motor-health-snapshot", str(HERE.parent / "readiness_audit_20260905_v1/motor_health.json"),
    "--warm-start", str(ASSETS / "sonic_release/g1_23dof_rev_1_0_low_latency_init.pt"),
    "--source-checkpoint", str(ASSETS / "low_latency/last.pt"),
    "--actor-checkpoint", str(checkpoint), "--expected-actor-sha256", checkpoint_hash,
    "--standing-bootstrap-report", str(BASE / "standing_lora500/report.json"),
    "--standing-teacher-report", str(BASE / "representable_standing/report.json"),
    "--output-dir", str(HERE / "smoke"), "--updates", "2", "--seed", "20260906"]
dump(HERE / "started.json", dict(command=command, inputs=inputs, requested_updates=2,
                                full_evaluation_before_and_after=True, hardware_authorized=False, deployment_ready=False))
print(json.dumps(dict(starting="full_lifecycle_smoke", requested_updates=2)), flush=True)
start = time.monotonic()
with (HERE / "smoke.log").open("x") as stream:
    result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
row = dict(command=command, return_code=result.returncode, elapsed_s=time.monotonic()-start,
           log=str(HERE / "smoke.log"), log_sha256=file_sha256(HERE / "smoke.log"))
dump(HERE / "smoke.stage.json", row)
print(json.dumps(dict(completed="full_lifecycle_smoke", return_code=result.returncode)), flush=True)
if result.returncode:
    raise SystemExit(result.returncode)
report_path = HERE / "smoke/report.json"
report = json.loads(report_path.read_text())
assert report["actual_new_updates"] == 2 and report["actual_active_policy_actions"] > 0
assert report["frozen_platform_and_std_unchanged"] is True
assert report["deployment_artifacts_emitted"] is False
assert len(report["learning"]) == 2 and all(len(row["episodes"]) == 8 for row in report["learning"])
assert report["initial_actor_sha256"] != report["final_actor_sha256"]
inputs[str(report_path)] = file_sha256(report_path)
for path, expected in inputs.items():
    assert file_sha256(path) == expected
dump(HERE / "experiment_report.json", dict(inputs=inputs, stage=row, actual_new_updates=2,
    actual_active_policy_actions=report["actual_active_policy_actions"],
    full_motion_qualified=False, hardware_authorized=False, deployment_ready=False))
