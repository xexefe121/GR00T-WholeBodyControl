"""Run four isolated current-checkpoint noise probes, serially and without learning."""

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
PREVIOUS = HERE.parent / "recovery_windows_20260906_v1/integrity_report.json"
assert file_sha256(PREVIOUS) == "2a4c0cbe9b6baf739709a8519eaddb6b5c05b4b2b857d6d583a37f1b13f1bb3a"
old = json.loads(PREVIOUS.read_text())
assert old["checked_files"] == 1373 and old["mismatches"] == []
inputs = {str(PREVIOUS): file_sha256(PREVIOUS), str(Path(__file__)): file_sha256(Path(__file__))}
checkpoint = HERE.parent / "ieee_motion_ppo_20260906_v2/breadth_serial/checkpoints/frozen_lora_model_100.pt"
source = ASSETS / "low_latency/last.pt"
warm = ASSETS / "sonic_release/g1_23dof_rev_1_0_low_latency_init.pt"
corpus = HERE.parent / "standing_motion_ppo_20260906_v1/corpus"
for path in (checkpoint, source, warm, corpus / "corpus.npz", corpus / "corpus.recovery.json", corpus / "corpus.spans.json"):
    digest = file_sha256(path)
    assert digest == old["inputs"][str(path)]
    inputs[str(path)] = digest
for path in (ROOT / "gear_sonic/utils/g1_true23_sensor_noise_scale.py",
             ROOT / "gear_sonic/scripts/audit_g1_true23_ieee_sensor_exploration.py",
             ROOT / "gear_sonic/tests/test_g1_true23_sensor_noise_scale.py"):
    inputs[str(path)] = file_sha256(path)
plan = [("original_noise", 1, 1), ("zero_action", 0, 1), ("zero_sensor", 1, 0), ("zero_both", 0, 0)]
dump(HERE / "started.json", dict(plan=plan, inputs=inputs, hardware_authorized=False, deployment_ready=False))
stages = []
for label, action_scale, sensor_scale in plan:
    command = [sys.executable, "-m", "gear_sonic.scripts.audit_g1_true23_ieee_sensor_exploration",
               "--source-checkpoint", str(source), "--warm-start", str(warm), "--checkpoint", str(checkpoint),
               "--expected-checkpoint-sha256", inputs[str(checkpoint)], "--motion-file", str(corpus / "corpus.npz"),
               "--motion-metadata", str(corpus / "corpus.recovery.json"), "--spans", str(corpus / "corpus.spans.json"),
               "--noise-scale", str(action_scale), "--sensor-noise-scale", str(sensor_scale),
               "--steps", "128", "--num-envs", "32", "--seed", "20260906", "--capture-actuation-failures",
               "--audit-reset-contacts", "--output-dir", str(HERE / label)]
    print(json.dumps(dict(starting_case=label)), flush=True)
    log = HERE / f"{label}.log"
    started = time.monotonic()
    with log.open("x") as stream:
        result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    row = dict(name=label, command=command, return_code=result.returncode, elapsed_s=time.monotonic()-started,
               log=str(log), log_sha256=file_sha256(log))
    stages.append(row)
    dump(HERE / f"{label}.stage.json", row)
    print(json.dumps(dict(completed_case=label, return_code=result.returncode)), flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)
    report = HERE / label / "ieee_sensor_report.json"
    inputs[str(report)] = file_sha256(report)
    inputs[str(log)] = row["log_sha256"]
for path, digest in inputs.items():
    assert file_sha256(path) == digest
dump(HERE / "experiment_report.json", dict(
    kind="g1_true23_ieee_current_sensor_action_noise_experiment_v1", inputs=inputs, plan=plan, stages=stages,
    checkpoint_sha256=inputs[str(checkpoint)], weights_updated=False, optimizer_steps=0,
    full_clip_fidelity_qualified=False, hardware_authorized=False, deployment_ready=False))
