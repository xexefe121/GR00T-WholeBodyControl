"""Evaluate saved evidence from the failed run; never call it 500 updates."""

import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time

from gear_sonic.scripts.record_g1_sonic_original29_baseline import dump
from gear_sonic.utils.g1_sonic_cpp_parameters import file_sha256

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ASSETS = ROOT.parent / "GR00T-WholeBodyControl"
BASE = HERE.parent / "interior_effort_20260906_v1"
inputs, stages = {}, []


def bind(path, expected=None):
    path = Path(path).resolve(strict=True)
    digest = file_sha256(path)
    if digest != inputs.get(str(path), digest) or (expected is not None and digest != expected):
        raise ValueError(f"interrupted experiment evidence changed: {path}")
    inputs[str(path)] = digest
    return path


def stage(name, command):
    log = HERE / f"{name}.log"
    print(json.dumps(dict(starting_stage=name)), flush=True)
    start = time.monotonic()
    with log.open("x") as stream:
        result = subprocess.run(list(map(str, command)), stdout=stream, stderr=subprocess.STDOUT, cwd=ROOT)
    row = dict(name=name, command=list(map(str, command)), return_code=result.returncode,
               elapsed_s=time.monotonic()-start, log=str(bind(log)))
    stages.append(row)
    dump(HERE / f"{name}.stage.json", row)
    print(json.dumps(dict(completed_stage=name, return_code=result.returncode)), flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)


bind(Path(__file__))
started = json.loads(bind(HERE / "started.json").read_text())
failed = json.loads(bind(HERE / "train500.stage.json").read_text())
assert failed["return_code"] == 1
log = bind(failed["log"], failed["log_sha256"]).read_text()
assert log.rstrip().endswith("ValueError: reset floor penetration exceeds diagnostic lift bound")
iterations = list(map(int, re.findall(r"Learning iteration (\d+)/500", log)))
assert iterations == list(range(410))
assert list(map(int, re.findall(r"Total steps:\s*(\d+)", log)))[-1] == 209920
assert not (HERE / "train500/checkpoints/frozen_lora_model_500.pt").exists()
assert not (HERE / "experiment_report.json").exists()
for path, expected in started["inputs"].items():
    bind(path, expected)
runtime = json.loads(bind(HERE / "train500/reset_curriculum_400.json").read_text())
assert runtime["new_update_count"] == 400 and runtime["actual_environment_control"] == 6400
assert runtime["action_std_unchanged"] is True
assert runtime["summary"]["maximum_scale"] == 1 and runtime["summary"]["full_disturbance_reset_rows"] > 0
failure_record = dict(
    requested_training_updates=500, completed_updates_logged=410,
    completed_update_rollout_transitions=209920,
    incomplete_final_rollout_transitions_not_claimed=True,
    last_saved_update=400, experiment_completed=False,
    failure="reset floor penetration exceeds diagnostic lift bound",
    maximum_allowed_reset_lift_m=0.2,
    failed_reset_state_not_captured=True,
    original_evaluation_plan=[100, 500],
    unavailable_planned_checkpoint=dict(update=500, not_executed=True, reason="training failed before checkpoint existed"),
    evaluation_updates=[100, 400],
    update400_role="additional last-saved-checkpoint diagnostic, not a replacement for planned update500",
    no_reward_based_checkpoint_selection=True, hardware_authorized=False, deployment_ready=False,
)
dump(HERE / "interrupted_run.json", dict(inputs=dict(inputs), **failure_record))
old = json.loads(bind(HERE.parent / "recovery_windows_20260906_v1/integrity_report.json",
                      "2a4c0cbe9b6baf739709a8519eaddb6b5c05b4b2b857d6d583a37f1b13f1bb3a").read_text())
for path in (ASSETS / "sonic_release/g1_23dof_rev_1_0_low_latency_init.pt", ASSETS / "low_latency/last.pt",
             BASE / "report.json", BASE / "stationary/report.json",
             HERE.parent / "readiness_audit_20260905_v1/motor_health.json"):
    bind(path, old["inputs"][str(path)])
for update in (100, 400):
    output = HERE / f"model_{update}"
    output.mkdir(exist_ok=False)
    policy = output / f"model_{update}.diagnostic.pt"
    script = bind(ROOT / "gear_sonic/scripts/materialize_g1_true23_frozen_lora_diagnostic.py")
    stage(f"interrupted.model_{update}.materialize", [sys.executable, script,
        "--checkpoint", bind(HERE / f"train500/checkpoints/frozen_lora_model_{update}.pt"),
        "--warm-start", ASSETS / "sonic_release/g1_23dof_rev_1_0_low_latency_init.pt",
        "--source-checkpoint", ASSETS / "low_latency/last.pt", "--output", policy])
    bind(policy)
    for component in ("encoder", "decoder"):
        model = output / f"model_{update}.diagnostic.{component}.onnx"
        report = model.with_suffix(".json")
        script = bind(ROOT / f"gear_sonic/scripts/export_g1_true23_frozen_lora_diagnostic_{component}.py")
        stage(f"interrupted.model_{update}.{component}", [sys.executable, script,
              "--diagnostic-policy", policy, "--output", model, "--report", report])
        bind(model)
        bind(report)
    script = bind(ROOT / "gear_sonic/scripts/evaluate_g1_true23_motion_ppo.py")
    stage(f"interrupted.model_{update}.evaluate", [sys.executable, script,
        "--encoder-report", output / f"model_{update}.diagnostic.encoder.json",
        "--decoder-report", output / f"model_{update}.diagnostic.decoder.json",
        "--full-request-report", BASE / "report.json", "--stationary-report", BASE / "stationary/report.json",
        "--asset-root", ASSETS, "--motor-health-snapshot", HERE.parent / "readiness_audit_20260905_v1/motor_health.json",
        "--output-dir", output / "evaluation"])
    report = json.loads(bind(output / "evaluation/report.json").read_text())
    assert len(report["records"]) == 11 and report["original_eight_request_set_preserved"] is True
    for path, expected in report["inputs"].items():
        bind(path, expected)
for path in list(inputs):
    bind(path)
dump(HERE / "interrupted_export_evaluation_report.json", dict(inputs=inputs, stages=stages, **failure_record))
stage("interrupted.regression", [sys.executable, bind(HERE / "run_regressions.py")])
