"""Evaluate fixed curriculum checkpoints against the complete unchanged request set."""

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
inputs, stages = {}, []


def bind(path, expected=None):
    path = Path(path).resolve(strict=True)
    digest = file_sha256(path)
    if digest != inputs.get(str(path), digest) or (expected is not None and digest != expected):
        raise ValueError(f"curriculum evaluation input changed: {path}")
    inputs[str(path)] = digest
    return path


def stage(name, script, arguments):
    command = [sys.executable, str(bind(ROOT / "gear_sonic/scripts" / script)), *map(str, arguments)]
    log = HERE / f"{name}.log"
    print(json.dumps(dict(starting_stage=name)), flush=True)
    start = time.monotonic()
    with log.open("x") as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, cwd=ROOT)
    row = dict(name=name, command=command, return_code=result.returncode, elapsed_s=time.monotonic()-start,
               log=str(bind(log)))
    stages.append(row)
    dump(HERE / f"{name}.stage.json", row)
    print(json.dumps(dict(completed_stage=name, return_code=result.returncode)), flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)


bind(Path(__file__))
experiment = json.loads(bind(HERE / "experiment_report.json").read_text())
assert experiment["new_training_updates"] == 500 and experiment["new_training_transitions"] == 256000
assert all(row["return_code"] == 0 for row in experiment["stages"])
for path, expected in experiment["inputs"].items():
    bind(path, expected)
old = json.loads(bind(HERE.parent / "recovery_windows_20260906_v1/integrity_report.json",
                      "2a4c0cbe9b6baf739709a8519eaddb6b5c05b4b2b857d6d583a37f1b13f1bb3a").read_text())
for path in (ASSETS / "sonic_release/g1_23dof_rev_1_0_low_latency_init.pt", ASSETS / "low_latency/last.pt",
             BASE / "report.json", BASE / "stationary/report.json",
             HERE.parent / "readiness_audit_20260905_v1/motor_health.json"):
    bind(path, old["inputs"][str(path)])
for update in (100, 500):
    output = HERE / f"model_{update}"
    output.mkdir(exist_ok=False)
    policy = output / f"model_{update}.diagnostic.pt"
    stage(f"model_{update}.materialize", "materialize_g1_true23_frozen_lora_diagnostic.py", [
        "--checkpoint", bind(HERE / f"train500/checkpoints/frozen_lora_model_{update}.pt"),
        "--warm-start", ASSETS / "sonic_release/g1_23dof_rev_1_0_low_latency_init.pt",
        "--source-checkpoint", ASSETS / "low_latency/last.pt", "--output", policy])
    bind(policy)
    for component in ("encoder", "decoder"):
        model = output / f"model_{update}.diagnostic.{component}.onnx"
        report = model.with_suffix(".json")
        stage(f"model_{update}.{component}", f"export_g1_true23_frozen_lora_diagnostic_{component}.py",
              ["--diagnostic-policy", policy, "--output", model, "--report", report])
        bind(model)
        bind(report)
    stage(f"model_{update}.evaluate", "evaluate_g1_true23_motion_ppo.py", [
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
dump(HERE / "export_evaluation_report.json", dict(
    inputs=inputs, stages=stages, checkpoints_fixed_in_advance=[100, 500],
    full_requests_and_standing_return_unchanged=True, hardware_authorized=False, deployment_ready=False))
