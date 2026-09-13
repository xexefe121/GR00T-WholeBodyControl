"""Serial trained export and unchanged full-request simulator evaluation."""

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
OLD = HERE.parent / "interior_effort_20260906_v1"
inputs, stages = {}, []


def bind(path):
    path = Path(path).resolve(strict=True)
    digest = file_sha256(path)
    if digest != inputs.get(str(path), digest):
        raise ValueError(f"retention export input changed: {path}")
    inputs[str(path)] = digest
    return path


def stage(name, script, arguments):
    command = [sys.executable, str(bind(ROOT / "gear_sonic/scripts" / script)), *map(str, arguments)]
    print(json.dumps(dict(starting_stage=name)), flush=True)
    log = HERE / f"{name}.log"
    start = time.monotonic()
    with log.open("x") as stream:
        code = subprocess.call(command, stdout=stream, stderr=subprocess.STDOUT, cwd=ROOT)
    record = dict(
        name=name, command=command, return_code=code, elapsed_s=time.monotonic() - start, log=str(bind(log))
    )
    stages.append(record)
    dump(HERE / f"{name}.stage.json", record)
    print(json.dumps(dict(completed_stage=name, return_code=code)), flush=True)
    if code:
        raise RuntimeError(f"retention export stage failed: {name}; inspect {log}")


bind(Path(__file__))
experiment = json.loads(bind(HERE / "experiment_report.json").read_text())
if len(experiment["stages"]) != 4 or any(row["return_code"] for row in experiment["stages"]):
    raise ValueError("retention training/resume stages must finish before evaluation")
training_directory = Path(experiment["training_directory"]).resolve(strict=True)
if training_directory.parent != HERE:
    raise ValueError("retention training directory is outside this experiment")
output = HERE / "model_100"
output.mkdir(exist_ok=False)
policy = output / "model_100.diagnostic.pt"
stage(
    "model_100.materialize",
    "materialize_g1_true23_frozen_lora_diagnostic.py",
    [
        "--checkpoint",
        bind(training_directory / "checkpoints/frozen_lora_model_100.pt"),
        "--warm-start",
        bind(ASSETS / "sonic_release/g1_23dof_rev_1_0_low_latency_init.pt"),
        "--source-checkpoint",
        bind(ASSETS / "low_latency/last.pt"),
        "--output",
        policy,
    ],
)
bind(policy)
for component in ("encoder", "decoder"):
    model = output / f"model_100.diagnostic.{component}.onnx"
    report = model.with_suffix(".json")
    stage(
        f"model_100.{component}",
        f"export_g1_true23_frozen_lora_diagnostic_{component}.py",
        ["--diagnostic-policy", policy, "--output", model, "--report", report],
    )
    bind(model)
    bind(report)
stage(
    "model_100.evaluate",
    "evaluate_g1_true23_motion_ppo.py",
    [
        "--encoder-report",
        output / "model_100.diagnostic.encoder.json",
        "--decoder-report",
        output / "model_100.diagnostic.decoder.json",
        "--full-request-report",
        bind(OLD / "report.json"),
        "--stationary-report",
        bind(OLD / "stationary/report.json"),
        "--asset-root",
        ASSETS,
        "--motor-health-snapshot",
        bind(HERE.parent / "readiness_audit_20260905_v1/motor_health.json"),
        "--output-dir",
        output / "evaluation",
    ],
)
bind(output / "evaluation/report.json")
for path in list(inputs):
    bind(path)
dump(
    HERE / "export_evaluation_report.json",
    dict(
        inputs=inputs,
        stages=stages,
        hardware_authorized=False,
        deployment_ready=False,
        full_motion_qualified=False,
    ),
)
