"""Export the trained checkpoint; retain the failed zero-update attempt."""

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
        raise ValueError(f"trained pipeline input changed: {path}")
    inputs[str(path)] = digest
    return path


def stage(name, script, arguments):
    command = [sys.executable, str(bind(ROOT / "gear_sonic/scripts" / script)), *map(str, arguments)]
    print(json.dumps({"starting_stage": name}), flush=True)
    start = time.monotonic()
    log = HERE / f"{name}.log"
    with log.open("x") as stream:
        code = subprocess.call(command, stdout=stream, stderr=subprocess.STDOUT, cwd=ROOT)
    stages.append(
        dict(name=name, command=command, return_code=code, elapsed_s=time.monotonic() - start, log=str(bind(log)))
    )
    dump(HERE / f"{name}.stage.json", stages[-1])
    print(json.dumps({"completed_stage": name, "return_code": code}), flush=True)
    if code:
        raise RuntimeError(f"pipeline stage failed: {name}; inspect {log}")


bind(Path(__file__))
bind(HERE / "export_and_evaluate.py")
bind(HERE / "model_0.materialize.log")
bind(HERE / "model_0.materialize.stage.json")
warm = bind(ASSETS / "sonic_release/g1_23dof_rev_1_0_low_latency_init.pt")
source = bind(ASSETS / "low_latency/last.pt")
checkpoint = bind(HERE / "breadth/checkpoints/frozen_lora_model_100.pt")
output = HERE / "model_100"
output.mkdir(exist_ok=False)
policy = output / "model_100.diagnostic.pt"
stage(
    "model_100.materialize",
    "materialize_g1_true23_frozen_lora_diagnostic.py",
    ["--checkpoint", checkpoint, "--warm-start", warm, "--source-checkpoint", source, "--output", policy],
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
    HERE / "trained_pipeline_report.json",
    dict(
        inputs=inputs,
        stages=stages,
        hardware_authorized=False,
        deployment_ready=False,
        zero_update_diagnostic_export_rejected=True,
        update_counter_not_falsified=True,
    ),
)
