"""Serial paired ONNX exports and unchanged full-request evaluations at 200/300."""

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
        raise ValueError(f"resumed export input changed: {path}")
    inputs[str(path)] = digest
    return path


def stage(name, script, arguments):
    command = [sys.executable, str(bind(ROOT / "gear_sonic/scripts" / script)), *map(str, arguments)]
    log = HERE / f"{name}.log"
    print(json.dumps(dict(starting_stage=name)), flush=True)
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
        raise RuntimeError(f"resumed export/evaluation failed: {name}")


bind(Path(__file__))
experiment = json.loads(bind(HERE / "experiment_report.json").read_text())
assert all(row["return_code"] == 0 for row in experiment["stages"])
directory = Path(experiment["training_directory"]).resolve(strict=True)
assert directory == HERE / "breadth"
for update in (200, 300):
    output = HERE / f"model_{update}"
    output.mkdir(exist_ok=False)
    policy = output / f"model_{update}.diagnostic.pt"
    stage(
        f"model_{update}.materialize",
        "materialize_g1_true23_frozen_lora_diagnostic.py",
        [
            "--checkpoint",
            bind(directory / f"checkpoints/frozen_lora_model_{update}.pt"),
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
        model = output / f"model_{update}.diagnostic.{component}.onnx"
        report = model.with_suffix(".json")
        stage(
            f"model_{update}.{component}",
            f"export_g1_true23_frozen_lora_diagnostic_{component}.py",
            ["--diagnostic-policy", policy, "--output", model, "--report", report],
        )
        bind(model)
        bind(report)
    stage(
        f"model_{update}.evaluate",
        "evaluate_g1_true23_motion_ppo.py",
        [
            "--encoder-report",
            output / f"model_{update}.diagnostic.encoder.json",
            "--decoder-report",
            output / f"model_{update}.diagnostic.decoder.json",
            "--full-request-report",
            bind(BASE / "report.json"),
            "--stationary-report",
            bind(BASE / "stationary/report.json"),
            "--asset-root",
            ASSETS,
            "--motor-health-snapshot",
            bind(HERE.parent / "readiness_audit_20260905_v1/motor_health.json"),
            "--output-dir",
            output / "evaluation",
        ],
    )
    report = json.loads(bind(output / "evaluation/report.json").read_text())
    for path, expected in report["inputs"].items():
        bind(path, expected)
for path in list(inputs):
    bind(path)
dump(
    HERE / "export_evaluation_report.json",
    dict(
        inputs=inputs,
        stages=stages,
        all_checkpoints_evaluated_not_reward_selected=True,
        hardware_authorized=False,
        deployment_ready=False,
        full_motion_qualified=False,
    ),
)
