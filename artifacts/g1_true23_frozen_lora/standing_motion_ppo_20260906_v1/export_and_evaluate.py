"""Serial export/evaluation; release each large source loader between stages."""

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
        raise ValueError(f"PPO pipeline input changed: {path}")
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
        {
            "name": name,
            "command": command,
            "return_code": code,
            "elapsed_s": time.monotonic() - start,
            "log": str(bind(log)),
        }
    )
    dump(HERE / f"{name}.stage.json", stages[-1])
    print(json.dumps({"completed_stage": name, "return_code": code}), flush=True)
    if code:
        raise RuntimeError(f"pipeline stage failed: {name}; inspect {log}")


bind(Path(__file__))
warm = bind(ASSETS / "sonic_release/g1_23dof_rev_1_0_low_latency_init.pt")
source = bind(ASSETS / "low_latency/last.pt")
for step in (0, 100):
    checkpoint = bind(HERE / f"breadth/checkpoints/frozen_lora_model_{step}.pt")
    output = HERE / f"model_{step}"
    output.mkdir(exist_ok=False)
    policy = output / f"model_{step}.diagnostic.pt"
    encoder = output / f"model_{step}.diagnostic.encoder.onnx"
    decoder = output / f"model_{step}.diagnostic.decoder.onnx"
    encoder_report, decoder_report = encoder.with_suffix(".json"), decoder.with_suffix(".json")
    stage(
        f"model_{step}.materialize",
        "materialize_g1_true23_frozen_lora_diagnostic.py",
        [
            "--checkpoint",
            checkpoint,
            "--warm-start",
            warm,
            "--source-checkpoint",
            source,
            "--output",
            policy,
        ],
    )
    bind(policy)
    stage(
        f"model_{step}.encoder",
        "export_g1_true23_frozen_lora_diagnostic_encoder.py",
        [
            "--diagnostic-policy",
            policy,
            "--output",
            encoder,
            "--report",
            encoder_report,
        ],
    )
    bind(encoder)
    bind(encoder_report)
    stage(
        f"model_{step}.decoder",
        "export_g1_true23_frozen_lora_diagnostic_decoder.py",
        [
            "--diagnostic-policy",
            policy,
            "--output",
            decoder,
            "--report",
            decoder_report,
        ],
    )
    bind(decoder)
    bind(decoder_report)
    stage(
        f"model_{step}.evaluate",
        "evaluate_g1_true23_motion_ppo.py",
        [
            "--encoder-report",
            encoder_report,
            "--decoder-report",
            decoder_report,
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
    HERE / "pipeline_report.json",
    {"inputs": inputs, "stages": stages, "hardware_authorized": False, "deployment_ready": False},
)
