"""Sequential drift measurement, real smoke/resume and bounded full-corpus PPO."""

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
PPO = HERE.parent / "standing_motion_ppo_20260906_v1"
inputs, stages = {}, []


def bind(path):
    path = Path(path).resolve(strict=True)
    digest = file_sha256(path)
    if digest != inputs.get(str(path), digest):
        raise ValueError(f"retention experiment input changed: {path}")
    inputs[str(path)] = digest
    return path


def stage(name, script, arguments):
    command = [sys.executable, str(bind(script)), *map(str, arguments)]
    print(json.dumps(dict(starting_stage=name)), flush=True)
    start = time.monotonic()
    log = HERE / f"{name}.log"
    with log.open("x") as stream:
        code = subprocess.call(command, stdout=stream, stderr=subprocess.STDOUT, cwd=ROOT)
    record = dict(
        name=name, command=command, return_code=code, elapsed_s=time.monotonic() - start, log=str(bind(log))
    )
    stages.append(record)
    dump(HERE / f"{name}.stage.json", record)
    print(json.dumps(dict(completed_stage=name, return_code=code)), flush=True)
    if code:
        raise RuntimeError(f"retention stage failed: {name}; inspect {log}")


bind(Path(__file__))
for relative in (
    "gear_sonic/utils/g1_true23_standing_retention.py",
    "gear_sonic/trl/mjlab/standing_retention_ppo.py",
):
    bind(ROOT / relative)
stage("previous_drift", HERE / "measure_previous_drift.py", [])
bind(HERE / "previous_drift.json")
script = ROOT / "gear_sonic/scripts/train_g1_true23_standing_retention.py"
common = [
    "--standing-bootstrap-report",
    bind(OLD / "standing_lora500/report.json"),
    "--standing-teacher-report",
    bind(OLD / "representable_standing/report.json"),
    "--standing-retention-weight",
    "10",
    "--standing-retention-batch-size",
    "128",
    "--source-checkpoint",
    bind(ASSETS / "low_latency/last.pt"),
    "--warm-start",
    bind(ASSETS / "sonic_release/g1_23dof_rev_1_0_low_latency_init.pt"),
    "--actuation-profile",
    "native_support_stateful_v2",
    "--motion-file",
    bind(PPO / "corpus/corpus.npz"),
    "--motion-metadata",
    bind(PPO / "corpus/corpus.recovery.json"),
    "--spans",
    bind(PPO / "corpus/corpus.spans.json"),
    "--learning-rate",
    "0.000005",
    "--seed",
    "20260906",
]
smoke = [
    "smoke",
    *common,
    "--run-dir",
    HERE / "smoke",
    "--num-envs",
    "4",
    "--iterations",
    "4",
    "--session-updates",
    "2",
    "--save-interval",
    "1",
]
stage("smoke_initial", script, smoke)
stage("smoke_resume", script, [*smoke, "--resume", bind(HERE / "smoke/checkpoints/frozen_lora_model_2.pt")])
stage(
    "breadth100",
    script,
    [
        "train",
        *common,
        "--run-dir",
        HERE / "breadth",
        "--num-envs",
        "32",
        "--iterations",
        "1000",
        "--session-updates",
        "100",
        "--save-interval",
        "100",
    ],
)
for path in list(inputs):
    bind(path)
dump(
    HERE / "experiment_report.json",
    dict(
        kind="g1_true23_standing_retention_bounded_ppo_experiment_v1",
        inputs=inputs,
        stages=stages,
        retention_weight_fixed_before_evaluation=10,
        held_out_rows_used_for_training=0,
        hardware_authorized=False,
        deployment_ready=False,
        full_motion_qualified=False,
    ),
)
