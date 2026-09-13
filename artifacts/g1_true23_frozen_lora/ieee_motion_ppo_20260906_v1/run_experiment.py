"""Serial IEEE repeat of the saved TF32 smoke/resume and 100-update commands."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PREVIOUS = HERE.parent / "standing_retention_ppo_20260906_v1"
inputs, stages = {}, []


def bind(path, expected=None):
    path = Path(path).resolve(strict=True)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != inputs.get(str(path), digest) or (expected is not None and digest != expected):
        raise ValueError(f"IEEE experiment input changed: {path}")
    inputs[str(path)] = digest
    return path


def dump(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


bind(Path(__file__))
previous_path = bind(
    PREVIOUS / "experiment_report.json", "ce690191534f1cb8fdf58ca60c311446becbb00f3aea8494fd27e7a9837b6dae"
)
previous = json.loads(previous_path.read_text())
script = bind(ROOT / "gear_sonic/scripts/train_g1_true23_ieee_standing_retention.py")
bind(ROOT / "gear_sonic/utils/g1_true23_training_precision.py")
for name in ("smoke_initial", "smoke_resume", "breadth100_serial"):
    source = next(row for row in previous["stages"] if row["name"] == name)
    assert source["return_code"] == 0
    command = [sys.executable, str(script)] + [
        value.replace(str(PREVIOUS) + "/", str(HERE) + "/") for value in source["command"][2:]
    ]
    for index, argument in enumerate(command):
        if argument in (
            "--standing-bootstrap-report",
            "--standing-teacher-report",
            "--source-checkpoint",
            "--warm-start",
            "--motion-file",
            "--motion-metadata",
            "--spans",
            "--resume",
        ):
            bind(command[index + 1])
    print(json.dumps(dict(starting_stage=name, precision="ieee")), flush=True)
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
        raise RuntimeError(f"IEEE stage failed: {name}; inspect {log}")
for path in list(inputs):
    bind(path)
dump(
    HERE / "experiment_report.json",
    dict(
        kind="g1_true23_ieee_bounded_motion_ppo_experiment_v1",
        inputs=inputs,
        stages=stages,
        training_directory=str(HERE / "breadth_serial"),
        previous_commands_reused_except_launcher_and_output_paths=True,
        precision="ieee",
        held_out_rows_used_for_training=0,
        hardware_authorized=False,
        deployment_ready=False,
        full_motion_qualified=False,
    ),
)
