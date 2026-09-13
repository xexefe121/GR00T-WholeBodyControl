"""Continue the already-running experiment, serially, without restarting it."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PROCESS = Path("/proc/505")
EXPECTED = b"artifacts/g1_true23_frozen_lora/reset_curriculum_20260906_v1/run_training.py"
inputs, stages = {}, []


def bind(path):
    path = Path(path).resolve(strict=True)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != inputs.get(str(path), digest):
        raise ValueError(f"continuation input changed: {path}")
    inputs[str(path)] = digest
    return path


def process_identity():
    try:
        stat = (PROCESS / "stat").read_text().rsplit(")", 1)[1].split()
        return stat[19], (PROCESS / "cmdline").read_bytes()
    except FileNotFoundError:
        return None


def save(name, value):
    with (HERE / name).open("x") as stream:
        json.dump(value, stream, indent=2)


bind(Path(__file__))
exporter = bind(HERE / "export_evaluate.py")
regressions = bind(HERE / "run_regressions.py")
identity = process_identity()
if identity is None or EXPECTED not in identity[1].split(b"\0"):
    raise ValueError("expected live training driver is absent; do not infer success or restart")
save("continuation_started.json", dict(
    process_pid=505, process_start_ticks=identity[0],
    command=identity[1].decode().split("\0"), inputs=inputs,
    sequence=["export_and_full_request_evaluation", "focused_regression", "critical_lint", "format_check"],
    restart_training=False, hardware_authorized=False, deployment_ready=False,
))
print(json.dumps(dict(waiting_for_verified_training_pid=505)), flush=True)
while process_identity() == identity:
    time.sleep(1)
experiment = json.loads(bind(HERE / "experiment_report.json").read_text())
if experiment["new_training_updates"] != 500 or any(row["return_code"] != 0 for row in experiment["stages"]):
    raise ValueError("training has no successful complete 500-update result")
sources = [
    ROOT / "gear_sonic" / section / name
    for section, name in (
        ("scripts", "audit_g1_true23_ieee_sensor_exploration.py"),
        ("scripts", "train_g1_true23_reset_curriculum.py"),
        ("utils", "g1_true23_sensor_noise_scale.py"),
        ("utils", "g1_true23_reset_curriculum.py"),
        ("tests", "test_g1_true23_sensor_noise_scale.py"),
        ("tests", "test_g1_true23_reset_curriculum.py"),
    )
]
for path in sources:
    bind(path)
commands = [
    ("export_and_full_request_evaluation", [sys.executable, str(exporter)]),
    ("focused_regression", [sys.executable, str(regressions)]),
    ("critical_lint", [sys.executable, "-m", "ruff", "check", "--select", "E9,F63,F7,F82", *map(str, sources)]),
    ("format_check", [sys.executable, "-m", "ruff", "format", "--check", *map(str, sources)]),
]
for name, command in commands:
    bind(exporter)
    bind(regressions)
    for path in sources:
        bind(path)
    log = HERE / f"{name}.log"
    print(json.dumps(dict(starting_stage=name)), flush=True)
    start = time.monotonic()
    with log.open("x") as stream:
        result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    row = dict(name=name, command=command, return_code=result.returncode,
               elapsed_s=time.monotonic() - start, log=str(bind(log)))
    stages.append(row)
    save(f"{name}.stage.json", row)
    print(json.dumps(dict(completed_stage=name, return_code=result.returncode)), flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)
save("continuation_report.json", dict(
    inputs=inputs, stages=stages, restarted_training=False,
    hardware_authorized=False, deployment_ready=False,
))
