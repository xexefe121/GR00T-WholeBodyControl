"""Retry only the terminal, zero-update host-memory failure in a new folder."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
inputs = {}


def bind(path):
    path = Path(path).resolve(strict=True)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != inputs.get(str(path), digest):
        raise ValueError(f"serial retry source changed: {path}")
    inputs[str(path)] = digest
    return path


def read(path):
    return json.loads(bind(path).read_text())


bind(Path(__file__))
bind(HERE / "run_experiment.py")
failed = read(HERE / "breadth100.stage.json")
failure = bind(failed["log"]).read_text()
if failed["return_code"] != 1 or "OSError: [Errno 12] Cannot allocate memory" not in failure:
    raise ValueError("retry requires the recorded terminal host-memory failure")
if (HERE / "breadth/checkpoints").exists():
    raise ValueError("failed attempt created checkpoints; do not repeat training blindly")
prime = read(HERE / "breadth/environment_prime_standing_lora.json")
if prime["physics_steps"] != 0 or prime["common_step_counter"] != 0:
    raise ValueError("failed attempt performed unexpected training/physics steps")
output = HERE / "breadth_serial"
if output.exists():
    raise FileExistsError("serial retry directory already exists")
command = list(failed["command"])
if command.count("--run-dir") != 1:
    raise ValueError("failed stage run-directory argument is ambiguous")
command[command.index("--run-dir") + 1] = str(output)
command[0] = sys.executable
for relative in (
    "gear_sonic/scripts/train_g1_true23_standing_retention.py",
    "gear_sonic/utils/g1_true23_standing_retention.py",
    "gear_sonic/trl/mjlab/standing_retention_ppo.py",
):
    bind(ROOT / relative)
stages = [read(HERE / f"{name}.stage.json") for name in ("previous_drift", "smoke_initial", "smoke_resume")]
assert all(row["return_code"] == 0 for row in stages)
for row in stages:
    bind(row["log"])
log = HERE / "breadth100_serial.log"
print(json.dumps(dict(starting_stage="breadth100_serial")), flush=True)
started = time.monotonic()
with log.open("x") as stream:
    code = subprocess.call(command, stdout=stream, stderr=subprocess.STDOUT, cwd=ROOT)
record = dict(
    name="breadth100_serial",
    command=command,
    return_code=code,
    elapsed_s=time.monotonic() - started,
    log=str(bind(log)),
)
with (HERE / "breadth100_serial.stage.json").open("x") as stream:
    json.dump(record, stream, indent=2)
print(json.dumps(dict(completed_stage="breadth100_serial", return_code=code)), flush=True)
if code:
    raise RuntimeError("serial breadth attempt failed; inspect its retained log")
stages.append(record)
for path in list(inputs):
    bind(path)
with (HERE / "experiment_report.json").open("x") as stream:
    json.dump(
        dict(
            kind="g1_true23_standing_retention_bounded_ppo_experiment_v1",
            inputs=inputs,
            stages=stages,
            failed_attempts=[failed],
            training_directory=str(output),
            failed_attempt_training_updates=0,
            retention_weight_fixed_before_evaluation=10,
            held_out_rows_used_for_training=0,
            hardware_authorized=False,
            deployment_ready=False,
            full_motion_qualified=False,
        ),
        stream,
        indent=2,
    )
