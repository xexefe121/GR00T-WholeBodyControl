"""Follow the completed noise factorial with two explicit reset controls."""

import json
from pathlib import Path
import subprocess
import time

from gear_sonic.scripts.record_g1_sonic_original29_baseline import dump
from gear_sonic.utils.g1_sonic_cpp_parameters import file_sha256

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
parent = HERE / "experiment_report.json"
experiment = json.loads(parent.read_text())
assert len(experiment["stages"]) == 4 and all(row["return_code"] == 0 for row in experiment["stages"])
inputs = {str(parent): file_sha256(parent), str(Path(__file__)): file_sha256(Path(__file__))}
for path, digest in experiment["inputs"].items():
    assert file_sha256(path) == digest
    inputs[path] = digest
base = experiment["stages"][-1]["command"]
assert base[base.index("--noise-scale")+1] == base[base.index("--sensor-noise-scale")+1] == "0"
stages = []
for label, lift in (("nominal_reset", False), ("nominal_reset_lift", True)):
    command = list(base)
    command[command.index("--output-dir")+1] = str(HERE / label)
    command += ["--reset-perturbation-scale", "0"]
    if lift:
        command.append("--lift-reset-floor-overlap")
    print(json.dumps(dict(starting_case=label)), flush=True)
    log = HERE / f"{label}.log"
    start = time.monotonic()
    with log.open("x") as stream:
        result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    row = dict(name=label, command=command, return_code=result.returncode, elapsed_s=time.monotonic()-start,
               log=str(log), log_sha256=file_sha256(log))
    dump(HERE / f"{label}.stage.json", row)
    stages.append(row)
    print(json.dumps(dict(completed_case=label, return_code=result.returncode)), flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)
    report = HERE / label / "ieee_sensor_report.json"
    inputs[str(report)] = file_sha256(report)
    inputs[str(log)] = row["log_sha256"]
for path, expected in inputs.items():
    assert file_sha256(path) == expected
dump(HERE / "reset_controls_report.json", dict(
    kind="g1_true23_ieee_current_reset_controls_v1", inputs=inputs, stages=stages,
    weights_updated=False, optimizer_steps=0, full_clip_fidelity_qualified=False,
    hardware_authorized=False, deployment_ready=False))
