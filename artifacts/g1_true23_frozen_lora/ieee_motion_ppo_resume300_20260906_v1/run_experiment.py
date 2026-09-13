"""Resume exactly checkpoint100 in a fresh directory; append 200 PPO updates."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PARENT = HERE.parent / "ieee_motion_ppo_20260906_v2"
WITNESS = HERE.parent / "recorded_training_boundary_20260906_v2"
inputs, stages = {}, []


def bind(path, expected=None):
    path = Path(path).resolve(strict=True)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != inputs.get(str(path), digest) or (expected is not None and digest != expected):
        raise ValueError(f"bounded resume input changed: {path}")
    inputs[str(path)] = digest
    return path


def dump(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def run(name, command):
    log = HERE / (name + ".log")
    print(json.dumps(dict(starting_stage=name)), flush=True)
    start = time.monotonic()
    with log.open("x") as stream:
        code = subprocess.call(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    record = dict(
        name=name, command=command, return_code=code, elapsed_s=time.monotonic() - start, log=str(bind(log))
    )
    stages.append(record)
    dump(HERE / (name + ".stage.json"), record)
    print(json.dumps(dict(completed_stage=name, return_code=code)), flush=True)
    if code:
        raise RuntimeError(f"bounded resume stage failed: {name}")


bind(Path(__file__))
witness = json.loads(bind(WITNESS / "audit/report.json").read_text())
for path, digest in witness["inputs"].items():
    bind(path, digest)
assert len(witness["records"]) == 22
assert sum(row.get("recorded_inference_calls", 0) for row in witness["records"]) == 2675
for row in witness["records"]:
    if row.get("not_executed"):
        continue
    for result in row["devices"].values():
        for key in (
            "encoder267",
            "history930",
            "fsq64",
            "raw23",
            "safe_target",
            "requested_target",
            "projected_target",
            "applied_effort",
        ):
            assert result[key]["within_tolerance"], (row["label"], key)
        assert result["invalid_successful_substeps"] == []
        assert result["terminal"] is None or result["terminal"]["rejected_by_training"]
for path in ("run_audit.py", "audit.log", "audit.stage.json"):
    bind(WITNESS / path)
assert json.loads((WITNESS / "audit.stage.json").read_text())["exit_code"] == 0
run(
    "boundary_tests",
    [
        sys.executable,
        "-m",
        "pytest",
        str(bind(ROOT / "gear_sonic/tests/test_g1_true23_recorded_training_boundary.py")),
        "-q",
        "--junitxml=" + str(HERE / "boundary_tests.xml"),
    ],
)
bind(HERE / "boundary_tests.xml")
previous = json.loads(bind(PARENT / "breadth100_serial.stage.json").read_text())
assert previous["return_code"] == 0
command = list(previous["command"])
command[0] = sys.executable
bind(command[1])
assert command[command.index("--session-updates") + 1] == "100"
command[command.index("--session-updates") + 1] = "200"
command[command.index("--run-dir") + 1] = str(HERE / "breadth")
checkpoint = bind(
    PARENT / "breadth_serial/checkpoints/frozen_lora_model_100.pt",
    "f20f82385dd7c652a7a74b6103a4753f0317b6fb10055452862d647e7dc14de5",
)
command += ["--resume", str(checkpoint)]
for index, argument in enumerate(command):
    if argument in (
        "--standing-bootstrap-report",
        "--standing-teacher-report",
        "--source-checkpoint",
        "--warm-start",
        "--motion-file",
        "--motion-metadata",
        "--spans",
    ):
        bind(command[index + 1])
dump(
    HERE / "started.json",
    dict(
        inputs=inputs,
        command=command,
        parent_checkpoint=str(checkpoint),
        planned_total_updates=1000,
        start_update=100,
        requested_additional_updates=200,
        simulator_reinitialized_at_process_start=True,
        uninterrupted_environment_trajectory_equivalence_claimed=False,
        hardware_authorized=False,
        deployment_ready=False,
    ),
)
run("resume100_to300", command)
for path in list(inputs):
    bind(path)
dump(
    HERE / "experiment_report.json",
    dict(
        kind="g1_true23_ieee_motion_ppo_continuation_v1",
        inputs=inputs,
        stages=stages,
        training_directory=str(HERE / "breadth"),
        parent_checkpoint=str(checkpoint),
        start_update=100,
        requested_additional_updates=200,
        simulation_reinitialized_before_exact_training_state_resume=True,
        actor_critic_adam_counters_rng_and_retention_cache_restored_by_checked_runner=True,
        hardware_authorized=False,
        deployment_ready=False,
        full_motion_qualified=False,
    ),
)
