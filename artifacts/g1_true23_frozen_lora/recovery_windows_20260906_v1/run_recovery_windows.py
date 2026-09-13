"""Serial, hash-bound early-return diagnostics for all three saved candidates."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ASSETS = ROOT.parent / "GR00T-WholeBodyControl"
inputs, stages, summaries = {}, [], []


def bind(path, expected=None):
    path = Path(path).resolve(strict=True)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != inputs.get(str(path), digest) or (expected is not None and digest != expected):
        raise ValueError(f"recovery experiment input changed: {path}")
    inputs[str(path)] = digest
    return path


def dump(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


bind(Path(__file__))
audit_path = bind(HERE.parent / "v14_native_ieee_20260906_v1/integrity_report.json",
                  "ef8ecd40ceaa24d2af0f1888783fc9b4507ec5ff9b0ee2b5155730f034c96b22")
audit = json.loads(audit_path.read_text())
assert audit["mismatches"] == [] and audit["checked_files"] == 1066
script = bind(ROOT / "gear_sonic/scripts/audit_g1_true23_recovery_window.py")
bind(ROOT / "gear_sonic/utils/g1_true23_recovery_window.py")
parents = (
    ("lora100", "ieee_motion_ppo_20260906_v2/model_100/evaluation/report.json"),
    ("lora300", "ieee_motion_ppo_resume300_20260906_v1/model_300/evaluation/report.json"),
    ("v14_100", "v14_native_ieee_20260906_v1/model_100/evaluation/report.json"),
)
for name, relative in parents:
    parent_path = (HERE.parent / relative).resolve(strict=True)
    parent_hash = audit["inputs"][str(parent_path)]
    bind(parent_path, parent_hash)
    command = [sys.executable, str(script), "--parent-report", str(parent_path),
               "--expected-parent-sha256", parent_hash, "--asset-root", str(ASSETS),
               "--output-dir", str(HERE / name)]
    print(json.dumps(dict(starting_stage=name)), flush=True)
    log = HERE / f"{name}.log"
    start = time.monotonic()
    with log.open("x") as stream:
        code = subprocess.call(command, stdout=stream, stderr=subprocess.STDOUT, cwd=ROOT)
    stage = dict(name=name, command=command, return_code=code, elapsed_s=time.monotonic() - start,
                 log=str(bind(log)))
    stages.append(stage)
    dump(HERE / f"{name}.stage.json", stage)
    print(json.dumps(dict(completed_stage=name, return_code=code)), flush=True)
    if code:
        raise RuntimeError(f"recovery-window stage failed: {name}; inspect retained {log}")
    report = json.loads(bind(HERE / name / "report.json").read_text())
    summaries.append(dict(candidate=name, summary=report["summary"]))
    print(json.dumps(summaries[-1]), flush=True)
for path in list(inputs):
    bind(path)
dump(HERE / "experiment_report.json", dict(
    kind="g1_true23_exhaustive_early_return_experiment_v1", inputs=inputs, stages=stages, summaries=summaries,
    three_original_full_request_replays=True, no_training_or_policy_change=True,
    full_dance_qualified=False, live_teleop_qualified=False, physical_fsm_handoff_proven=False,
    hardware_authorized=False, deployment_ready=False))
