"""Export and evaluate actual v14 update 100 using the existing gated exporter."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ASSETS = ROOT.parent / "GR00T-WholeBodyControl"
BASE = HERE.parent / "interior_effort_20260906_v1"
inputs, stages = {}, []


def bind(path, expected=None):
    path = Path(path).resolve(strict=True)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != inputs.get(str(path), digest) or (expected is not None and digest != expected):
        raise ValueError(f"v14 export input changed: {path}")
    inputs[str(path)] = digest
    return path


def dump(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def stage(name, script, arguments):
    command = [sys.executable, str(bind(ROOT / "gear_sonic/scripts" / script)), *map(str, arguments)]
    log = HERE / f"{name}.log"
    print(json.dumps(dict(starting_stage=name)), flush=True)
    start = time.monotonic()
    with log.open("x") as stream:
        code = subprocess.call(command, stdout=stream, stderr=subprocess.STDOUT, cwd=ROOT)
    row = dict(name=name, command=command, return_code=code, elapsed_s=time.monotonic() - start,
               log=str(bind(log)))
    stages.append(row)
    dump(HERE / f"{name}.stage.json", row)
    print(json.dumps(dict(completed_stage=name, return_code=code)), flush=True)
    if code:
        raise RuntimeError(f"v14 stage failed: {name}; retained {log}")


bind(Path(__file__))
experiment = json.loads(bind(HERE / "experiment_report.json").read_text())
assert all(row["return_code"] == 0 for row in experiment["stages"])
lineage = json.loads(bind(HERE / "train100/lineage.json").read_text())
checkpoint = bind(HERE / "train100/checkpoints/causal_model_100.pt")
output = HERE / "model_100"
output.mkdir(exist_ok=False)
prefix = output / "causal_model_100"
stage("export", "export_g1_23dof_mjlab_diagnostic_onnx.py", [
    "--checkpoint", checkpoint, "--output-prefix", prefix, "--expected-lineage-sha256", lineage["lineage_sha256"],
    "--expected-reference-profile", "true23_causal_step1_history_0p02s_v1", "--json"])
metadata = bind(output / "causal_model_100.diagnostic.json")
stage("evaluate", "evaluate_g1_true23_v14_motion_ppo.py", [
    "--checkpoint", checkpoint, "--diagnostic-metadata", metadata,
    "--full-request-report", bind(BASE / "report.json", "a9b91e753111a3541605bac9177043fac67e732bba03bd48d7b14ca2c140d8e1"),
    "--stationary-report", bind(BASE / "stationary/report.json", "1c342ba441f302d25d2e8e068d64e1a5e985cfe1b90c94ff8a01663a5daf9c17"),
    "--asset-root", ASSETS, "--motor-health-snapshot", bind(HERE.parent / "readiness_audit_20260905_v1/motor_health.json"),
    "--output-dir", output / "evaluation"])
report = json.loads(bind(output / "evaluation/report.json").read_text())
for path, expected in report["inputs"].items():
    bind(path, expected)
for path in list(inputs):
    bind(path)
dump(HERE / "export_evaluation_report.json", dict(
    kind="g1_true23_original_v14_native_ieee_export_evaluation_v1", inputs=inputs, stages=stages,
    original_minimum_update_export_gate_unchanged=True, checkpoint_zero_not_exported_or_gate_bypassed=True,
    both_network_components_from_same_checkpoint=True, hardware_authorized=False,
    deployment_ready=False, full_motion_qualified=False))
