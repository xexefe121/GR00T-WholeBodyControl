"""Serial fresh v14 smoke and budget-matched 100-update simulator run."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ASSETS = ROOT.parent / "GR00T-WholeBodyControl"
CORPUS = HERE.parent / "standing_motion_ppo_20260906_v1/corpus"
inputs, stages = {}, []


def bind(path, expected=None):
    path = Path(path).resolve(strict=True)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != inputs.get(str(path), digest) or (expected is not None and digest != expected):
        raise ValueError(f"v14 input changed: {path}")
    inputs[str(path)] = digest
    return path


def dump(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


bind(Path(__file__))
script = bind(ROOT / "gear_sonic/scripts/train_g1_true23_v14_native_ieee.py")
warm = bind(ASSETS / "sonic_release/g1_23dof_rev_1_0_low_latency_init.pt",
            "5e5be23982f15eaf2eb1d52b2433d081b5f43c260ecb5055457edf222b77c9bb")
motion = bind(CORPUS / "corpus.npz", "e285ca71295bc047ed076abfeda9348be9567475f4fca21c30e112ac62976c06")
metadata = bind(CORPUS / "corpus.recovery.json", "8b73012e8e9774d25e4e238dc7a8ff7480993683c4ee1e6c57d6c9c0061d36c1")
spans = bind(CORPUS / "corpus.spans.json", "78fb47539191a7cf02db09b7e0867ac8762969f89062ba1c6d57f1f59f7dcdeb")
bind("/root/g1_true23_runs/causal_history_stand_acquisition_v3_smoke100/checkpoints/causal_model_100.pt",
     "d13f47eff7348a7fce1277233a1d1795a2bafe12cd8000b2e101351c73c63bcc")
for name, mode, envs, updates, planned in (("smoke", "smoke", 4, 2, 2), ("train100", "train", 32, 100, 1000)):
    command = [sys.executable, str(script), mode, "--warm-start", str(warm), "--motion-file", str(motion),
               "--motion-metadata", str(metadata), "--spans", str(spans), "--learning-rate", "5e-6",
               "--seed", "20260906", "--run-dir", str(HERE / name), "--num-envs", str(envs),
               "--iterations", str(planned), "--session-updates", str(updates), "--save-interval", str(updates)]
    print(json.dumps(dict(starting_stage=name)), flush=True)
    log = HERE / f"{name}.log"
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
for path in list(inputs):
    bind(path)
dump(HERE / "experiment_report.json", dict(
    kind="g1_true23_original_v14_native_ieee_bounded_experiment_v1", inputs=inputs, stages=stages,
    fresh_training_runs=True, smoke_transitions=64, train_transitions=51200,
    actual_train_updates=100, planned_train_updates=1000, remaining_updates_not_run=900,
    prior_recovery_actor_training_budget_not_counted_as_new_updates=True,
    one_variable_ablation=False, resume_performed=False, hardware_authorized=False,
    deployment_ready=False, full_motion_qualified=False))
