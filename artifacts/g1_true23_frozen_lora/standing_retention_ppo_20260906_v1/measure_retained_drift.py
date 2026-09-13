"""Reconstruct anchors and measure trained drift without loading held-out rows."""

import json
from pathlib import Path

import torch

from gear_sonic.scripts.record_g1_sonic_original29_baseline import dump
from gear_sonic.trl.mjlab.frozen_platform_lora_actor import FrozenPlatformTrue23Core
from gear_sonic.trl.mjlab.frozen_platform_lora_runner import load_frozen_platform_lora_checkpoint
from gear_sonic.utils.g1_sonic_cpp_parameters import file_sha256
from gear_sonic.utils.g1_true23_standing_initialization import read_standing_initialization
from gear_sonic.utils.g1_true23_standing_retention import StandingOutputAnchor, load_training_states

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ASSETS = ROOT.parent / "GR00T-WholeBodyControl"
OLD = HERE.parent / "interior_effort_20260906_v1"
inputs = {}


def bind(path):
    path = Path(path).resolve(strict=True)
    digest = file_sha256(path)
    if digest != inputs.get(str(path), digest):
        raise ValueError(f"retained drift input changed: {path}")
    inputs[str(path)] = digest
    return path


bind(Path(__file__))
experiment = json.loads(bind(HERE / "experiment_report.json").read_text())
directory = Path(experiment["training_directory"]).resolve(strict=True)
assert directory.parent == HERE
runtime = json.loads(bind(directory / "standing_retention.json").read_text())
previous = json.loads(bind(HERE / "previous_drift.json").read_text())
payload, descriptor = read_standing_initialization(bind(OLD / "standing_lora500/report.json"))
data, training = load_training_states(bind(OLD / "representable_standing/report.json"), descriptor)
contract = descriptor["frozen_platform_contract"]
torch.set_num_threads(1)
core = FrozenPlatformTrue23Core(
    warm_start_path=bind(ASSETS / "sonic_release/g1_23dof_rev_1_0_low_latency_init.pt"),
    source_checkpoint_path=bind(ASSETS / "low_latency/last.pt"),
    lora_rank=contract["lora_rank"],
    lora_alpha=contract["lora_alpha"],
).eval()
cpu_anchor = StandingOutputAnchor(core, data, payload, batch_size=128, seed=20260906)
assert cpu_anchor.cache_sha256 == previous["anchor_cache_sha256"]
core.cuda()
gpu_anchor = StandingOutputAnchor(core, data, payload, batch_size=128, seed=20260906)
assert gpu_anchor.cache_sha256 == runtime["anchor_cache_sha256"]
backend = dict(
    token_differing_elements=int((cpu_anchor.inputs[:, :64] != gpu_anchor.inputs[:, :64].cpu()).sum()),
    proprio_max_abs_error=float((cpu_anchor.inputs[:, 64:] - gpu_anchor.inputs[:, 64:].cpu()).abs().max()),
    raw_max_abs_error=float((cpu_anchor.raw - gpu_anchor.raw.cpu()).abs().max()),
    safe_target_max_abs_error_rad=float((cpu_anchor.target - gpu_anchor.target.cpu()).abs().max()),
    actual_training_gpu_cache_reconstructed=True,
)
assert backend["token_differing_elements"] == 0 and backend["proprio_max_abs_error"] == 0
assert backend["raw_max_abs_error"] < 1e-5
core.cpu()
del gpu_anchor
rows = []
for step in (0, 100):
    path = bind(directory / f"checkpoints/frozen_lora_model_{step}.pt")
    checkpoint = load_frozen_platform_lora_checkpoint(path, expected_contract=core.adapter_contract())
    core.load_lora_state_dict(checkpoint["adapter_state_dict"], strict=True)
    core.assert_frozen_platform_unchanged()
    row = dict(update_count=step, adapter_sha256=checkpoint["adapter_state_sha256"], metrics=cpu_anchor.metrics())
    if step == 0:
        assert row["metrics"]["target_rmse_rad"] == row["metrics"]["raw_rmse"] == 0
    rows.append(row)
    print(json.dumps(row), flush=True)
for path in list(inputs):
    bind(path)
dump(
    HERE / "retained_drift.json",
    dict(
        kind="g1_true23_standing_retention_training_state_drift_v1",
        inputs=inputs,
        records=rows,
        training=training,
        backend=backend,
        cpu_anchor_cache_sha256=cpu_anchor.cache_sha256,
        optimization_performed=False,
        held_out_arrays_loaded=False,
        full_motion_qualified=False,
        hardware_authorized=False,
        deployment_ready=False,
    ),
)
print(json.dumps(dict(backend_comparison=backend)), flush=True)
