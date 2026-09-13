"""Measure rejected PPO100 drift on training states only; no optimization."""

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
PPO = HERE.parent / "standing_motion_ppo_20260906_v1"
inputs = {}


def bind(path):
    path = Path(path).resolve(strict=True)
    digest = file_sha256(path)
    if digest != inputs.get(str(path), digest):
        raise ValueError(f"drift input changed: {path}")
    inputs[str(path)] = digest
    return path


bind(Path(__file__))
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
anchor = StandingOutputAnchor(core, data, payload, batch_size=128, seed=20260906)
rows = []
for step in (0, 100):
    path = bind(PPO / f"breadth/checkpoints/frozen_lora_model_{step}.pt")
    checkpoint = load_frozen_platform_lora_checkpoint(path, expected_contract=core.adapter_contract())
    core.load_lora_state_dict(checkpoint["adapter_state_dict"], strict=True)
    core.assert_frozen_platform_unchanged()
    row = dict(update_count=step, adapter_sha256=checkpoint["adapter_state_sha256"], metrics=anchor.metrics())
    if step == 0:
        assert row["metrics"]["target_rmse_rad"] == row["metrics"]["raw_rmse"] == 0
    rows.append(row)
    print(json.dumps(row), flush=True)
for path in list(inputs):
    bind(path)
dump(
    HERE / "previous_drift.json",
    dict(
        kind="g1_true23_standing_training_states_ppo_drift_v1",
        inputs=inputs,
        records=rows,
        training=training,
        anchor_cache_sha256=anchor.cache_sha256,
        optimization_performed=False,
        held_out_arrays_loaded=False,
        full_motion_qualified=False,
        hardware_authorized=False,
        deployment_ready=False,
    ),
)
