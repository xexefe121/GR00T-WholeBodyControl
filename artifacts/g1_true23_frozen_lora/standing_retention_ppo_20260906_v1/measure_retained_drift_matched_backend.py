"""Reproduce the inherited TF32 training setting and retain IEEE comparison."""

import inspect
import json
from pathlib import Path

from mjlab.utils.torch import configure_torch_backends
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
        raise ValueError(f"matched backend input changed: {path}")
    inputs[str(path)] = digest
    return path


bind(Path(__file__))
bind(HERE / "measure_retained_drift.py")
bind(HERE / "post_measure_retained_drift.log")
bind(ROOT / "gear_sonic/scripts/train_g1_23dof_mjlab_causal_history.py")
bind(inspect.getfile(configure_torch_backends))
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
comparisons = []
for enabled in (False, True):
    # Exact function and defaults used by the checked training launcher.
    configure_torch_backends(allow_tf32=enabled, deterministic=False)
    gpu_anchor = StandingOutputAnchor(core, data, payload, batch_size=128, seed=20260906)
    actual = dict(
        precision=torch.backends.cuda.matmul.fp32_precision,
        token_differing_elements=int((cpu_anchor.inputs[:, :64] != gpu_anchor.inputs[:, :64].cpu()).sum()),
        token_differing_rows=int((cpu_anchor.inputs[:, :64] != gpu_anchor.inputs[:, :64].cpu()).any(dim=1).sum()),
        proprio_max_abs_error=float((cpu_anchor.inputs[:, 64:] - gpu_anchor.inputs[:, 64:].cpu()).abs().max()),
        raw_max_abs_error=float((cpu_anchor.raw - gpu_anchor.raw.cpu()).abs().max()),
        safe_target_max_abs_error_rad=float((cpu_anchor.target - gpu_anchor.target.cpu()).abs().max()),
        gpu_cache_sha256=gpu_anchor.cache_sha256,
        matches_recorded_training_cache=gpu_anchor.cache_sha256 == runtime["anchor_cache_sha256"],
    )
    print(json.dumps(actual), flush=True)
    if enabled:
        assert actual["matches_recorded_training_cache"] is True
    comparisons.append(actual)
    del gpu_anchor
core.cpu()
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
        backend=dict(actual_training_gpu_cache_reconstructed=True),
        backend_comparisons=comparisons,
        actual_training_precision="tf32",
        cpu_anchor_cache_sha256=cpu_anchor.cache_sha256,
        previous_default_fp32_reconstruction_failed=True,
        optimization_performed=False,
        held_out_arrays_loaded=False,
        full_motion_qualified=False,
        hardware_authorized=False,
        deployment_ready=False,
    ),
)
