"""Matched CPU PyTorch comparison; checkpoint zero is not a trained export."""

import json
from pathlib import Path
import sys

import numpy as np
import onnxruntime as ort
import torch

from gear_sonic.scripts import evaluate_g1_true23_deployment_envelope as envelope
from gear_sonic.scripts.evaluate_g1_true23_motion_ppo import evaluation_plan
from gear_sonic.scripts.record_g1_sonic_original29_baseline import dump
from gear_sonic.trl.mjlab.frozen_platform_lora_actor import FrozenPlatformTrue23Core
from gear_sonic.trl.mjlab.frozen_platform_lora_runner import _state_sha256, load_frozen_platform_lora_checkpoint
from gear_sonic.utils.g1_sonic_cpp_parameters import file_sha256
from gear_sonic.utils.g1_true23_actuation_profile import NativeSupportActuationProfile
from gear_sonic.utils.g1_true23_clean_mujoco_teleop import UnitreeZeroVelocityFallbackPolicy
from gear_sonic.utils.g1_true23_interior_target_filter import run_interior_case

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ASSETS = ROOT.parent / "GR00T-WholeBodyControl"
OLD = HERE.parent / "interior_effort_20260906_v1"
inputs = {}


def bind(path, expected=None):
    path = Path(path).resolve(strict=True)
    digest = file_sha256(path)
    if digest != inputs.get(str(path), digest) or (expected is not None and digest != expected):
        raise ValueError(f"matched checkpoint comparison input changed: {path}")
    inputs[str(path)] = digest
    return path


suite = json.loads(bind(OLD / "report.json").read_text())
stationary = json.loads(bind(OLD / "stationary/report.json").read_text())
receipt = json.loads(bind(HERE / "breadth/standing_initialization.json").read_text())
lineage = json.loads(bind(HERE / "breadth/lineage.json").read_text())
for report in (suite, stationary):
    for path, digest in report["inputs"].items():
        bind(path, digest)
plan = evaluation_plan(suite, bind(OLD / "stationary/stationary_reference.npz"))
for case in plan:
    if not case["unavailable"]:
        bind(case["source"], case["source_sha256"])
torch.set_num_threads(1)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
contract = receipt["frozen_platform_contract"]
core = FrozenPlatformTrue23Core(
    warm_start_path=bind(ASSETS / "sonic_release/g1_23dof_rev_1_0_low_latency_init.pt"),
    source_checkpoint_path=bind(ASSETS / "low_latency/last.pt"),
    lora_rank=contract["lora_rank"],
    lora_alpha=contract["lora_alpha"],
).eval()
if core.adapter_contract() != contract:
    raise ValueError("training and matched evaluation frozen platforms differ")
encoder_hash = _state_sha256(
    {
        f"actor_module.encoders.teleop.module.{index * 2}.{name}": getattr(layer, name)
        for index, layer in enumerate(core.encoder.layers)
        for name in ("weight", "bias")
    }
)
if encoder_hash != suite["pair"]["paired_encoder_state_sha256"]:
    raise ValueError("matched evaluation frozen encoder differs from original baseline")


class Policy:
    def infer(self, semantic, history):
        with torch.no_grad():
            token = core.encode(torch.from_numpy(semantic[None]))
            encoded = core.codec.encode_proprioception(torch.from_numpy(history[None]))
            raw = core.codec.decode_action(core.decoder(torch.cat((token, encoded), dim=-1)))
        return raw.numpy()[0], token.numpy()[0]


options = ort.SessionOptions()
options.intra_op_num_threads = options.inter_op_num_threads = 1
balance = UnitreeZeroVelocityFallbackPolicy(
    bind(
        ASSETS
        / "artifacts/external/unitree_rl_mjlab/deploy/robots/g1/config/policy/velocity/v0/exported/policy.onnx"
    ),
    session_options=options,
)
measured = envelope.load_measured_initial_state(
    bind(HERE.parent / "readiness_audit_20260905_v1/motor_health.json")
)
profile = NativeSupportActuationProfile.from_sim_config(bind(ROOT / envelope.PHYSICS))
bind(Path(__file__))
for name, module in list(sys.modules.items()):
    path = getattr(module, "__file__", None)
    if name.startswith("gear_sonic.") and path and Path(path).suffix == ".py":
        bind(path)
flags = dict(hardware_authorized=False, deployment_ready=False, promotion_eligible=False)
records = []
for step in (0, 100):
    path = bind(HERE / f"breadth/checkpoints/frozen_lora_model_{step}.pt")
    checkpoint = load_frozen_platform_lora_checkpoint(path, expected_contract=contract, expected_lineage=lineage)
    if checkpoint["update_count"] != step:
        raise ValueError("checkpoint name and real completed update count differ")
    if step == 0 and checkpoint["adapter_state_sha256"] != receipt["adapter_state_sha256"]:
        raise ValueError("checkpoint zero differs from imported standing adapter")
    core.load_lora_state_dict(checkpoint["adapter_state_dict"], strict=True)
    core.assert_frozen_platform_unchanged()
    if core.merged_true23_policy_sha256(core.initial_std) != checkpoint["merged_true23_policy_sha256"]:
        raise ValueError("reconstructed checkpoint merged policy hash mismatch")
    output = HERE / f"torch_{step}"
    output.mkdir(exist_ok=False)
    rows = []
    for case in plan:
        row = {**case, "update_count": step, **flags}
        if case["unavailable"]:
            rows.append({**row, "not_executed": True})
            continue
        with np.load(case["source"], allow_pickle=False) as archive:
            motion = {key: archive[key].copy() for key in archive.files}
        lifecycle, historical = case["lifecycle"], case["historical_start"]
        result, arrays = run_interior_case(
            fraction=1.0,
            root=ROOT,
            asset_root=ASSETS,
            policy=Policy(),
            motion=motion,
            kp=np.asarray(profile.kp),
            kd=np.asarray(profile.kd),
            joint_scale=np.ones(23),
            ankle_effort=35.0,
            slew_rate=5.0,
            initial_state="measured" if historical else "reference",
            maximum_steps=None,
            measured_state=measured if historical else None,
            startup_hold_s=5.0 if lifecycle else 0.0,
            return_hold_s=5.0 if lifecycle else 0.0,
            transition_policy=balance if lifecycle else None,
            align_reference_start=lifecycle,
            project_transition_effort=lifecycle,
            project_active_effort=True,
            stateful_native_controller=True,
            trace_active_actuation=True,
        )
        if (
            result["requested_transitions"] != case["expected_transitions"]
            or result["compiled_native_model_sha256"]
            != stationary["records"][0]["result"]["compiled_native_model_sha256"]
        ):
            raise ValueError("matched comparison changed requested frames or compiled model")
        trace = output / f"{case['label']}.npz"
        with trace.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        bind(trace)
        row.update(trace_path=str(trace), result=result)
        rows.append(row)
        dump(output / f"{case['label']}.json", row)
        bind(output / f"{case['label']}.json")
        print(
            json.dumps(
                dict(
                    update=step,
                    case=case["label"],
                    completed=result["completed_transitions"],
                    requested=result["requested_transitions"],
                    fidelity=result["motion_fidelity"]["passed"],
                    failure=None if result["failure"] is None else result["failure"]["type"],
                    return_completed=result["return_hold"].get("completed_transitions"),
                )
            ),
            flush=True,
        )
    core.assert_frozen_platform_unchanged()
    dump(
        output / "report.json",
        dict(
            kind="g1_true23_motion_ppo_matched_torch_evaluation_v1",
            inputs=dict(inputs),
            records=rows,
            update_count=step,
            adapter_state_sha256=checkpoint["adapter_state_sha256"],
            merged_true23_policy_sha256=checkpoint["merged_true23_policy_sha256"],
            source_checkpoint_path=str(path),
            source_checkpoint_sha256=file_sha256(path),
            paired_encoder_state_sha256=encoder_hash,
            same_cpu_pytorch_inference_path=True,
            torch_adapter_not_deployment_onnx=True,
            original_eight_request_set_preserved=True,
            full_eight_clip_qualification=False,
            default_candidate_selected=False,
            **flags,
        ),
    )
    bind(output / "report.json")
    records.extend(rows)
for path in list(inputs):
    bind(path)
dump(
    HERE / "matched_torch_comparison.json",
    dict(
        kind="g1_true23_standing_initialized_motion_ppo_comparison_v1",
        inputs=inputs,
        records=records,
        full_eight_clip_qualification=False,
        robot_connected=False,
        **flags,
    ),
)
