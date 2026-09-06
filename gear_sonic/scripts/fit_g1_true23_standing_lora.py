"""Offline standing-only decoder-LoRA prerequisite, never a deployment export.

Uses three explicit bounded-teacher episodes and a fourth held-out episode.
Does not call these a dance behavior bank or reuse their metrics as full-motion
qualification. The released platform stays frozen; the starting adapter must
reconstruct its existing paired SONIC policy exactly before any fitting.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import onnxruntime as ort
import torch

from gear_sonic.scripts import evaluate_g1_true23_deployment_envelope as envelope
from gear_sonic.scripts.record_g1_sonic_original29_baseline import dump
from gear_sonic.trl.mjlab.frozen_platform_lora_actor import FrozenPlatformTrue23Core, _tensor_state_sha256
from gear_sonic.trl.mjlab.frozen_platform_lora_runner import load_frozen_platform_lora_checkpoint
from gear_sonic.utils.g1_23dof_contract import HARDWARE_23_ACTION_SCALE
from gear_sonic.utils.g1_23dof_safe_target_transform import (
    safe_target_transform_numpy,
    safe_target_transform_torch,
)
from gear_sonic.utils.g1_sonic_cpp_observation_trace import audit_engine_trace
from gear_sonic.utils.g1_sonic_cpp_parameters import file_sha256
from gear_sonic.utils.g1_true23_actuation_profile import NativeSupportActuationProfile
from gear_sonic.utils.g1_true23_clean_mujoco_teleop import UnitreeZeroVelocityFallbackPolicy
from gear_sonic.utils.g1_true23_interior_target_filter import run_interior_case

FLAGS = dict(hardware_authorized=False, deployment_ready=False, promotion_eligible=False)


def validate_standing_labels(arrays):
    """Reject incomplete, relabelled or nonfinite supervised episodes."""
    for key, width in (("encoder267", 267), ("history930", 930), ("raw_native23", 23), ("target_hardware23", 23)):
        if np.shape(arrays.get(key)) != (500, width) or not np.isfinite(arrays[key]).all():
            raise ValueError(f"standing labels require 500 finite rows of {key}")
    if np.max(np.abs(arrays["raw_native23"])) > 10:
        raise ValueError("standing raw labels exceed unchanged action bound")
    np.testing.assert_array_equal(
        safe_target_transform_numpy(arrays["raw_native23"].astype(np.float32))[1], arrays["target_hardware23"]
    )
    for key, width in (
        ("pre_qpos", 30),
        ("post_qpos", 30),
        ("pre_qvel", 29),
        ("post_qvel", 29),
        ("effort", 23),
        ("generalized_actuator_force", 23),
    ):
        value = arrays.get("physics_" + key)
        if np.shape(value) != (5000, width) or not np.isfinite(value).all():
            raise ValueError(f"standing labels require 5000 finite physics rows of {key}")
    if len(arrays.get("physics_pre_qpos", [])) != 5000 or not audit_engine_trace(arrays)["passed"]:
        raise ValueError("standing labels require full continuous engine evidence")
    np.testing.assert_array_equal(arrays["physics_pre_qpos"][1:], arrays["physics_post_qpos"][:-1])
    np.testing.assert_array_equal(arrays["physics_pre_qvel"][1:], arrays["physics_post_qvel"][:-1])
    np.testing.assert_array_equal(arrays["physics_effort"], arrays["physics_generalized_actuator_force"])


def standing_loss(raw, label_raw, label_target):
    """Fit representable requested targets; raw term keeps saturated gradients."""
    _, target = safe_target_transform_torch(raw)
    scale = raw.new_tensor(HARDWARE_23_ACTION_SCALE)
    return ((target - label_target) / scale).square().mean() + 0.01 * (raw - label_raw).square().mean()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-report", type=Path, required=True)
    parser.add_argument("--stationary-report", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--warm-start", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if not 1 <= args.steps <= 10000 or not 1 <= args.batch_size <= 500:
        parser.error("standing diagnostic requires 1..10000 steps and 1..500 batch size")
    if not np.isfinite(args.learning_rate) or not 0 < args.learning_rate <= 1e-3:
        parser.error("standing diagnostic learning rate must be within (0, 0.001]")
    root, assets = Path(__file__).resolve().parents[2], args.asset_root.resolve(strict=True)
    output = args.output_dir.resolve()
    inputs = {}

    def bind(path, expected=None):
        path = Path(path).resolve(strict=True)
        digest = file_sha256(path)
        if digest != inputs.get(str(path), digest) or (expected is not None and digest != expected):
            raise ValueError(f"standing fit input changed: {path}")
        inputs[str(path)] = digest
        return path

    teacher = json.loads(bind(args.teacher_report).read_text())
    stationary = json.loads(bind(args.stationary_report).read_text())
    if (
        teacher.get("kind") != "g1_true23_representable_standing_teacher_comparison_v1"
        or teacher.get("full_motion_teacher_accepted") is not False
        or teacher.get("held_out_episode_separate_from_training") is not True
        or stationary.get("kind") != "g1_true23_stationary_actor_comparison_diagnostic_v1"
    ):
        raise ValueError("requires explicit standing-only teacher and paired baseline reports")
    for report in (teacher, stationary):
        if report.get("hardware_authorized") is not False or report.get("deployment_ready") is not False:
            raise ValueError("standing reports must remain diagnostic-only")
        for path, digest in report["inputs"].items():
            bind(path, digest)
    wanted = {
        "standing_train": ["bounded_nominal", "bounded_plus", "bounded_minus"],
        "held_out_episode": ["bounded_holdout"],
    }
    groups = {}
    for role, names in wanted.items():
        rows = [row for row in teacher["records"] if row["role"] == role]
        if [row["name"] for row in rows] != names:
            raise ValueError("standing train/holdout episode selection changed")
        batches = []
        for row in rows:
            result = row["result"]
            if (
                result["standing_only_labels_usable"] is not True
                or result["original_request_projected_before_actuation"] is not True
            ):
                raise ValueError("standing teacher has not passed its actual bounded closed loop")
            with np.load(bind(row["arrays"]), allow_pickle=False) as archive:
                arrays = {key: archive[key].copy() for key in archive.files}
            validate_standing_labels(arrays)
            batches.append(arrays)
        groups[role] = {
            key: np.concatenate([batch[key] for batch in batches])
            for key in ("encoder267", "history930", "raw_native23", "target_hardware23")
        }

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    resume_path = bind(args.checkpoint)
    peek = torch.load(resume_path, map_location="cpu", weights_only=True)
    contract = peek["adapter_contract"]
    core = FrozenPlatformTrue23Core(
        warm_start_path=bind(args.warm_start),
        source_checkpoint_path=bind(args.source_checkpoint),
        lora_rank=contract["lora_rank"],
        lora_alpha=contract["lora_alpha"],
    )
    checkpoint = load_frozen_platform_lora_checkpoint(resume_path, expected_contract=core.adapter_contract())
    core.load_lora_state_dict(checkpoint["adapter_state_dict"])
    initial_policy_hash = core.merged_true23_policy_sha256(core.initial_std)
    if (
        initial_policy_hash != checkpoint["merged_true23_policy_sha256"]
        or initial_policy_hash != stationary["pair"]["source"]["policy_state_sha256"]
    ):
        raise ValueError("standing fit adapter does not reconstruct the compared SONIC pair")
    baseline = envelope._policy(
        Path(stationary["pair"]["encoder"]["path"]),
        Path(stationary["pair"]["decoder"]["path"]),
        stationary["pair"]["decoder"]["sha256"],
        encoder_hash=stationary["pair"]["encoder"]["sha256"],
    )
    parity = []
    with torch.no_grad():
        for index in (0, 1, 9, 10, 499):
            semantic, history = [groups["held_out_episode"][key][index] for key in ("encoder267", "history930")]
            actual = core(torch.from_numpy(semantic[None]), torch.from_numpy(history[None])).numpy()[0]
            expected, _ = baseline.infer(semantic, history)
            np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)
            parity.append(float(np.max(np.abs(actual - expected))))
    del checkpoint, peek, baseline
    core.assert_frozen_platform_unchanged()
    trainable = [(name, parameter) for name, parameter in core.named_parameters() if parameter.requires_grad]
    if not trainable or any(not name.endswith(("lora_a", "lora_b")) for name, _ in trainable):
        raise ValueError("standing fitting may train decoder LoRA only")
    core.to(args.device)
    data = {}
    with torch.no_grad():
        for role, arrays in groups.items():
            semantic = torch.from_numpy(arrays["encoder267"]).to(args.device)
            history = torch.from_numpy(arrays["history930"]).to(args.device)
            core.codec.validate_padded_proprioception(history)
            tokens = torch.cat([core.encode(chunk) for chunk in semantic.split(128)])
            data[role] = {
                "input": torch.cat((tokens, core.codec.encode_proprioception(history)), dim=-1),
                "raw": torch.from_numpy(arrays["raw_native23"]).to(args.device),
                "target": torch.from_numpy(arrays["target_hardware23"]).to(args.device),
            }

    def metrics():
        result = {}
        with torch.no_grad():
            for role, batch in data.items():
                raw = torch.cat(
                    [core.codec.decode_action(core.decoder(chunk)) for chunk in batch["input"].split(128)]
                )
                target = safe_target_transform_torch(raw)[1]
                result[role] = dict(
                    target_rmse_rad=float(torch.mean((target - batch["target"]).square()).sqrt().item()),
                    raw_rmse=float(torch.mean((raw - batch["raw"]).square()).sqrt().item()),
                    maximum_raw_action=float(raw.abs().max().item()),
                )
        return result

    bind(Path(__file__))
    for name, module in list(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if name.startswith("gear_sonic.") and path and Path(path).suffix == ".py":
            bind(path)
    output.mkdir(parents=True, exist_ok=False)
    started = {
        "kind": "g1_true23_standing_only_lora_fit_diagnostic_v1",
        "inputs": dict(inputs),
        "adapter_contract": core.adapter_contract(),
        "initial_merged_policy_sha256": initial_policy_hash,
        "initial_paired_ort_max_error": max(parity),
        "train_rows": 1500,
        "held_out_rows": 500,
        "split_unit": "whole_synthetic_episode",
        "steps": args.steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "device": args.device,
        "selection": "fixed_budget_not_holdout_selection",
        "before": metrics(),
        "encoder_tokens_cached_without_gradient": True,
        "full_motion_teacher_accepted": False,
        "full_motion_suite_not_replaced": True,
        "ppo_resume_checkpoint": False,
        **FLAGS,
    }
    dump(output / "started.json", started)
    optimizer = torch.optim.Adam([parameter for _, parameter in trainable], lr=args.learning_rate)
    batch = data["standing_train"]
    losses = []
    for step in range(args.steps):
        indices = torch.randint(1500, (args.batch_size,), device=args.device)
        raw = core.codec.decode_action(core.decoder(batch["input"][indices]))
        loss = standing_loss(raw, batch["raw"][indices], batch["target"][indices])
        if not torch.isfinite(loss):
            raise ValueError("standing loss became nonfinite")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([parameter for _, parameter in trainable], 1.0, error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(loss.detach().item()))
        if (step + 1) % 100 == 0 or step + 1 == args.steps:
            print(json.dumps({"standing_fit_step": step + 1, "loss": losses[-1]}), flush=True)
    after = metrics()
    core.cpu().eval()
    core.assert_frozen_platform_unchanged()
    state = core.lora_state_dict()
    candidate = {
        "kind": "g1_true23_standing_only_lora_adapter_diagnostic_v1",
        "adapter_contract": core.adapter_contract(),
        "adapter_state_dict": state,
        "adapter_state_sha256": _tensor_state_sha256(state),
        "source_checkpoint_sha256": inputs[str(resume_path)],
        "optimizer_steps": args.steps,
        "ppo_resume_checkpoint": False,
        "full_motion_teacher_accepted": False,
        **FLAGS,
    }
    with (output / "standing_lora.pt").open("xb") as stream:
        torch.save(candidate, stream)
    bind(output / "standing_lora.pt")
    with (output / "training_losses.npz").open("xb") as stream:
        np.savez_compressed(stream, loss=np.array(losses))
    bind(output / "training_losses.npz")

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
        assets
        / "artifacts/external/unitree_rl_mjlab/deploy/robots/g1/config/policy/velocity/v0/exported/policy.onnx",
        session_options=options,
    )
    profile = NativeSupportActuationProfile.from_sim_config(root / envelope.PHYSICS)
    with np.load(args.stationary_report.parent / "stationary_reference.npz", allow_pickle=False) as archive:
        motion = {key: archive[key].copy() for key in archive.files}
    records = []
    for acquisition in (False, True):
        name = "standing_after_acquisition" if acquisition else "standing_synthetic_start"
        result, arrays = run_interior_case(
            fraction=1.0,
            root=root,
            asset_root=assets,
            policy=Policy(),
            motion=motion,
            kp=np.asarray(profile.kp),
            kd=np.asarray(profile.kd),
            joint_scale=np.ones(23),
            ankle_effort=35.0,
            slew_rate=5.0,
            initial_state="reference",
            maximum_steps=None,
            startup_hold_s=5.0 if acquisition else 0.0,
            return_hold_s=5.0 if acquisition else 0.0,
            transition_policy=balance if acquisition else None,
            align_reference_start=acquisition,
            project_transition_effort=acquisition,
            project_active_effort=True,
            stateful_native_controller=True,
            trace_active_actuation=True,
        )
        with (output / f"{name}.npz").open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        bind(output / f"{name}.npz")
        records.append({"name": name, "result": result, **FLAGS})
        dump(output / f"{name}.json", records[-1])
        bind(output / f"{name}.json")
        print(
            json.dumps(
                {
                    "case": name,
                    "completed": result["completed_transitions"],
                    "requested": result["requested_transitions"],
                    "failure": result["failure"],
                    "return_completed": result["return_hold"].get("completed_transitions"),
                }
            ),
            flush=True,
        )
    for path in list(inputs):
        bind(path)
    dump(
        output / "report.json",
        {
            **started,
            "inputs": inputs,
            "after": after,
            "records": records,
            "adapter_state_sha256": candidate["adapter_state_sha256"],
            "frozen_platform_unchanged": True,
            "full_motion_qualified": False,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
