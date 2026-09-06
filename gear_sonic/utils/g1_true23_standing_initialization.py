"""Import checked standing-only LoRA weights into a fresh simulator trainer.

The diagnostic adapter is not relabelled as PPO resume or hardware policy.
Its actual stationary lifecycle evidence and frozen-platform contract remain
bound to the new training lineage. No failed motion actions are teacher labels.
"""

from __future__ import annotations

from collections.abc import Mapping
import copy
import json
from pathlib import Path

import numpy as np
import torch

from gear_sonic.trl.mjlab.frozen_platform_lora_actor import _tensor_state_sha256
from gear_sonic.utils.g1_sonic_cpp_observation_trace import audit_engine_trace
from gear_sonic.utils.g1_sonic_cpp_parameters import file_sha256


def validate_standing_payload(value, fit, *, expected_contract=None):
    if not isinstance(value, Mapping) or value.get("kind") != "g1_true23_standing_only_lora_adapter_diagnostic_v1":
        raise ValueError("standing initialization requires its distinct diagnostic adapter header")
    for flag in (
        "hardware_authorized",
        "deployment_ready",
        "promotion_eligible",
        "ppo_resume_checkpoint",
        "full_motion_teacher_accepted",
    ):
        if value.get(flag) is not False:
            raise ValueError(f"standing initialization must not claim {flag}")
    contract = value.get("adapter_contract")
    if not isinstance(contract, Mapping) or contract != fit.get("adapter_contract"):
        raise ValueError("standing initialization frozen-platform contract mismatch")
    if expected_contract is not None and dict(contract) != dict(expected_contract):
        raise ValueError("standing initialization differs from the trainer's frozen platform")
    state = value.get("adapter_state_dict")
    if not isinstance(state, Mapping) or not state:
        raise ValueError("standing initialization adapter state is missing")
    if any(
        not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float32 or not torch.isfinite(tensor).all()
        for tensor in state.values()
    ):
        raise ValueError("standing initialization adapter tensors must be finite float32")
    digest = _tensor_state_sha256(state)
    if digest != value.get("adapter_state_sha256") or digest != fit.get("adapter_state_sha256"):
        raise ValueError("standing initialization adapter tensor hash mismatch")
    if (
        type(value.get("optimizer_steps")) is not int
        or value["optimizer_steps"] != fit.get("steps")
        or value["optimizer_steps"] < 1
    ):
        raise ValueError("standing initialization fitting-step evidence mismatch")
    return value


def read_standing_initialization(report_path: Path):
    report_path = Path(report_path).expanduser()
    if report_path.is_symlink():
        raise ValueError("standing fit report may not be a symlink")
    report_path = report_path.resolve(strict=True)
    fit = json.loads(report_path.read_text())
    if (
        fit.get("kind") != "g1_true23_standing_only_lora_fit_diagnostic_v1"
        or fit.get("frozen_platform_unchanged") is not True
    ):
        raise ValueError("standing initialization requires completed frozen-platform fitting evidence")
    if any(
        fit.get(flag) is not False
        for flag in ("hardware_authorized", "deployment_ready", "promotion_eligible", "full_motion_qualified")
    ):
        raise ValueError("standing fitting must remain an unqualified simulator diagnostic")
    pins = dict(fit["inputs"])
    for path, digest in pins.items():
        if file_sha256(Path(path)) != digest:
            raise ValueError(f"standing initialization source changed: {path}")
    adapter_path = report_path.parent / "standing_lora.pt"
    if adapter_path.is_symlink() or str(adapter_path) not in pins:
        raise ValueError("standing adapter must be directly hash-bound by its fit report")
    payload = torch.load(adapter_path, map_location="cpu", weights_only=True)
    validate_standing_payload(payload, fit)
    if payload.get("source_checkpoint_sha256") not in pins.values():
        raise ValueError("standing initialization lost its prior adapter provenance")
    records = fit.get("records")
    if not isinstance(records, list) or [row["name"] for row in records] != [
        "standing_synthetic_start",
        "standing_after_acquisition",
    ]:
        raise ValueError("standing initialization needs both stationary lifecycle attempts")
    for row in records:
        result = row["result"]
        if (
            result.get("completed_transitions") != 500
            or result.get("requested_transitions") != 500
            or result.get("failure") is not None
        ):
            raise ValueError("standing initialization active phase is incomplete")
        if not result.get("upright_physical_bounds_passed") or not result.get("motion_fidelity", {}).get("passed"):
            raise ValueError("standing initialization lacks stationary tracking/physical screens")
        path = report_path.parent / f"{row['name']}.npz"
        if str(path) not in pins:
            raise ValueError("standing initialization trace is not hash-bound")
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        if not audit_engine_trace(arrays)["passed"]:
            raise ValueError("standing initialization has an invalid engine trace")
        for kind in ("qpos", "qvel"):
            before, after = arrays[f"physics_pre_{kind}"], arrays[f"physics_post_{kind}"]
            if not np.isfinite(before).all() or not np.isfinite(after).all():
                raise ValueError("standing initialization has nonfinite simulated state")
            np.testing.assert_array_equal(before[1:], after[:-1])
        np.testing.assert_array_equal(arrays["physics_effort"], arrays["physics_generalized_actuator_force"])
        expected_phase_counts = [2500, 5000, 2500] if row["name"] == "standing_after_acquisition" else [0, 5000, 0]
        if [int(np.sum(arrays["physics_phase"] == phase)) for phase in range(3)] != expected_phase_counts:
            raise ValueError("standing initialization phase lengths disagree with actual physics")
        if row["name"] == "standing_after_acquisition":
            if result.get("lifecycle_simulator_screen_passed") is not True:
                raise ValueError("standing initialization lifecycle did not pass")
            for name in ("startup_hold", "return_hold"):
                hold = result[name]
                if (
                    hold.get("completed_transitions") != 250
                    or hold.get("existing_guard_screen_passed") is not True
                ):
                    raise ValueError("standing initialization acquisition/return is incomplete")
    descriptor = {
        "kind": "g1_true23_standing_only_training_initialization_v1",
        "report_path": str(report_path),
        "report_sha256": file_sha256(report_path),
        "adapter_path": str(adapter_path),
        "adapter_file_sha256": pins[str(adapter_path)],
        "adapter_state_sha256": payload["adapter_state_sha256"],
        "frozen_platform_contract": copy.deepcopy(payload["adapter_contract"]),
        "source_optimizer_steps": payload["optimizer_steps"],
        "critic_reused": False,
        "optimizer_reused": False,
        "counters_reused": False,
        "standing_lifecycle_simulator_evidence_only": True,
        "full_motion_teacher_accepted": False,
        "hardware_authorized": False,
        "deployment_ready": False,
    }
    return payload, descriptor


def apply_standing_initialization(runner, payload, descriptor):
    """Transactional adapter-only import; never restore critic/PPO state."""
    if runner._require_counter_coherence() != 0 or runner.alg.optimizer.state:
        raise ValueError("standing initialization requires a fresh zero-update optimizer/runner")
    actor = runner.alg.get_policy()
    if actor.core.adapter_contract() != descriptor["frozen_platform_contract"]:
        raise ValueError("standing initialization differs from the runner's frozen platform")
    validate_standing_payload(
        payload,
        {
            "adapter_contract": descriptor["frozen_platform_contract"],
            "adapter_state_sha256": descriptor["adapter_state_sha256"],
            "steps": descriptor["source_optimizer_steps"],
        },
        expected_contract=actor.core.adapter_contract(),
    )
    before = actor.core.lora_state_dict()
    critic_before = _tensor_state_sha256(runner.alg.critic.state_dict())
    try:
        actor.core.load_lora_state_dict(payload["adapter_state_dict"], strict=True)
        actor.core.assert_frozen_platform_unchanged()
        runner._assert_boundary()
        if (
            runner._require_counter_coherence() != 0
            or runner.alg.optimizer.state
            or _tensor_state_sha256(runner.alg.critic.state_dict()) != critic_before
        ):
            raise ValueError("standing initialization changed fresh PPO state")
        merged_hash = actor.core.merged_true23_policy_sha256(actor.distribution.std_param)
    except BaseException:
        actor.core.load_lora_state_dict(before, strict=True)
        raise
    runner._frozen_lora_runtime["standing_initialization"] = {
        **descriptor,
        "merged_true23_policy_sha256": merged_hash,
    }
    return copy.deepcopy(runner._frozen_lora_runtime["standing_initialization"])
