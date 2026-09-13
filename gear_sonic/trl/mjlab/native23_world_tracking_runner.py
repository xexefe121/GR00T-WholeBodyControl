"""Fresh SIM snapshots with unchanged bounded reward plus world tracking failure."""

import copy
import os
from pathlib import Path
import tempfile

import numpy as np
import torch

from gear_sonic.envs.mjlab.sonic_true23_bounded_progress import install_progress_step, verify_runtime
from gear_sonic.envs.mjlab.sonic_true23_world_tracking_termination import (
    termination_contract,
    verify_runtime as verify_world_runtime,
)
from gear_sonic.trl.mjlab.native23_decoder_lora_runner import (
    CHECKPOINT_HEADER as DECODER_HEADER,
    Native23DecoderLoraRunner,
    validate_decoder_lora_checkpoint,
)
from gear_sonic.utils.g1_true23_bounded_progress import GAMMA, NEGATIVE_WEIGHTS, reward_contract
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file
from gear_sonic.utils.g1_true23_training_precision import write_runtime

CHECKPOINT_HEADER = {
    **DECODER_HEADER,
    "kind": "g1_native23_decoder_lora_world_tracking_termination_training_snapshot",
}


def require_reward_contract(resolved):
    if resolved.get("native23_world_tracking_termination") != termination_contract():
        raise ValueError("world tracking termination contract missing or changed")
    if resolved.get("native23_bounded_progress") != reward_contract():
        raise ValueError("bounded progress reward contract missing or changed")
    if resolved.get("agent", {}).get("algorithm", {}).get("gamma") != GAMMA:
        raise ValueError("bounded progress requires the bound PPO discount")


def decoder_schema_view(checkpoint):
    """Validate common tensor schema without relabelling any on-disk artifact."""
    if not isinstance(checkpoint, dict) or checkpoint.get("header") != CHECKPOINT_HEADER:
        raise ValueError("bounded progress requires its distinct research header")
    resolved = checkpoint.get("lineage", {}).get("materials", {}).get("resolved_config", {}).get("payload", {})
    require_reward_contract(resolved)
    return {**checkpoint, "header": copy.deepcopy(DECODER_HEADER)}


def validate_checkpoint(checkpoint, *, actor, lineage, minimum_update_count=0):
    require_reward_contract(lineage["materials"]["resolved_config"]["payload"])
    validate_decoder_lora_checkpoint(
        decoder_schema_view(checkpoint), actor=actor, lineage=lineage, minimum_update_count=minimum_update_count
    )
    return checkpoint


class Native23WorldTrackingRunner(Native23DecoderLoraRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        resolved = self.training_lineage["materials"]["resolved_config"]["payload"]
        require_reward_contract(resolved)
        if self.alg.gamma != GAMMA or self.alg.rnd:
            raise ValueError("bounded progress requires the unchanged discount and no intrinsic reward")
        spec = resolved["native23_root_feedback"]["original_intent_spec"]
        verify_world_runtime(self.env.unwrapped)
        self.env.unwrapped._world_root_failure_capture = []
        self._reward_step = install_progress_step(self.env.unwrapped, spec)
        previous_process = self.alg.process_env_step

        def capture_process(obs, rewards, dones, extras):
            row = self._reward_step.rows[-1]
            if row["ppo_recorded"].any():
                raise ValueError("PPO attempted to store one reward capture twice")
            value = self.alg.transition.values.detach().cpu().clone().squeeze(-1)
            index = self.alg.storage.step
            previous_process(obs, rewards, dones, extras)
            if self.alg.storage.step != index + 1:
                raise ValueError("unexpected PPO transition-storage advancement")
            row["critic_value_before_bootstrap"] = value
            row["stored_reward_with_timeout_bootstrap"] = (
                self.alg.storage.rewards[index].detach().cpu().clone().squeeze(-1)
            )
            row["stored_done"] = self.alg.storage.dones[index].detach().cpu().clone().squeeze(-1).bool()
            row["ppo_recorded"][:] = True

        self.alg.process_env_step = capture_process

    def require_runtime(self):
        require_reward_contract(self.training_lineage["materials"]["resolved_config"]["payload"])
        verify_runtime(self.env.unwrapped)
        verify_world_runtime(self.env.unwrapped)
        if self.env.unwrapped.step is not self._reward_step or self.alg.gamma != GAMMA or self.alg.rnd:
            raise ValueError("bounded progress runtime changed")

    def _numbered_checkpoint_path(self, update_count):
        # Reuse the inherited strict counter validation, but a separate name.
        old = super()._numbered_checkpoint_path(update_count)
        return old.with_name(old.name.replace("decoder_lora_model_", "world_tracking_model_"))

    def _checkpoint_payload(self):
        result = super()._checkpoint_payload()
        result["header"] = copy.deepcopy(CHECKPOINT_HEADER)
        return result

    def save(self, path, infos=None):
        self.require_runtime()
        if infos is not None:
            raise ValueError("bounded progress snapshots forbid stock RSL infos")
        self._require_checkpointable()
        output = Path(path).expanduser()
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"refusing to overwrite bounded progress snapshot: {output}")
        output = output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = self._checkpoint_payload()
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=output.parent, prefix=f".{output.name}.", suffix=".tmp", delete=False
            ) as stream:
                temporary = Path(stream.name)
            torch.save(payload, temporary)
            loaded = torch.load(temporary, map_location="cpu", weights_only=True)
            validate_checkpoint(loaded, actor=self.alg.get_policy(), lineage=self.training_lineage)
            with temporary.open("rb+") as stream:
                os.fsync(stream.fileno())
            os.link(temporary, output)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
        self._last_checkpoint_path = output
        self._last_checkpoint_update_count = self._require_counter_coherence()

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        self.require_runtime()
        try:
            return super().learn(num_learning_iterations, init_at_random_ep_len)
        finally:
            # Preserve even a failed partial run. This is an arithmetic capture,
            # not a successful rollout, policy export or physics qualification.
            count = self.completed_update_count
            arrays = self._reward_step.capture()
            world_rows = self.env.unwrapped._world_root_failure_capture
            for key in ("desired_position_w", "measured_position_w", "error_m", "failure", "reference_q0"):
                arrays["world_tracking_" + key] = (
                    torch.stack([row[key] for row in world_rows]).numpy() if world_rows else np.empty((0,))
                )
            arrays["world_tracking_common_step_counter"] = np.asarray(
                [row["common_step_counter"] for row in world_rows], dtype=np.int64
            )
            target = self.checkpoint_dir.parent / f"reward_steps_after_{count}_updates.npz"
            with target.open("xb") as stream:
                np.savez_compressed(stream, **arrays)
            write_runtime(
                target.with_suffix(".json"),
                {
                    "reward_contract": reward_contract(),
                    "world_tracking_termination": termination_contract(),
                    "actual_world_termination_calls": len(world_rows),
                    "world_and_reward_capture_counts_equal": len(world_rows) == len(self._reward_step.rows),
                    "completed_updates": count,
                    "actual_controls_per_env": len(self._reward_step.rows),
                    "num_envs": self.env.num_envs,
                    "raw_cost_names": list(NEGATIVE_WEIGHTS),
                    "base_component_names": list(self.env.unwrapped.reward_manager.active_terms),
                    "arrays_path": str(target),
                    "arrays_sha256": sha256_file(target),
                    "lineage_sha256": self.lineage_sha256,
                    "training_state_poisoned": bool(self._training_state_poisoned),
                    "independent_audit_passed": False,
                    "hardware_authorized": False,
                    "deployment_ready": False,
                },
            )
