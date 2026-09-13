"""Fresh-only native23 research runner with explicit joint world-quality bonus."""

import copy
import os
from pathlib import Path
import tempfile

import numpy as np
import torch

from gear_sonic.envs.mjlab.sonic_true23_world_tracking_termination import termination_contract
from gear_sonic.trl.mjlab.native23_decoder_lora_runner import Native23DecoderLoraRunner
from gear_sonic.trl.mjlab.native23_world_tracking_runner import (
    CHECKPOINT_HEADER as WORLD_HEADER,
    Native23WorldTrackingRunner,
    validate_checkpoint as validate_world_checkpoint,
)
from gear_sonic.utils.g1_true23_bounded_progress import NEGATIVE_WEIGHTS, reward_contract
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file
from gear_sonic.utils.g1_true23_training_precision import write_runtime
from gear_sonic.utils.g1_true23_world_quality import WorldQualityStep, quality_contract

CHECKPOINT_HEADER = {**WORLD_HEADER, "kind": "g1_native23_joint_world_quality_training_snapshot"}


def require_quality_contract(resolved):
    if resolved.get("native23_world_quality_bonus") != quality_contract():
        raise ValueError("joint world-quality contract missing or changed")


def world_schema_view(checkpoint):
    if not isinstance(checkpoint, dict) or checkpoint.get("header") != CHECKPOINT_HEADER:
        raise ValueError("world-quality reader requires its distinct research header")
    resolved = checkpoint.get("lineage", {}).get("materials", {}).get("resolved_config", {}).get("payload", {})
    require_quality_contract(resolved)
    # Validate the unchanged common tensor schema in memory, never relabel disk.
    return {**checkpoint, "header": copy.deepcopy(WORLD_HEADER)}


def validate_checkpoint(checkpoint, *, actor, lineage, minimum_update_count=0):
    require_quality_contract(lineage["materials"]["resolved_config"]["payload"])
    validate_world_checkpoint(
        world_schema_view(checkpoint), actor=actor, lineage=lineage, minimum_update_count=minimum_update_count
    )
    return checkpoint


class Native23WorldQualityRunner(Native23WorldTrackingRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        require_quality_contract(self.training_lineage["materials"]["resolved_config"]["payload"])
        previous = self._reward_step
        self._reward_step = WorldQualityStep(previous)
        self.env.unwrapped.step = self._reward_step
        self.require_runtime()

    def require_runtime(self):
        super().require_runtime()
        require_quality_contract(self.training_lineage["materials"]["resolved_config"]["payload"])
        if not isinstance(self._reward_step, WorldQualityStep):
            raise ValueError("world-quality step is not installed")
        if self._reward_step.original is not self.env.unwrapped._bounded_progress_step:
            raise ValueError("original bounded reward step changed")

    def _numbered_checkpoint_path(self, update_count):
        old = super()._numbered_checkpoint_path(update_count)
        return old.with_name(old.name.replace("world_tracking_model_", "world_quality_model_"))

    def _checkpoint_payload(self):
        result = super()._checkpoint_payload()
        result["header"] = copy.deepcopy(CHECKPOINT_HEADER)
        return result

    def save(self, path, infos=None):
        self.require_runtime()
        if infos is not None:
            raise ValueError("world-quality snapshots forbid stock RSL infos")
        self._require_checkpointable()
        output = Path(path).expanduser()
        if output.exists() or output.is_symlink():
            raise FileExistsError("world-quality snapshot refuses overwrite")
        output = output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=output.parent, prefix=f".{output.name}.", suffix=".tmp", delete=False
            ) as stream:
                temporary = Path(stream.name)
            torch.save(self._checkpoint_payload(), temporary)
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
            # Use the same actual optimization loop, but this version owns its
            # complete capture metadata instead of mislabelling a parent file.
            return Native23DecoderLoraRunner.learn(self, num_learning_iterations, init_at_random_ep_len)
        finally:
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
                dict(
                    kind="native23_world_quality_actual_reward_capture_v1",
                    reward_contract=reward_contract(),
                    world_quality_bonus=quality_contract(),
                    world_tracking_termination=termination_contract(),
                    actual_world_termination_calls=len(world_rows),
                    world_and_reward_capture_counts_equal=len(world_rows) == len(self._reward_step.rows),
                    completed_updates=count,
                    actual_controls_per_env=len(self._reward_step.rows),
                    num_envs=self.env.num_envs,
                    raw_cost_names=list(NEGATIVE_WEIGHTS),
                    base_component_names=list(self.env.unwrapped.reward_manager.active_terms),
                    arrays_path=str(target),
                    arrays_sha256=sha256_file(target),
                    lineage_sha256=self.lineage_sha256,
                    training_state_poisoned=bool(self._training_state_poisoned),
                    independent_audit_passed=False,
                    hardware_authorized=False,
                    deployment_ready=False,
                ),
            )
