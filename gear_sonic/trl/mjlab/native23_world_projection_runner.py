"""Fresh world-quality LoRA training with audited reachable-mean auxiliary loss."""

import copy
import os
from pathlib import Path
import tempfile

import numpy as np
import torch

from gear_sonic.trl.mjlab.native23_projected_target_ppo import (
    PROFILE,
    WEIGHT,
    ProjectedTargetPPO,
    projection_objective_contract,
)
from gear_sonic.trl.mjlab.native23_world_quality_runner import (
    CHECKPOINT_HEADER as QUALITY_HEADER,
    Native23WorldQualityRunner,
    validate_checkpoint as validate_quality_checkpoint,
)
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file
from gear_sonic.utils.g1_true23_training_precision import write_runtime

ALGORITHM = "gear_sonic.trl.mjlab.native23_projected_target_ppo:ProjectedTargetPPO"
CHECKPOINT_HEADER = {**QUALITY_HEADER, "kind": "g1_native23_world_quality_projected_means_training_snapshot"}


def training_contract():
    return dict(
        kind="world_quality_plus_reachable_mean_loss_v1",
        auxiliary_objective=projection_objective_contract(PROFILE),
        algorithm=ALGORITHM,
        reward_action_likelihood_and_physical_limits_unchanged=True,
        fresh_same_initialization_as_world_quality=True,
        rollout_and_minibatch_mean_capture=True,
        checkpoint_pattern="world_projection_model_N.pt",
        resume_or_transfer_supported=False,
        hardware_authorized=False,
        deployment_ready=False,
    )


def require_contract(resolved):
    if resolved.get("native23_world_projection") != training_contract():
        raise ValueError("world-projection objective contract missing or changed")
    if resolved.get("agent", {}).get("algorithm", {}).get("class_name") != ALGORITHM:
        raise ValueError("world-projection algorithm does not match declared objective")
    if resolved.get("ppo_auxiliary_objective") != projection_objective_contract(PROFILE):
        raise ValueError("world-projection auxiliary loss not declared")


def quality_schema_view(checkpoint):
    if not isinstance(checkpoint, dict) or checkpoint.get("header") != CHECKPOINT_HEADER:
        raise ValueError("world-projection reader requires its distinct research header")
    require_contract(
        checkpoint.get("lineage", {}).get("materials", {}).get("resolved_config", {}).get("payload", {})
    )
    return {**checkpoint, "header": copy.deepcopy(QUALITY_HEADER)}


def validate_checkpoint(checkpoint, *, actor, lineage, minimum_update_count=0):
    require_contract(lineage["materials"]["resolved_config"]["payload"])
    validate_quality_checkpoint(
        quality_schema_view(checkpoint),
        actor=actor,
        lineage=lineage,
        minimum_update_count=minimum_update_count,
    )
    return checkpoint


class Native23WorldProjectionRunner(Native23WorldQualityRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.require_runtime()
        self._projection_learning = False
        self._projection_updating = False
        self._projection_rollouts = []
        self._projection_minibatches = []
        self._projection_updates = []
        self._projection_hook = self.alg.actor.register_forward_hook(self._capture_mean, with_kwargs=True)
        original_update = self.alg.update

        def update():
            before = len(self._projection_minibatches)
            self._projection_updating = True
            try:
                result = original_update()
            finally:
                self._projection_updating = False
            count = len(self._projection_minibatches) - before
            if count != self.alg.num_learning_epochs * self.alg.num_mini_batches:
                raise ValueError("actual projection minibatch capture count differs")
            self._projection_updates.append(dict(result))
            return result

        self.alg.update = update

    def _capture_mean(self, module, args, kwargs, output):
        del args
        if not self._projection_learning or kwargs.get("stochastic_output") is not True:
            return
        mean, std = module.output_distribution_params
        row = dict(mean=mean.detach().cpu().clone(), std=std.detach().cpu().clone())
        if self._projection_updating:
            self._projection_minibatches.append(row)
        else:
            row["sample"] = output.detach().cpu().clone()
            row["common_step_counter"] = int(self.env.unwrapped.common_step_counter)
            self._projection_rollouts.append(row)

    def require_runtime(self):
        super().require_runtime()
        require_contract(self.training_lineage["materials"]["resolved_config"]["payload"])
        if type(self.alg) is not ProjectedTargetPPO or self.alg.mean_projection_weight != WEIGHT:
            raise ValueError("actual projection algorithm or coefficient changed")

    def _numbered_checkpoint_path(self, update_count):
        old = super()._numbered_checkpoint_path(update_count)
        return old.with_name(old.name.replace("world_quality_model_", "world_projection_model_"))

    def _checkpoint_payload(self):
        result = super()._checkpoint_payload()
        result["header"] = copy.deepcopy(CHECKPOINT_HEADER)
        return result

    def save(self, path, infos=None):
        self.require_runtime()
        if infos is not None:
            raise ValueError("world-projection snapshot forbids stock RSL infos")
        self._require_checkpointable()
        output = Path(path).expanduser()
        if output.exists() or output.is_symlink():
            raise FileExistsError("world-projection snapshot refuses overwrite")
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
        self._projection_learning = True
        try:
            return super().learn(num_learning_iterations, init_at_random_ep_len)
        finally:
            self._projection_learning = False
            count = self.completed_update_count
            arrays = {}
            for label, rows in (
                ("rollout", self._projection_rollouts),
                ("minibatch", self._projection_minibatches),
            ):
                keys = ("mean", "std", "sample") if label == "rollout" else ("mean", "std")
                for key in keys:
                    arrays[label + "_" + key] = (
                        torch.stack([row[key] for row in rows]).numpy() if rows else np.empty((0,))
                    )
            arrays["rollout_common_step_counter"] = np.asarray(
                [r["common_step_counter"] for r in self._projection_rollouts]
            )
            target = self.checkpoint_dir.parent / f"projection_capture_after_{count}_updates.npz"
            with target.open("xb") as stream:
                np.savez_compressed(stream, **arrays)
            write_runtime(
                target.with_suffix(".json"),
                dict(
                    kind="native23_world_projection_actual_mean_capture_v1",
                    training_contract=training_contract(),
                    completed_updates=count,
                    algorithm_receipt=self.alg.projection_runtime_receipt(),
                    actual_rollout_calls=len(self._projection_rollouts),
                    actual_minibatch_calls=len(self._projection_minibatches),
                    update_losses=self._projection_updates,
                    arrays_path=str(target),
                    arrays_sha256=sha256_file(target),
                    lineage_sha256=self.lineage_sha256,
                    training_state_poisoned=bool(self._training_state_poisoned),
                    hardware_authorized=False,
                    deployment_ready=False,
                ),
            )
