"""Simulation-only causal PPO runner for fully trainable native23 decoders.

Keeps the existing lineage/counter-aware learning loop, but checkpoints the
bounded exploration parameter itself rather than trying to invert exported
standard deviations. Legacy SONIC deployment/resume loaders cannot confuse
these artifacts with qualified deployable checkpoints.
"""

from __future__ import annotations

from collections.abc import Mapping
import copy
import os
from pathlib import Path
import tempfile
from typing import Any

import torch

from gear_sonic.trl.mjlab.causal_history_runner import CausalHistoryMjlabOnPolicyRunner
from gear_sonic.trl.mjlab.frozen_platform_lora_runner import _state_sha256
from gear_sonic.trl.mjlab.native23_generalist_actor import True23Native23GeneralistActorModel
from gear_sonic.trl.mjlab.runner import (
    _require_learning_rate,
    _require_nonnegative_integer,
    _set_environment_common_step_counter,
)
from gear_sonic.utils.g1_23dof_artifact import inspect_true23_policy_state
from gear_sonic.utils.g1_23dof_mjlab_training import (
    _validate_optimizer_state,
    validate_mjlab_training_lineage,
)

CHECKPOINT_HEADER = {
    "schema_version": 1,
    "kind": "g1_native23_generalist_training_resume",
    "role": "simulator_training_resume_only",
    "deployment_ready": False,
    "promotion_eligible": False,
    "hardware_authorized": False,
}


def generalist_optimizer_groups(
    actor: True23Native23GeneralistActorModel, critic: torch.nn.Module
) -> list[dict[str, Any]]:
    """Exactly one copy of every decoder/noise/critic trainable parameter."""
    groups = [{"name": name, "params": values} for name, values in actor.parameter_groups().items()]
    groups.append({"name": "critic", "params": [p for p in critic.parameters() if p.requires_grad]})
    if any(not group["params"] for group in groups):
        raise ValueError("generalist optimizer requires decoder, exploration and critic")
    parameters = [p for group in groups for p in group["params"]]
    actual_ids = [id(p) for p in parameters]
    expected_ids = {id(p) for p in (*actor.parameters(), *critic.parameters()) if p.requires_grad}
    if len(actual_ids) != len(set(actual_ids)) or set(actual_ids) != expected_ids:
        raise ValueError("generalist optimizer parameter partition is incomplete or duplicated")
    return groups


def validate_generalist_checkpoint(
    value: Any,
    *,
    actor: True23Native23GeneralistActorModel,
    lineage: Mapping[str, Any],
    minimum_update_count: int = 0,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or value.get("header") != CHECKPOINT_HEADER:
        raise ValueError("generalist checkpoint header mismatch")
    if set(value) != {
        "header",
        "actor",
        "critic_state_dict",
        "critic_state_sha256",
        "optimizer_state_dict",
        "trainer_state",
        "lineage",
        "lineage_sha256",
    }:
        raise ValueError("generalist checkpoint fields mismatch")
    actor.validate_training_artifact(value["actor"])
    actual_lineage = validate_mjlab_training_lineage(value["lineage"])
    expected = validate_mjlab_training_lineage(lineage)
    if (
        actual_lineage["lineage_sha256"] != expected["lineage_sha256"]
        or value["lineage_sha256"] != expected["lineage_sha256"]
    ):
        raise ValueError("generalist checkpoint lineage mismatch")
    if value["critic_state_sha256"] != _state_sha256(value["critic_state_dict"]):
        raise ValueError("generalist critic state hash mismatch")
    if not isinstance(value["optimizer_state_dict"], Mapping):
        raise ValueError("generalist checkpoint optimizer state missing")
    optimizer = _validate_optimizer_state(value["optimizer_state_dict"])
    if [group.get("name") for group in optimizer["param_groups"]] != ["decoder", "exploration", "critic"]:
        raise ValueError("generalist optimizer checkpoint groups mismatch")
    ids = [p for group in optimizer["param_groups"] for p in group["params"]]
    if len(ids) != len(set(ids)) or set(optimizer["state"]) - set(ids):
        raise ValueError("generalist optimizer checkpoint parameter ids mismatch")
    del optimizer
    trainer = value["trainer_state"]
    if not isinstance(trainer, Mapping) or set(trainer) != {
        "completed_update_count",
        "current_learning_iteration",
        "env_common_step_counter",
        "algorithm_learning_rate",
    }:
        raise ValueError("generalist trainer state fields mismatch")
    completed = _require_nonnegative_integer(trainer["completed_update_count"], "completed_update_count")
    current = _require_nonnegative_integer(trainer["current_learning_iteration"], "current_learning_iteration")
    _require_nonnegative_integer(trainer["env_common_step_counter"], "env_common_step_counter")
    _require_learning_rate(trainer["algorithm_learning_rate"], "algorithm_learning_rate")
    if completed != current or completed < minimum_update_count:
        raise ValueError("generalist checkpoint counters mismatch or predate runner")
    return value


class Native23GeneralistRunner(CausalHistoryMjlabOnPolicyRunner):
    """Full decoder + bounded exploration + fresh critic, exact resume state."""

    def _validate_initial_policy(self) -> None:
        actor = self.alg.get_policy()
        if not isinstance(actor, True23Native23GeneralistActorModel):
            raise TypeError("generalist runner requires True23Native23GeneralistActorModel")
        # Mean network must exactly match the pinned initialization. Training
        # exploration starts at explicit .1, not the release's own noise.
        actual_hash = inspect_true23_policy_state(
            {"policy_state_dict": actor.core.export_policy_state(actor.core.initial_std)},
            reference_profile=actor.reference_profile,
        )
        if actual_hash != self._training_lineage["warm_start"]["initial_policy_state_sha256"]:
            raise ValueError("generalist paired-source initialization mismatch")

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        actor = self.alg.get_policy()
        lr = _require_learning_rate(self.cfg["algorithm"]["learning_rate"], "generalist learning_rate")
        defaults = dict(self.alg.optimizer.defaults)
        defaults["lr"] = lr
        self.alg.optimizer = type(self.alg.optimizer)(
            generalist_optimizer_groups(actor, self.alg.critic), **defaults
        )
        self.alg.learning_rate = lr
        self._generalist_contract = actor.artifact_contract()
        self._generalist_optimizer_ids = tuple(id(p) for g in self.alg.optimizer.param_groups for p in g["params"])
        self._assert_boundary()

    def _assert_boundary(self) -> None:
        actor = self.alg.get_policy()
        actor.core.assert_frozen_encoder_unchanged()
        if actor.artifact_contract() != self._generalist_contract:
            raise RuntimeError("generalist actor contract changed during training")
        expected = tuple(id(p) for g in generalist_optimizer_groups(actor, self.alg.critic) for p in g["params"])
        actual = tuple(id(p) for g in self.alg.optimizer.param_groups for p in g["params"])
        if actual != expected or actual != self._generalist_optimizer_ids:
            raise RuntimeError("generalist optimizer partition changed")
        self._algorithm_learning_rate()

    def _numbered_checkpoint_path(self, update_count: int) -> Path:
        count = _require_nonnegative_integer(update_count, "generalist update_count")
        return self.checkpoint_dir / f"native23_generalist_model_{count}.pt"

    def _checkpoint_payload(self) -> dict[str, Any]:
        self._assert_boundary()
        critic = {
            name: value.detach().cpu().contiguous().clone() for name, value in self.alg.critic.state_dict().items()
        }
        return {
            "header": copy.deepcopy(CHECKPOINT_HEADER),
            "actor": self.alg.get_policy().export_training_artifact(),
            "critic_state_dict": critic,
            "critic_state_sha256": _state_sha256(critic),
            "optimizer_state_dict": copy.deepcopy(self.alg.optimizer.state_dict()),
            "trainer_state": self._trainer_state(),
            "lineage": copy.deepcopy(self._training_lineage),
            "lineage_sha256": self._lineage_sha256,
        }

    def save(self, path: str, infos: dict | None = None) -> None:
        if infos is not None:
            raise ValueError("generalist checkpoints forbid stock RSL infos")
        self._require_checkpointable()
        output = Path(path).expanduser()
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"refusing to overwrite generalist checkpoint: {output}")
        output = output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = self._checkpoint_payload()
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=output.parent, prefix=f".{output.name}.", suffix=".tmp", delete=False
            ) as handle:
                temporary = Path(handle.name)
            torch.save(payload, temporary)
            loaded = torch.load(temporary, map_location="cpu", weights_only=True)
            validate_generalist_checkpoint(loaded, actor=self.alg.get_policy(), lineage=self._training_lineage)
            with temporary.open("rb+") as stream:
                os.fsync(stream.fileno())
            # Atomic no-replace publication. Unlike os.replace, another writer
            # cannot be overwritten between a pre-check and the publication.
            os.link(temporary, output)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
        self._last_checkpoint_path = output
        self._last_checkpoint_update_count = self._require_counter_coherence()

    def load(
        self, path: str, load_cfg: dict | None = None, strict: bool = True, map_location: str | None = None
    ) -> dict[str, Any]:
        if load_cfg is not None or strict is not True:
            raise ValueError("generalist resume must be complete and strict")
        # A failed PPO minibatch may corrupt optimizer state and rollout
        # storage. Require a new runner instead of claiming in-place recovery.
        self._require_checkpointable()
        requested = Path(path).expanduser()
        if requested.is_symlink():
            raise ValueError("generalist resume may not be a symlink")
        requested = requested.resolve()
        actor = self.alg.get_policy()
        loaded = torch.load(requested, map_location=map_location or "cpu", weights_only=True)
        checkpoint = validate_generalist_checkpoint(
            loaded,
            actor=actor,
            lineage=self._training_lineage,
            minimum_update_count=self._require_counter_coherence(),
        )
        before = self._checkpoint_payload()
        self._training_state_poisoned = True
        try:
            self._restore_payload(checkpoint)
        except BaseException:
            try:
                self._restore_payload(before)
            except BaseException as rollback_error:
                raise RuntimeError("generalist restore rollback failed; discard runner") from rollback_error
            self._training_state_poisoned = False
            raise
        self._training_state_poisoned = False
        self._last_checkpoint_path = requested
        self._last_checkpoint_update_count = self._require_counter_coherence()
        return {
            "trainer_state": dict(checkpoint["trainer_state"]),
            "lineage_sha256": self._lineage_sha256,
            "actor_state_sha256": checkpoint["actor"]["state_sha256"],
        }

    def _restore_payload(self, checkpoint: Mapping[str, Any]) -> None:
        self.alg.get_policy().load_training_artifact(checkpoint["actor"])
        self.alg.critic.load_state_dict(checkpoint["critic_state_dict"], strict=True)
        self.alg.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        trainer = checkpoint["trainer_state"]
        self.completed_update_count = trainer["completed_update_count"]
        self.current_learning_iteration = trainer["current_learning_iteration"]
        self.alg.learning_rate = trainer["algorithm_learning_rate"]
        _set_environment_common_step_counter(self.env, trainer["env_common_step_counter"])
        self._assert_boundary()
        self._require_counter_coherence()


__all__ = ["Native23GeneralistRunner", "generalist_optimizer_groups", "validate_generalist_checkpoint"]
