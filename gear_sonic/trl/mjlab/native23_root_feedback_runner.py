"""Distinct root-feedback PPO checkpoint format; old single-input artifacts stay valid."""

from __future__ import annotations

from collections.abc import Mapping
import copy
import math
import os
from pathlib import Path
import tempfile

import torch

from gear_sonic.trl.mjlab.native23_generalist_runner import Native23GeneralistRunner
from gear_sonic.trl.mjlab.native23_root_feedback_actor import True23RootFeedbackActorModel
from gear_sonic.trl.mjlab.frozen_platform_lora_runner import _state_sha256
from gear_sonic.trl.mjlab.runner import _require_learning_rate, _require_nonnegative_integer
from gear_sonic.utils.g1_23dof_mjlab_training import (
    _validate_optimizer_state,
    validate_mjlab_training_lineage,
)

CHECKPOINT_HEADER = {
    "schema_version": 2,
    "kind": "g1_native23_root_feedback_training_resume",
    "role": "simulator_training_resume_only",
    "deployment_ready": False,
    "promotion_eligible": False,
    "hardware_authorized": False,
}

OPTIMIZER_PROFILES = {
    "legacy_uniform": {"decoder": 1.0, "root_conditioner": 1.0, "exploration": 1.0, "critic": 1.0},
    "feedback_priority": {"decoder": 1.0, "root_conditioner": 200.0, "exploration": 1.0, "critic": 600.0},
}


def optimizer_multipliers(resolved_config):
    config = resolved_config.get("native23_root_feedback", {})
    profile = config.get("optimizer_profile", "legacy_uniform")
    if profile not in OPTIMIZER_PROFILES:
        raise ValueError("unknown root-feedback optimizer profile")
    expected = OPTIMIZER_PROFILES[profile]
    actual = config.get("optimizer_learning_rate_multipliers", expected)
    if actual != expected:
        raise ValueError("root-feedback optimizer multipliers differ from versioned profile")
    return dict(expected)


def validate_optimizer_rates(groups, base_rate, multipliers):
    base = _require_learning_rate(base_rate, "root-feedback base learning rate")
    if [group.get("name") for group in groups] != list(multipliers):
        raise ValueError("root-feedback rate group names or ordering changed")
    for group in groups:
        value = _require_learning_rate(group.get("lr"), "root-feedback group learning rate")
        if not math.isclose(value, base * multipliers[group["name"]], rel_tol=1e-12, abs_tol=0):
            raise ValueError("root-feedback optimizer rate differs from bound profile")
    return base


def validate_root_feedback_checkpoint(value, *, actor, lineage, minimum_update_count=0):
    if not isinstance(actor, True23RootFeedbackActorModel):
        raise TypeError("root-feedback checkpoint requires versioned root-feedback actor")
    if not isinstance(value, Mapping) or value.get("header") != CHECKPOINT_HEADER:
        raise ValueError("root-feedback checkpoint header mismatch")
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
        raise ValueError("root-feedback checkpoint fields mismatch")
    actor.validate_training_artifact(value["actor"])
    actual = validate_mjlab_training_lineage(value["lineage"])
    expected = validate_mjlab_training_lineage(lineage)
    if (
        actual["lineage_sha256"] != expected["lineage_sha256"]
        or value["lineage_sha256"] != expected["lineage_sha256"]
    ):
        raise ValueError("root-feedback checkpoint lineage mismatch")
    if value["critic_state_sha256"] != _state_sha256(value["critic_state_dict"]):
        raise ValueError("root-feedback critic state hash mismatch")
    if not isinstance(value["optimizer_state_dict"], Mapping):
        raise ValueError("root-feedback checkpoint optimizer missing")
    optimizer = _validate_optimizer_state(value["optimizer_state_dict"])
    if [g.get("name") for g in optimizer["param_groups"]] != [
        "decoder",
        "root_conditioner",
        "exploration",
        "critic",
    ]:
        raise ValueError("root-feedback optimizer groups mismatch")
    offset = 0
    for group, parameters in zip(optimizer["param_groups"], actor.parameter_groups().values()):
        if group["params"] != list(range(offset, offset + len(parameters))):
            raise ValueError("root-feedback optimizer group parameter ownership mismatch")
        offset += len(parameters)
    critic_ids = optimizer["param_groups"][-1]["params"]
    if not critic_ids or critic_ids != list(range(offset, offset + len(critic_ids))):
        raise ValueError("root-feedback optimizer critic parameter ownership mismatch")
    ids = [p for group in optimizer["param_groups"] for p in group["params"]]
    if len(ids) != len(set(ids)) or set(optimizer["state"]) - set(ids):
        raise ValueError("root-feedback optimizer parameter ids mismatch")
    trainer = value["trainer_state"]
    if not isinstance(trainer, Mapping) or set(trainer) != {
        "completed_update_count",
        "current_learning_iteration",
        "env_common_step_counter",
        "algorithm_learning_rate",
    }:
        raise ValueError("root-feedback trainer state fields mismatch")
    count = _require_nonnegative_integer(trainer["completed_update_count"], "completed_update_count")
    current = _require_nonnegative_integer(trainer["current_learning_iteration"], "current_learning_iteration")
    _require_nonnegative_integer(trainer["env_common_step_counter"], "env_common_step_counter")
    validate_optimizer_rates(
        optimizer["param_groups"],
        trainer["algorithm_learning_rate"],
        optimizer_multipliers(expected["materials"]["resolved_config"]["payload"]),
    )
    del optimizer
    if count != current or count < minimum_update_count:
        raise ValueError("root-feedback checkpoint counters mismatch or predate runner")
    return value


class Native23RootFeedbackRunner(Native23GeneralistRunner):
    def __init__(self, *args, **kwargs):
        # Parent initially builds uniform groups; activate declared ratios only
        # after its initialization boundary, before any PPO step/checkpoint.
        self._root_group_rates_pending = True
        super().__init__(*args, **kwargs)
        multipliers = optimizer_multipliers(self._training_lineage["materials"]["resolved_config"]["payload"])
        if getattr(self.alg, "schedule", None) != "fixed":
            raise ValueError("versioned group rates require fixed PPO schedule")
        for group in self.alg.optimizer.param_groups:
            group["lr"] = self.alg.learning_rate * multipliers[group["name"]]
        self._root_group_rates_pending = False
        self._assert_boundary()

    def _algorithm_learning_rate(self):
        if getattr(self, "_root_group_rates_pending", False):
            return super()._algorithm_learning_rate()
        return validate_optimizer_rates(
            self.alg.optimizer.param_groups,
            self.alg.learning_rate,
            optimizer_multipliers(self._training_lineage["materials"]["resolved_config"]["payload"]),
        )

    def _validate_initial_policy(self):
        actor = self.alg.get_policy()
        if not isinstance(actor, True23RootFeedbackActorModel):
            raise TypeError("root-feedback runner requires True23RootFeedbackActorModel")
        if torch.count_nonzero(actor.root_conditioner.weight).item():
            raise ValueError("root-feedback initial conditioner must be zero")
        super()._validate_initial_policy()

    def _numbered_checkpoint_path(self, update_count):
        count = _require_nonnegative_integer(update_count, "root-feedback update_count")
        return self.checkpoint_dir / f"root_feedback_model_{count}.pt"

    def _checkpoint_payload(self):
        payload = super()._checkpoint_payload()
        payload["header"] = copy.deepcopy(CHECKPOINT_HEADER)
        return payload

    def save(self, path, infos=None):
        if infos is not None:
            raise ValueError("root-feedback checkpoints forbid stock RSL infos")
        self._require_checkpointable()
        output = Path(path).expanduser()
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"refusing to overwrite root-feedback checkpoint: {output}")
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
            validate_root_feedback_checkpoint(loaded, actor=self.alg.get_policy(), lineage=self._training_lineage)
            with temporary.open("rb+") as stream:
                os.fsync(stream.fileno())
            os.link(temporary, output)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
        self._last_checkpoint_path = output
        self._last_checkpoint_update_count = self._require_counter_coherence()

    def load(self, path, load_cfg=None, strict=True, map_location=None):
        if load_cfg is not None or strict is not True:
            raise ValueError("root-feedback resume must be complete and strict")
        self._require_checkpointable()
        requested = Path(path).expanduser()
        if requested.is_symlink():
            raise ValueError("root-feedback resume may not be a symlink")
        requested = requested.resolve()
        loaded = torch.load(requested, map_location=map_location or "cpu", weights_only=True)
        checkpoint = validate_root_feedback_checkpoint(
            loaded,
            actor=self.alg.get_policy(),
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
            except BaseException as error:
                raise RuntimeError("root-feedback restore rollback failed; discard runner") from error
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


__all__ = ["Native23RootFeedbackRunner", "validate_root_feedback_checkpoint", "CHECKPOINT_HEADER"]
