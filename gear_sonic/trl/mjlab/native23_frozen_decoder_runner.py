"""Fresh-only, versioned PPO runner for a genuinely frozen SONIC mean decoder."""

from collections.abc import Mapping
import copy
import os
from pathlib import Path
import tempfile

import torch

from gear_sonic.trl.mjlab.frozen_platform_lora_runner import _state_sha256
from gear_sonic.trl.mjlab.native23_frozen_decoder_actor import TRAINABLE, True23FrozenDecoderActorModel
from gear_sonic.trl.mjlab.native23_generalist_runner import Native23GeneralistRunner
from gear_sonic.trl.mjlab.native23_root_feedback_runner import validate_optimizer_rates
from gear_sonic.trl.mjlab.runner import _require_nonnegative_integer
from gear_sonic.utils.g1_23dof_mjlab_training import (
    _validate_optimizer_state,
    validate_mjlab_training_lineage,
)

CHECKPOINT_HEADER = {
    "schema_version": 1,
    "kind": "g1_native23_frozen_decoder_original_intent_training_snapshot",
    "role": "simulator_research_fresh_run_snapshot_only",
    "deployment_ready": False,
    "promotion_eligible": False,
    "hardware_authorized": False,
}
PROFILE = "frozen_decoder_root_priority_v1"
MULTIPLIERS = {"root_conditioner": 200.0, "exploration": 1.0, "critic": 600.0}


def training_contract():
    return {
        "kind": PROFILE,
        "trainable": TRAINABLE,
        "encoder_and_decoder_frozen": True,
        "decoder_optimizer_group_present": False,
        "learning_rate_multipliers": dict(MULTIPLIERS),
        "initialization": "fresh_released_row_mapping_and_zero_root_conditioner",
        "resume_or_transfer_supported": False,
        "hardware_authorized": False,
        "deployment_ready": False,
    }


def validate_frozen_decoder_checkpoint(value, *, actor, lineage, minimum_update_count=0):
    if not isinstance(actor, True23FrozenDecoderActorModel):
        raise TypeError("frozen-decoder checkpoint requires its distinct actor")
    if not isinstance(value, Mapping) or value.get("header") != CHECKPOINT_HEADER:
        raise ValueError("frozen-decoder checkpoint header mismatch")
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
        raise ValueError("frozen-decoder checkpoint fields mismatch")
    actor.validate_training_artifact(value["actor"])
    actual = validate_mjlab_training_lineage(value["lineage"])
    expected = validate_mjlab_training_lineage(lineage)
    if (
        actual["lineage_sha256"] != expected["lineage_sha256"]
        or value["lineage_sha256"] != expected["lineage_sha256"]
    ):
        raise ValueError("frozen-decoder checkpoint lineage mismatch")
    resolved = expected["materials"]["resolved_config"]["payload"]
    if resolved.get("native23_frozen_decoder") != training_contract():
        raise ValueError("frozen-decoder training contract missing or changed")
    if value["critic_state_sha256"] != _state_sha256(value["critic_state_dict"]):
        raise ValueError("frozen-decoder critic state hash mismatch")
    if not isinstance(value["optimizer_state_dict"], Mapping):
        raise ValueError("frozen-decoder optimizer state missing")
    optimizer = _validate_optimizer_state(value["optimizer_state_dict"])
    if [g.get("name") for g in optimizer["param_groups"]] != list(MULTIPLIERS):
        raise ValueError("frozen-decoder optimizer groups mismatch")
    offset = 0
    for group, parameters in zip(optimizer["param_groups"], actor.parameter_groups().values()):
        if group["params"] != list(range(offset, offset + len(parameters))):
            raise ValueError("frozen-decoder optimizer parameter ownership mismatch")
        offset += len(parameters)
    critic_ids = optimizer["param_groups"][-1]["params"]
    if not critic_ids or critic_ids != list(range(offset, offset + len(critic_ids))):
        raise ValueError("frozen-decoder critic parameter ownership mismatch")
    ids = [p for group in optimizer["param_groups"] for p in group["params"]]
    if len(ids) != len(set(ids)) or set(optimizer["state"]) - set(ids):
        raise ValueError("frozen-decoder optimizer parameter ids mismatch")
    trainer = value["trainer_state"]
    if not isinstance(trainer, Mapping) or set(trainer) != {
        "completed_update_count",
        "current_learning_iteration",
        "env_common_step_counter",
        "algorithm_learning_rate",
    }:
        raise ValueError("frozen-decoder trainer state fields mismatch")
    count = _require_nonnegative_integer(trainer["completed_update_count"], "completed_update_count")
    current = _require_nonnegative_integer(trainer["current_learning_iteration"], "current_learning_iteration")
    _require_nonnegative_integer(trainer["env_common_step_counter"], "env_common_step_counter")
    if count != current or count < minimum_update_count:
        raise ValueError("frozen-decoder checkpoint counters mismatch")
    validate_optimizer_rates(optimizer["param_groups"], trainer["algorithm_learning_rate"], MULTIPLIERS)
    return value


class Native23FrozenDecoderRunner(Native23GeneralistRunner):
    """Reuse checked PPO counters/partitioning, but never optimize decoder weights."""

    def __init__(self, *args, **kwargs):
        self._frozen_group_rates_pending = True
        super().__init__(*args, **kwargs)
        resolved = self._training_lineage["materials"]["resolved_config"]["payload"]
        if resolved.get("native23_frozen_decoder") != training_contract():
            raise ValueError("frozen-decoder runner requires its bound training contract")
        if getattr(self.alg, "schedule", None) != "fixed":
            raise ValueError("frozen-decoder rates require fixed PPO schedule")
        for group in self.alg.optimizer.param_groups:
            group["lr"] = self.alg.learning_rate * MULTIPLIERS[group["name"]]
        self._frozen_group_rates_pending = False
        self._assert_boundary()

    def _validate_initial_policy(self):
        actor = self.alg.get_policy()
        if not isinstance(actor, True23FrozenDecoderActorModel) or not actor.initial_conditioner_is_zero():
            raise ValueError("frozen-decoder runner requires fresh zero-conditioner actor")
        super()._validate_initial_policy()

    def _algorithm_learning_rate(self):
        if getattr(self, "_frozen_group_rates_pending", False):
            return super()._algorithm_learning_rate()
        return validate_optimizer_rates(self.alg.optimizer.param_groups, self.alg.learning_rate, MULTIPLIERS)

    def _numbered_checkpoint_path(self, update_count):
        count = _require_nonnegative_integer(update_count, "frozen-decoder update_count")
        return self.checkpoint_dir / f"frozen_decoder_model_{count}.pt"

    def _checkpoint_payload(self):
        payload = super()._checkpoint_payload()
        payload["header"] = copy.deepcopy(CHECKPOINT_HEADER)
        return payload

    def save(self, path, infos=None):
        if infos is not None:
            raise ValueError("frozen-decoder snapshots forbid stock RSL infos")
        self._require_checkpointable()
        output = Path(path).expanduser()
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"refusing to overwrite frozen-decoder snapshot: {output}")
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
            validate_frozen_decoder_checkpoint(loaded, actor=self.alg.get_policy(), lineage=self._training_lineage)
            with temporary.open("rb+") as stream:
                os.fsync(stream.fileno())
            os.link(temporary, output)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
        self._last_checkpoint_path = output
        self._last_checkpoint_update_count = self._require_counter_coherence()

    def load(self, *args, **kwargs):
        del args, kwargs
        raise ValueError("frozen-decoder experiments are fresh-only; resume is not supported")

    def initialize_evaluated_continuation(self, contract):
        del contract
        raise ValueError("frozen-decoder experiments cannot continue a previous actor")
