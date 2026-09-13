"""Versioned learner-state continuation with new foot objective, SIM only.

Restores actor, critic, Adam state and counters. Creates fresh environment/RNG
state explicitly; this is not an exact uninterrupted simulator resume.
"""

import copy
import os
from pathlib import Path
import tempfile

import torch

from gear_sonic.envs.mjlab.sonic_true23_bounded_progress import install_progress_step, verify_runtime
from gear_sonic.trl.mjlab.frozen_platform_lora_runner import _state_sha256
from gear_sonic.trl.mjlab.native23_generalist_runner import Native23GeneralistRunner
from gear_sonic.trl.mjlab.native23_normal_lora_actor import True23NormalLoraActorModel
from gear_sonic.trl.mjlab.native23_normal_lora_runner import (
    MULTIPLIERS,
    normal_training_contract,
    validate_normal_checkpoint,
)
from gear_sonic.trl.mjlab.native23_root_feedback_runner import validate_optimizer_rates
from gear_sonic.trl.mjlab.runner import _require_nonnegative_integer
from gear_sonic.utils.g1_23dof_mjlab_training import _validate_optimizer_state, validate_mjlab_training_lineage
from gear_sonic.utils.g1_true23_foot_precision_reward import FootPrecisionStep, foot_precision_contract
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file
from gear_sonic.utils.g1_true23_normal_reference import NORMAL_TIMING
from gear_sonic.utils.g1_true23_world_quality import WorldQualityStep

HEADER = dict(
    schema_version=1,
    kind="native23_pico_foot_precision_learner_continuation_v1",
    role="simulation_training_snapshot_not_exact_environment_resume",
    deployment_ready=False,
    promotion_eligible=False,
    hardware_authorized=False,
)
PARENT_SHA256 = "047b7387f8268d2ff8032c8b74764de6a3f0321f4245814c0f5a909e805dc065"


def training_contract():
    return dict(
        kind="native23_pico_foot_precision_training_v1",
        prior_recipe=normal_training_contract(),
        overlay=foot_precision_contract(),
        parent_checkpoint_sha256=PARENT_SHA256,
        parent_completed_updates=500,
        actor_critic_and_adam_preserved=True,
        environment_and_rng_reinitialized=True,
        uninterrupted_simulator_resume_claimed=False,
        fresh_actor_or_critic=False,
        optimizer_multipliers=dict(MULTIPLIERS),
        policy_architecture_unchanged=True,
        source_encoder_and_decoder_frozen=True,
        deployment_ready=False,
        hardware_authorized=False,
    )


def equal_state(left, right):
    """Device-independent exact tensor/state equality, including Adam moments."""
    if isinstance(left, torch.Tensor):
        return (
            isinstance(right, torch.Tensor)
            and left.dtype == right.dtype
            and left.shape == right.shape
            and torch.equal(left.detach().cpu(), right.detach().cpu())
        )
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(equal_state(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(equal_state(a, b) for a, b in zip(left, right, strict=True))
    return left == right


def validate_checkpoint(value, *, actor, lineage):
    if type(actor) is not True23NormalLoraActorModel or value.get("header") != HEADER:
        raise ValueError("foot precision checkpoint has wrong actor/header")
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
        raise ValueError("foot precision checkpoint fields differ")
    actor.validate_training_artifact(value["actor"])
    actual, expected = (validate_mjlab_training_lineage(x) for x in (value["lineage"], lineage))
    if actual != expected or value["lineage_sha256"] != expected["lineage_sha256"]:
        raise ValueError("foot precision checkpoint lineage differs")
    if expected["materials"]["resolved_config"]["payload"].get("foot_precision_adaptation") != training_contract():
        raise ValueError("foot precision objective/continuation contract differs")
    if value["critic_state_sha256"] != _state_sha256(value["critic_state_dict"]):
        raise ValueError("foot precision critic hash differs")
    optimizer = _validate_optimizer_state(value["optimizer_state_dict"])
    if [g.get("name") for g in optimizer["param_groups"]] != list(MULTIPLIERS):
        raise ValueError("foot precision optimizer groups differ")
    offset = 0
    for group, parameters in zip(optimizer["param_groups"], actor.parameter_groups().values()):
        if group["params"] != list(range(offset, offset + len(parameters))):
            raise ValueError("foot precision actor optimizer ownership differs")
        offset += len(parameters)
    ids = optimizer["param_groups"][-1]["params"]
    if not ids or ids != list(range(offset, offset + len(ids))):
        raise ValueError("foot precision critic optimizer ownership differs")
    all_ids = [p for g in optimizer["param_groups"] for p in g["params"]]
    if len(set(all_ids)) != len(all_ids) or set(optimizer["state"]) - set(all_ids):
        raise ValueError("foot precision Adam state ownership differs")
    trainer = value["trainer_state"]
    if set(trainer) != {
        "completed_update_count",
        "current_learning_iteration",
        "env_common_step_counter",
        "algorithm_learning_rate",
    }:
        raise ValueError("foot precision trainer state fields differ")
    for name in ("completed_update_count", "current_learning_iteration", "env_common_step_counter"):
        _require_nonnegative_integer(trainer[name], name)
    if (
        trainer["completed_update_count"] < 500
        or trainer["current_learning_iteration"] != trainer["completed_update_count"]
    ):
        raise ValueError("foot precision counters predate parent or disagree")
    validate_optimizer_rates(optimizer["param_groups"], trainer["algorithm_learning_rate"], MULTIPLIERS)
    return value


class Native23PicoFootPrecisionRunner(Native23GeneralistRunner):
    @property
    def semantic_profile(self):
        return NORMAL_TIMING

    def __init__(self, *args, **kwargs):
        self._rates_pending = True
        self._parent_imported = False
        super().__init__(*args, **kwargs)
        resolved = self.training_lineage["materials"]["resolved_config"]["payload"]
        if resolved.get("foot_precision_adaptation") != training_contract():
            raise ValueError("foot precision runner requires explicit objective and parent")
        if self.alg.schedule != "fixed" or self.alg.use_clipped_value_loss or self.alg.clip_param != 0.2:
            raise ValueError("foot precision requires unchanged PPO setup")
        for group in self.alg.optimizer.param_groups:
            group["lr"] = self.alg.learning_rate * MULTIPLIERS[group["name"]]
        self._rates_pending = False
        self._assert_boundary()
        original = install_progress_step(self.env.unwrapped, resolved["original_intent_spec"])
        self._reward_step = FootPrecisionStep(WorldQualityStep(original))
        self.env.unwrapped.step = self._reward_step
        self.env.unwrapped._world_root_failure_capture = []
        previous_process = self.alg.process_env_step

        def capture_process(obs, rewards, dones, extras):
            row = self._reward_step.rows[-1]
            if row["ppo_recorded"].any():
                raise ValueError("foot precision PPO reward stored twice")
            value = self.alg.transition.values.detach().cpu().clone().squeeze(-1)
            index = self.alg.storage.step
            previous_process(obs, rewards, dones, extras)
            if self.alg.storage.step != index + 1:
                raise ValueError("foot precision storage did not advance once")
            row["critic_value_before_bootstrap"] = value
            row["stored_reward_with_timeout_bootstrap"] = (
                self.alg.storage.rewards[index].detach().cpu().clone().squeeze(-1)
            )
            row["stored_done"] = self.alg.storage.dones[index].detach().cpu().clone().squeeze(-1).bool()
            row["ppo_recorded"][:] = True

        self.alg.process_env_step = capture_process
        verify_runtime(self.env.unwrapped)

    def _validate_initial_policy(self):
        actor = self.alg.get_policy()
        if type(actor) is not True23NormalLoraActorModel or not actor.initial_conditioner_is_zero():
            raise ValueError("foot precision constructs known source actor before parent import")
        super()._validate_initial_policy()

    def _algorithm_learning_rate(self):
        if getattr(self, "_rates_pending", False):
            return super()._algorithm_learning_rate()
        return validate_optimizer_rates(self.alg.optimizer.param_groups, self.alg.learning_rate, MULTIPLIERS)

    def import_parent(self, path):
        self._require_checkpointable()
        if self._parent_imported or self.completed_update_count != 0 or self.alg.storage.step != 0:
            raise ValueError("parent import requires a fresh empty runner, once only")
        path = Path(path).resolve(strict=True)
        if sha256_file(path) != PARENT_SHA256:
            raise ValueError("foot precision parent differs from evaluated500 checkpoint")
        parent = torch.load(path, map_location="cpu", weights_only=True)
        validate_normal_checkpoint(parent, actor=self.alg.get_policy(), lineage=parent["lineage"])
        for kind in ("robot_assets", "motion_dataset"):
            if parent["lineage"]["materials"][kind] != self.training_lineage["materials"][kind]:
                raise ValueError("foot precision changes parent assets or reference data")
        if parent["trainer_state"]["completed_update_count"] != 500:
            raise ValueError("foot precision parent must have500 completed updates")
        self._training_state_poisoned = True
        # Failure deliberately leaves this runner poisoned; never silently
        # downgrade to actor-only initialization or fresh Adam/critic state.
        self._restore_payload(parent)
        actual = super()._checkpoint_payload()
        for key in ("actor", "critic_state_dict", "optimizer_state_dict", "trainer_state"):
            if not equal_state(parent[key], actual[key]):
                raise ValueError("imported parent learner state differs: " + key)
        self._parent_imported = True
        self._training_state_poisoned = False
        return dict(
            parent_checkpoint_sha256=PARENT_SHA256,
            exact_learner_state=True,
            imported_updates=500,
            imported_env_control_counter=parent["trainer_state"]["env_common_step_counter"],
            simulator_and_rng_state_reinitialized=True,
            exact_simulation_resume=False,
        )

    def _numbered_checkpoint_path(self, update_count):
        count = _require_nonnegative_integer(update_count, "foot precision update count")
        return self.checkpoint_dir / f"foot_precision_model_{count}.pt"

    def _checkpoint_payload(self):
        if not self._parent_imported:
            raise ValueError("foot precision snapshot requires verified parent import")
        value = super()._checkpoint_payload()
        value["header"] = copy.deepcopy(HEADER)
        return value

    def save(self, path, infos=None):
        if infos is not None:
            raise ValueError("foot precision snapshots forbid stock RSL infos")
        self._require_checkpointable()
        output = Path(path).expanduser()
        if output.exists() or output.is_symlink():
            raise FileExistsError("foot precision snapshot refuses overwrite")
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

    def load(self, *args, **kwargs):
        raise ValueError("use explicit parent import; foot precision exact environment resume is unsupported")
