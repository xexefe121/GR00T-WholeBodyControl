"""Fresh, separately versioned normal-core native23 simulator training."""

import copy
import os
from pathlib import Path
import tempfile

import torch

from gear_sonic.envs.mjlab.sonic_true23_bounded_progress import install_progress_step, verify_runtime
from gear_sonic.envs.mjlab.sonic_true23_world_tracking_termination import termination_contract
from gear_sonic.trl.mjlab.frozen_platform_lora_runner import _state_sha256
from gear_sonic.trl.mjlab.native23_generalist_runner import Native23GeneralistRunner
from gear_sonic.trl.mjlab.native23_normal_lora_actor import True23NormalLoraActorModel, normal_adapter_contract
from gear_sonic.trl.mjlab.native23_root_feedback_runner import validate_optimizer_rates
from gear_sonic.trl.mjlab.runner import _require_nonnegative_integer
from gear_sonic.utils.g1_23dof_mjlab_training import _validate_optimizer_state, validate_mjlab_training_lineage
from gear_sonic.utils.g1_true23_bounded_progress import reward_contract
from gear_sonic.utils.g1_true23_normal_reference import NORMAL_TIMING, normal_reference_contract
from gear_sonic.utils.g1_true23_world_quality import WorldQualityStep, quality_contract

HEADER = dict(
    schema_version=1,
    kind="g1_native23_normal_core_lora_root9_training_snapshot_v1",
    role="simulator_fresh_run_snapshot_only",
    deployment_ready=False,
    promotion_eligible=False,
    hardware_authorized=False,
)
MULTIPLIERS = {"root_conditioner": 200.0, "decoder_adapters": 200.0, "exploration": 1.0, "critic": 600.0}


def normal_training_contract():
    return dict(
        kind="native23_original_core_lora_root9_world_quality_training_v1",
        reference=normal_reference_contract(),
        adapters=normal_adapter_contract(),
        bounded_reward=reward_contract(),
        quality=quality_contract(),
        termination=termination_contract(),
        optimizer_multipliers=dict(MULTIPLIERS),
        actor_clip=0.2,
        critic_value_clipped=False,
        same_low_latency_checkpoint_or_optimizer=False,
        fresh_only=True,
        hardware_authorized=False,
        deployment_ready=False,
    )


def validate_normal_checkpoint(value, *, actor, lineage):
    if type(actor) is not True23NormalLoraActorModel or value.get("header") != HEADER:
        raise ValueError("normal checkpoint requires its exact actor and distinct header")
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
        raise ValueError("normal checkpoint fields differ")
    actor.validate_training_artifact(value["actor"])
    actual, expected = (validate_mjlab_training_lineage(item) for item in (value["lineage"], lineage))
    if actual != expected or value["lineage_sha256"] != expected["lineage_sha256"]:
        raise ValueError("normal checkpoint lineage differs")
    if expected["materials"]["resolved_config"]["payload"].get("normal_adaptation") != normal_training_contract():
        raise ValueError("normal training contract missing or changed")
    if value["critic_state_sha256"] != _state_sha256(value["critic_state_dict"]):
        raise ValueError("normal checkpoint critic hash differs")
    optimizer = _validate_optimizer_state(value["optimizer_state_dict"])
    if [group.get("name") for group in optimizer["param_groups"]] != list(MULTIPLIERS):
        raise ValueError("normal optimizer groups differ")
    offset = 0
    for group, parameters in zip(optimizer["param_groups"], actor.parameter_groups().values()):
        if group["params"] != list(range(offset, offset + len(parameters))):
            raise ValueError("normal optimizer parameter ownership differs")
        offset += len(parameters)
    critic_ids = optimizer["param_groups"][-1]["params"]
    if not critic_ids or critic_ids != list(range(offset, offset + len(critic_ids))):
        raise ValueError("normal optimizer critic ownership differs")
    ids = [p for group in optimizer["param_groups"] for p in group["params"]]
    if len(set(ids)) != len(ids) or set(optimizer["state"]) - set(ids):
        raise ValueError("normal optimizer state ids differ")
    trainer = value["trainer_state"]
    if set(trainer) != {
        "completed_update_count",
        "current_learning_iteration",
        "env_common_step_counter",
        "algorithm_learning_rate",
    }:
        raise ValueError("normal trainer state fields differ")
    counters = {
        name: _require_nonnegative_integer(trainer[name], name)
        for name in ("completed_update_count", "current_learning_iteration", "env_common_step_counter")
    }
    if counters["completed_update_count"] != counters["current_learning_iteration"]:
        raise ValueError("normal checkpoint update counters differ")
    validate_optimizer_rates(optimizer["param_groups"], trainer["algorithm_learning_rate"], MULTIPLIERS)
    return value


class Native23NormalLoraRunner(Native23GeneralistRunner):
    @property
    def semantic_profile(self):
        return NORMAL_TIMING

    def __init__(self, *args, **kwargs):
        self._normal_rates_pending = True
        super().__init__(*args, **kwargs)
        resolved = self.training_lineage["materials"]["resolved_config"]["payload"]
        if resolved.get("normal_adaptation") != normal_training_contract():
            raise ValueError("normal runner requires its explicitly bound training recipe")
        if self.alg.schedule != "fixed" or self.alg.use_clipped_value_loss or self.alg.clip_param != 0.2:
            raise ValueError("normal runner requires fixed rates and unchanged actor/unclipped critic setup")
        for group in self.alg.optimizer.param_groups:
            group["lr"] = self.alg.learning_rate * MULTIPLIERS[group["name"]]
        self._normal_rates_pending = False
        self._assert_boundary()
        original_step = install_progress_step(self.env.unwrapped, resolved["original_intent_spec"])
        self._reward_step = WorldQualityStep(original_step)
        self.env.unwrapped.step = self._reward_step
        self.env.unwrapped._world_root_failure_capture = []
        previous_process = self.alg.process_env_step

        def capture_process(obs, rewards, dones, extras):
            row = self._reward_step.rows[-1]
            if row["ppo_recorded"].any():
                raise ValueError("normal PPO attempted to record one reward twice")
            value = self.alg.transition.values.detach().cpu().clone().squeeze(-1)
            index = self.alg.storage.step
            previous_process(obs, rewards, dones, extras)
            if self.alg.storage.step != index + 1:
                raise ValueError("normal PPO storage did not advance once")
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
            raise ValueError("normal runner requires fresh zero-effect normal-core adaptation")
        super()._validate_initial_policy()

    def _algorithm_learning_rate(self):
        if getattr(self, "_normal_rates_pending", False):
            return super()._algorithm_learning_rate()
        return validate_optimizer_rates(self.alg.optimizer.param_groups, self.alg.learning_rate, MULTIPLIERS)

    def _numbered_checkpoint_path(self, update_count):
        count = _require_nonnegative_integer(update_count, "normal update count")
        return self.checkpoint_dir / f"normal_lora_model_{count}.pt"

    def _checkpoint_payload(self):
        value = super()._checkpoint_payload()
        value["header"] = copy.deepcopy(HEADER)
        return value

    def save(self, path, infos=None):
        if infos is not None:
            raise ValueError("normal snapshots forbid stock RSL infos")
        self._require_checkpointable()
        output = Path(path).expanduser()
        if output.exists() or output.is_symlink():
            raise FileExistsError("normal snapshot refuses overwrite")
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
            validate_normal_checkpoint(loaded, actor=self.alg.get_policy(), lineage=self.training_lineage)
            with temporary.open("rb+") as stream:
                os.fsync(stream.fileno())
            os.link(temporary, output)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
        self._last_checkpoint_path = output
        self._last_checkpoint_update_count = self._require_counter_coherence()

    def load(self, *args, **kwargs):
        raise ValueError("normal adaptation is fresh-only; evaluated continuation not yet implemented")
