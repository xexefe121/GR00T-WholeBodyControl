"""Fresh, checked actor-initialized IEEE LoRA PPO with a reset-only curriculum.

Keep the original standing output anchor, all motion sources, action/sensor
noise and actuation profile. Existing checkpoints and trainers are immutable.
This launcher intentionally rejects resume rather than misrepresenting reset
state restoration. Export/evaluation still require actual new PPO updates.
"""

import json
from pathlib import Path
import sys

import torch

from gear_sonic.scripts import train_g1_true23_ieee_standing_retention as ieee
from gear_sonic.scripts import train_g1_true23_standing_retention as retention
from gear_sonic.scripts.record_g1_sonic_original29_baseline import dump
from gear_sonic.trl.mjlab.frozen_platform_lora_runner import _state_sha256, load_frozen_platform_lora_checkpoint
from gear_sonic.utils.g1_sonic_cpp_parameters import file_sha256
from gear_sonic.utils.g1_true23_reset_curriculum import (
    reset_curriculum_contract,
    resample_with_curriculum,
    summarize_reset_events,
)
from gear_sonic.utils.g1_true23_training_precision import ieee_training_precision


def install_hooks(checkpoint_path, checkpoint_hash, contract):
    standing, frozen = retention.standing, retention.standing.frozen
    original_standing = standing.install_standing_hooks

    def with_curriculum(payload, descriptor, *, initialization_mode, run_directory):
        if not initialization_mode:
            raise ValueError("reset curriculum requires a fresh actor-only initialization")
        original_standing(payload, descriptor, initialization_mode=True, run_directory=run_directory)
        parent = frozen.FrozenPlatformLoraRunner
        prior = load_frozen_platform_lora_checkpoint(
            checkpoint_path, expected_contract=descriptor["frozen_platform_contract"]
        )
        if file_sha256(checkpoint_path) != checkpoint_hash:
            raise ValueError("curriculum initialization checkpoint changed")
        actor_initialization = dict(
            path=str(checkpoint_path),
            sha256=checkpoint_hash,
            prior_lineage_sha256=prior["lineage_sha256"],
            prior_update_count=prior["update_count"],
            adapter_state_sha256=prior["adapter_state_sha256"],
            merged_policy_sha256=prior["merged_true23_policy_sha256"],
            critic_reused=False,
            optimizer_reused=False,
            prior_updates_counted_as_new=False,
            source_candidate_qualified=False,
            hardware_authorized=False,
            deployment_ready=False,
        )

        class ResetCurriculumRunner(parent):
            def load(self, path, *args, **kwargs):
                # Existing checked standing bootstrap creates fresh PPO state.
                # Its adapter is then replaced explicitly by this checked actor.
                receipt = super().load(path, *args, **kwargs)
                if self._require_counter_coherence() != 0 or self.alg.optimizer.state:
                    raise ValueError("curriculum actor import requires fresh zero-update PPO state")
                actor = self.alg.get_policy()
                critic_before = _state_sha256(self.alg.critic.state_dict())
                actor.core.load_lora_state_dict(prior["adapter_state_dict"], strict=True)
                if (
                    actor.core.merged_true23_policy_sha256(actor.distribution.std_param)
                    != prior["merged_true23_policy_sha256"]
                ):
                    raise ValueError("curriculum actor import differs from its checked current policy")
                if _state_sha256(self.alg.critic.state_dict()) != critic_before or self.alg.optimizer.state:
                    raise ValueError("curriculum actor import changed fresh critic or optimizer")
                self._assert_boundary()
                dump(
                    run_directory / "curriculum_actor_initialization.json",
                    {
                        **actor_initialization,
                        "new_update_count": 0,
                        "fresh_critic_sha256": critic_before,
                        "standing_bootstrap_receipt_is_intermediate": True,
                    },
                )
                return {**receipt, "subsequent_curriculum_actor_initialization": actor_initialization}

            def save(self, path, infos=None):
                if not torch.equal(
                    self.alg.get_policy().distribution.std_param.detach().cpu(),
                    self.alg.get_policy().core.initial_std,
                ):
                    raise ValueError("reset curriculum changed the frozen action-noise distribution")
                super().save(path, infos)
                command = self.env.unwrapped.command_manager.get_term("motion")
                events = command._reset_curriculum_events
                update = self._require_counter_coherence()
                dump(
                    run_directory / f"reset_curriculum_{update}.json",
                    dict(
                        contract=contract,
                        actor_initialization=actor_initialization,
                        new_update_count=update,
                        actual_environment_control=int(self.env.unwrapped.common_step_counter),
                        summary=summarize_reset_events(events, contract),
                        events=events,
                        action_std_unchanged=torch.equal(
                            self.alg.get_policy().distribution.std_param.detach().cpu(),
                            self.alg.get_policy().core.initial_std,
                        ),
                        hardware_authorized=False,
                        deployment_ready=False,
                    ),
                )

        frozen.FrozenPlatformLoraRunner = ResetCurriculumRunner
        original_install = frozen._install_frozen_lora_hooks

        def checked_install(**kwargs):
            original_install(**kwargs)
            from gear_sonic.envs.mjlab import sonic_true23_causal_history as task

            original_resample = task.CausalHistoryMotionCommand._resample_command

            def resample(command, env_ids):
                return resample_with_curriculum(command, env_ids, original_resample, contract)

            task.CausalHistoryMotionCommand._resample_command = resample

        frozen._install_frozen_lora_hooks = checked_install
        original_resolved = frozen.base._resolved_training_config

        def resolved(**kwargs):
            return {
                **original_resolved(**kwargs),
                "reset_curriculum": contract,
                "curriculum_actor_initialization": actor_initialization,
            }

        frozen.base._resolved_training_config = resolved
        root = Path(__file__).resolve().parents[2]
        frozen.base.CAUSAL_SOURCE_FILES += (
            Path(__file__).resolve(),
            root / "gear_sonic/utils/g1_true23_reset_curriculum.py",
            root / "gear_sonic/utils/g1_true23_contact_geometry.py",
        )

    standing.install_standing_hooks = with_curriculum


def main(argv=None):
    values = []
    for value in sys.argv[1:] if argv is None else argv:
        values.extend(value.split("=", 1) if value.startswith("--") and "=" in value else [value])
    frozen = retention.standing.frozen
    if retention.standing.option(values, "--resume") is not None:
        raise ValueError("reset curriculum is fresh-only; no partial reset-state resume claim")
    checkpoint_arg = frozen._pop_option(values, "--curriculum-actor-checkpoint")
    expected = frozen._pop_option(values, "--expected-curriculum-actor-sha256")
    if checkpoint_arg is None or expected is None:
        raise ValueError("reset curriculum requires a hash-bound current actor checkpoint")
    checkpoint = Path(checkpoint_arg).expanduser()
    if checkpoint.is_symlink():
        raise ValueError("curriculum actor checkpoint may not be a symlink")
    checkpoint = checkpoint.resolve(strict=True)
    if file_sha256(checkpoint) != expected:
        raise ValueError("curriculum actor checkpoint hash mismatch")
    contract = reset_curriculum_contract(
        int(frozen._pop_option(values, "--reset-warmup-controls", default="1600")),
        int(frozen._pop_option(values, "--reset-ramp-end-controls", default="6400")),
    )
    with ieee_training_precision() as (precision, guard):
        ieee.install_hooks(precision, guard)
        install_hooks(checkpoint, expected, contract)
        return retention.main(values)


if __name__ == "__main__":
    raise SystemExit(main())
