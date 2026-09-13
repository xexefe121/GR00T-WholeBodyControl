"""Fresh original-source SIM training with a frozen SONIC encoder and decoder.

This entry point must run in its own process, like its source-aware predecessor.
It changes the actor/optimizer contract, not references, rewards or physics.
"""

from pathlib import Path

from gear_sonic.scripts import train_g1_true23_original_intent as intent
from gear_sonic.trl.mjlab.native23_frozen_decoder_actor import TRAINABLE
from gear_sonic.trl.mjlab.native23_frozen_decoder_runner import (
    MULTIPLIERS,
    PROFILE,
    Native23FrozenDecoderRunner,
    training_contract,
)
from gear_sonic.utils.g1_true23_generalist_source_closure import collect_local_source_closure


def validate_frozen_recipe(args):
    intent.validate_fresh_recipe(args)
    if args.optimizer_profile != "feedback_priority" or args.ppo_auxiliary_objective != "none":
        raise ValueError("frozen-decoder recipe requires fixed root-priority rates and no PPO auxiliary loss")
    if not 1 <= args.iterations <= 100 or not 1 <= args.session_updates <= 100:
        raise ValueError("frozen-decoder fresh experiment is capped at100 updates")


def main(argv=None):
    legacy = intent.legacy
    previous_main = legacy.main
    previous_feedback = legacy.feedback_training_contract

    def feedback(args, curriculum):
        result = previous_feedback(args, curriculum)
        result.update(
            trainable=TRAINABLE,
            optimizer_profile=PROFILE,
            optimizer_learning_rate_multipliers=dict(MULTIPLIERS),
            checkpoint_pattern="frozen_decoder_model_N.pt",
            frozen_decoder_training=training_contract(),
        )
        return result

    def dispatch(remaining):
        # The original-intent launcher has now installed its own complete
        # source/critic/reward hooks. Wrap their result, preserving that work.
        intent_hooks = legacy.install_hooks

        def install_hooks(args, inputs, curriculum, precision, guard):
            from gear_sonic.trl.mjlab import config, native23_root_feedback_runner

            validate_frozen_recipe(args)
            old_runner = native23_root_feedback_runner.Native23RootFeedbackRunner
            native23_root_feedback_runner.Native23RootFeedbackRunner = Native23FrozenDecoderRunner
            try:
                base = intent_hooks(args, inputs, curriculum, precision, guard)
            finally:
                native23_root_feedback_runner.Native23RootFeedbackRunner = old_runner
            previous_agent = config.true23_mjlab_ppo_runner_cfg

            def agent():
                cfg = previous_agent()
                cfg.actor.class_name = (
                    "gear_sonic.trl.mjlab.native23_frozen_decoder_actor:True23FrozenDecoderActorModel"
                )
                cfg.experiment_name = "sonic_native23_frozen_decoder_original_intent_simulation"
                return cfg

            config.true23_mjlab_ppo_runner_cfg = agent
            previous_resolved = base._resolved_training_config

            def resolved(*values, **kwargs):
                result = previous_resolved(*values, **kwargs)
                result["native23_frozen_decoder"] = training_contract()
                result["native23_generalist"].update(
                    trainable=TRAINABLE,
                    decoder_frozen=True,
                    checkpoint_pattern="frozen_decoder_model_N.pt",
                )
                return result

            base._resolved_training_config = resolved
            previous_preflight = base.preflight

            def preflight(values):
                result = previous_preflight(values)
                if "native23_generalist" in result:
                    result["pretrained_base_initialization_preflight"] = result.pop("native23_generalist")
                result["native23_frozen_decoder"] = training_contract()
                result["actual_frozen_actor_and_partition_verification_required_at_runner_construction"] = True
                return result

            base.preflight = preflight
            previous_sources = base._source_files

            def sources():
                result = previous_sources()
                roots = [Path(__file__)] + [
                    legacy.generalist.ROOT / name
                    for name in (
                        "gear_sonic/trl/mjlab/native23_frozen_decoder_actor.py",
                        "gear_sonic/trl/mjlab/native23_frozen_decoder_runner.py",
                    )
                ]
                result.update(
                    collect_local_source_closure(legacy.generalist.ROOT, roots).as_source_files(
                        legacy.generalist.ROOT
                    )
                )
                return result

            base._source_files = sources
            return base

        legacy.install_hooks = install_hooks
        return previous_main(remaining)

    legacy.feedback_training_contract = feedback
    legacy.main = dispatch
    try:
        return intent.main(argv)
    finally:
        legacy.main = previous_main
        legacy.feedback_training_contract = previous_feedback


if __name__ == "__main__":
    raise SystemExit(main())
