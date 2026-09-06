"""Separate IEEE motion-PPO experiment; original TF32 runs remain immutable."""

from pathlib import Path

from gear_sonic.scripts import train_g1_true23_standing_retention as retention
from gear_sonic.utils.g1_true23_training_precision import (
    backend_state,
    guard_algorithm,
    ieee_training_precision,
    write_runtime,
)


def install_hooks(contract, guard):
    frozen = retention.standing.frozen
    original_install = frozen._install_frozen_lora_hooks

    def with_precision(**kwargs):
        # Called after the standing and retention wrappers construct their
        # runner, before the frozen launcher publishes it to the base trainer.
        parent = frozen.FrozenPlatformLoraRunner

        class IeeeStandingRetentionRunner(parent):
            def __init__(self, *args, **kwargs):
                guard()
                super().__init__(*args, **kwargs)
                guard_algorithm(self.alg, guard)
                write_runtime(
                    self.checkpoint_dir.parent / "training_precision.json",
                    dict(
                        contract=contract,
                        actual_state=backend_state(),
                        guarded_boundaries=["construction", "actor", "critic", "ppo_update", "load", "save"],
                        hardware_authorized=False,
                        deployment_ready=False,
                    ),
                )

            def load(self, *args, **kwargs):
                guard()
                result = super().load(*args, **kwargs)
                guard()
                return result

            def _checkpoint_payload(self):
                guard()
                return super()._checkpoint_payload()

        frozen.FrozenPlatformLoraRunner = IeeeStandingRetentionRunner
        original_install(**kwargs)
        original_resolved = frozen.base._resolved_training_config

        def resolved(*args, **kwargs):
            guard()
            return {**original_resolved(*args, **kwargs), "training_precision": contract}

        frozen.base._resolved_training_config = resolved
        frozen.base.CAUSAL_SOURCE_FILES += (
            Path(__file__).resolve(),
            Path(__file__).resolve().parents[1] / "utils/g1_true23_training_precision.py",
        )

    frozen._install_frozen_lora_hooks = with_precision


def main(argv=None):
    with ieee_training_precision() as (contract, guard):
        install_hooks(contract, guard)
        return retention.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
