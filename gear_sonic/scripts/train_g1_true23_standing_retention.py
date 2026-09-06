"""Motion PPO with checked standing-only output retention, no hardware path."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys

from gear_sonic.scripts import train_g1_true23_standing_warm_start as standing
from gear_sonic.trl.mjlab.standing_retention_ppo import install, optimizer_step
from gear_sonic.utils.g1_true23_standing_retention import StandingOutputAnchor, load_training_states


def install_hooks(data, inputs, *, weight, batch_size, seed):
    original_install = standing.install_standing_hooks

    def with_retention(payload, descriptor, *, initialization_mode, run_directory):
        original_install(payload, descriptor, initialization_mode=initialization_mode, run_directory=run_directory)
        parent = standing.frozen.FrozenPlatformLoraRunner
        loss_contract = dict(
            kind="g1_true23_standing_output_retention_ppo_v1",
            inputs=inputs,
            weight=weight,
            batch_size=batch_size,
            seed=seed,
            anchor_adapter_sha256=descriptor["adapter_state_sha256"],
            target="checked_standing_adapter_outputs_on_three_training_episodes",
            loss="mean_scaled_safe_target_squared_error_plus_0p01_raw_squared_error",
            sampling="cpu_generator_seed_plus_actual_adam_minibatch_step",
            held_out_training_rows=0,
            joint_loss_before_gradient_clipping=True,
            full_motion_teacher_accepted=False,
            original_motion_sampler_retained=True,
            hardware_authorized=False,
            deployment_ready=False,
        )

        class StandingRetentionRunner(parent):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                anchor = StandingOutputAnchor(
                    self.alg.get_policy().core, data, payload, batch_size=batch_size, seed=seed
                )
                runtime = install(self.alg, anchor, weight=weight)
                document = {**loss_contract, "anchor_cache_sha256": anchor.cache_sha256, "runtime": runtime}
                path = run_directory / "standing_retention.json"
                encoded = json.dumps(document, indent=2, sort_keys=True) + "\n"
                if path.exists():
                    if path.read_text() != encoded:
                        raise ValueError("standing retention runtime/anchor changed on resume")
                else:
                    with path.open("x") as stream:
                        stream.write(encoded)

            def load(self, path, *args, **kwargs):
                result = super().load(path, *args, **kwargs)
                expected = (
                    self._require_counter_coherence() * self.alg.num_learning_epochs * self.alg.num_mini_batches
                )
                if optimizer_step(self.alg.optimizer) != expected:
                    raise ValueError("standing retention resume has incoherent PPO/Adam counters")
                self.alg.standing_minibatch_step = expected
                return result

            def _checkpoint_payload(self):
                expected = (
                    self._require_counter_coherence() * self.alg.num_learning_epochs * self.alg.num_mini_batches
                )
                if self.alg.standing_minibatch_step != expected or optimizer_step(self.alg.optimizer) != expected:
                    raise ValueError("standing retention checkpoint lost exact minibatch accounting")
                return super()._checkpoint_payload()

        standing.frozen.FrozenPlatformLoraRunner = StandingRetentionRunner
        original_resolved = standing.frozen.base._resolved_training_config

        def resolved(**kwargs):
            return {**original_resolved(**kwargs), "standing_output_retention": loss_contract}

        standing.frozen.base._resolved_training_config = resolved
        root = Path(__file__).resolve().parents[2]
        standing.frozen.base.CAUSAL_SOURCE_FILES += (
            Path(__file__).resolve(),
            root / "gear_sonic/utils/g1_true23_standing_retention.py",
            root / "gear_sonic/trl/mjlab/standing_retention_ppo.py",
        )

    standing.install_standing_hooks = with_retention


def main(argv=None):
    values = []
    for value in sys.argv[1:] if argv is None else argv:
        values.extend(value.split("=", 1) if value.startswith("--") and "=" in value else [value])
    teacher = standing.frozen._pop_option(values, "--standing-teacher-report")
    weight = float(standing.frozen._pop_option(values, "--standing-retention-weight", default="10"))
    batch_size = int(standing.frozen._pop_option(values, "--standing-retention-batch-size", default="128"))
    seed = int(standing.option(values, "--seed", "20260906"))
    bootstrap = standing.option(values, "--standing-bootstrap-report")
    if teacher is None or bootstrap is None:
        raise SystemExit("requires --standing-teacher-report and --standing-bootstrap-report")
    if (
        not math.isfinite(weight)
        or not 0 <= weight <= 1000
        or not 1 <= batch_size <= 1500
        or not 0 <= seed < 2**63
    ):
        raise SystemExit("invalid standing retention weight, batch size or seed")
    _, descriptor = standing.read_standing_initialization(Path(bootstrap))
    data, inputs = load_training_states(Path(teacher), descriptor)
    install_hooks(data, inputs, weight=weight, batch_size=batch_size, seed=seed)
    return standing.main(values)


if __name__ == "__main__":
    raise SystemExit(main())
