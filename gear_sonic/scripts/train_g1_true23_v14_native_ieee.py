"""Fresh original-v14 comparator under the current simulator controller.

Keep the original recovery initialization, four trainable actor tensors,
exploration, rewards, sampler and adaptive PPO schedule. Change only the
explicitly recorded native-support actuation and IEEE precision boundaries.
This is a method comparison, not an unmodified historical reproduction or
a one-variable LoRA ablation. Run in a separate process and output directory.
"""

from dataclasses import replace
import json
from pathlib import Path
import sys

from gear_sonic.scripts import train_g1_23dof_mjlab_teleop_v14 as v14
from gear_sonic.trl.mjlab.sonic_recovery_blend_policy import (
    RECOVERY_CHECKPOINT_PATH,
    RECOVERY_CHECKPOINT_SHA256,
    RECOVERY_POLICY_SHA256,
)
from gear_sonic.utils.g1_sonic_cpp_parameters import file_sha256
from gear_sonic.utils.g1_true23_actuation_profile import SIM_CONFIG, NativeSupportActuationProfile
from gear_sonic.utils.g1_true23_training_precision import (
    backend_state,
    guard_algorithm,
    ieee_training_precision,
    write_runtime,
)

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_KEY = "original_v14_native_ieee_comparison"
KIND = "g1_true23_original_v14_native_ieee_comparison_v1"
SOURCE_FILES = (
    Path(__file__).resolve(),
    ROOT / SIM_CONFIG,
    ROOT / "gear_sonic/utils/g1_true23_actuation_profile.py",
    ROOT / "gear_sonic/utils/g1_true23_projected_controller_state.py",
    ROOT / "gear_sonic/envs/mjlab/sonic_true23_stage_one_actuation.py",
    ROOT / "gear_sonic/utils/g1_true23_training_precision.py",
    ROOT / "gear_sonic/trl/mjlab/sonic_recovery_blend_policy.py",
)


def fresh_arguments(argv):
    values = list(argv)
    if any(token.split("=", 1)[0] == "--resume" for token in values):
        raise ValueError("original-v14 comparator requires a fresh run; resume is not supported")
    # The original runner overrides this setting. Bind the same value in the
    # resolved configuration rather than claiming a different initial rate.
    rates = []
    for index, token in enumerate(values):
        if token == "--learning-rate":
            if index + 1 == len(values):
                raise ValueError("--learning-rate requires a value")
            rates.append(values[index + 1])
        elif token.startswith("--learning-rate="):
            rates.append(token.split("=", 1)[1])
    if len(rates) > 1 or (rates and float(rates[0]) != 5.0e-6):
        raise ValueError("original-v14 initial learning rate must be exactly 5e-6")
    if not rates:
        values.extend(("--learning-rate", "5e-6"))
    return values


def comparison_contract(span_sidecar, profile):
    spans = json.loads(Path(span_sidecar).read_text())
    cursor = 0
    if spans.get("kind") != "g1_true23_motion_corpus_spans_v1" or spans.get("fps") != 50:
        raise ValueError("v14 comparison requires the unchanged 50-Hz corpus")
    for row in spans["spans"]:
        if type(row["start"]) is not int or row["start"] != cursor:
            raise ValueError("corpus spans must be contiguous and ordered")
        if type(row["length"]) is not int or row["length"] <= 15:
            raise ValueError("each corpus clip must contain the causal margins")
        cursor += row["length"]
    if (
        type(spans.get("clip_count")) is not int
        or spans["clip_count"] != len(spans["spans"])
        or spans["clip_count"] < 1
        or type(spans.get("total_frames")) is not int
        or spans["total_frames"] != cursor
    ):
        raise ValueError("corpus span count or total differs")
    if not profile.consistent_controller_state:
        raise ValueError("v14 comparison requires the stateful native-support controller")
    return dict(
        kind=KIND,
        recovery_checkpoint_path=str(RECOVERY_CHECKPOINT_PATH),
        recovery_checkpoint_sha256=RECOVERY_CHECKPOINT_SHA256,
        recovery_policy_sha256_before_std_pin=RECOVERY_POLICY_SHA256,
        original_v14_trainable_actor_tensor_count=4,
        fixed_exploration_std=0.10,
        initial_learning_rate=5.0e-6,
        learning_rate_schedule="original_v14_adaptive",
        sampling="original_v14_clip_constrained_uniform_including_adaptive_fallback",
        standing_retention_loss_used=False,
        standing_lora_bootstrap_used=False,
        frozen_sonic_lora_used=False,
        raw_decoder_output="native_il23_raw_action",
        safe_target_transform="external_once_then_stateful_native_projection",
        stage_one_actuation=profile.contract(),
        spans=dict(path=str(Path(span_sidecar).resolve()), sha256=file_sha256(span_sidecar)),
        intentional_changes_from_original_v14=["native_support_stateful_v2", "ieee_float32"],
        one_variable_lora_ablation=False,
        historical_run_reproduced_unmodified=False,
        fresh_critic_optimizer_counters=True,
        resume_supported=False,
        hardware_authorized=False,
        deployment_ready=False,
        promotion_eligible=False,
    )


def guarded_runner(parent, contract, precision, guard):
    class V14NativeIeeeRunner(parent):
        def __init__(self, *args, **kwargs):
            guard()
            super().__init__(*args, **kwargs)
            self.assert_frozen_actor_unchanged()
            guard_algorithm(self.alg, guard)
            actor = self.alg.get_policy()
            trainable = {
                name: dict(shape=list(parameter.shape), elements=parameter.numel())
                for name, parameter in actor.named_parameters()
                if parameter.requires_grad
            }
            if len(trainable) != 4:
                raise ValueError("original-v14 comparator actor trainable set changed")
            write_runtime(
                self.checkpoint_dir.parent / "v14_native_ieee_runtime.json",
                dict(
                    comparison=contract,
                    precision=precision,
                    actual_precision=backend_state(),
                    trainable_actor_tensors=trainable,
                    trainable_actor_elements=sum(row["elements"] for row in trainable.values()),
                    actual_initial_learning_rate=self.alg.learning_rate,
                    fresh_critic_optimizer_counters=True,
                    checkpoint_frozen_actor_guard=True,
                    hardware_authorized=False,
                    deployment_ready=False,
                ),
            )

        def load(self, *args, **kwargs):
            raise ValueError("original-v14 comparator is fresh-only; no resume or warm relabel")

        def save(self, *args, **kwargs):
            guard()
            self.assert_frozen_actor_unchanged()
            return super().save(*args, **kwargs)

    return V14NativeIeeeRunner


def install_hooks(precision, guard):
    original_install = v14._install_isolated_v14_hooks

    def install(span_sidecar):
        from gear_sonic.envs.mjlab import sonic_true23_causal_history as task
        from gear_sonic.envs.mjlab.sonic_true23_stage_one_actuation import apply_stage_one_actuation_profile

        if file_sha256(RECOVERY_CHECKPOINT_PATH) != RECOVERY_CHECKPOINT_SHA256:
            raise ValueError("original-v14 recovery checkpoint changed")
        profile = replace(
            NativeSupportActuationProfile.from_sim_config(ROOT / SIM_CONFIG),
            consistent_controller_state=True,
        )
        contract = comparison_contract(span_sidecar, profile)
        original_install(span_sidecar)
        original_builder = task.make_causal_history_recovery_env_cfg

        def profiled(**kwargs):
            return apply_stage_one_actuation_profile(original_builder(**kwargs), profile)

        task.make_causal_history_recovery_env_cfg = profiled
        runner = guarded_runner(v14.CausalTeleopRunnerV14, contract, precision, guard)
        v14.runner_module.CausalHistoryMjlabOnPolicyRunner = runner
        v14.base.CausalHistoryMjlabOnPolicyRunner = runner
        for source in SOURCE_FILES:
            if source not in v14.base.CAUSAL_SOURCE_FILES:
                v14.base.CAUSAL_SOURCE_FILES += (source,)
        original_resolved = v14.base._resolved_training_config

        def resolved(*args, **kwargs):
            guard()
            return {
                **original_resolved(*args, **kwargs),
                CONTRACT_KEY: contract,
                "stage_one_actuation": profile.contract(),
                "training_precision": precision,
            }

        v14.base._resolved_training_config = resolved

    v14._install_isolated_v14_hooks = install


def main(argv=None):
    values = fresh_arguments(sys.argv[1:] if argv is None else argv)
    with ieee_training_precision() as (precision, guard):
        install_hooks(precision, guard)
        return v14.main(values)


if __name__ == "__main__":
    raise SystemExit(main())
