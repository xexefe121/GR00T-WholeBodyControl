"""Bounded pose-LoRA experiment changing only the critic value-loss clipping.

Actor/checkpoint schema and offline inference remain identical to pose-LoRA;
the full agent configuration and this distinct diagnostic-backed training
lineage record the optimizer change. No resume, deployment or hardware access.
"""

import argparse
import copy
import json
from pathlib import Path

from gear_sonic.scripts import train_g1_true23_pose_lora as pose
from gear_sonic.trl.mjlab.native23_pose_lora_runner import Native23PoseLoraRunner
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file
from gear_sonic.utils.g1_true23_generalist_source_closure import collect_local_source_closure

DIAGNOSTIC_SHA256 = "95f31d3b715ef6f34add62837fd74eb84d816d9c3ce7a056eed47a15e4442e0e"
KIND = "native23_pose_unclipped_critic_value_loss_v1"


def value_contract():
    return {
        "kind": KIND,
        "use_clipped_value_loss": False,
        "previous_use_clipped_value_loss": True,
        "critic_loss": "mean_squared_error_to_same_stock_GAE_returns",
        "policy_clip_param": 0.2,
        "critic_max_grad_norm": 0.5,
        "reward_scale_or_weights_changed": False,
        "actor_architecture_or_inference_changed": False,
        "motion_or_physics_changed": False,
        "diagnostic_audit_sha256": DIAGNOSTIC_SHA256,
        "maximum_fresh_updates": 100,
        "resume_supported": False,
        "control_improvement_proven": False,
        "deployment_ready": False,
        "hardware_authorized": False,
    }


def configure_agent(cfg):
    result = copy.deepcopy(cfg)
    algorithm = result.algorithm
    if algorithm.use_clipped_value_loss is not True or algorithm.clip_param != 0.2:
        raise ValueError("unclipped-value experiment requires the known clipped0.2 predecessor")
    if algorithm.value_loss_coef != 1.0:
        raise ValueError("unclipped-value experiment must preserve critic loss coefficient")
    # The shared CLI applies max_grad_norm=0.5 AFTER calling this factory.
    # Preserve the factory default here; require_unclipped checks the final
    # resolved and actual runner values before any saved checkpoint/update.
    algorithm.use_clipped_value_loss = False
    return result


def require_unclipped(runner):
    resolved = runner.training_lineage["materials"]["resolved_config"]["payload"]
    if resolved.get("native23_unclipped_value") != value_contract():
        raise ValueError("unclipped critic objective missing from executed lineage")
    actual = runner.alg
    expected = resolved["agent"]["algorithm"]
    for name, value in (
        ("use_clipped_value_loss", False),
        ("clip_param", 0.2),
        ("max_grad_norm", 0.5),
        ("value_loss_coef", 1.0),
    ):
        if getattr(actual, name) != value or expected.get(name) != value:
            raise ValueError(f"unclipped critic runtime/config mismatch: {name}")


class UnclippedValueRunner(Native23PoseLoraRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        require_unclipped(self)

    def save(self, *args, **kwargs):
        require_unclipped(self)
        return super().save(*args, **kwargs)


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--value-diagnostic-audit", type=Path, required=True)
    extra, remaining = parser.parse_known_args(argv)
    audit_path = extra.value_diagnostic_audit.resolve(strict=True)
    if sha256_file(audit_path) != DIAGNOSTIC_SHA256:
        raise ValueError("unclipped-value experiment requires the exact verified8192-transition diagnostic")
    audit = json.loads(audit_path.read_text())
    if (
        audit.get("passed") is not True
        or audit.get("actual_simulator_transitions") != 8192
        or audit.get("deployment_ready") is not False
        or audit.get("hardware_authorized") is not False
    ):
        raise ValueError("diagnostic proof is incomplete or claims hardware authority")
    legacy = pose.intent.legacy
    previous_main = legacy.main
    previous_runner = pose.Native23PoseLoraRunner

    def dispatch(values):
        # This runs after pose.main has installed the pose-specific wrappers.
        # Wrap their final output so only the final algorithm boolean changes.
        previous_hooks = legacy.install_hooks

        def install_hooks(args, inputs, curriculum, precision, guard):
            from gear_sonic.trl.mjlab import config

            base = previous_hooks(args, inputs, curriculum, precision, guard)
            previous_agent = config.true23_mjlab_ppo_runner_cfg
            config.true23_mjlab_ppo_runner_cfg = lambda: configure_agent(previous_agent())
            previous_resolved = base._resolved_training_config

            def resolved(*values, **kwargs):
                result = previous_resolved(*values, **kwargs)
                result["native23_unclipped_value"] = value_contract()
                return result

            base._resolved_training_config = resolved
            previous_sources = base._source_files

            def sources():
                result = previous_sources()
                root = legacy.generalist.ROOT
                result.update(collect_local_source_closure(root, [Path(__file__)]).as_source_files(root))
                result["native23_unclipped_value/diagnostic_audit.json"] = audit_path
                return result

            base._source_files = sources
            return base

        legacy.install_hooks = install_hooks
        return previous_main(values)

    legacy.main = dispatch
    pose.Native23PoseLoraRunner = UnclippedValueRunner
    try:
        return pose.main(remaining)
    finally:
        legacy.main = previous_main
        pose.Native23PoseLoraRunner = previous_runner


if __name__ == "__main__":
    raise SystemExit(main())
