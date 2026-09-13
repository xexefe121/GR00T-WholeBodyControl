"""Fresh training experiment; preserve source and physics, change reset sampling."""

import argparse
from pathlib import Path

from gear_sonic.envs.mjlab.sonic_true23_failure_sampling import FailureAdaptiveRootFeedbackCommand
from gear_sonic.scripts import train_g1_true23_world_quality as quality
from gear_sonic.trl.mjlab.native23_failure_sampling_runner import Native23FailureSamplingRunner
from gear_sonic.utils.g1_true23_failure_sampling import sampling_contract
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file
from gear_sonic.utils.g1_true23_generalist_source_closure import collect_local_source_closure

MEASUREMENT_AUDIT_SHA256 = "bbd23cc260b45595c66c9b440623dbe5eaffc111d057596911d5578a3469f381"


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--training-failure-audit", type=Path, required=True)
    parser.add_argument("--failure-sampling-experiment", type=Path, required=True)
    options, remaining = parser.parse_known_args(argv)
    audit = options.training_failure_audit.resolve(strict=True)
    experiment = options.failure_sampling_experiment.resolve(strict=True)
    if sha256_file(audit) != MEASUREMENT_AUDIT_SHA256:
        raise ValueError(
            "failure sampling requires the completed measured failure audit, including failed identity flags"
        )
    legacy = quality.world.bounded.decoder.intent.legacy
    previous_main, previous_runner = legacy.main, quality.Native23WorldQualityRunner
    previous_feedback, previous_command = legacy.feedback_training_contract, legacy.install_root_feedback_command

    def feedback(args, curriculum):
        if args.start_schedule != "mixed_reference_reset_v1" or args.mode not in ("smoke", "regression"):
            raise ValueError("failure sampling is fresh mixed-layout SIM regression only")
        result = previous_feedback(args, curriculum)
        result["start_schedule"] = sampling_contract()
        result["checkpoint_pattern"] = "failure_sampling_model_N.pt"
        return result

    def install_command(spans, *, start_schedule):
        if start_schedule != "mixed_reference_reset_v1":
            raise ValueError("failure sampling requires original standing/source environment allocation")
        from gear_sonic.envs.mjlab import sonic_true23_causal_history as task

        task.CausalHistoryMotionCommandCfg.build = lambda cfg, env: FailureAdaptiveRootFeedbackCommand(
            cfg, env, spans=spans
        )

    def dispatch(values):
        previous_hooks = legacy.install_hooks

        def install_hooks(args, inputs, curriculum, precision, guard):
            base = previous_hooks(args, inputs, curriculum, precision, guard)
            previous_resolved, previous_sources = base._resolved_training_config, base._source_files

            def resolved(*values, **kwargs):
                result = previous_resolved(*values, **kwargs)
                result["native23_failure_sampling"] = sampling_contract()
                for key in ("native23_generalist", "native23_root_feedback"):
                    result[key]["checkpoint_pattern"] = "failure_sampling_model_N.pt"
                return result

            def sources():
                result = previous_sources()
                root = legacy.generalist.ROOT
                roots = [Path(__file__), root / "gear_sonic/utils/g1_true23_failure_sampling_checkpoint.py"]
                result.update(collect_local_source_closure(root, roots).as_source_files(root))
                result["native23_failure_sampling/training_failure_audit.json"] = audit
                result["native23_failure_sampling/EXPERIMENT.md"] = experiment
                return result

            base._resolved_training_config, base._source_files = resolved, sources
            return base

        legacy.install_hooks = install_hooks
        try:
            return previous_main(values)
        finally:
            legacy.install_hooks = previous_hooks

    legacy.main, legacy.feedback_training_contract, legacy.install_root_feedback_command = (
        dispatch,
        feedback,
        install_command,
    )
    quality.Native23WorldQualityRunner = Native23FailureSamplingRunner
    try:
        return quality.main(remaining)
    finally:
        legacy.main, legacy.feedback_training_contract, legacy.install_root_feedback_command = (
            previous_main,
            previous_feedback,
            previous_command,
        )
        quality.Native23WorldQualityRunner = previous_runner


if __name__ == "__main__":
    raise SystemExit(main())
