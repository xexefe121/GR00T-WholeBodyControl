"""Fresh controlled SIM experiment: add world tracking failure, keep reward."""

import argparse
from pathlib import Path

from gear_sonic.envs.mjlab.sonic_true23_world_tracking_termination import termination_contract
from gear_sonic.scripts import train_g1_true23_bounded_progress as bounded
from gear_sonic.trl.mjlab.native23_world_tracking_runner import Native23WorldTrackingRunner
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file
from gear_sonic.utils.g1_true23_generalist_source_closure import collect_local_source_closure

FULL_LOOP_SHA256 = "796358dee11223f808f8a301adbf4d269b507a326f187e4ab32746192d259cd4"


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--full-loop-diagnostic", type=Path, required=True)
    parser.add_argument("--world-experiment", type=Path, required=True)
    extra, remaining = parser.parse_known_args(argv)
    diagnostic = extra.full_loop_diagnostic.resolve(strict=True)
    experiment = extra.world_experiment.resolve(strict=True)
    if sha256_file(diagnostic) != FULL_LOOP_SHA256:
        raise ValueError("world failure experiment requires the actual four-motion diagnostic")
    legacy = bounded.decoder.intent.legacy
    previous_main, previous_runner = legacy.main, bounded.Native23BoundedProgressRunner

    def dispatch(values):
        previous_hooks = legacy.install_hooks

        def install_hooks(args, inputs, curriculum, precision, guard):
            from gear_sonic.envs.mjlab import sonic_true23_causal_history as task
            from gear_sonic.envs.mjlab.sonic_true23_world_tracking_termination import configure_environment

            base = previous_hooks(args, inputs, curriculum, precision, guard)
            previous_builder = task.make_causal_history_recovery_env_cfg
            task.make_causal_history_recovery_env_cfg = lambda **kw: configure_environment(previous_builder(**kw))
            previous_resolved = base._resolved_training_config

            def resolved(*values, **kwargs):
                result = previous_resolved(*values, **kwargs)
                result["native23_world_tracking_termination"] = termination_contract()
                result["native23_root_feedback"]["additional_training_failure"] = termination_contract()
                for key in ("native23_generalist", "native23_root_feedback"):
                    result[key]["checkpoint_pattern"] = "world_tracking_model_N.pt"
                return result

            base._resolved_training_config = resolved
            previous_sources = base._source_files

            def sources():
                result = previous_sources()
                root = legacy.generalist.ROOT
                roots = [Path(__file__), root / "gear_sonic/utils/g1_true23_world_tracking_checkpoint.py"]
                result.update(collect_local_source_closure(root, roots).as_source_files(root))
                result["native23_world_tracking/full_loop_diagnostic.json"] = diagnostic
                result["native23_world_tracking/EXPERIMENT.md"] = experiment
                return result

            base._source_files = sources
            return base

        legacy.install_hooks = install_hooks
        return previous_main(values)

    legacy.main = dispatch
    bounded.Native23BoundedProgressRunner = Native23WorldTrackingRunner
    try:
        return bounded.main(remaining)
    finally:
        legacy.main = previous_main
        bounded.Native23BoundedProgressRunner = previous_runner


if __name__ == "__main__":
    raise SystemExit(main())
