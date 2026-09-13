"""Fresh SIM training: unchanged world-tracking recipe plus one joint task bonus."""

import argparse
from pathlib import Path

from gear_sonic.scripts import train_g1_true23_world_tracking as world
from gear_sonic.trl.mjlab.native23_world_quality_runner import Native23WorldQualityRunner
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file
from gear_sonic.utils.g1_true23_generalist_source_closure import collect_local_source_closure
from gear_sonic.utils.g1_true23_world_quality import quality_contract

LEARNING_SIGNAL_SHA256 = "38ed659972c0303f75c5e55e6278d1a8b3ba6c2e083b6ad3836abc64ec4cf146"


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--learning-signal", type=Path, required=True)
    parser.add_argument("--quality-experiment", type=Path, required=True)
    extra, remaining = parser.parse_known_args(argv)
    signal, experiment = extra.learning_signal.resolve(strict=True), extra.quality_experiment.resolve(strict=True)
    if sha256_file(signal) != LEARNING_SIGNAL_SHA256:
        raise ValueError("world-quality trial requires the completed transition-level signal audit")
    legacy = world.bounded.decoder.intent.legacy
    previous_main, previous_runner = legacy.main, world.Native23WorldTrackingRunner

    def dispatch(values):
        previous_hooks = legacy.install_hooks

        def install_hooks(args, inputs, curriculum, precision, guard):
            base = previous_hooks(args, inputs, curriculum, precision, guard)
            previous_resolved = base._resolved_training_config

            def resolved(*values, **kwargs):
                result = previous_resolved(*values, **kwargs)
                result["native23_world_quality_bonus"] = quality_contract()
                for key in ("native23_generalist", "native23_root_feedback"):
                    result[key]["checkpoint_pattern"] = "world_quality_model_N.pt"
                return result

            base._resolved_training_config = resolved
            previous_sources = base._source_files

            def sources():
                result = previous_sources()
                root = legacy.generalist.ROOT
                roots = [Path(__file__), root / "gear_sonic/utils/g1_true23_world_quality_checkpoint.py"]
                result.update(collect_local_source_closure(root, roots).as_source_files(root))
                result["native23_world_quality/learning_signal.json"] = signal
                result["native23_world_quality/EXPERIMENT.md"] = experiment
                return result

            base._source_files = sources
            return base

        legacy.install_hooks = install_hooks
        try:
            return previous_main(values)
        finally:
            legacy.install_hooks = previous_hooks

    legacy.main = dispatch
    world.Native23WorldTrackingRunner = Native23WorldQualityRunner
    try:
        return world.main(remaining)
    finally:
        legacy.main = previous_main
        world.Native23WorldTrackingRunner = previous_runner


if __name__ == "__main__":
    raise SystemExit(main())
