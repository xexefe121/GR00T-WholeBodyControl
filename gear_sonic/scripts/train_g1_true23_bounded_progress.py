"""Explicit, fresh, capped reward experiment; no controller or limit changes."""

import argparse
import json
from pathlib import Path

from gear_sonic.scripts import train_g1_true23_decoder_lora as decoder
from gear_sonic.trl.mjlab.native23_bounded_progress_runner import Native23BoundedProgressRunner
from gear_sonic.utils.g1_true23_bounded_progress import reward_contract
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file
from gear_sonic.utils.g1_true23_generalist_source_closure import collect_local_source_closure

DIAGNOSTIC_SHA256 = "6f96fc17f99107f52e56af3ec69ff6b2f8ca4db4f248fd0419c61a590ff3178c"


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--reward-incentive-diagnostic", type=Path, required=True)
    extra, remaining = parser.parse_known_args(argv)
    diagnostic_path = extra.reward_incentive_diagnostic.resolve(strict=True)
    if sha256_file(diagnostic_path) != DIAGNOSTIC_SHA256:
        raise ValueError("bounded reward requires the exact saved-trajectory diagnostic")
    diagnostic = json.loads(diagnostic_path.read_text())
    if len(diagnostic.get("cases", [])) != 8:
        raise ValueError("reward diagnostic must cover both complete four-motion candidates")
    legacy = decoder.intent.legacy
    previous_main, previous_runner = legacy.main, decoder.Native23DecoderLoraRunner

    def dispatch(values):
        previous_hooks = legacy.install_hooks

        def install_hooks(args, inputs, curriculum, precision, guard):
            from gear_sonic.envs.mjlab import sonic_true23_causal_history as task
            from gear_sonic.envs.mjlab.sonic_true23_bounded_progress import configure_environment

            base = previous_hooks(args, inputs, curriculum, precision, guard)
            previous_builder = task.make_causal_history_recovery_env_cfg
            task.make_causal_history_recovery_env_cfg = lambda **kw: configure_environment(previous_builder(**kw))
            previous_resolved = base._resolved_training_config

            def resolved(*values, **kwargs):
                result = previous_resolved(*values, **kwargs)
                result["native23_bounded_progress"] = reward_contract()
                result["native23_generalist"]["checkpoint_pattern"] = "bounded_progress_model_N.pt"
                result["native23_root_feedback"]["checkpoint_pattern"] = "bounded_progress_model_N.pt"
                result["native23_root_feedback"]["post_reward_transform"] = reward_contract()
                return result

            base._resolved_training_config = resolved
            previous_sources = base._source_files

            def sources():
                result = previous_sources()
                root = legacy.generalist.ROOT
                result.update(collect_local_source_closure(root, [Path(__file__)]).as_source_files(root))
                result["native23_bounded_progress/incentive_diagnostic.json"] = diagnostic_path
                return result

            base._source_files = sources
            return base

        legacy.install_hooks = install_hooks
        return previous_main(values)

    legacy.main = dispatch
    decoder.Native23DecoderLoraRunner = Native23BoundedProgressRunner
    try:
        return decoder.main(remaining)
    finally:
        legacy.main = previous_main
        decoder.Native23DecoderLoraRunner = previous_runner


if __name__ == "__main__":
    raise SystemExit(main())
