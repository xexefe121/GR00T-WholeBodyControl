"""Fresh SIM-only experiment adding the existing physical mean-overreach loss."""

import argparse
from pathlib import Path

from gear_sonic.scripts import train_g1_true23_world_quality as quality
from gear_sonic.trl.mjlab.native23_projected_target_ppo import PROFILE, projection_objective_contract
from gear_sonic.trl.mjlab.native23_world_projection_runner import (
    ALGORITHM,
    Native23WorldProjectionRunner,
    training_contract,
)
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file
from gear_sonic.utils.g1_true23_generalist_source_closure import collect_local_source_closure

REACHABILITY_AUDIT_SHA256 = "0c7e2c65bf8ca9a323e65c3147993d9206e6962a5fe12d87d32aef60d70f0041"


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--reachability-audit", type=Path, required=True)
    parser.add_argument("--projection-experiment", type=Path, required=True)
    extra, remaining = parser.parse_known_args(argv)
    audit, experiment = (
        extra.reachability_audit.resolve(strict=True),
        extra.projection_experiment.resolve(strict=True),
    )
    if sha256_file(audit) != REACHABILITY_AUDIT_SHA256:
        raise ValueError("world-projection trial requires the completed current-policy reachability audit")
    legacy = quality.world.bounded.decoder.intent.legacy
    previous_main, previous_runner = legacy.main, quality.Native23WorldQualityRunner

    def dispatch(values):
        previous_hooks = legacy.install_hooks

        def install_hooks(args, inputs, curriculum, precision, guard):
            from gear_sonic.trl.mjlab import config

            base = previous_hooks(args, inputs, curriculum, precision, guard)
            previous_agent = config.true23_mjlab_ppo_runner_cfg

            def agent():
                cfg = previous_agent()
                if cfg.algorithm.class_name != "PPO":
                    raise ValueError("unexpected previous world-quality algorithm")
                cfg.algorithm.class_name = ALGORITHM
                return cfg

            config.true23_mjlab_ppo_runner_cfg = agent
            previous_resolved = base._resolved_training_config

            def resolved(*values, **kwargs):
                result = previous_resolved(*values, **kwargs)
                result["native23_world_projection"] = training_contract()
                result["ppo_auxiliary_objective"] = projection_objective_contract(PROFILE)
                for key in ("native23_generalist", "native23_root_feedback"):
                    result[key]["checkpoint_pattern"] = "world_projection_model_N.pt"
                return result

            base._resolved_training_config = resolved
            previous_sources = base._source_files

            def sources():
                result = previous_sources()
                root = legacy.generalist.ROOT
                roots = [Path(__file__), root / "gear_sonic/utils/g1_true23_world_projection_checkpoint.py"]
                result.update(collect_local_source_closure(root, roots).as_source_files(root))
                result["native23_world_projection/reachability_audit.json"] = audit
                result["native23_world_projection/EXPERIMENT.md"] = experiment
                return result

            base._source_files = sources
            return base

        legacy.install_hooks = install_hooks
        try:
            return previous_main(values)
        finally:
            legacy.install_hooks = previous_hooks

    legacy.main = dispatch
    quality.Native23WorldQualityRunner = Native23WorldProjectionRunner
    try:
        return quality.main(remaining)
    finally:
        legacy.main = previous_main
        quality.Native23WorldQualityRunner = previous_runner


if __name__ == "__main__":
    raise SystemExit(main())
