"""Paired-initial-state sensor/action-noise probe of a checked LoRA checkpoint.

Run one combination per process. Reuse the actual existing no-learning
MJLab audit, retain the full corpus and guards, and explicitly guard IEEE
precision at every actor call. Later adaptive resets are not paired.
"""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
import sys

import numpy as np

from gear_sonic.scripts import audit_g1_true23_exploration_rollout as original
from gear_sonic.scripts import train_g1_23dof_mjlab_frozen_lora as launcher
from gear_sonic.scripts.record_g1_sonic_original29_baseline import dump
from gear_sonic.scripts.train_g1_true23_standing_warm_start import option
from gear_sonic.trl.mjlab.frozen_platform_lora_actor import FrozenPlatformTrue23Core
from gear_sonic.utils.g1_sonic_cpp_parameters import file_sha256
from gear_sonic.utils.g1_true23_sensor_noise_scale import UniformNoiseCfg, scale_sensor_noise
from gear_sonic.utils.g1_true23_training_precision import backend_state, ieee_training_precision


def main(argv=None):
    values = []
    for value in sys.argv[1:] if argv is None else argv:
        values.extend(value.split("=", 1) if value.startswith("--") and "=" in value else [value])
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--sensor-noise-scale", type=float, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    args, remaining = parser.parse_known_args(values)
    checkpoint_arg, output_arg = option(remaining, "--checkpoint"), option(remaining, "--output-dir")
    if checkpoint_arg is None or output_arg is None:
        raise ValueError("sensor probe requires an explicit checkpoint and new output directory")
    checkpoint, output = Path(checkpoint_arg).expanduser(), Path(output_arg).expanduser()
    if checkpoint.is_symlink() or output.is_symlink() or output.exists():
        raise ValueError("sensor evidence needs nonsymlink inputs and a new output directory")
    checkpoint, output = checkpoint.resolve(strict=True), output.resolve()
    if file_sha256(checkpoint) != args.expected_checkpoint_sha256:
        raise ValueError("sensor probe checkpoint hash mismatch")
    inputs = {str(checkpoint): args.expected_checkpoint_sha256}
    for path in (Path(__file__), Path(inspect.getfile(UniformNoiseCfg))):
        inputs[str(path.resolve())] = file_sha256(path)
    configurations, observations = [], []
    old_install, old_forward, old_argv = (
        launcher._install_frozen_lora_hooks,
        FrozenPlatformTrue23Core.forward,
        sys.argv,
    )
    with ieee_training_precision() as (precision, guard):

        def install(**kwargs):
            old_install(**kwargs)
            from gear_sonic.envs.mjlab import sonic_true23_causal_history as task

            old_builder = task.make_causal_history_recovery_env_cfg

            def build(**kwargs):
                guard()
                cfg = old_builder(**kwargs)
                configurations.append(scale_sensor_noise(cfg, args.sensor_noise_scale))
                return cfg

            task.make_causal_history_recovery_env_cfg = build

        def forward(self, semantic, proprio):
            guard()
            observations.append((semantic.detach().cpu().numpy().copy(), proprio.detach().cpu().numpy().copy()))
            return old_forward(self, semantic, proprio)

        try:
            launcher._install_frozen_lora_hooks = install
            FrozenPlatformTrue23Core.forward = forward
            sys.argv = [str(Path(original.__file__)), *remaining]
            status = original.main()
            guard()
            actual_precision = backend_state()
        finally:
            launcher._install_frozen_lora_hooks = old_install
            FrozenPlatformTrue23Core.forward = old_forward
            sys.argv = old_argv
    if status != 0 or not configurations or any(row != configurations[0] for row in configurations):
        raise ValueError("sensor probe failed or configuration changed across builds")
    report_path = output / "summary.json"
    report = json.loads(report_path.read_text())
    if (
        report["weights_updated"] is not False
        or report["optimizer_steps"] != 0
        or report["observation_corruption_enabled"] != {"tokenizer": True, "policy": True}
        or len(observations) != report["episode_audit"]["control_steps_per_environment"]
    ):
        raise ValueError("sensor probe changed no-learning or actual-observation accounting")
    with (output / "actor_observations.npz").open("xb") as stream:
        np.savez_compressed(
            stream,
            tokenizer=np.asarray([row[0] for row in observations]),
            policy=np.asarray([row[1] for row in observations]),
        )
    for collection in (report["sources"], report["inputs"]):
        for path, digest in collection.items():
            if digest != inputs.get(path, digest):
                raise ValueError("sensor source binding conflicts with base rollout")
            inputs[path] = digest
    for path in (report_path, output / "rollout.npz", output / "actor_observations.npz"):
        inputs[str(path)] = file_sha256(path)
    for name, module in list(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if name.startswith("gear_sonic.") and path and Path(path).suffix == ".py":
            path = str(Path(path).resolve())
            digest = file_sha256(path)
            if digest != inputs.get(path, digest):
                raise ValueError("sensor imported source changed during rollout")
            inputs[path] = digest
    for path, expected in inputs.items():
        if file_sha256(path) != expected:
            raise ValueError(f"sensor source/evidence changed: {path}")
    result = dict(
        kind="g1_true23_ieee_sensor_exploration_probe_v1",
        inputs=inputs,
        checkpoint_sha256=args.expected_checkpoint_sha256,
        checkpoint_lineage_sha256=report["checkpoint_lineage_sha256"],
        sensor_noise=configurations[0],
        action_noise_scale=report["noise_scale"],
        precision=precision,
        actual_precision=actual_precision,
        guarded_actor_calls=len(observations),
        episode_audit=report["episode_audit"],
        base_report=str(report_path),
        initial_physics_sha256=report["initial_physics_qpos_qvel_sha256"],
        initial_observation_state_sha256=report["initial_state_sha256"],
        later_adaptive_resets_paired=False,
        weights_updated=False,
        optimizer_steps=0,
        full_clip_fidelity_qualified=False,
        hardware_authorized=False,
        deployment_ready=False,
    )
    dump(output / "ieee_sensor_report.json", result)
    print(
        json.dumps({key: result[key] for key in ("action_noise_scale", "guarded_actor_calls", "episode_audit")}),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
