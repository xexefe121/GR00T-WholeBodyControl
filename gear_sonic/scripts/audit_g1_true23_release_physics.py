"""Same-state, same-torque audit of the executed corrected training environment.

Samples saved CPU rollout states, not invented poses. No learning or hardware.
All model changes below are isolated diagnostic counterfactual copies.
"""

import argparse
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
import torch

from gear_sonic.scripts.audit_g1_true23_training_replay_parity import (
    compare_model_parameters,
    differences,
    model_counterfactuals,
    validate_native_layout,
)
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("asset-root", "curriculum-directory", "source-geometry", "trace", "output-directory"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument(
        "--training-physics-profile",
        choices=("legacy_training_asset", "pinned_cpu_referee_scene_v1"),
        default="legacy_training_asset",
    )
    parser.add_argument(
        "--release-action-convention",
        choices=("released_bounded_linear", "released29_scale_bounded_linear_v2"),
        default="released_bounded_linear",
    )
    args = parser.parse_args(argv)
    output = args.output_directory.resolve()
    if output.exists():
        raise FileExistsError(output)
    from mjlab.envs import ManagerBasedRlEnv

    from gear_sonic.envs.mjlab.sonic_true23_causal_multimotion_v14 import make_causal_multimotion_v14_env_cfg
    from gear_sonic.envs.mjlab.sonic_true23_native_model_actuation import apply_native_model_actuation_profile
    from gear_sonic.envs.mjlab.sonic_true23_release_compatible import configure_release_compatible_environment
    from gear_sonic.envs.mjlab.sonic_true23_root_feedback import (
        configure_root_feedback_environment,
        install_root_feedback_command,
    )
    from gear_sonic.utils.g1_true23_native_model_actuation import NativeModelActuationProfile
    from gear_sonic.utils.g1_true23_step1b_mujoco import prepare_true23_model

    root = Path(__file__).resolve().parents[2]
    physics = root / "gear_sonic/config/sim_validation/g1_23dof_mujoco_sim2sim.json"
    motion = args.curriculum_directory / "curriculum.npz"
    spans_file = args.curriculum_directory / "curriculum.spans.json"
    spans = json.loads(spans_file.read_text())["spans"]
    inputs = {
        str(p.resolve()): sha256_file(p) for p in (physics, motion, spans_file, args.source_geometry, args.trace)
    }
    samples = np.array([0, 350, 700, 1200, 1800])
    with np.load(args.trace, allow_pickle=False) as data:
        qpos = data["qpos"][samples].copy()
        qvel = data["qvel"][samples].copy()
        raw = data["released_model_raw23"][samples].copy()
    install_root_feedback_command(spans)
    cfg = make_causal_multimotion_v14_env_cfg(motion_file=str(motion), num_envs=len(samples), play=False)
    cfg = apply_native_model_actuation_profile(cfg, NativeModelActuationProfile.from_sim_config(physics))
    cfg = configure_root_feedback_environment(cfg, spans, objective_profile="root_and_posture_v1")
    cfg = configure_release_compatible_environment(
        cfg, args.source_geometry.resolve(), args.release_action_convention
    )
    scene_parity = None
    native_model_path = args.asset_root / "gear_sonic/data/robots/g1/g1_23dof_rev_1_0.xml"
    if args.training_physics_profile == "pinned_cpu_referee_scene_v1":
        from gear_sonic.envs.mjlab.sonic_true23_nominal_scene import configure_nominal_scene, verify_nominal_scene

        cfg = configure_nominal_scene(cfg, native_model_path, physics)
    cfg.seed = 20260803
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    try:
        if args.training_physics_profile == "pinned_cpu_referee_scene_v1":
            scene_parity = verify_nominal_scene(env.sim.mj_model, native_model_path, physics)
            inputs.update(scene_parity["profile"]["inputs"])
        for name, module in list(sys.modules.items()):
            path = getattr(module, "__file__", None)
            if name.startswith("gear_sonic.") and path and Path(path).suffix == ".py":
                inputs[str(Path(path).resolve())] = sha256_file(Path(path))
        env.reset()
        env.sim.data.qpos[:] = torch.as_tensor(qpos, dtype=torch.float32, device=env.device)
        env.sim.data.qvel[:] = torch.as_tensor(qvel, dtype=torch.float32, device=env.device)
        env.sim.data.qacc_warmstart.zero_()
        env.sim.forward()
        env.scene.update(dt=env.physics_dt)
        # Compare exactly the float32 state that actually entered Warp.
        qpos = env.sim.data.qpos.cpu().numpy().copy()
        qvel = env.sim.data.qvel.cpu().numpy().copy()
        training = env.sim.mj_model
        _, replay, _ = prepare_true23_model(
            args.asset_root / "gear_sonic/data/robots/g1/g1_23dof_rev_1_0.xml", physics
        )
        validate_native_layout(training, "robot/")
        validate_native_layout(replay, "")
        models = {"training": training, "replay": replay, **model_counterfactuals(training, replay)}
        env.action_manager.process_action(torch.as_tensor(raw, device=env.device))
        env.action_manager.apply_action()
        env.scene.write_data_to_sim()
        torque = env.sim.data.ctrl.cpu().numpy().copy()
        env.sim.step()
        warp_next = env.sim.data.qvel.cpu().numpy().copy()
        rows, arrays = [], dict(qpos=qpos, qvel=qvel, raw23=raw, torque23=torque, warp_next_qvel=warp_next)
        for i, sample in enumerate(samples):
            next_states, contacts = {}, {}
            for name, model in models.items():
                state = mujoco.MjData(model)
                state.qpos[:], state.qvel[:], state.ctrl[:] = qpos[i], qvel[i], torque[i]
                state.qacc_warmstart[:] = 0
                mujoco.mj_step(model, state)
                next_states[name] = state.qvel.copy()
                contacts[name] = int(state.ncon)
                arrays.setdefault(name + "_next_qvel", []).append(state.qvel.copy())
            rows.append(
                dict(
                    control=int(sample),
                    contacts=contacts,
                    warp_vs_same_model_cpu=differences(warp_next[i], next_states["training"], 1e-3),
                    cpu_training_vs_replay=differences(next_states["training"], next_states["replay"], 1e-3),
                    counterfactuals={
                        name: differences(next_states["training"], value, 1e-3)
                        for name, value in next_states.items()
                        if name not in ("training", "replay")
                    },
                )
            )
        report = dict(
            kind="native23_release_compatible_actual_physics_comparison_v1",
            inputs=inputs,
            samples=rows,
            training_scene_parameter_parity=scene_parity,
            release_action_convention=args.release_action_convention,
            model_parameters=compare_model_parameters(training, replay),
            scope="five_saved_rollout_states_one_2ms_step_not_full_lifecycle_parity",
            deployment_ready=False,
            hardware_authorized=False,
        )
        for path, digest in inputs.items():
            if sha256_file(Path(path)) != digest:
                raise ValueError("physics audit input changed during execution")
        output.mkdir(parents=True, exist_ok=False)
        with (output / "trace.npz").open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        mujoco.mj_saveModel(training, str(output / "training.mjb"))
        mujoco.mj_saveModel(replay, str(output / "replay.mjb"))
        with (output / "report.json").open("x") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
        print(json.dumps(rows), flush=True)
        if scene_parity is not None and not all(
            row["warp_vs_same_model_cpu"]["within_tolerance"]
            and row["cpu_training_vs_replay"]["within_tolerance"]
            for row in rows
        ):
            raise ValueError("nominal scene numerical parity failed; training must not start")
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
