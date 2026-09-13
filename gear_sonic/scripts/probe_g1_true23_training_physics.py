"""32 same-state, same-target20ms CPU/Warp probes, never robot control.

Recover real solver warmstarts by bit-exact CPU reexecution of all completed
controls. Compare the original CPU model, attached training model with native
PD, and attached CPU model receiving exactly the GPU motor controls. GPU state
injection occurs once per independent probe, never during its ten substeps.
This isolates one-control engine differences; it is not a full closed-loop test.
"""

import argparse
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
import torch

from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file

NAMES = ("walk002", "walk003", "walk008", "dance")


def sample_controls(count):
    if type(count) is not int or count < 8:
        raise ValueError("physics probe requires at least eight completed controls")
    selected = np.linspace(0, count - 1, 8).astype(int)
    if len(np.unique(selected)) != 8 or selected[0] != 0 or selected[-1] != count - 1:
        raise ValueError("physics sampling lost an endpoint or duplicated a control")
    return selected


def main(precision, guard):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bounded-run", type=Path, required=True)
    parser.add_argument("--input-proof", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError("training physics probe refuses overwrite")
    output.mkdir(parents=True)
    inputs = {}

    def bind(path, expected=None):
        path = Path(path).resolve(strict=True)
        digest = sha256_file(path)
        if expected is not None and digest != expected:
            raise ValueError(f"physics input changed: {path}")
        inputs[str(path)] = digest
        return path

    def read_npz(path):
        with np.load(bind(path), allow_pickle=False) as data:
            return {k: data[k].copy() for k in data.files}

    bind(__file__)
    bind(args.experiment)
    proof = json.loads(
        bind(args.input_proof, "1c68e3e7062e29c046b474c71f0be77c535c7f85f317da63d4341966e3352d62").read_text()
    )
    run = args.bounded_run.resolve(strict=True)
    resolved = json.loads(bind(run / "train/resolved_training.json").read_text())
    runtime = json.loads(bind(run / "train/root_feedback_runtime.json").read_text())
    spans = runtime["training_inputs"]["derived_curriculum"]["derived_spans"]["spans"]
    spec = resolved["native23_root_feedback"]["original_intent_spec"]

    from mjlab.envs import ManagerBasedRlEnv

    from gear_sonic.envs.mjlab import (
        sonic_true23_bounded_progress as bounded,
        sonic_true23_original_intent as intent,
    )
    from gear_sonic.envs.mjlab.sonic_true23_buffered_source import configure_buffered_source_environment
    from gear_sonic.envs.mjlab.sonic_true23_causal_multimotion_v14 import make_causal_multimotion_v14_env_cfg
    from gear_sonic.envs.mjlab.sonic_true23_native_model_actuation import apply_native_model_actuation_profile
    from gear_sonic.envs.mjlab.sonic_true23_nominal_scene import configure_nominal_scene, verify_nominal_scene
    from gear_sonic.envs.mjlab.sonic_true23_release_compatible import configure_release_compatible_environment
    from gear_sonic.envs.mjlab.sonic_true23_root_feedback import (
        configure_root_feedback_environment,
        install_root_feedback_command,
    )
    from gear_sonic.utils.g1_true23_clean_mujoco_teleop import CleanTrue23MujocoController
    from gear_sonic.utils.g1_true23_native_model_actuation import (
        NativeModelActuationProfile,
        native_model_pd_numpy,
    )
    from gear_sonic.utils.g1_true23_reference_floor import compiled_model_sha256
    from gear_sonic.utils.g1_true23_source_action_codec import SOURCE_ACTION_CONVENTION

    guard()
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    physics = Path(__file__).resolve().parents[2] / "gear_sonic/config/sim_validation/g1_23dof_mujoco_sim2sim.json"
    native = bind(spec["files"]["native_model"]["path"], spec["files"]["native_model"]["sha256"])
    profile = NativeModelActuationProfile.from_sim_config(bind(physics))
    captures, rows = [], []
    reexecuted = 0
    for name in NAMES:
        directory = run / ("cpu100_" + name)
        if (directory / "nominal.npz").exists():
            trace = read_npz(directory / "nominal.npz")
            result = json.loads(bind(directory / "report.json").read_text())["records"][0]["result"]
        else:
            trace = read_npz(directory / "failed_physics.npz")
            result = json.loads(bind(directory / "failed_physics.json").read_text())["result"]
        count = len(trace["target23"])
        selected = set(sample_controls(count).tolist())
        controller = CleanTrue23MujocoController(model_path=native, physics_path=physics, policy=None)
        model, data = controller.model, controller.data
        if compiled_model_sha256(model) != result["compiled_model_sha256"]:
            raise ValueError("CPU recovery model differs from the saved trajectory")
        q, v = trace["qpos"][0], trace["qvel"][0]
        controller.reset(
            base_position=q[:3],
            base_quaternion_wxyz=q[3:7],
            joint_position_hardware=q[7:],
            root_velocity=v[:6],
            joint_velocity_hardware=v[6:],
        )
        for control in range(count):
            controller.refresh_observation_kinematics()
            if control in selected:
                captures.append(
                    dict(
                        qpos=data.qpos.copy(),
                        qvel=data.qvel.copy(),
                        warmstart=data.qacc_warmstart.copy(),
                        target=trace["target23"][control].copy(),
                        expected_qpos=trace["physics_post_qpos"][control * 10 : control * 10 + 10].copy(),
                        expected_qvel=trace["physics_post_qvel"][control * 10 : control * 10 + 10].copy(),
                    )
                )
                rows.append(dict(case=name, control=control))
            for substep in range(10):
                index = control * 10 + substep
                np.testing.assert_array_equal(data.qpos, trace["physics_pre_qpos"][index])
                np.testing.assert_array_equal(data.qvel, trace["physics_pre_qvel"][index])
                _, applied, invalid, _ = native_model_pd_numpy(
                    trace["target23"][control].astype(np.float64), data.qpos[7:], data.qvel[6:], profile
                )
                if invalid or np.any(trace["physics_external_force_world_n"][index]):
                    raise ValueError("physics probe requires valid original targets and zero external forces")
                data.ctrl[:] = applied
                mujoco.mj_step(model, data)
                np.testing.assert_array_equal(data.qpos, trace["physics_post_qpos"][index])
                np.testing.assert_array_equal(data.qvel, trace["physics_post_qvel"][index])
                reexecuted += 1
        print(json.dumps({"warmstart_recovery": name, "bit_exact_substeps": count * 10}), flush=True)
    arrays = {key: np.stack([r[key] for r in captures]) for key in captures[0]}
    if len(rows) != 32:
        raise ValueError("physics probe must contain all four cases with eight states each")
    install_root_feedback_command(spans, start_schedule="mixed_reference_reset_v1")
    cfg = make_causal_multimotion_v14_env_cfg(
        motion_file=spec["files"]["native_motion"]["path"], num_envs=32, play=False
    )
    cfg = apply_native_model_actuation_profile(cfg, profile)
    cfg = configure_root_feedback_environment(cfg, spans, objective_profile="root_and_upper_feet_world_v4")
    cfg = configure_release_compatible_environment(
        cfg, spec["files"]["source_model"]["path"], SOURCE_ACTION_CONVENTION
    )
    cfg = configure_nominal_scene(cfg, native, physics)
    cfg = configure_buffered_source_environment(cfg)
    cfg = intent.configure_original_intent_environment(cfg, spec)
    cfg = bounded.configure_environment(cfg)
    cfg.seed = resolved["seed"]
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    try:
        parity = verify_nominal_scene(env.sim.mj_model, native, physics)
        bounded.verify_runtime(env)
        for name, module in tuple(sys.modules.items()):
            path = getattr(module, "__file__", None)
            if (
                name.startswith(("gear_sonic.", "mjlab.", "src.tasks.tracking", "mujoco_warp."))
                and path
                and str(path).endswith(".py")
            ):
                bind(path, proof["inputs"].get(str(Path(path).resolve())))
        origins = env.scene.env_origins.cpu().numpy().copy()
        arrays["environment_origins"] = origins
        qpos = arrays["qpos"].astype(np.float32)
        qpos[:, :3] += origins
        qvel = arrays["qvel"].astype(np.float32)
        warmstart = arrays["warmstart"].astype(np.float32)
        env.sim.reset()
        env.sim.data.qpos.copy_(torch.as_tensor(qpos, device=env.device))
        env.sim.data.qvel.copy_(torch.as_tensor(qvel, device=env.device))
        env.sim.data.qacc_warmstart.copy_(torch.as_tensor(warmstart, device=env.device))
        env.sim.forward()
        arrays["gpu_initial_qpos"] = env.sim.data.qpos.cpu().numpy().copy()
        arrays["gpu_initial_qvel"] = env.sim.data.qvel.cpu().numpy().copy()
        arrays["gpu_initial_warmstart"] = env.sim.data.qacc_warmstart.cpu().numpy().copy()
        action = env.action_manager.get_term("joint_pos")
        action.reset()
        # Already validated, actually-applied physical targets; no new policy
        # action, relaxation or unverified guard fallback is being generated.
        action._processed_actions.copy_(torch.as_tensor(arrays["target"], device=env.device))
        attached_native_pd, attached_same_ctrl = [], []
        for index in range(32):
            pd, same = mujoco.MjData(env.sim.mj_model), mujoco.MjData(env.sim.mj_model)
            for data in (pd, same):
                data.qpos[:] = arrays["gpu_initial_qpos"][index]
                data.qvel[:] = arrays["gpu_initial_qvel"][index]
                data.qacc_warmstart[:] = arrays["gpu_initial_warmstart"][index]
                mujoco.mj_forward(env.sim.mj_model, data)
            attached_native_pd.append(pd)
            attached_same_ctrl.append(same)
        recorded = {
            key: []
            for key in (
                "gpu_qpos",
                "gpu_qvel",
                "gpu_ctrl",
                "gpu_invalid",
                "cpu_attached_pd_qpos",
                "cpu_attached_pd_qvel",
                "cpu_attached_pd_ctrl",
                "cpu_same_ctrl_qpos",
                "cpu_same_ctrl_qvel",
            )
        }
        for _ in range(10):
            guard()
            env.action_manager.apply_action()
            env.scene.write_data_to_sim()
            ctrl = env.sim.data.ctrl.cpu().numpy().copy()
            recorded["gpu_ctrl"].append(ctrl)
            recorded["gpu_invalid"].append(action.invalid_actuation.cpu().numpy().copy())
            env.sim.step()
            env.scene.update(dt=env.physics_dt)
            recorded["gpu_qpos"].append(env.sim.data.qpos.cpu().numpy().copy())
            recorded["gpu_qvel"].append(env.sim.data.qvel.cpu().numpy().copy())
            for index, (pd, same) in enumerate(zip(attached_native_pd, attached_same_ctrl, strict=True)):
                _, applied, invalid, _ = native_model_pd_numpy(
                    arrays["target"][index].astype(np.float64), pd.qpos[7:], pd.qvel[6:], profile
                )
                if invalid:
                    raise ValueError("attached CPU native PD became invalid")
                pd.ctrl[:] = applied
                same.ctrl[:] = ctrl[index]
                mujoco.mj_step(env.sim.mj_model, pd)
                mujoco.mj_step(env.sim.mj_model, same)
            for key, values in (("cpu_attached_pd", attached_native_pd), ("cpu_same_ctrl", attached_same_ctrl)):
                recorded[key + "_qpos"].append(np.stack([v.qpos.copy() for v in values]))
                recorded[key + "_qvel"].append(np.stack([v.qvel.copy() for v in values]))
            recorded["cpu_attached_pd_ctrl"].append(np.stack([v.ctrl.copy() for v in attached_native_pd]))
        arrays.update({key: np.stack(value, axis=1) for key, value in recorded.items()})
        if arrays["gpu_invalid"].any() or np.max(np.abs(arrays["gpu_ctrl"]) - np.asarray(profile.effort)) > 1e-5:
            raise ValueError("sampled GPU actuation violated the existing motor boundary")
        for key in ("gpu_qpos", "cpu_attached_pd_qpos", "cpu_same_ctrl_qpos"):
            arrays[key + "_local"] = arrays[key].astype(np.float64).copy()
            arrays[key + "_local"][..., :3] -= origins[:, None]
        for index, row in enumerate(rows):
            for label, actual, expected in (
                ("gpu_vs_original_qpos", "gpu_qpos_local", "expected_qpos"),
                ("gpu_vs_original_qvel", "gpu_qvel", "expected_qvel"),
                ("attached_cpu_vs_original_qpos", "cpu_attached_pd_qpos_local", "expected_qpos"),
                ("attached_cpu_vs_original_qvel", "cpu_attached_pd_qvel", "expected_qvel"),
                ("gpu_vs_cpu_same_controls_qpos", "gpu_qpos", "cpu_same_ctrl_qpos"),
                ("gpu_vs_cpu_same_controls_qvel", "gpu_qvel", "cpu_same_ctrl_qvel"),
            ):
                error = np.abs(arrays[actual][index].astype(np.float64) - arrays[expected][index])
                row[label] = dict(max_abs=float(error.max()), final_max_abs=float(error[-1].max()))
        mujoco.mj_saveModel(env.sim.mj_model, str(output / "training.mjb"))
    finally:
        env.close()
    for path, digest in inputs.items():
        if sha256_file(Path(path)) != digest:
            raise ValueError("physics probe source/input changed during execution")
    with (output / "samples.npz").open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    report = dict(
        kind="native23_same_state_one_control_training_physics_probe_v1",
        inputs=inputs,
        samples=rows,
        precision=precision,
        scene_parameter_parity=parity,
        bit_exact_original_cpu_recovery_substeps=reexecuted,
        independent_gpu_probes=32,
        physics_substeps_per_probe=10,
        diagnostic_state_injection_once_per_probe=True,
        intermediate_qpos_qvel_or_warmstart_writes=False,
        hold_actual_saved_targets=True,
        same_gpu_controls_replayed_on_attached_cpu_model=True,
        policy_inference_or_updates=0,
        full_closed_loop_or_hardware_parity_proven=False,
        deployment_ready=False,
        hardware_authorized=False,
        artifacts={name: sha256_file(output / name) for name in ("samples.npz", "training.mjb")},
    )
    with (output / "report.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(
        json.dumps(
            {
                "completed": True,
                "report_sha256": sha256_file(output / "report.json"),
                "maxima": {
                    key: max(row[key]["max_abs"] for row in rows)
                    for key in rows[0]
                    if key.endswith(("qpos", "qvel"))
                },
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    from gear_sonic.utils.g1_true23_training_precision import ieee_training_precision

    with ieee_training_precision() as (precision, guard):
        main(precision, guard)
