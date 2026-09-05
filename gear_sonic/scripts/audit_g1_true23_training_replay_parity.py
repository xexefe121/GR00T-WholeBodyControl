"""Same-state training/replay diagnostic, not learning or robot qualification.

Samples every corpus clip without reset rejection. Observation corruption and
reset perturbations are disabled explicitly to isolate deterministic mappings.
The actual MJLab action manager and one 2 ms Warp step are compared with CPU
MuJoCo on the same compiled training model and the pinned replay mesh model.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

OPTION_FIELDS = (
    "timestep",
    "integrator",
    "cone",
    "solver",
    "iterations",
    "ls_iterations",
    "tolerance",
    "ls_tolerance",
    "ccd_iterations",
    "ccd_tolerance",
    "impratio",
    "gravity",
    "wind",
    "density",
    "viscosity",
    "disableflags",
    "enableflags",
    "jacobian",
)
ROOT_DOF_FIELDS = ("dof_armature", "dof_damping", "dof_frictionloss")


def clip_sample_plan(spans, total):
    """Four real history/proof-complete samples per clip, including endpoints."""
    rows, cursor = [], 0
    for span in spans:
        start, length = span["start"], span["length"]
        if type(start) is not int or type(length) is not int or start != cursor or length < 16:
            raise ValueError("parity sampling requires contiguous complete clips of at least 16 frames")
        for phase in np.linspace(start + 12, start + length - 3, 4).astype(int):
            rows.append({"clip": span["name"], "phase": int(phase), "stop": start + length - 2})
        cursor += length
    if not rows or cursor != total:
        raise ValueError("parity span coverage does not match corpus")
    return rows


def differences(left, right, atol):
    if isinstance(atol, bool) or not np.isscalar(atol) or not np.isfinite(atol) or atol < 0:
        raise ValueError("parity tolerance must be finite and nonnegative")
    left, right = np.asarray(left), np.asarray(right)
    if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("parity comparison requires equal-shape finite arrays")
    delta = np.abs(left.astype(float) - right.astype(float))
    return {
        "maximum_absolute_difference": float(delta.max(initial=0)),
        "absolute_tolerance": atol,
        "within_tolerance": bool(np.all(delta <= atol)),
    }


def file_hashes(paths):
    return {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def validate_free_root(model):
    if model.njnt < 1 or model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE or model.jnt_dofadr[0] != 0:
        raise ValueError("diagnostic requires a leading six-DoF free joint")


def validate_native_layout(model, prefix):
    from gear_sonic.utils.g1_23dof_contract import HARDWARE_23_JOINT_NAMES

    validate_free_root(model)
    if (model.nq, model.nv, model.nu) != (30, 29, 23):
        raise ValueError("parity requires exact native23 dimensions")
    ids = np.array([model.joint(prefix + name).id for name in HARDWARE_23_JOINT_NAMES])
    if (
        not np.array_equal(model.jnt_qposadr[ids], np.arange(7, 30))
        or not np.array_equal(model.jnt_dofadr[ids], np.arange(6, 29))
        or not np.array_equal(model.actuator_trnid[:, 0], ids)
        or not np.all(model.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT)
        or not np.all(model.jnt_type[ids] == mujoco.mjtJoint.mjJNT_HINGE)
    ):
        raise ValueError("parity joint/actuator state layout differs from hardware23 order")


def model_counterfactuals(training, replay):
    """Change diagnostic copies only, never the supplied models or XML pins.

    Root-only, solver/options-only and combined ablations do not claim that
    remaining differences are geometry alone. Contact parameters also differ.
    """
    for model in (training, replay):
        validate_free_root(model)
    models = {}
    for name in ("root_only", "options_only", "root_and_options"):
        model = copy.copy(replay)
        if name in ("root_only", "root_and_options"):
            for field in ROOT_DOF_FIELDS:
                getattr(model, field)[:6] = getattr(training, field)[:6]
        if name in ("options_only", "root_and_options"):
            for field in OPTION_FIELDS:
                value = getattr(training.opt, field)
                if isinstance(value, np.ndarray):
                    getattr(model.opt, field)[:] = value
                else:
                    setattr(model.opt, field, value)
        mujoco.mj_setConst(model, mujoco.MjData(model))
        models[name] = model
    return models


def collision_parameters(model):
    rows = []
    for i in range(model.ngeom):
        if not model.geom_contype[i] and not model.geom_conaffinity[i]:
            continue
        row = {"name": model.geom(i).name, "body": model.body(int(model.geom_bodyid[i])).name}
        for field in (
            "type",
            "size",
            "pos",
            "quat",
            "contype",
            "conaffinity",
            "condim",
            "priority",
            "solmix",
            "solref",
            "solimp",
            "friction",
            "margin",
            "gap",
        ):
            row[field] = np.asarray(getattr(model, "geom_" + field)[i]).tolist()
        rows.append(row)
    return rows


def compare_model_parameters(training, replay):
    from gear_sonic.utils.g1_23dof_contract import HARDWARE_23_JOINT_NAMES

    joints = [
        [model.joint(prefix + name).id for name in HARDWARE_23_JOINT_NAMES]
        for model, prefix in ((training, "robot/"), (replay, ""))
    ]
    dofs = [model.jnt_dofadr[ids] for model, ids in zip((training, replay), joints)]
    reports = {}
    for field in ROOT_DOF_FIELDS:
        a, b = [getattr(model, field)[ids] for model, ids in zip((training, replay), dofs)]
        reports[field] = {**differences(a, b, 1e-9), "training": a.tolist(), "replay": b.tolist()}
        a, b = [getattr(model, field)[:6] for model in (training, replay)]
        reports["free_root_" + field] = {**differences(a, b, 1e-9), "training": a.tolist(), "replay": b.tolist()}
    names = [replay.body(i).name for i in range(1, replay.nbody)]
    body_ids = [
        [model.body(prefix + name).id for name in names] for model, prefix in ((training, "robot/"), (replay, ""))
    ]
    for field in ("body_mass", "body_inertia", "body_ipos", "body_iquat"):
        a, b = [getattr(model, field)[ids] for model, ids in zip((training, replay), body_ids)]
        reports[field] = differences(a, b, 1e-9)
    for field in (
        "actuator_gear",
        "actuator_gainprm",
        "actuator_biasprm",
        "actuator_forcerange",
        "actuator_forcelimited",
        "actuator_ctrlrange",
        "actuator_ctrllimited",
        "actuator_dyntype",
        "actuator_gaintype",
        "actuator_biastype",
    ):
        reports[field] = differences(getattr(training, field), getattr(replay, field), 1e-9)
    for field in ("jnt_actfrclimited", "jnt_actfrcrange", "jnt_range", "jnt_solref", "jnt_solimp"):
        a, b = [getattr(model, field)[ids] for model, ids in zip((training, replay), joints)]
        reports[field] = differences(a, b, 1e-9)
    for field in OPTION_FIELDS:
        reports[field] = {
            "training": np.asarray(getattr(training.opt, field)).tolist(),
            "replay": np.asarray(getattr(replay.opt, field)).tolist(),
        }
    reports["collision_geometry_and_parameters"] = {
        "training": collision_parameters(training),
        "replay": collision_parameters(replay),
    }
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--motion-file", type=Path, required=True)
    parser.add_argument("--spans", type=Path, required=True)
    parser.add_argument("--encoder-report", type=Path, required=True)
    parser.add_argument("--decoder-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("parity audit requires a new output directory")

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.utils.torch import configure_torch_backends
    import torch

    from gear_sonic.envs.mjlab import sonic_true23_causal_history as causal_task
    from gear_sonic.scripts import train_g1_23dof_mjlab_frozen_lora as launcher
    from gear_sonic.scripts.audit_g1_true23_exploration_rollout import scale_reset_perturbations
    from gear_sonic.scripts.evaluate_g1_true23_deployment_envelope import _policy, load_diagnostic_pair
    from gear_sonic.utils.g1_23dof_safe_target_transform import safe_target_transform_numpy
    from gear_sonic.utils.g1_true23_actuation_profile import SIM_CONFIG, NativeSupportActuationProfile
    from gear_sonic.utils.g1_true23_clean_mujoco_teleop import (
        CleanTrue23MujocoController,
        encoder267_from_reference,
        motion_reference_terms,
        term_major_history,
    )
    from gear_sonic.utils.g1_true23_projected_controller_state import applied_target_native_numpy
    from gear_sonic.utils.g1_true23_reference_floor import compiled_model_sha256
    from gear_sonic.utils.g1_true23_sim_acquisition import TargetIntersectionError, effort_feasible_target
    from gear_sonic.utils.g1_true23_step1b_mujoco import _body_velocity, world_angular_velocity_to_body

    root = Path(__file__).resolve().parents[2]
    spans = launcher._validate_span_sidecar(args.spans)
    if hashlib.sha256(args.motion_file.read_bytes()).hexdigest() != spans["corpus_sha256"]:
        raise ValueError("parity corpus bytes do not match span sidecar")
    with np.load(args.motion_file, allow_pickle=False) as data:
        motion = {key: data[key].copy() for key in data.files}
    samples = clip_sample_plan(spans["spans"], len(motion["joint_pos"]))
    pair = load_diagnostic_pair(args.encoder_report, args.decoder_report)
    policy = _policy(
        Path(pair["encoder"]["path"]),
        Path(pair["decoder"]["path"]),
        pair["decoder"]["sha256"],
        encoder_hash=pair["encoder"]["sha256"],
    )
    profile = replace(
        NativeSupportActuationProfile.from_sim_config(root / SIM_CONFIG), consistent_controller_state=True
    )
    launcher._install_frozen_lora_hooks(
        source_checkpoint=args.asset_root / "low_latency/last.pt",
        lora_rank=8,
        lora_alpha=8,
        phase="breadth",
        span_sidecar=args.spans.resolve(),
        behavior_bank=None,
        bank_indices=(),
        adapter_initialization=None,
        adapter_initialization_mode=False,
        actuation_profile=profile,
    )
    configure_torch_backends()
    torch.cuda.set_device(0)
    torch.manual_seed(20260906)
    cfg = causal_task.make_causal_history_recovery_env_cfg(
        motion_file=str(args.motion_file.resolve()), num_envs=len(samples), play=False
    )
    cfg.seed = 20260906
    cfg.commands["motion"].sampling_mode = "uniform"
    scale_reset_perturbations(cfg.commands["motion"], 0.0)
    for name in ("tokenizer", "policy"):
        cfg.observations[name].enable_corruption = False
    source_paths = [
        Path(__file__),
        *launcher.base.CAUSAL_SOURCE_FILES,
        *launcher.base._source_files().values(),
        Path(inspect.getfile(ManagerBasedRlEnv)),
        root / "gear_sonic/scripts/evaluate_g1_true23_deployment_envelope.py",
        root / "gear_sonic/utils/g1_true23_clean_mujoco_teleop.py",
        root / "gear_sonic/utils/g1_true23_sim_acquisition.py",
        root / "gear_sonic/utils/g1_23dof_safe_target_transform.py",
        root / "gear_sonic/utils/g1_true23_projected_controller_state.py",
        root / "gear_sonic/utils/g1_true23_step1b_mujoco.py",
        root / "gear_sonic/utils/g1_23dof_mujoco_sim2sim.py",
        root / "gear_sonic/utils/g1_true23_reference_floor.py",
    ]
    input_paths = [
        args.motion_file,
        args.spans,
        args.encoder_report,
        args.decoder_report,
        Path(pair["encoder"]["path"]),
        Path(pair["decoder"]["path"]),
        root / SIM_CONFIG,
        args.asset_root / "gear_sonic/data/robots/g1/g1_23dof_rev_1_0.xml",
    ]
    replay_xml = input_paths[-1]
    input_paths.append(
        args.asset_root / "external_dependencies/unitree_rl_mjlab/src/assets/robots/unitree_g1/xmls/g1_23dof.xml"
    )
    inputs, sources = file_hashes(input_paths), file_hashes(source_paths)
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    arrays, rows = {}, []
    try:
        # Bind the modules actually imported, including editable MJLab/assets,
        # not only the launcher's nominal sibling source-path manifest.
        runtime_sources = {
            Path(module.__file__)
            for name, module in tuple(sys.modules.items())
            if name.startswith(("mjlab.", "src.assets.robots.unitree_g1", "src.tasks.tracking"))
            and getattr(module, "__file__", "")
            and str(module.__file__).endswith(".py")
        }
        source_paths.extend(sorted(runtime_sources))
        sources.update(file_hashes(runtime_sources))
        command = env.command_manager.get_term("motion")
        phases = torch.tensor([s["phase"] for s in samples], device=env.device)
        stops = torch.tensor([s["stop"] for s in samples], device=env.device)

        def fixed_samples(ids):
            command.time_steps[ids] = phases[ids]
            command._env_clip_stop[ids] = stops[ids]

        command._uniform_sampling = fixed_samples
        env.reset()
        command.refresh_relative_body_targets_after_reset()
        ids = torch.arange(len(samples), device=env.device)
        env.observation_manager.reset(ids)
        obs = env.observation_manager.compute(update_history=True)
        action = env.action_manager.get_term("joint_pos")
        cpu_training = env.sim.mj_model
        replay = CleanTrue23MujocoController(model_path=replay_xml, physics_path=root / SIM_CONFIG, policy=policy)
        validate_native_layout(cpu_training, "robot/")
        validate_native_layout(replay.model, "")
        initial_model_hash = compiled_model_sha256(cpu_training)
        replay_model_hash = compiled_model_sha256(replay.model)
        counterfactuals = model_counterfactuals(cpu_training, replay.model)
        arrays["qpos"] = env.sim.data.qpos.cpu().numpy().copy()
        arrays["qvel"] = env.sim.data.qvel.cpu().numpy().copy()
        arrays["qacc_warmstart"] = env.sim.data.qacc_warmstart.cpu().numpy().copy()
        if torch.any(env.sim.data.qfrc_applied != 0) or torch.any(env.sim.data.xfrc_applied != 0):
            raise ValueError("same-state parity requires no unmodeled external forces")
        arrays["target"] = action._previous_targets.cpu().numpy().copy()
        arrays["training_semantic"] = obs["tokenizer"][:, 1:].cpu().numpy().copy()
        arrays["training_policy"] = obs["policy"].cpu().numpy().copy()
        arrays["initial_invalid"] = action.envelope_violation.cpu().numpy().copy()
        replay_semantic, replay_proprio, means, replay_means, tokens, replay_tokens = [], [], [], [], [], []

        def infer(semantic, proprio):
            token = policy.encoder.run(None, {policy.encoder_input: semantic.reshape(1, 267)})[0]
            raw = policy.decoder.run(
                None, {policy.decoder_input: np.concatenate((token.reshape(64), proprio)).reshape(1, 994)}
            )[0]
            return token.reshape(64), raw.reshape(23)

        for index, sample in enumerate(samples):
            qpos, qvel = arrays["qpos"][index], arrays["qvel"][index]
            replay.reset(
                base_position=qpos[:3],
                base_quaternion_wxyz=qpos[3:7],
                joint_position_hardware=qpos[7:],
                joint_velocity_hardware=qvel[6:],
                root_velocity=qvel[:6],
            )
            replay.previous_safe_native = applied_target_native_numpy(arrays["target"][index])
            packet = motion_reference_terms(motion, sample["phase"])
            semantic = encoder267_from_reference(packet, command.causal_robot_anchor_quat_w[index].cpu().numpy())
            proprio = term_major_history([replay._policy_frame() for _ in range(10)])
            token, raw = infer(arrays["training_semantic"][index], arrays["training_policy"][index])
            other_token, other_raw = infer(semantic, proprio)
            replay_semantic.append(semantic)
            replay_proprio.append(proprio)
            means.append(raw)
            replay_means.append(other_raw)
            tokens.append(token)
            replay_tokens.append(other_token)
            rows.append(
                {
                    **sample,
                    "initial_controller_invalid": bool(arrays["initial_invalid"][index]),
                    "semantic": differences(arrays["training_semantic"][index], semantic, 1e-5),
                    "proprioception": differences(arrays["training_policy"][index], proprio, 1e-5),
                    "tokens": differences(token, other_token, 0),
                    "raw_action": differences(raw, other_raw, 1e-4),
                }
            )
        arrays.update(
            replay_semantic=np.array(replay_semantic),
            replay_policy=np.array(replay_proprio),
            training_raw=np.array(means),
            replay_raw=np.array(replay_means),
            training_token=np.array(tokens),
            replay_token=np.array(replay_tokens),
        )
        env.action_manager.process_action(torch.as_tensor(arrays["training_raw"], device=env.device))
        arrays["requested_target"] = action._requested_targets.cpu().numpy().copy()
        env.action_manager.apply_action()
        env.scene.write_data_to_sim()
        arrays["applied_target"] = action._previous_targets.cpu().numpy().copy()
        arrays["ctrl"] = env.sim.data.ctrl.cpu().numpy().copy()
        arrays["step_invalid"] = action.envelope_violation.cpu().numpy().copy()
        env.sim.step()
        env.scene.update(dt=env.physics_dt)
        env.sim.forward()
        arrays["warp_next_qpos"] = env.sim.data.qpos.cpu().numpy().copy()
        arrays["warp_next_qvel"] = env.sim.data.qvel.cpu().numpy().copy()
        training_next, replay_next = [], []
        counterfactual_next = {name: [] for name in counterfactuals}
        for index, row in enumerate(rows):
            row["training_controller_invalid_after_step"] = bool(arrays["step_invalid"][index])
            _, requested = safe_target_transform_numpy(arrays["training_raw"][index])
            row["requested_target"] = differences(arrays["requested_target"][index], requested, 2e-6)
            try:
                target = effort_feasible_target(
                    arrays["requested_target"][index],
                    arrays["target"][index],
                    arrays["qpos"][index, 7:],
                    arrays["qvel"][index, 6:],
                    np.array(profile.kp),
                    np.array(profile.kd),
                    np.array(profile.effort),
                    dt=profile.timestep_s,
                    slew_rate=profile.slew_rad_s,
                )
            except TargetIntersectionError:
                row["numpy_projection_invalid"] = True
            else:
                row["numpy_projection_invalid"] = False
                row["applied_target"] = differences(arrays["applied_target"][index], target, 2e-6)
                torque = (
                    np.array(profile.kp) * (target - arrays["qpos"][index, 7:])
                    - np.array(profile.kd) * arrays["qvel"][index, 6:]
                )
                row["actuator_torque"] = differences(arrays["ctrl"][index], torque, 2e-5)
            row["controller_status_agrees"] = (
                row["training_controller_invalid_after_step"] == row["numpy_projection_invalid"]
            )
            cases = {"training": cpu_training, "replay": replay.model, **counterfactuals}
            row["cpu_contact_count_at_step_start"] = {}
            for name, model in cases.items():
                data = mujoco.MjData(model)
                data.qpos[:] = arrays["qpos"][index]
                data.qvel[:] = arrays["qvel"][index]
                data.qacc_warmstart[:] = arrays["qacc_warmstart"][index]
                data.ctrl[:] = arrays["ctrl"][index]
                mujoco.mj_step(model, data)
                row["cpu_contact_count_at_step_start"][name] = int(data.ncon)
                collection = (
                    training_next
                    if name == "training"
                    else replay_next
                    if name == "replay"
                    else counterfactual_next[name]
                )
                collection.append(data.qvel.copy())
                if name == "replay":
                    _, old_gyro_world = _body_velocity(mujoco, model, data, "pelvis", (0.0, 0.0, 0.0))
                    old_gyro = world_angular_velocity_to_body(data.qpos[3:7], old_gyro_world)
                    replay.data = data
                    row["cached_gyro_vs_current_state"] = differences(old_gyro, data.qvel[3:6], 1e-5)
                    row["policy_gyro_vs_current_state"] = differences(
                        replay._policy_frame()[:3], data.qvel[3:6], 1e-5
                    )
            row["warp_vs_cpu_training_next_qvel"] = differences(
                arrays["warp_next_qvel"][index], training_next[-1], 1e-3
            )
            row["training_vs_replay_next_qvel"] = differences(training_next[-1], replay_next[-1], 1e-3)
            row["training_vs_counterfactual_next_qvel"] = {
                name: differences(training_next[-1], values[-1], 1e-3)
                for name, values in counterfactual_next.items()
            }
        arrays.update(cpu_training_next_qvel=np.array(training_next), cpu_replay_next_qvel=np.array(replay_next))
        arrays.update(
            {"cpu_" + name + "_next_qvel": np.array(values) for name, values in counterfactual_next.items()}
        )
        parameters = compare_model_parameters(cpu_training, replay.model)
        if compiled_model_sha256(cpu_training) != initial_model_hash:
            raise RuntimeError("parity audit changed the compiled training model")
        if compiled_model_sha256(replay.model) != replay_model_hash:
            raise RuntimeError("parity audit changed the pinned replay model")
    finally:
        env.close()
    if inputs != file_hashes(input_paths) or sources != file_hashes(source_paths):
        raise RuntimeError("parity inputs or sources changed during execution")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "trace.npz").open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    mujoco.mj_saveModel(cpu_training, str(args.output_dir / "training.mjb"))
    mujoco.mj_saveModel(replay.model, str(args.output_dir / "replay.mjb"))
    report = {
        "kind": "g1_true23_sampled_training_replay_parity_v2",
        "samples": rows,
        "inputs": inputs,
        "sources": sources,
        "diagnostic_pair": pair,
        "model_parameters": parameters,
        "training_model_sha256": initial_model_hash,
        "replay_model_sha256": replay_model_hash,
        "runtime_versions": {
            name: importlib.metadata.version(name)
            for name in ("mujoco", "mujoco-warp", "warp-lang", "torch", "mjlab", "onnxruntime", "numpy")
        },
        "output_hashes": file_hashes(
            [args.output_dir / name for name in ("trace.npz", "training.mjb", "replay.mjb")]
        ),
        "actuation_profile": profile.contract(),
        "scope": "four samples per clip and one 2ms step, not full-clip tracking",
        "reset_samples_rejected_or_dropped": 0,
        "observation_corruption": False,
        "reset_perturbation_scale": 0.0,
        "learning_performed": False,
        "hardware_authorized": False,
        "deployment_ready": False,
    }
    with (args.output_dir / "report.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(
        json.dumps(
            {
                "samples": len(rows),
                "observation_disagreements": sum(
                    not r["semantic"]["within_tolerance"] or not r["proprioception"]["within_tolerance"]
                    for r in rows
                ),
                "token_disagreements": sum(not r["tokens"]["within_tolerance"] for r in rows),
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
