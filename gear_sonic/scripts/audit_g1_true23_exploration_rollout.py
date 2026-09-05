"""No-learning MJLab rollout audit of frozen SONIC exploration noise.

Uses the complete supplied corpus and the unchanged native-support V2 profile.
This is a sampled training-distribution diagnostic, not a full-clip, measured
acquisition, hardware, or original-v14 parity qualification. Run each noise
scale in a separate process so process-local training hooks remain isolated.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path

import numpy as np


def summarize_episodes(dones, terms, invalid_before_step):
    """Count actual done episodes; retain unfinished lengths as censored data."""
    dones = np.asarray(dones)
    invalid_before_step = np.asarray(invalid_before_step)
    if dones.ndim != 2 or dones.dtype != bool or invalid_before_step.shape != dones.shape:
        raise ValueError("rollout masks require matching [steps, environments] boolean arrays")
    if invalid_before_step.dtype != bool or not dones.size or not terms:
        raise ValueError("rollout masks require nonempty boolean evidence")
    arrays = {key: np.asarray(value) for key, value in terms.items()}
    if any(value.shape != dones.shape or value.dtype != bool for value in arrays.values()):
        raise ValueError("termination masks differ from done shape or dtype")
    if not np.array_equal(np.logical_or.reduce(list(arrays.values())), dones):
        raise ValueError("termination union differs from actual done signals")
    lengths = np.zeros(dones.shape[1], dtype=np.int64)
    completed = []
    initial_invalid = np.zeros_like(dones)
    for step, row in enumerate(dones):
        initial_invalid[step] = (lengths == 0) & invalid_before_step[step]
        lengths += 1
        completed.extend(lengths[row].tolist())
        lengths[row] = 0
    completed = np.asarray(completed, dtype=np.int64)
    guard = arrays.get("stage_one_actuation_guard", np.zeros_like(dones))
    return {
        "control_steps_per_environment": len(dones),
        "sampled_transitions": int(dones.size),
        "completed_episodes": int(len(completed)),
        "completed_episode_mean_steps": float(completed.mean()) if completed.size else None,
        "completed_episode_median_steps": float(np.median(completed)) if completed.size else None,
        "completed_episode_maximum_steps": int(completed.max()) if completed.size else None,
        "one_step_completed_episodes": int(np.count_nonzero(completed == 1)),
        "unfinished_episode_lengths_steps": lengths.tolist(),
        "termination_counts_nonexclusive": {key: int(value.sum()) for key, value in arrays.items()},
        "invalid_controller_at_episode_start": int(initial_invalid.sum()),
        "guard_terminations_from_invalid_episode_start": int((guard & initial_invalid).sum()),
        "guard_terminations_without_invalid_start_that_step": int((guard & ~initial_invalid).sum()),
    }


def _noise_scale(value):
    result = float(value)
    if not np.isfinite(result) or not 0 <= result <= 1:
        raise argparse.ArgumentTypeError("noise scale must be finite and within 0..1")
    return result


def _positive(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def _hash_files(paths):
    return {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def projection_failure_details(requested, previous, q, dq, profile):
    """Independent NumPy diagnostic of a Torch-flagged state, not new control."""
    from gear_sonic.utils.g1_true23_sim_acquisition import TargetIntersectionError, effort_feasible_target

    try:
        effort_feasible_target(
            *[np.asarray(value, dtype=np.float64) for value in (requested, previous, q, dq)],
            *[np.asarray(getattr(profile, name)) for name in ("kp", "kd", "effort")],
            dt=profile.timestep_s,
            slew_rate=profile.slew_rad_s,
        )
    except TargetIntersectionError as error:
        return {"numpy_confirms_empty_interval": True, "joints": error.details}
    return {"numpy_confirms_empty_interval": False, "joints": []}


def audit_reset_contacts(model, qpos):
    """Inspect actual compiled geometry in separate CPU data, without stepping."""
    import mujoco

    qpos = np.asarray(qpos)
    if qpos.ndim != 2 or qpos.shape[1] != model.nq or not np.isfinite(qpos).all():
        raise ValueError("contact audit requires finite [environments, nq] positions")
    probe = mujoco.MjData(model)
    plane_ids = set(np.flatnonzero(model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE).tolist())
    rows = []
    for index, position in enumerate(qpos):
        probe.qpos[:] = position
        mujoco.mj_fwdPosition(model, probe)
        contacts = []
        for contact in probe.contact[: probe.ncon]:
            first, second = map(int, contact.geom)
            contacts.append(
                {
                    "geom1": model.geom(first).name,
                    "geom2": model.geom(second).name,
                    "floor_contact": first in plane_ids or second in plane_ids,
                    "distance_m": float(contact.dist),
                }
            )
        floor = [row for row in contacts if row["floor_contact"]]
        rows.append(
            {
                "env_index": index,
                "minimum_floor_contact_distance_m": min((row["distance_m"] for row in floor), default=None),
                "floor_penetration_contact_count": sum(row["distance_m"] < 0 for row in floor),
                "worst_contacts": sorted(contacts, key=lambda row: row["distance_m"])[:6],
            }
        )
    return {
        "method": "compiled_model_cpu_position_forward_on_independent_data_no_physics_step",
        "negative_distance_means_geometric_overlap": True,
        "dynamics_or_impulse_causation_proven": False,
        "rows": rows,
    }


def lift_reset_floor_overlap(model, qpos, *, clearance_m=1e-5, maximum_lift_m=0.2):
    """Diagnostic reset intervention, not contact retargeting or robot control.

    Lift only penetrated states of a single floating articulation over a flat
    world plane. Preserve every joint, orientation and horizontal coordinate;
    velocities are not inputs and must be preserved by the caller. Never lower
    airborne states. Reject unsupported geometry or excessive corrections.
    """
    import mujoco

    if not np.isfinite([clearance_m, maximum_lift_m]).all() or not 0 < clearance_m < maximum_lift_m:
        raise ValueError("floor-lift bounds must be finite, positive and ordered")
    free = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    planes = np.flatnonzero(model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE)
    if len(free) != 1 or len(planes) != 1:
        raise ValueError("floor lift requires one free articulation and one static world plane")
    plane_body = int(model.geom_bodyid[planes[0]])
    while plane_body != 0:
        if model.body_jntnum[plane_body] != 0 or model.body_mocapid[plane_body] >= 0:
            raise ValueError("floor lift requires a plane welded to the world")
        plane_body = int(model.body_parentid[plane_body])
    probe = mujoco.MjData(model)
    mujoco.mj_fwdPosition(model, probe)
    if not np.allclose(probe.geom_xmat[planes[0]].reshape(3, 3)[:, 2], [0, 0, 1], atol=1e-8, rtol=0):
        raise ValueError("floor lift requires a horizontal upward-facing plane")
    root_body = int(model.jnt_bodyid[free[0]])
    for geom_body in model.geom_bodyid[np.arange(model.ngeom) != planes[0]]:
        body = int(geom_body)
        while body not in (0, root_body):
            body = int(model.body_parentid[body])
        if body != root_body:
            raise ValueError("floor lift requires all non-plane geometry on the free articulation")
    before = audit_reset_contacts(model, qpos)
    lifted = np.array(qpos, copy=True)
    if lifted.dtype.kind != "f":
        raise ValueError("floor-lift positions must have floating dtype")
    root_z = int(model.jnt_qposadr[free[0]]) + 2
    lifts = np.array(
        [
            max(0.0, clearance_m - row["minimum_floor_contact_distance_m"])
            if row["minimum_floor_contact_distance_m"] is not None and row["minimum_floor_contact_distance_m"] < 0
            else 0.0
            for row in before["rows"]
        ]
    )
    if np.any(lifts > maximum_lift_m):
        raise ValueError("reset floor penetration exceeds diagnostic lift bound")
    lifted[:, root_z] += lifts
    after = audit_reset_contacts(model, lifted)
    if any(
        row["minimum_floor_contact_distance_m"] is not None and row["minimum_floor_contact_distance_m"] < 0
        for row in after["rows"]
    ):
        raise ValueError("floor-lift correction left geometric penetration")
    return lifted, {
        "clearance_m": clearance_m,
        "maximum_lift_m": maximum_lift_m,
        "root_z_qpos_index": root_z,
        "actual_lifts_m": (lifted[:, root_z] - np.asarray(qpos)[:, root_z]).tolist(),
        "before": before,
        "after": after,
        "reference_motion_changed": False,
        "dynamic_contact_consistency_proven": False,
    }


def scale_reset_perturbations(cfg, scale):
    """Ablate reset perturbations only; source state/velocity stay unchanged."""
    scale = _noise_scale(str(scale))
    cfg.pose_range = {key: tuple(scale * x for x in value) for key, value in cfg.pose_range.items()}
    cfg.velocity_range = {key: tuple(scale * x for x in value) for key, value in cfg.velocity_range.items()}
    cfg.joint_position_range = tuple(scale * x for x in cfg.joint_position_range)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--warm-start", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--motion-file", type=Path, required=True)
    parser.add_argument("--motion-metadata", type=Path, required=True)
    parser.add_argument("--spans", type=Path, required=True)
    parser.add_argument("--noise-scale", type=_noise_scale, required=True)
    parser.add_argument("--steps", type=_positive, default=128)
    parser.add_argument("--num-envs", type=_positive, default=32)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--capture-actuation-failures", action="store_true")
    parser.add_argument("--audit-reset-contacts", action="store_true")
    parser.add_argument("--reset-perturbation-scale", type=_noise_scale, default=1.0)
    parser.add_argument("--lift-reset-floor-overlap", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("rollout audit requires a new output directory")
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.utils.torch import configure_torch_backends
    import torch

    from gear_sonic.envs.mjlab import (
        sonic_true23_causal_history as causal_task,
        sonic_true23_stage_one_actuation as actuation_module,
    )
    from gear_sonic.envs.mjlab.sonic_true23 import prime_sonic_true23_training_environment
    from gear_sonic.scripts import train_g1_23dof_mjlab_frozen_lora as launcher
    from gear_sonic.trl.mjlab.frozen_platform_lora_actor import FrozenPlatformTrue23Core
    from gear_sonic.trl.mjlab.frozen_platform_lora_runner import (
        _state_sha256,
        load_frozen_platform_lora_checkpoint,
    )
    from gear_sonic.utils.g1_true23_actuation_profile import SIM_CONFIG, NativeSupportActuationProfile

    profile = replace(
        NativeSupportActuationProfile.from_sim_config(launcher.REPO_ROOT / SIM_CONFIG),
        consistent_controller_state=True,
    )
    launcher._validate_span_sidecar(args.spans)
    launcher._install_frozen_lora_hooks(
        source_checkpoint=args.source_checkpoint.resolve(),
        lora_rank=8,
        lora_alpha=8.0,
        phase="breadth",
        span_sidecar=args.spans.resolve(),
        behavior_bank=None,
        bank_indices=(),
        adapter_initialization=None,
        adapter_initialization_mode=False,
        actuation_profile=profile,
    )
    preflight_args = launcher.base._parser().parse_args(
        [
            "preflight",
            "--warm-start",
            str(args.warm_start),
            "--motion-file",
            str(args.motion_file),
            "--motion-metadata",
            str(args.motion_metadata),
            "--run-dir",
            str(args.output_dir),
            "--num-envs",
            str(args.num_envs),
            "--seed",
            str(args.seed),
        ]
    )
    preflight = launcher.base.preflight(preflight_args)
    if not preflight["ready"]:
        raise ValueError(f"training-material preflight failed: {preflight['problems']}")
    source_paths = list(launcher.base._source_files().values()) + list(launcher.base.CAUSAL_SOURCE_FILES)
    source_paths += [
        Path(__file__),
        Path(inspect.getfile(ManagerBasedRlEnv)),
        Path(inspect.getfile(RslRlVecEnvWrapper)),
        launcher.REPO_ROOT / "gear_sonic/utils/g1_true23_sim_acquisition.py",
    ]
    sources = _hash_files(source_paths)
    input_paths = [args.source_checkpoint, args.warm_start, args.motion_file, args.motion_metadata, args.spans]
    if args.checkpoint is not None:
        input_paths.append(args.checkpoint)
    inputs = _hash_files(input_paths)

    configure_torch_backends()
    torch.cuda.set_device(0)
    torch.manual_seed(args.seed)
    device = "cuda:0"
    cfg = causal_task.make_causal_history_recovery_env_cfg(
        motion_file=str(args.motion_file.resolve()), num_envs=args.num_envs, play=False
    )
    cfg.seed = args.seed
    scale_reset_perturbations(cfg.commands["motion"], args.reset_perturbation_scale)
    env = ManagerBasedRlEnv(cfg=cfg, device=device)
    original_pd_step = actuation_module.native_support_pd_step
    failure_records = []
    reset_lifts = []
    command = env.command_manager.get_term("motion")
    original_resample = command._resample_command
    try:
        if args.lift_reset_floor_overlap:

            def resample_with_floor_lift(env_ids):
                original_resample(env_ids)
                if not len(env_ids):
                    return
                qpos = env.sim.data.qpos[env_ids].detach().cpu().numpy().copy()
                qvel_before = env.sim.data.qvel[env_ids].clone()
                lifted, evidence = lift_reset_floor_overlap(env.sim.mj_model, qpos)
                root_start = evidence["root_z_qpos_index"] - 2
                root_pose = torch.as_tensor(lifted[:, root_start : root_start + 7], device=device)
                command.robot.write_root_link_pose_to_sim(root_pose, env_ids=env_ids)
                command.robot.clear_state(env_ids=env_ids)
                if not torch.equal(env.sim.data.qvel[env_ids], qvel_before):
                    raise RuntimeError("diagnostic floor lift changed reset velocities")
                if not torch.equal(env.sim.data.qpos[env_ids], torch.as_tensor(lifted, device=device)):
                    raise RuntimeError("diagnostic floor lift changed non-root-z reset positions")
                reset_lifts.append(
                    {
                        "control_step": int(env.common_step_counter),
                        "env_indices": env_ids.cpu().tolist(),
                        "source_phases": command.time_steps[env_ids].cpu().tolist(),
                        "applied_positions_and_unchanged_velocities_verified": True,
                        **evidence,
                    }
                )

            command._resample_command = resample_with_floor_lift
        wrapped = RslRlVecEnvWrapper(env, clip_actions=10.0)
        prime = prime_sonic_true23_training_environment(wrapped)
        core = FrozenPlatformTrue23Core(
            warm_start_path=args.warm_start,
            source_checkpoint_path=args.source_checkpoint,
            lora_rank=8,
            lora_alpha=8.0,
        )
        checkpoint_lineage = None
        if args.checkpoint is not None:
            checkpoint = load_frozen_platform_lora_checkpoint(
                args.checkpoint, expected_contract=core.adapter_contract()
            )
            core.load_lora_state_dict(checkpoint["adapter_state_dict"], strict=True)
            if core.merged_true23_policy_sha256(core.initial_std) != checkpoint["merged_true23_policy_sha256"]:
                raise ValueError("rollout checkpoint merged-policy hash mismatch")
            checkpoint_lineage = checkpoint["lineage_sha256"]
        adapter_before = _state_sha256(core.lora_state_dict())
        core.to(device).eval()
        std = core.initial_std.to(device)
        generator = torch.Generator(device=device).manual_seed(args.seed + 1)
        obs = wrapped.get_observations()
        action = env.action_manager.get_term("joint_pos")
        command = env.command_manager.get_term("motion")
        if args.capture_actuation_failures:

            def observed_pd_step(requested, previous, q, dq, current_profile):
                result = original_pd_step(requested, previous, q, dq, current_profile)
                newly_invalid = result[2] & ~action.envelope_violation
                for index in torch.where(newly_invalid)[0].tolist():
                    if len(failure_records) >= 128:
                        break
                    diagnostic = projection_failure_details(
                        *[value[index].detach().cpu().numpy().copy() for value in (requested, previous, q, dq)],
                        current_profile,
                    )
                    failure_records.append(
                        {
                            "env_index": index,
                            "control_step": int(env.common_step_counter),
                            "physics_step": int(env._sim_step_counter),
                            "substep": (int(env._sim_step_counter) - 1) % 10,
                            "source_phase": int(command.time_steps[index]),
                            **diagnostic,
                        }
                    )
                return result

            actuation_module.native_support_pd_step = observed_pd_step
        initial = {
            "q": action._entity.data.joint_pos[:, action._target_ids],
            "dq": action._entity.data.joint_vel[:, action._target_ids],
            "target": action._previous_targets,
            "tokenizer": obs["tokenizer"],
            "policy": obs["policy"],
            "phase": command.time_steps,
        }
        initial_hash = _state_sha256(initial)
        contacts = (
            audit_reset_contacts(env.sim.mj_model, env.sim.data.qpos.detach().cpu().numpy().copy())
            if args.audit_reset_contacts
            else None
        )
        initial_qpos = env.sim.data.qpos.detach().cpu().numpy().copy()
        initial_qvel = env.sim.data.qvel.detach().cpu().numpy().copy()
        initial_physics_hash = _state_sha256(
            {"qpos": torch.from_numpy(initial_qpos), "qvel": torch.from_numpy(initial_qvel)}
        )
        traces = {key: [] for key in ("done", "invalid_before_step", "phase", "q", "dq", "target", "mean", "raw")}
        terms = {name: [] for name in env.termination_manager.active_terms}
        with torch.inference_mode():
            for step in range(args.steps):
                semantic = obs["tokenizer"]
                if semantic.shape[-1] == 268:
                    semantic = semantic[:, 1:]
                mean = core(semantic, obs["policy"])
                # Dedicated RNG consumes the same Gaussian sequence per case;
                # later reset states still diverge when termination times differ.
                noise = torch.randn(mean.shape, device=device, generator=generator)
                raw = mean + args.noise_scale * std * noise
                for key, value in (
                    ("invalid_before_step", action.envelope_violation),
                    ("phase", command.time_steps),
                    ("q", action._entity.data.joint_pos[:, action._target_ids]),
                    ("dq", action._entity.data.joint_vel[:, action._target_ids]),
                    ("target", action._previous_targets),
                    ("mean", mean),
                    ("raw", raw),
                ):
                    traces[key].append(value.detach().cpu().numpy().copy())
                obs, _, done, _ = wrapped.step(raw)
                traces["done"].append(done.detach().cpu().numpy().astype(bool))
                # The installed manager preserves these computed masks across
                # reset; assert their union against the wrapper's actual dones.
                for name in terms:
                    terms[name].append(env.termination_manager.get_term(name).detach().cpu().numpy().copy())
                if (step + 1) % 64 == 0:
                    print(
                        json.dumps({"completed_control_steps": step + 1, "noise_scale": args.noise_scale}),
                        flush=True,
                    )
        arrays = {key: np.asarray(value) for key, value in traces.items()}
        arrays.update(initial_qpos=initial_qpos, initial_qvel=initial_qvel)
        arrays.update({f"termination_{name}": np.asarray(value) for name, value in terms.items()})
        summary = summarize_episodes(arrays["done"], terms, arrays["invalid_before_step"])
        core.assert_frozen_platform_unchanged()
        if _state_sha256(core.lora_state_dict()) != adapter_before:
            raise RuntimeError("no-learning audit changed adapter weights")
    finally:
        command._resample_command = original_resample
        actuation_module.native_support_pd_step = original_pd_step
        env.close()
    if _hash_files(source_paths) != sources or _hash_files(input_paths) != inputs:
        raise RuntimeError("rollout sources or inputs changed during audit")
    report = {
        "kind": "g1_true23_no_learning_exploration_rollout_v1",
        "noise_scale": args.noise_scale,
        "seed": args.seed,
        "noise_seed": args.seed + 1,
        "initial_state_sha256": initial_hash,
        "initial_physics_qpos_qvel_sha256": initial_physics_hash,
        "source_std_native23": std.cpu().tolist(),
        "checkpoint_lineage_sha256": checkpoint_lineage,
        "weights_updated": False,
        "optimizer_steps": 0,
        "initial_episode_lengths_randomized": False,
        "later_resets_paired_across_noise_cases": False,
        "sampling_mode": "adaptive_full_corpus",
        "reset_distribution": {
            "perturbation_scale": args.reset_perturbation_scale,
            "pose_range": cfg.commands["motion"].pose_range,
            "velocity_range": cfg.commands["motion"].velocity_range,
            "joint_position_range": cfg.commands["motion"].joint_position_range,
        },
        "observation_corruption_enabled": {
            name: bool(cfg.observations[name].enable_corruption) for name in ("tokenizer", "policy")
        },
        "initial_contact_audit": contacts,
        "reset_floor_lift_intervention": {
            "enabled": args.lift_reset_floor_overlap,
            "scope": "all_synthetic_resets_before_observation_and_controller_state_refresh",
            "production_training_changed": False,
            "events": reset_lifts,
        },
        "actuation_profile": profile.contract(),
        "episode_audit": summary,
        "actuation_failure_capture": {
            "enabled": args.capture_actuation_failures,
            "maximum_records": 128,
            "scope": "first_newly_infeasible_rows_before_training_guard_latches",
            "numeric_detail": "independent_float64_numpy_check_of_float32_torch_flagged_states",
            "records": failure_records,
        },
        "environment_prime": prime,
        "sources": sources,
        "inputs": inputs,
        "runtime": {
            name: importlib.metadata.version(name) for name in ("torch", "mjlab", "mujoco", "mujoco-warp")
        },
        "full_clip_fidelity_qualified": False,
        "hardware_authorized": False,
        "deployment_ready": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "rollout.npz").open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    report["rollout_sha256"] = hashlib.sha256((args.output_dir / "rollout.npz").read_bytes()).hexdigest()
    with (args.output_dir / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({"noise_scale": args.noise_scale, **summary}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
