"""SIM-only same-state observation probe; no policy update or control rollout.

Feed completed CPU trajectory states through the shared MJLab training config
and actual observation manager. Rebuild ten-frame histories from measured
states and applied commands, never by copying the expected history. Exclude
the first ten controls because their synthetic prehistory is not measured.
State injection is diagnostic-only and is NOT a successful dynamic trajectory.
"""

import argparse
import inspect
import json
from pathlib import Path
import sys

import numpy as np
import torch

from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file

CASES = (("walk002", "walk002"), ("walk003", "turn003"), ("dance", "sonic_happy_dance"))
TERMS = {
    "angular_velocity": (0, 30),
    "joint_position": (30, 320),
    "joint_velocity": (320, 610),
    "applied_action": (610, 900),
    "gravity": (900, 930),
}


def control_batches(count, batch_size=32):
    if type(count) is not int or count <= 10 or type(batch_size) is not int or batch_size <= 0:
        raise ValueError("probe requires more than ten measured controls and a positive batch size")
    for start in range(10, count, batch_size):
        real = np.arange(start, min(start + batch_size, count))
        yield np.pad(real, (0, batch_size - len(real)), mode="edge"), len(real)


def error_summary(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape or actual.ndim != 2:
        raise ValueError("probe requires matching [control, feature] matrices")
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError("probe input or result is not finite")
    error = np.abs(actual.astype(np.float64) - expected)
    worst = np.unravel_index(error.argmax(), error.shape)
    return dict(
        max_abs=float(error.max()),
        row_max_p95=float(np.percentile(error.max(axis=1), 95)),
        row_max_median=float(np.median(error.max(axis=1))),
        different_rows=int(np.any(error != 0, axis=1).sum()),
        worst_row=int(worst[0]),
        worst_feature=int(worst[1]),
    )


def main(precision, guard):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--bounded-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError("training boundary probe refuses overwrite")
    output.mkdir(parents=True)
    inputs = {}

    def bind(path, expected=None):
        path = Path(path).resolve(strict=True)
        digest = sha256_file(path)
        if expected is not None and digest != expected:
            raise ValueError(f"probe input changed: {path}")
        inputs[str(path)] = digest
        return path

    def read_npz(path):
        with np.load(bind(path), allow_pickle=False) as value:
            return {key: value[key].copy() for key in value.files}

    bind(args.experiment)
    bind(__file__)
    run = args.bounded_run.resolve(strict=True)
    audit = json.loads(
        bind(
            run / "cpu_audit.json", "fa31b9a3f1f426de0c674ef6d864198d41f9133e41dd7028242fbee5ec96b3a6"
        ).read_text()
    )
    # Revalidate predecessor proof inputs, not just a stale pass label.
    for path, digest in audit["inputs"].items():
        bind(path, digest)
    resolved = json.loads(bind(run / "train/resolved_training.json").read_text())
    runtime = json.loads(bind(run / "train/root_feedback_runtime.json").read_text())
    rows = runtime["training_inputs"]["derived_curriculum"]["derived_spans"]["spans"]
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
    from gear_sonic.utils.g1_23dof_safe_target_transform import safe_target_transform_torch
    from gear_sonic.utils.g1_true23_bounded_progress_checkpoint import load_cpu_actor
    from gear_sonic.utils.g1_true23_native_model_actuation import NativeModelActuationProfile
    from gear_sonic.utils.g1_true23_source_action_codec import SOURCE_ACTION_CONVENTION

    guard()
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    torch.manual_seed(resolved["seed"])
    physics = Path(__file__).resolve().parents[2] / "gear_sonic/config/sim_validation/g1_23dof_mujoco_sim2sim.json"
    profile = NativeModelActuationProfile.from_sim_config(bind(physics))
    install_root_feedback_command(rows, start_schedule="mixed_reference_reset_v1")
    cfg = make_causal_multimotion_v14_env_cfg(
        motion_file=spec["files"]["native_motion"]["path"], num_envs=32, play=False
    )
    cfg = apply_native_model_actuation_profile(cfg, profile)
    cfg = configure_root_feedback_environment(cfg, rows, objective_profile="root_and_upper_feet_world_v4")
    cfg = configure_release_compatible_environment(
        cfg, spec["files"]["source_model"]["path"], SOURCE_ACTION_CONVENTION
    )
    cfg = configure_nominal_scene(cfg, spec["files"]["native_model"]["path"], physics)
    cfg = configure_buffered_source_environment(cfg)
    cfg = intent.configure_original_intent_environment(cfg, spec)
    cfg = bounded.configure_environment(cfg)
    cfg.seed = resolved["seed"]
    noise = {name: cfg.observations[name].enable_corruption for name in ("tokenizer", "policy")}
    for name in noise:
        cfg.observations[name].enable_corruption = False
    checkpoint = bind(
        run / "train/checkpoints/bounded_progress_model_100.pt",
        "6efd755d281eb2d718e2a5c7ed1da75f4b9d3a5593c679fd855aae39d19c1cca",
    )
    original_root = Path(spec["files"]["native_model"]["path"]).parents[4]
    reader, identity, semantics = load_cpu_actor(
        checkpoint,
        warm_start_path=bind(original_root / "sonic_release/g1_23dof_rev_1_0_low_latency_init.pt"),
        source_checkpoint_path=bind(original_root / "low_latency/last.pt"),
    )
    inputs.update(semantics["reverified_repository_sources"])
    actor = reader.actor.to("cuda:0").eval()
    before_actor = {k: v.detach().cpu().clone() for k, v in actor.state_dict().items()}
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    summary = {}
    try:
        scene_proof = verify_nominal_scene(env.sim.mj_model, spec["files"]["native_model"]["path"], physics)
        bounded.verify_runtime(env)
        command = env.command_manager.get_term("motion")
        action = env.action_manager.get_term("joint_pos")
        ids = torch.arange(env.num_envs, device=env.device)
        intent._cache(command, spec)
        for name, module in tuple(sys.modules.items()):
            path = getattr(module, "__file__", None)
            if (
                name.startswith(("gear_sonic.", "mjlab.", "src.tasks.tracking"))
                and path
                and str(path).endswith(".py")
            ):
                bind(path)
        if env.sim.data.qpos.shape != (32, 30) or env.sim.data.qvel.shape != (32, 29):
            raise ValueError("constructed probe does not have native23 state layout")
        expected_groups = {"tokenizer": 268, "policy": 930, "root_feedback": 9}

        for case, span_name in CASES:
            span_id = next(i for i, row in enumerate(rows) if row["name"] == span_name)
            span = rows[span_id]
            directory = run / ("cpu100_" + case)
            if (directory / "nominal.npz").exists():
                trace = read_npz(directory / "nominal.npz")
            else:
                trace = read_npz(directory / "failed_physics.npz")
                trace.update(read_npz(directory / "failed_adapter_attempts.npz"))
            count = len(trace["target23"])
            if len(trace["qpos"]) != count + 1 or len(trace["qvel"]) != count + 1:
                raise ValueError("probe needs every completed control boundary")
            if span["length"] < count + 11:
                raise ValueError("probe crosses the original training span")
            kept = {
                name: []
                for name in (
                    "control",
                    "tokenizer",
                    "policy",
                    "root_feedback",
                    "gpu_raw",
                    "gpu_saved_input_raw",
                    "gpu_token",
                    "gpu_saved_input_token",
                )
            }
            command._lifecycle_choice[:] = span_id
            command._lifecycle_last_anchor[:] = span["start"] + span["length"] - 2
            command._env_clip_stop[:] = command._lifecycle_last_anchor + 1
            for indices, valid in control_batches(count):
                guard()
                env.observation_manager.reset(ids)
                for offset in range(-9, 1):
                    controls = indices + offset
                    poses = torch.as_tensor(
                        trace["qpos"][controls], device=env.device, dtype=torch.float32
                    ).clone()
                    poses[:, :3] += env.scene.env_origins
                    velocity = torch.as_tensor(trace["qvel"][controls], device=env.device, dtype=torch.float32)
                    # Explicit same-state diagnostic injection, never env.step().
                    env.sim.data.qpos.copy_(poses)
                    env.sim.data.qvel.copy_(velocity)
                    previous = torch.as_tensor(trace["raw23"][controls - 1], device=env.device)
                    safe, targets = safe_target_transform_torch(previous)
                    action._safe_native_actions.copy_(safe)
                    action._processed_actions.copy_(targets)
                    command.time_steps[:] = torch.as_tensor(controls + span["start"] + 9, device=env.device)
                    env.sim.forward()
                    env.sim.sense()
                    obs = {
                        name: env.observation_manager.compute_group(name, update_history=True)
                        for name in expected_groups
                    }
                    if any(obs[name].shape != (32, size) for name, size in expected_groups.items()):
                        raise ValueError("executed observation shape differs")
                    if not torch.equal(env.sim.data.qpos, poses) or not torch.equal(env.sim.data.qvel, velocity):
                        raise ValueError("observation probe unexpectedly integrated physical state")
                saved_obs = {
                    "tokenizer": torch.cat(
                        (
                            obs["tokenizer"][:, :1],
                            torch.as_tensor(trace["actual_policy_encoder267"][indices], device=env.device),
                        ),
                        -1,
                    ),
                    "policy": torch.as_tensor(trace["history930"][indices], device=env.device),
                    "root_feedback": torch.as_tensor(trace["root_feedback9"][indices], device=env.device),
                }
                with torch.inference_mode():
                    for name, value in obs.items():
                        kept[name].append(value[:valid].cpu().numpy().copy())
                    kept["gpu_raw"].append(actor(obs)[:valid].cpu().numpy())
                    kept["gpu_saved_input_raw"].append(actor(saved_obs)[:valid].cpu().numpy())
                    kept["gpu_token"].append(actor.core.encode(obs["tokenizer"][:, 1:])[:valid].cpu().numpy())
                    kept["gpu_saved_input_token"].append(
                        actor.core.encode(saved_obs["tokenizer"][:, 1:])[:valid].cpu().numpy()
                    )
                kept["control"].append(indices[:valid])
            arrays = {name: np.concatenate(value) for name, value in kept.items()}
            selected = arrays["control"]
            for name, key in (
                ("expected_encoder", "actual_policy_encoder267"),
                ("expected_policy", "history930"),
                ("expected_root", "root_feedback9"),
                ("expected_raw", "released_model_raw23"),
                ("expected_decoder", "decoder994"),
            ):
                arrays[name] = trace[key][selected]
            result = dict(
                completed_controls=count,
                measured_controls_tested=len(selected),
                excluded_synthetic_history_controls=list(range(10)),
                encoder=error_summary(arrays["tokenizer"][:, 1:], arrays["expected_encoder"]),
                history=error_summary(arrays["policy"], arrays["expected_policy"]),
                history_terms={
                    name: error_summary(arrays["policy"][:, a:b], arrays["expected_policy"][:, a:b])
                    for name, (a, b) in TERMS.items()
                },
                root_feedback=error_summary(arrays["root_feedback"], arrays["expected_root"]),
                raw_total=error_summary(arrays["gpu_raw"], arrays["expected_raw"]),
                raw_observation_effect_same_gpu_batch=error_summary(
                    arrays["gpu_raw"], arrays["gpu_saved_input_raw"]
                ),
                raw_backend_effect_same_saved_inputs=error_summary(
                    arrays["gpu_saved_input_raw"], arrays["expected_raw"]
                ),
                token_observation_effect=error_summary(arrays["gpu_token"], arrays["gpu_saved_input_token"]),
                token_backend_effect=error_summary(
                    arrays["gpu_saved_input_token"], arrays["expected_decoder"][:, :64]
                ),
            )
            artifact = output / (case + ".npz")
            with artifact.open("xb") as stream:
                np.savez_compressed(stream, **arrays)
            result.update(arrays_path=str(artifact), arrays_sha256=sha256_file(artifact))
            summary[case] = result
            print(
                json.dumps(
                    {
                        "case": case,
                        "history": result["history"],
                        "encoder": result["encoder"],
                        "root": result["root_feedback"],
                        "raw": result["raw_total"],
                    }
                ),
                flush=True,
            )
        if env.common_step_counter or env._sim_step_counter:
            raise ValueError("same-state input diagnostic unexpectedly stepped the environment")
        for key, value in actor.state_dict().items():
            torch.testing.assert_close(value.cpu(), before_actor[key], atol=0, rtol=0)
        guard()
    finally:
        env.close()
    # Include installed runtime files actually used, not just repository imports.
    for name, module in tuple(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if name.startswith(("gear_sonic.", "mjlab.", "src.tasks.tracking")) and path and str(path).endswith(".py"):
            bind(path)
    bind(inspect.getfile(ManagerBasedRlEnv))
    bind(__file__)
    for path, digest in inputs.items():
        if sha256_file(Path(path)) != digest:
            raise ValueError("probe input changed during execution")
    report = dict(
        kind="native23_measured_training_replay_input_probe_v1",
        cases=summary,
        inputs=inputs,
        identity=identity,
        scene_parameter_parity=scene_proof,
        precision=precision,
        original_observation_corruption=noise,
        diagnostic_corruption_disabled=True,
        state_injection_used=True,
        environment_steps=0,
        physics_integration_steps=0,
        training_updates=0,
        trained_actor_tensors_unchanged=True,
        tested_complete_measured_prefixes_not_full_success=True,
        initial_synthetic_history_or_return_success_proven=False,
        floating_point_parity_is_not_tracking_or_hardware_causality=True,
        deployment_ready=False,
        hardware_authorized=False,
    )
    with (output / "report.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(
        json.dumps(
            {
                "completed": True,
                "report": str(output / "report.json"),
                "report_sha256": sha256_file(output / "report.json"),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    from gear_sonic.utils.g1_true23_training_precision import ieee_training_precision

    with ieee_training_precision() as (precision, guard):
        main(precision, guard)
