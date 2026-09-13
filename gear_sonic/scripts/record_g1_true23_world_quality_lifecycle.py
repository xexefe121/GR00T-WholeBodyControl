"""Uninterrupted SIM-only joint world-quality candidate in the same MJLab/Warp plant.

Derived from frozen lifecycle version2; uses four independently scored worlds.
The new candidate and additive training failure are bound to its update audit.
The version1 smoke disproved its assumption of bit-identical physical replicas.
No physical world is compressed, reset, or silently discarded in this version.
Actor inference retains training batch32 by repeating each measured observation
eight times; every repeated raw/token output is saved, only the first is applied.
GPU physics uses four worlds rather than training's32. This is not a claim of
bit-exact batch-size invariance. Motor limits and CPU referees are unchanged.
Training episode predicates are recorded without automatic reset or promotion.
Stopped/finished worlds are torque-disabled and excluded from subsequent traces.
No scored state is written after the one initial standing-state installation.
"""

import argparse
import copy
import json
from pathlib import Path
import sys
import time

import mujoco
import numpy as np
import torch

from gear_sonic.envs.mjlab import sonic_true23_world_tracking_termination as world_tracking
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file
from gear_sonic.utils.g1_true23_world_quality import quality_contract

NAMES = ("walk002", "walk003", "walk008", "dance")
WORLDS_PER_CASE = 1
INFERENCE_REPEATS = 8
HISTORY_TERMS = {
    "base_ang_vel": (0, 30),
    "joint_pos_rel": (30, 320),
    "joint_vel": (320, 610),
    "previous_action": (610, 900),
    "projected_gravity": (900, 930),
}


def actor_observations(obs):
    """CPU-reader actor expects semantic267; training manager supplies route+267."""
    semantic = obs["tokenizer"]
    if semantic.ndim != 2 or semantic.shape[-1] != 268 or not torch.isfinite(semantic).all():
        raise ValueError("training tokenizer group must be finite [batch,268]")
    return {**obs, "tokenizer": semantic[:, 1:]}


def scored_world(value, case_index):
    rows = np.asarray(value)
    if rows.shape[0] != len(NAMES) or not 0 <= case_index < len(NAMES):
        raise ValueError("expected four independently scored physical worlds")
    row = rows[case_index]
    if not np.isfinite(row).all():
        raise ValueError("nonfinite scored simulation world")
    return row.copy()


def training_inference_batch(obs):
    if any(value.ndim != 2 or value.shape[0] != len(NAMES) for value in obs.values()):
        raise ValueError("actor observations must contain four measured rows")
    return {key: value.repeat_interleave(INFERENCE_REPEATS, dim=0) for key, value in obs.items()}


def selected_inference_rows(value):
    rows = np.asarray(value)
    if rows.ndim != 2 or len(rows) != len(NAMES) * INFERENCE_REPEATS or not np.isfinite(rows).all():
        raise ValueError("expected finite batch32 inference")
    return rows[::INFERENCE_REPEATS].copy()


def fail(case, message, *, kind="RuntimeError", stage):
    if case["failure"] is None:
        case["failure"] = dict(
            type=kind,
            message=str(message),
            stage=stage,
            completed_controls=case["completed"],
            physics_steps=len(case["trace"]["physics_post_qpos"]),
        )
    case["active"] = False


def verified_update_checkpoint(run, update):
    """Bind only the audited100-update world-tracking SIM snapshot."""
    if (
        update.get("kind") != "native23_world_quality_actual_update_audit_v1"
        or update.get("passed") is not True
        or update.get("completed_updates") != 100
        or update.get("actual_transitions") != 204800
        or update.get("initial_actor_and_critic_equal_matched_world_predecessor") is not True
        or update.get("additional_training_failure") != world_tracking.termination_contract()
        or update.get("world_quality_bonus") != quality_contract()
        or update.get("hardware_authorized") is not False
        or update.get("deployment_ready") is not False
    ):
        raise ValueError("lifecycle requires the complete audited world-tracking regression")
    path = (Path(run) / "train/checkpoints/world_quality_model_100.pt").resolve()
    digest = update.get("inputs", {}).get(str(path))
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("update audit does not bind this world-tracking checkpoint")
    return path, digest


def captured_world_record(env):
    capture = env._world_root_failure_capture
    if (
        not capture
        or len(capture) != env.common_step_counter
        or capture[-1]["common_step_counter"] != env.common_step_counter
    ):
        raise ValueError("actual world termination capture skipped or duplicated a control")
    return {
        key: value.detach().cpu().numpy().copy()
        for key, value in capture[-1].items()
        if key != "common_step_counter"
    }


def main(precision, precision_guard):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-run", type=Path, required=True)
    parser.add_argument("--update-audit", type=Path, required=True)
    parser.add_argument("--boundary-proof", type=Path, required=True)
    parser.add_argument("--physics-proof", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke-controls", type=int)
    args = parser.parse_args()
    if args.smoke_controls is not None and not 1 <= args.smoke_controls <= 10:
        raise ValueError("declared startup smoke must have1..10 controls")
    output = args.output.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError("training-engine lifecycle refuses overwrite or automatic retry")
    output.mkdir(parents=True)
    inputs = {}

    def bind(path, expected=None):
        path = Path(path).resolve(strict=True)
        digest = sha256_file(path)
        if expected is not None and digest != expected:
            raise ValueError(f"lifecycle input changed: {path}")
        if str(path) in inputs and inputs[str(path)] != digest:
            raise ValueError("lifecycle source changed during preparation")
        inputs[str(path)] = digest
        return path

    def read(path, expected=None):
        return json.loads(bind(path, expected).read_text())

    def arrays(path, expected=None):
        with np.load(bind(path, expected), allow_pickle=False) as data:
            return {key: data[key].copy() for key in data.files}

    def write(path, value):
        with path.open("x") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)

    bind(__file__)
    bind(args.experiment)
    boundary = read(args.boundary_proof, "1c68e3e7062e29c046b474c71f0be77c535c7f85f317da63d4341966e3352d62")
    physics_proof = read(args.physics_proof, "965b21ada924bba04ca3a763d642b24d7fb9b8c7ac399ac60b91f2503927942e")
    previous_bindings = {**boundary["inputs"], **physics_proof["inputs"]}
    run = args.world_run.resolve(strict=True)
    update = read(args.update_audit)
    checkpoint_path, checkpoint_digest = verified_update_checkpoint(run, update)
    resolved = read(run / "train/resolved_training.json")
    if resolved.get("native23_world_quality_bonus") != quality_contract():
        raise ValueError("resolved training quality contract changed")
    if resolved.get("native23_world_tracking_termination") != world_tracking.termination_contract():
        raise ValueError("resolved training world-failure contract changed")
    trained_spec = resolved["native23_root_feedback"]["original_intent_spec"]
    native = bind(trained_spec["files"]["native_model"]["path"], trained_spec["files"]["native_model"]["sha256"])
    geometry_path = bind(
        trained_spec["files"]["source_model"]["path"], trained_spec["files"]["source_model"]["sha256"]
    )
    geometry = mujoco.MjModel.from_xml_path(str(geometry_path))

    from mjlab.envs import ManagerBasedRlEnv

    from gear_sonic.envs.mjlab import (
        sonic_true23_bounded_progress as bounded,
        sonic_true23_original_intent as intent,
    )
    from gear_sonic.envs.mjlab.sonic_true23_buffered_source import configure_buffered_source_environment
    from gear_sonic.envs.mjlab.sonic_true23_causal_multimotion_v14 import make_causal_multimotion_v14_env_cfg
    from gear_sonic.envs.mjlab.sonic_true23_generalist_curriculum import advance_lifecycle_command
    from gear_sonic.envs.mjlab.sonic_true23_native_model_actuation import (
        NativeModelActuationAction,
        apply_native_model_actuation_profile,
    )
    from gear_sonic.envs.mjlab.sonic_true23_nominal_scene import configure_nominal_scene, verify_nominal_scene
    from gear_sonic.envs.mjlab.sonic_true23_release_compatible import configure_release_compatible_environment
    from gear_sonic.envs.mjlab.sonic_true23_root_feedback import (
        configure_root_feedback_environment,
        install_root_feedback_command,
    )
    from gear_sonic.scripts.record_g1_true23_original_intent import task_metrics
    from gear_sonic.teleop.buffered_source_horizon import ReceivedSourceHorizon
    from gear_sonic.utils.g1_23dof_contract import HARDWARE_23_JOINT_NAMES
    from gear_sonic.utils.g1_23dof_safe_target_transform import (
        SAFE_TARGET_HARD_LOWER_HARDWARE,
        SAFE_TARGET_HARD_UPPER_HARDWARE,
    )
    from gear_sonic.utils.g1_true23_buffered_reference import continued_standing_source
    from gear_sonic.utils.g1_true23_generalist_benchmark import task_points
    from gear_sonic.utils.g1_true23_generalist_lifecycle import assess_lifecycle_diagnostic
    from gear_sonic.utils.g1_true23_native_model_actuation import NativeModelActuationProfile
    from gear_sonic.utils.g1_true23_original29_reference import (
        build_original29_reference,
        verify_unmodified_native_pair,
    )
    from gear_sonic.utils.g1_true23_range_preview import Native23RangePreview
    from gear_sonic.utils.g1_true23_sonic_library_replay import _reference_policy_frame
    from gear_sonic.utils.g1_true23_source_action_codec import (
        SOURCE_ACTION_CONVENTION,
        source_action_history_numpy,
        source_scaled_precompensation,
    )
    from gear_sonic.utils.g1_true23_step1b_mujoco import _projected_gravity, term_major_history
    from gear_sonic.utils.g1_true23_world_quality_checkpoint import load_cpu_actor

    physics = bind(
        Path(__file__).resolve().parents[2] / "gear_sonic/config/sim_validation/g1_23dof_mujoco_sim2sim.json"
    )
    profile = NativeModelActuationProfile.from_sim_config(physics)
    cases, spans, offset = [], [], 0
    for name in NAMES:
        command_args = read(run / f"cpu100_{name}.command.json")["command"]

        def get_arg(key):
            return Path(command_args[command_args.index("--" + key) + 1])

        baseline = read(get_arg("baseline-directory") / "report.json")
        timeline = baseline["timeline"]
        motion = arrays(timeline["timeline_path"], timeline["timeline_sha256"])
        source_file = get_arg("original-reference")
        original = arrays(source_file, baseline["inputs"][str(source_file)])
        reference = build_original29_reference(geometry, original["source_qpos29"])
        for key, value in reference.arrays().items():
            np.testing.assert_array_equal(value, original[key])
        verify_unmodified_native_pair(reference, motion)
        directory = run / ("cpu100_" + name)
        cpu = arrays(directory / ("nominal.npz" if (directory / "nominal.npz").exists() else "failed_physics.npz"))
        if "actual_policy_encoder267" not in cpu:
            cpu.update(arrays(directory / "failed_adapter_attempts.npz"))
        length = len(motion["joint_pos"])
        if length - 11 != timeline["total_requested_controls"]:
            raise ValueError("unchanged lifecycle has a different timing convention")
        spans.append(dict(name=name, start=offset, length=length, timeline=copy.deepcopy(timeline)))
        trace = {
            key: []
            for key in (
                "qpos",
                "qvel",
                "history930",
                "actual_policy_encoder267",
                "root_feedback9",
                "decoder994",
                "released_model_raw23",
                "batch32_raw_repeats",
                "batch32_token_repeats",
                "raw23",
                "target23",
                "physics_pre_qpos",
                "physics_pre_qvel",
                "physics_post_qpos",
                "physics_post_qvel",
                "requested_torque23",
                "applied_torque23",
                "physics_time",
                "training_episode_terminated",
                "training_episode_timeout",
                "world_tracking_desired_position_w",
                "world_tracking_measured_position_w",
                "world_tracking_error_m",
                "world_tracking_failure",
                "world_tracking_reference_q0",
                "world_tracking_common_step_counter",
            )
        }
        cases.append(
            dict(
                name=name,
                span=spans[-1],
                motion=motion,
                reference=reference,
                cpu=cpu,
                source=continued_standing_source(motion, reference.virtual_vr21),
                buffer=ReceivedSourceHorizon(),
                trace=trace,
                attempted=[],
                completed=0,
                active=True,
                failure=None,
                max_encoder_difference=0.0,
                max_root_difference=0.0,
                guard_interventions=[],
                initial_proof=None,
            )
        )
        offset += length
    combined = {
        key: (
            cases[0]["motion"][key].copy() if key == "fps" else np.concatenate([c["motion"][key] for c in cases])
        )
        for key in cases[0]["motion"]
    }
    original_combined = build_original29_reference(
        geometry, np.concatenate([c["reference"].source_qpos29 for c in cases])
    )
    for name, data in (("motion.npz", combined), ("original_reference.npz", original_combined.arrays())):
        with (output / name).open("xb") as stream:
            np.savez_compressed(stream, **data)
    spec = intent.make_spec(
        original_reference=output / "original_reference.npz",
        native_motion=output / "motion.npz",
        source_model=geometry_path,
        native_model=native,
    )
    intent.load_reference(spec)
    for entry in spec["files"].values():
        bind(entry["path"], entry["sha256"])
    write(
        output / "evaluation_source_spec.json",
        dict(spec=spec, spans=spans, training_data_changed=False, evaluation_only_includes_public008=True),
    )
    install_root_feedback_command(spans, start_schedule="mixed_reference_reset_v1")
    cfg = make_causal_multimotion_v14_env_cfg(
        motion_file=str(output / "motion.npz"), num_envs=len(NAMES), play=False
    )
    cfg = apply_native_model_actuation_profile(cfg, profile)
    cfg = configure_root_feedback_environment(cfg, spans, objective_profile="root_and_upper_feet_world_v4")
    cfg = configure_release_compatible_environment(cfg, geometry_path, SOURCE_ACTION_CONVENTION)
    cfg = configure_nominal_scene(cfg, native, physics)
    cfg = configure_buffered_source_environment(cfg)
    cfg = intent.configure_original_intent_environment(cfg, spec)
    cfg = bounded.configure_environment(cfg)
    cfg = world_tracking.configure_environment(cfg)
    cfg.seed = resolved["seed"]
    for name in ("policy", "tokenizer"):
        cfg.observations[name].enable_corruption = False
    checkpoint = bind(checkpoint_path, checkpoint_digest)
    original_root = native.parents[4]
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    precision_guard()
    reader, identity, semantics = load_cpu_actor(
        checkpoint,
        warm_start_path=bind(original_root / "sonic_release/g1_23dof_rev_1_0_low_latency_init.pt"),
        source_checkpoint_path=bind(original_root / "low_latency/last.pt"),
    )
    inputs.update(semantics["reverified_repository_sources"])
    actor = reader.actor.to("cuda:0").eval()
    if actor.tokenizer_has_encoder_index:
        raise ValueError("strict CPU reader no longer uses semantic267 construction")
    initial_actor = {key: value.detach().cpu().clone() for key, value in actor.state_dict().items()}
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    guard = Native23RangePreview(model_path=native, physics_path=physics)
    started_at = time.monotonic()
    fatal = None

    def lanes():
        return torch.as_tensor(np.repeat([case["active"] for case in cases], WORLDS_PER_CASE), device=env.device)

    def push(case, frame):
        source = case["source"]
        return case["buffer"].push(
            source_timestamp_s=frame * 0.02,
            arrival_timestamp_s=frame * 0.02,
            joint_names=HARDWARE_23_JOINT_NAMES,
            joint_position23=source["joint_pos"][frame],
            root_position_w=source["root_position_w"][frame],
            root_quaternion_wxyz=source["root_quaternion_wxyz"][frame],
            virtual_source_vr21=source["virtual_vr21"][frame],
        )

    try:
        model_proof = verify_nominal_scene(env.sim.mj_model, native, physics)
        bounded.verify_runtime(env)
        world_tracking.verify_runtime(env)
        env._world_root_failure_capture = []
        command = env.command_manager.get_term("motion")
        if command.cfg.anchor_body_name != "pelvis":
            raise ValueError("world-root measurement requires the configured pelvis frame")
        action = env.action_manager.get_term("joint_pos")
        for name, module in tuple(sys.modules.items()):
            path = getattr(module, "__file__", None)
            if (
                name.startswith(("gear_sonic.", "mjlab.", "src.tasks.tracking", "mujoco_warp."))
                and path
                and str(path).endswith(".py")
            ):
                path = Path(path).resolve()
                bind(path, previous_bindings.get(str(path)))
        env.scene.env_origins.zero_()  # Four independent worlds, identical CPU world frames; not a pose write.
        env.sim.reset()
        action.reset()
        q = np.repeat(np.stack([c["cpu"]["qpos"][0] for c in cases]), WORLDS_PER_CASE, axis=0).astype(np.float32)
        v = np.repeat(np.stack([c["cpu"]["qvel"][0] for c in cases]), WORLDS_PER_CASE, axis=0).astype(np.float32)
        env.sim.data.qpos.copy_(torch.as_tensor(q, device=env.device))
        env.sim.data.qvel.copy_(torch.as_tensor(v, device=env.device))
        for index, case in enumerate(cases):
            group = slice(index * WORLDS_PER_CASE, (index + 1) * WORLDS_PER_CASE)
            command._lifecycle_choice[group] = index
            command.time_steps[group] = case["span"]["start"] + 9
            command._lifecycle_last_anchor[group] = case["span"]["start"] + case["span"]["length"] - 2
            command._env_clip_stop[group] = command._lifecycle_last_anchor[group] + 1
            case["trace"]["qpos"].append(q[group][0].copy())
            case["trace"]["qvel"].append(v[group][0].copy())
            for frame in range(19):
                push(case, frame)
        command._causal_resampled.zero_()
        env.sim.forward()
        command._refresh_targets_from_causal_anchor()
        env.sim.sense()
        ids = torch.arange(len(NAMES), device=env.device)
        env.observation_manager.reset(ids)
        # Original CPU startup contains reference frames1..9, then measured0.
        # Append each term to the actual manager buffers; no expected trace copy.
        for frame in range(1, 10):
            per_case = [
                source_action_history_numpy(term_major_history([_reference_policy_frame(c["motion"], frame)] * 10))
                for c in cases
            ]
            for name, (a, b) in HISTORY_TERMS.items():
                width = (b - a) // 10
                values = np.repeat(np.stack([row[a : a + width] for row in per_case]), WORLDS_PER_CASE, axis=0)
                env.observation_manager._group_obs_term_history_buffer["policy"][name].append(
                    torch.as_tensor(values, device=env.device)
                )
        obs = {
            name: env.observation_manager.compute_group(name, update_history=True)
            for name in ("tokenizer", "policy", "root_feedback")
        }
        write(
            output / "started.json",
            dict(
                inputs=inputs,
                identity=identity,
                model=model_proof,
                mode="full" if args.smoke_controls is None else "startup_smoke",
                declared_smoke_controls=args.smoke_controls,
                original28_base_and_all_trained_adapter_tensors_unchanged=True,
                four_independently_scored_physical_worlds=True,
                physical_world_count=4,
                inference_batch_size=32,
                episode_predicates_recorded_without_resets=True,
                hardware_authorized=False,
                deployment_ready=False,
            ),
        )
        for control in range(max(c["span"]["timeline"]["total_requested_controls"] for c in cases)):
            if not any(c["active"] for c in cases):
                break
            precision_guard()
            qpos = env.sim.data.qpos.cpu().numpy().copy()
            qvel = env.sim.data.qvel.cpu().numpy().copy()
            with torch.inference_mode():
                actor_obs = actor_observations(obs)
                batch_obs = training_inference_batch(actor_obs)
                raw32 = actor(batch_obs).cpu().numpy()
                token32 = actor.core.encode(batch_obs["tokenizer"]).cpu().numpy()
                raw = selected_inference_rows(raw32)
                token = selected_inference_rows(token32)
            numpy_obs = {key: value.detach().cpu().numpy().copy() for key, value in actor_obs.items()}
            applied = np.zeros((len(NAMES), 23), dtype=np.float32)
            for index, case in enumerate(cases):
                if not case["active"]:
                    continue
                if control != case["completed"]:
                    raise ValueError("active motion skipped a control")
                state_q, state_v = scored_world(qpos, index), scored_world(qvel, index)
                values = {key: scored_world(value, index) for key, value in numpy_obs.items()}
                proposal = scored_world(raw, index)
                encoded = scored_world(token, index)
                window = push(case, 19 + control)
                encoder_error = float(np.max(np.abs(values["tokenizer"] - window.encoder267(state_q[3:7]))))
                root_error = float(
                    np.max(
                        np.abs(
                            values["root_feedback"] - window.root_feedback9(state_q[:3], state_v[:3], state_q[3:7])
                        )
                    )
                )
                case["max_encoder_difference"] = max(case["max_encoder_difference"], encoder_error)
                case["max_root_difference"] = max(case["max_root_difference"], root_error)
                if encoder_error > 2e-6 or root_error > 2e-6:
                    raise ValueError("actual GPU observation diverges from independently received source")
                case["attempted"].append(
                    dict(control=control, raw23=proposal.tolist(), qpos=state_q.tolist(), qvel=state_v.tolist())
                )
                if control == 0:
                    initial = dict(
                        history_max_abs=float(np.max(np.abs(values["policy"] - case["cpu"]["history930"][0]))),
                        encoder_max_abs=float(
                            np.max(np.abs(values["tokenizer"] - case["cpu"]["actual_policy_encoder267"][0]))
                        ),
                        root_max_abs=float(
                            np.max(np.abs(values["root_feedback"] - case["cpu"]["root_feedback9"][0]))
                        ),
                        raw_max_abs=float(np.max(np.abs(proposal - case["cpu"]["released_model_raw23"][0]))),
                    )
                    if max(initial.values()) > 1e-5:
                        raise ValueError(f"startup is not matched to CPU: {case['name']} {initial}")
                    np.testing.assert_array_equal(encoded, case["cpu"]["decoder994"][0, :64])
                    case["initial_proof"] = initial
                try:
                    native_request, _ = source_scaled_precompensation(proposal)
                    accepted = guard.filter(native_request, state_q.astype(np.float64), state_v.astype(np.float64))
                except ValueError as exc:
                    fail(case, exc, stage="before_motor_target")
                    print(json.dumps({"stopped": case["name"], "failure": case["failure"]}), flush=True)
                    continue
                group = slice(index * WORLDS_PER_CASE, (index + 1) * WORLDS_PER_CASE)
                applied[group] = accepted
                if guard.records[-1]["intervened"]:
                    case["guard_interventions"].append(dict(control=control, details=guard.records[-1]))
                for key, value in (
                    ("history930", values["policy"]),
                    ("actual_policy_encoder267", values["tokenizer"]),
                    ("root_feedback9", values["root_feedback"]),
                    ("decoder994", np.r_[encoded, values["policy"]]),
                    ("released_model_raw23", proposal),
                    ("batch32_raw_repeats", raw32[index * INFERENCE_REPEATS : (index + 1) * INFERENCE_REPEATS]),
                    (
                        "batch32_token_repeats",
                        token32[index * INFERENCE_REPEATS : (index + 1) * INFERENCE_REPEATS],
                    ),
                    ("raw23", accepted),
                ):
                    case["trace"][key].append(value.copy())
            # Source scaling and guard ran above. Execute the unchanged native
            # action transform once, not ReleaseCompatibleAction a second time.
            NativeModelActuationAction.process_actions(action, torch.as_tensor(applied, device=env.device))
            action.invalid_actuation[~lanes()] = True
            targets = action.processed_action.cpu().numpy().copy()
            for index, case in enumerate(cases):
                if case["active"]:
                    case["trace"]["target23"].append(scored_world(targets, index))
            if not any(c["active"] for c in cases):
                break
            for substep in range(10):
                before_q = env.sim.data.qpos.cpu().numpy().copy()
                before_v = env.sim.data.qvel.cpu().numpy().copy()
                before_time = env.sim.data.time.cpu().numpy().copy()
                env.action_manager.apply_action()
                env.scene.write_data_to_sim()
                requested = action.requested_torque.cpu().numpy().copy()
                torque = env.sim.data.ctrl.cpu().numpy().copy()
                invalid = action.invalid_actuation.cpu().numpy().copy()
                env.sim.step()
                env.scene.update(dt=env.physics_dt)
                env._sim_step_counter += 1
                after_q = env.sim.data.qpos.cpu().numpy().copy()
                after_v = env.sim.data.qvel.cpu().numpy().copy()
                after_time = env.sim.data.time.cpu().numpy().copy()
                for index, case in enumerate(cases):
                    if not case["active"]:
                        continue
                    first = index * WORLDS_PER_CASE
                    sq, sv = scored_world(after_q, index), scored_world(after_v, index)
                    for key, value in (
                        ("physics_pre_qpos", scored_world(before_q, index)),
                        ("physics_pre_qvel", scored_world(before_v, index)),
                        ("physics_post_qpos", sq),
                        ("physics_post_qvel", sv),
                        ("requested_torque23", scored_world(requested, index)),
                        ("applied_torque23", scored_world(torque, index)),
                        ("physics_time", np.array([before_time[first], after_time[first]])),
                    ):
                        case["trace"][key].append(value.copy())
                    excess = float(
                        np.maximum(
                            np.maximum(
                                np.asarray(SAFE_TARGET_HARD_LOWER_HARDWARE) - sq[7:],
                                sq[7:] - np.asarray(SAFE_TARGET_HARD_UPPER_HARDWARE),
                            ),
                            0,
                        ).max()
                    )
                    if (
                        invalid[first]
                        or np.max(np.abs(torque[first]) - np.asarray(profile.effort)) > 1e-5
                        or excess > 0
                    ):
                        fail(
                            case,
                            f"actual native motor/range guard: invalid={bool(invalid[first])}, "
                            f"range_excess={excess}",
                            stage=f"substep_{substep}",
                        )
                        action.invalid_actuation[first : first + WORLDS_PER_CASE] = True
            env.episode_length_buf += lanes().long()
            env.common_step_counter += 1
            # Observe old and new training predicates at their original stale
            # pre-final-forward phase; record actual calls without automatic resets.
            env.termination_manager.compute()
            world_record = captured_world_record(env)
            terminated = env.termination_manager.terminated.cpu().numpy()
            timeouts = env.termination_manager.time_outs.cpu().numpy()
            for index, case in enumerate(cases):
                if not case["active"]:
                    continue
                first = index * WORLDS_PER_CASE
                case["completed"] += 1
                case["trace"]["qpos"].append(scored_world(after_q, index))
                case["trace"]["qvel"].append(scored_world(after_v, index))
                case["trace"]["training_episode_terminated"].append(bool(terminated[first]))
                case["trace"]["training_episode_timeout"].append(bool(timeouts[first]))
                for key in ("desired_position_w", "measured_position_w", "error_m", "failure", "reference_q0"):
                    case["trace"]["world_tracking_" + key].append(scored_world(world_record[key], index))
                case["trace"]["world_tracking_common_step_counter"].append(int(env.common_step_counter))
                tilt = float(np.arccos(np.clip(-_projected_gravity(after_q[first, 3:7])[2], -1, 1)))
                if after_q[first, 2] < 0.12 or tilt > 2.2:
                    fail(case, "absolute height/tilt diagnostic stop", stage="after_control")
                elif case["completed"] == case["span"]["timeline"]["total_requested_controls"]:
                    case["active"] = False
                elif args.smoke_controls is not None and case["completed"] == args.smoke_controls:
                    fail(case, "declared startup-only smoke cap", kind="SmokeCap", stage="after_control")
            action.invalid_actuation[~lanes()] = True
            if not any(c["active"] for c in cases):
                break
            env.sim.forward()
            advance_lifecycle_command(command, hold_reference=~lanes())
            command.command_update_count += 1
            env.sim.sense()
            obs = {
                name: env.observation_manager.compute_group(name, update_history=True)
                for name in ("tokenizer", "policy", "root_feedback")
            }
            if (control + 1) % 100 == 0:
                print(
                    json.dumps(
                        {
                            "control": control + 1,
                            "active": [c["name"] for c in cases if c["active"]],
                            "elapsed_s": round(time.monotonic() - started_at, 1),
                        }
                    ),
                    flush=True,
                )
        for key, value in actor.state_dict().items():
            torch.testing.assert_close(value.cpu(), initial_actor[key], rtol=0, atol=0)
    except Exception as exc:
        fatal = dict(type=type(exc).__name__, message=str(exc))
        for case in cases:
            if case["active"]:
                fail(case, exc, kind=type(exc).__name__, stage="global_runtime")
    finally:
        env.close()
    reports = []
    for case in cases:
        name, count = case["name"], case["completed"]
        path = output / name
        path.mkdir()
        trace = {key: np.asarray(value) for key, value in case["trace"].items()}
        # Full physical substeps and attempted controls remain untouched. Only
        # completed control-boundary states feed unchanged lifecycle metrics.
        actual_points = []
        data = mujoco.MjData(guard.probe.model)
        for pose in trace["qpos"][1:]:
            data.qpos[:] = pose
            mujoco.mj_fwdPosition(guard.probe.model, data)
            actual_points.append(task_points(data.xpos[1:], data.xquat[1:]))
        actual_points = np.asarray(actual_points).reshape(-1, 5, 3)
        reference_points = np.asarray(
            [
                task_points(case["motion"]["body_pos_w"][11 + i], case["motion"]["body_quat_w"][11 + i])
                for i in range(count)
            ]
        ).reshape(-1, 5, 3)
        trace["landmark_error_m"] = np.linalg.norm(actual_points - reference_points, axis=-1)
        requested = case["span"]["timeline"]["total_requested_controls"]
        result = dict(
            available_controls=requested,
            requested_controls=requested,
            completed_controls=count,
            failure=case["failure"],
        )
        lifecycle = assess_lifecycle_diagnostic(case["span"]["timeline"], result, trace)
        phase = next(p for p in case["span"]["timeline"]["phases"] if p["name"] == "source_motion")
        metrics = task_metrics(guard.probe.model, geometry, case["reference"], case["motion"], trace, phase)
        with (path / "trace.npz").open("xb") as stream:
            np.savez_compressed(stream, **trace)
        write(path / "attempted_controls.json", case["attempted"])
        row = dict(
            name=name,
            result=result,
            lifecycle=lifecycle,
            original_task_metrics=metrics,
            actual_integrated_substeps=len(trace["physics_post_qpos"]),
            initial_comparison=case["initial_proof"],
            training_episode_failure_controls=np.flatnonzero(trace["training_episode_terminated"]).tolist(),
            training_episode_timeout_controls=np.flatnonzero(trace["training_episode_timeout"]).tolist(),
            world_tracking_failure_controls=np.flatnonzero(trace["world_tracking_failure"]).tolist(),
            additional_training_failure=world_tracking.termination_contract(),
            received_encoder_max_abs_error=case["max_encoder_difference"],
            received_root_max_abs_error=case["max_root_difference"],
            range_guard_interventions=case["guard_interventions"],
            trace_sha256=sha256_file(path / "trace.npz"),
            hardware_authorized=False,
            deployment_ready=False,
            full_lifecycle_qualification=False,
        )
        write(path / "report.json", row)
        reports.append(row)
    for path, digest in inputs.items():
        if sha256_file(Path(path)) != digest:
            raise ValueError("lifecycle input changed before completion")
    report = dict(
        kind="native23_world_quality_full_lifecycle_diagnostic_v1",
        world_quality_bonus=quality_contract(),
        additional_training_failure=world_tracking.termination_contract(),
        audited_training_checkpoint_sha256=checkpoint_digest,
        cases=reports,
        inputs=inputs,
        fatal_runtime_error=fatal,
        identity=identity,
        precision=precision,
        mode="full" if args.smoke_controls is None else "startup_smoke",
        smoke_controls=args.smoke_controls,
        actual_four_motion_observation_physics_pipeline=True,
        actor_tensors_changed=False,
        native_range_guard_unchanged=True,
        source_speed_factor=1.0,
        automatic_training_episode_resets=False,
        initial_physical_state_installations_per_lane=1,
        subsequent_physical_qpos_qvel_warmstart_writes=0,
        stopped_lanes_torque_disabled_and_never_reenter=True,
        unique_motion_count=4,
        physical_world_count=4,
        physical_worlds_per_motion=1,
        inference_batch_size=32,
        repeated_inference_outputs_saved=True,
        bit_exact_physics_batch_size_invariance_claimed=False,
        physical_replica_compression=False,
        hardware_authorized=False,
        deployment_ready=False,
        simulator_qualified=False,
    )
    write(output / "report.json", report)
    print(
        json.dumps(
            {
                "finished": True,
                "fatal": fatal,
                "report_sha256": sha256_file(output / "report.json"),
                "cases": [{"name": r["name"], **r["result"]} for r in reports],
            }
        ),
        flush=True,
    )
    if fatal:
        raise RuntimeError("training-engine lifecycle failed; preserved full attempted/physical evidence")


if __name__ == "__main__":
    from gear_sonic.utils.g1_true23_training_precision import ieee_training_precision

    with ieee_training_precision() as (precision, precision_guard):
        main(precision, precision_guard)
