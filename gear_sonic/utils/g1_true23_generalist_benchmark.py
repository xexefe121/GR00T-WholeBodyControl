"""Shared native-model CPU diagnostics; not lifecycle or hardware qualification.

Reference reset reproduces only the original SONIC reset boundary, not its
historical stale-cvel observation frontend. Both gain profiles use synchronized
post-integration observations. Reset alone does not demonstrate acquiring or
returning to standing. All evaluated state after reset is produced by MuJoCo
integration, never motion playback pose writes.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from gear_sonic.utils.g1_23dof_contract import HARDWARE_23_JOINT_NAMES
from gear_sonic.utils.g1_23dof_safe_target_transform import safe_target_transform_numpy
from gear_sonic.utils.g1_true23_clean_mujoco_teleop import (
    CleanTrue23MujocoController,
    encoder267_from_reference,
    motion_reference_terms,
    sha256_file,
)
from gear_sonic.utils.g1_true23_reference_floor import compiled_model_sha256
from gear_sonic.utils.g1_true23_native_model_actuation import (
    NativeModelActuationProfile,
    native_model_pd_numpy,
)
from gear_sonic.utils.g1_true23_sonic_library_replay import (
    RELEASED_RETAINED_KD,
    RELEASED_RETAINED_KP,
    _quaternion_error_rad,
    _reference_policy_frame,
    validate_library_motion,
)
from gear_sonic.utils.g1_true23_step1b_mujoco import (
    _projected_gravity,
    _quaternion_matrix,
    term_major_history,
)

MODEL = "gear_sonic/data/robots/g1/g1_23dof_rev_1_0.xml"
PHYSICS = "gear_sonic/config/sim_validation/g1_23dof_mujoco_sim2sim.json"
PROFILES = ("native_model", "historical_released_gains")
LANDMARKS = (
    ("left_ankle_origin", 6, (0.0, 0.0, 0.0)),
    ("right_ankle_origin", 12, (0.0, 0.0, 0.0)),
    ("left_hand_point", 18, (0.18, -0.025, 0.0)),
    ("right_hand_point", 23, (0.18, 0.025, 0.0)),
    ("head_point", 13, (0.0, 0.0, 0.35)),
)
FLAGS = dict(hardware_authorized=False, deployment_ready=False, simulator_qualified=False)


def observation_phase_contract():
    """Separate current sensor timing from the archived original diagnostic."""
    return dict(
        observation_refresh_phase="post_integration_before_measured_policy_history",
        refresh_operations=["mj_kinematics", "mj_comPos", "mj_comVel"],
        current_cvel_phase="same_post_integration_qpos_qvel_instant",
        angular_velocity_body="R(current_root_quaternion).T @ current_pelvis_angular_velocity_world",
        joint_position_velocity_and_angular_velocity_synchronized=True,
        extra_integration_or_contact_solve=False,
        qacc_warmstart_modified=False,
        historical_original_frontend_reproduced=False,
        historical_original_cvel_phase="previous_physics_substep_before_final_2ms_integration",
        historical_first_difference="control_index_1_history930_indices_27_28_29_newest_angular_velocity",
        historical_first10_evidence=dict(
            repository_relative_path="artifacts/g1_true23_generalist/first10_frontend_audit_20260907_v1/comparison.json",
            sha256="e156406fbf1a625121d74c79e3fc242684fdde946dd142ea049722c6eab473b7",
            original_actual_matches_archived_first11_qpos_exactly=True,
            current_actual_matches_shared_first10_controls_and100_substeps_exactly=True,
            cause="observation_timing_not_floating_point_precision",
            full_horizon_historical_reproduction_proven=False,
        ),
    )


def task_points(body_pos, body_quat):
    """Identical named geometric landmarks for actual and reference channels."""
    pos, quat = np.asarray(body_pos), np.asarray(body_quat)
    if pos.shape != (24, 3) or quat.shape != (24, 4):
        raise ValueError("landmarks require native23 body channels in declared order")
    return np.asarray([pos[index] + _quaternion_matrix(quat[index]) @ offset for _, index, offset in LANDMARKS])


def summarize_tracking(errors, *, completed, available, requested, failure):
    values = np.asarray(errors, dtype=np.float64).reshape(-1, 5)
    if len(values) != completed or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("tracking summary requires every finite completed-control observation")
    full = completed == requested == available and available > 0 and failure is None
    p95 = None if not len(values) else np.percentile(values, 95, axis=0).tolist()
    tracked = p95 is not None and all(error <= limit for error, limit in zip(p95, (0.05, 0.05, 0.10, 0.10, 0.10)))
    return dict(
        full_source_motion_completed=full,
        landmark_position_p95_m=None if p95 is None else dict(zip((row[0] for row in LANDMARKS), p95)),
        provisional_reference_landmark_screen_passed=full and tracked,
        reference_landmark_thresholds_m=dict(zip((row[0] for row in LANDMARKS), (0.05, 0.05, 0.10, 0.10, 0.10))),
        foot_metric_is_ankle_origin_not_contact_or_slip=True,
        original_29dof_source_fidelity_measured=False,
        contact_timing_or_forbidden_contact_qualified=False,
        lifecycle_qualified=False,
        generalization_qualified=False,
        **FLAGS,
    )


def validate_lifecycle_checkpoint(payload):
    """Verify the actual CPU lifecycle format, not the different MJLab resume."""
    from gear_sonic.trl.mjlab.frozen_platform_lora_actor import _tensor_state_sha256

    if not isinstance(payload, dict) or payload.get("kind") != "g1_true23_cpu_lifecycle_lora_ppo_checkpoint_v1":
        raise ValueError("requires CPU lifecycle LoRA checkpoint, not MJLab resume")
    if any(
        payload.get(flag) is not False
        for flag in ("hardware_authorized", "deployment_ready", "promotion_eligible")
    ):
        raise ValueError("lifecycle checkpoint must remain unqualified")
    if payload.get("adapter_state_sha256") != _tensor_state_sha256(payload.get("adapter_state_dict", {})):
        raise ValueError("lifecycle adapter state hash mismatch")
    if not isinstance(payload.get("adapter_contract"), dict):
        raise ValueError("lifecycle checkpoint lacks paired platform contract")
    return payload


def load_lifecycle_policy(checkpoint, *, expected_sha256, warm_start, source_checkpoint):
    """Load latest CPU LoRA actor with its own frozen encoder, never original's."""
    import torch

    from gear_sonic.trl.mjlab.frozen_platform_lora_actor import FrozenPlatformTrue23Core

    path = Path(checkpoint).resolve(strict=True)
    if sha256_file(path) != expected_sha256:
        raise ValueError("lifecycle checkpoint file hash mismatch")
    payload = validate_lifecycle_checkpoint(torch.load(path, map_location="cpu", weights_only=True))
    contract = payload["adapter_contract"]
    core = FrozenPlatformTrue23Core(
        warm_start_path=warm_start,
        source_checkpoint_path=source_checkpoint,
        lora_rank=contract["lora_rank"],
        lora_alpha=contract["lora_alpha"],
    )
    if core.adapter_contract() != contract:
        raise ValueError("lifecycle frozen encoder/decoder platform differs from paired source")
    core.load_lora_state_dict(payload["adapter_state_dict"], strict=True)
    if not torch.equal(payload["source_std"], core.initial_std):
        raise ValueError("lifecycle source action scale/std mismatch")
    if core.merged_true23_policy_sha256(core.initial_std) != payload["merged_true23_policy_sha256"]:
        raise ValueError("lifecycle merged policy hash mismatch")
    core.cpu().eval()

    class Policy:
        def infer(self, encoder267, history930):
            if encoder267.shape != (267,) or history930.shape != (930,):
                raise ValueError("policy requires exact causal 267/930 boundaries")
            if encoder267.dtype != np.float32 or history930.dtype != np.float32:
                raise ValueError("policy boundary must be float32")
            with torch.inference_mode():
                semantic, history = torch.from_numpy(encoder267[None]), torch.from_numpy(history930[None])
                padded = core.codec.encode_proprioception(history)
                if not torch.equal(padded, history):
                    raise ValueError("absent joint history slots are not zero")
                decoder = torch.cat((core.encode(semantic), padded), dim=-1)
                raw = core.codec.decode_action(core.decoder(decoder))
            return raw[0].numpy().copy(), decoder[0].numpy().copy()

    return Policy(), dict(
        kind="cpu_lifecycle_lora_with_own_frozen_encoder",
        checkpoint_sha256=expected_sha256,
        paired_frozen_state_sha256=contract["frozen_state_sha256"],
        merged_true23_policy_sha256=payload["merged_true23_policy_sha256"],
        original_causal_encoder_substituted=False,
        stochastic=False,
    )


def load_generalist_pair(manifest_path, *, session_options=None):
    """Consume completed diagnostic exporter manifests with graph metadata pins."""
    from gear_sonic.envs.mjlab.sonic_true23_causal_history import causal_history_profile_contract
    from gear_sonic.utils.g1_true23_sonic_library_replay import ExactHashSonicPolicy

    manifest_path = Path(manifest_path).resolve(strict=True)
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema_version") != 1
        or manifest.get("kind") != "g1_native23_generalist_diagnostic_pair"
        or manifest.get("diagnostic_only") is not True
        or manifest.get("semantic_profile") != causal_history_profile_contract()
        or any(
            manifest.get(key) is not False
            for key in (
                "deployment_ready",
                "promotion_eligible",
                "hardware_authorized",
                "active_motor_control_authorized",
                "completed_motion_qualification",
            )
        )
    ):
        raise ValueError("requires completed unqualified generalist pair with exact causal semantics")
    paths = {}
    for key, sizes in (("encoder", (267, 64)), ("decoder", (994, 23))):
        part = manifest[key]
        filename = part["filename"]
        if not isinstance(filename, str) or Path(filename).name != filename or filename in ("", ".", ".."):
            raise ValueError("pair component must name a sibling file")
        paths[key] = manifest_path.parent / filename
        if part.get("input_shape") != [1, sizes[0]] or part.get("output_shape") != [1, sizes[1]]:
            raise ValueError("generalist manifest component ABI mismatch")
        if key == "encoder" and part.get("parity", {}).get("parity_max_abs_error") != 0:
            raise ValueError("generalist encoder requires exact token parity evidence")
    policy = ExactHashSonicPolicy(
        paths["encoder"],
        paths["decoder"],
        expected_encoder_sha256=manifest["encoder"]["sha256"],
        expected_decoder_sha256=manifest["decoder"]["sha256"],
        session_options=session_options,
    )
    for key, session in (("encoder", policy.encoder), ("decoder", policy.decoder)):
        metadata = session.get_modelmeta().custom_metadata_map
        expected = {
            "source_checkpoint_sha256": manifest["source"]["checkpoint_sha256"],
            "actor_state_sha256": manifest["source"]["actor_state_sha256"],
            "semantic_contract_sha256": manifest["semantic_profile"]["contract_sha256"],
            "artifact_role": "native23_generalist_diagnostic_" + key,
            "hardware_authorized": "false",
            "deployment_ready": "false",
        }
        if any(metadata.get(name) != value for name, value in expected.items()):
            raise ValueError("ONNX components do not share declared checkpoint/encoder semantics")
        if (
            session.get_inputs()[0].name != manifest[key]["input_name"]
            or session.get_outputs()[0].name != manifest[key]["output_name"]
        ):
            raise ValueError("generalist ONNX declared tensor names differ")
    return policy, dict(
        manifest_path=str(manifest_path),
        manifest_sha256=sha256_file(manifest_path),
        source=manifest["source"],
        encoder_sha256=manifest["encoder"]["sha256"],
        decoder_sha256=manifest["decoder"]["sha256"],
        diagnostic_only=True,
        component_paths={key: str(path) for key, path in paths.items()},
    )


def run_reference_diagnostic(
    *, root, asset_root, motion_path, policy, profile="native_model", maximum_controls=None
):
    """One uninterrupted causal rollout; prefixes explicitly cannot pass."""
    if profile not in PROFILES:
        raise ValueError("unknown benchmark actuator profile")
    if maximum_controls is not None and (type(maximum_controls) is not int or maximum_controls <= 0):
        raise ValueError("maximum_controls must be a positive diagnostic-only limit")
    root, assets, source = Path(root), Path(asset_root), Path(motion_path).resolve(strict=True)
    source_sha = sha256_file(source)
    with np.load(source, allow_pickle=False) as archive:
        motion = {key: archive[key].copy() for key in archive.files}
    count = validate_library_motion(motion)
    available = count - 11
    requested = available if maximum_controls is None else min(available, maximum_controls)
    controller = CleanTrue23MujocoController(model_path=assets / MODEL, physics_path=root / PHYSICS, policy=policy)
    module, model, data, physics = controller.module, controller.model, controller.data, controller.physics
    actuation = NativeModelActuationProfile.from_sim_config(root / PHYSICS)
    if profile == "historical_released_gains":
        np.copyto(physics.kp, RELEASED_RETAINED_KP)
        np.copyto(physics.kd, RELEASED_RETAINED_KD)
        actuation = replace(actuation, kp=tuple(physics.kp), kd=tuple(physics.kd))
    if (model.nq, model.nv, model.nu, model.nbody) != (30, 29, 23, 25):
        raise ValueError("benchmark requires exact native23 body topology")
    for _, index, _ in LANDMARKS:
        if index >= model.nbody - 1:
            raise ValueError("landmark body missing")
    model_sha = compiled_model_sha256(model)
    controller.reset(
        base_position=motion["body_pos_w"][10, 0],
        base_quaternion_wxyz=motion["body_quat_w"][10, 0],
        joint_position_hardware=motion["joint_pos"][10],
        root_velocity=np.concatenate(
            (
                motion["body_lin_vel_w"][10, 0],
                _quaternion_matrix(motion["body_quat_w"][10, 0]).T @ motion["body_ang_vel_w"][10, 0],
            )
        ),
        joint_velocity_hardware=motion["joint_vel"][10],
        buffered_robot_pelvis_q9=motion["body_quat_w"][9, 0],
    )
    controller.history = [_reference_policy_frame(motion, index) for index in range(10)]
    initial_state_sha = hashlib.sha256(
        data.qpos.tobytes() + data.qvel.tobytes() + np.asarray(controller.history).tobytes()
    ).hexdigest()
    probe = module.MjData(model)  # Metrics-only FK; never integrated or used by controller.
    arrays = {
        key: []
        for key in (
            "qpos",
            "qvel",
            "encoder267",
            "history930",
            "decoder994",
            "raw23",
            "target23",
            "physics_pre_qpos",
            "physics_post_qpos",
            "physics_pre_qvel",
            "physics_post_qvel",
            "physics_time",
            "requested_torque23",
            "applied_torque23",
            "engine_actuator_force23",
            "torque_saturated23",
            "physics_contact_count",
            "landmark_error_m",
            "relative_landmark_error_m",
            "joint_tracking_rmse_rad",
            "physics_velocity_ratio23",
            "physics_hard_limit_excess23",
            "requested_effort_excess_cost",
            "pelvis_error_m",
            "pelvis_orientation_error_rad",
        )
    }
    arrays["qpos"].append(data.qpos.copy())
    arrays["qvel"].append(data.qvel.copy())
    failure, completed = None, 0
    warning_before = np.asarray([row.number for row in data.warning], dtype=np.int64)
    try:
        for transition in range(requested):
            q9, reference_index = 9 + transition, 11 + transition
            packet = motion_reference_terms(motion, q9)
            encoder = encoder267_from_reference(packet, controller.buffered_robot_pelvis_q9)
            current_pelvis = data.qpos[3:7].copy()
            controller.history = [*controller.history[1:], controller._policy_frame()]
            history = term_major_history(controller.history)
            raw, decoder = policy.infer(encoder, history)
            if (
                raw.shape != (23,)
                or decoder.shape != (994,)
                or raw.dtype != np.float32
                or decoder.dtype != np.float32
            ):
                raise ValueError("actor returned different native23 ABI")
            if not np.isfinite(raw).all() or not np.isfinite(decoder).all() or np.max(np.abs(raw)) >= 10:
                raise RuntimeError("actor emitted nonfinite or raw-clipped action")
            np.testing.assert_array_equal(decoder[64:], history)
            safe, target = safe_target_transform_numpy(raw)
            for key, value in (
                ("encoder267", encoder),
                ("history930", history),
                ("decoder994", decoder),
                ("raw23", raw),
                ("target23", target),
            ):
                arrays[key].append(value.copy())
            for _ in range(physics.decimation):
                arrays["physics_pre_qpos"].append(data.qpos.copy())
                arrays["physics_pre_qvel"].append(data.qvel.copy())
                torque, applied, invalid, excess_cost = native_model_pd_numpy(
                    target.astype(np.float64), data.qpos[7:], data.qvel[6:], actuation
                )
                if invalid:
                    raise RuntimeError("invalid PD request before integration")
                data.ctrl[:] = applied
                start_time = float(data.time)
                module.mj_step(model, data)
                for key, value in (
                    ("physics_post_qpos", data.qpos.copy()),
                    ("physics_post_qvel", data.qvel.copy()),
                    ("physics_time", (start_time, float(data.time))),
                    ("requested_torque23", torque.copy()),
                    ("applied_torque23", applied.copy()),
                    ("engine_actuator_force23", data.qfrc_actuator[6:].copy()),
                    ("torque_saturated23", np.abs(torque) > physics.effort),
                    ("physics_contact_count", int(data.ncon)),
                    ("requested_effort_excess_cost", float(excess_cost)),
                    ("physics_velocity_ratio23", np.abs(data.qvel[6:]) / np.asarray(actuation.velocity)),
                    (
                        "physics_hard_limit_excess23",
                        np.maximum(
                            np.maximum(
                                model.jnt_range[1:, 0] - data.qpos[7:], data.qpos[7:] - model.jnt_range[1:, 1]
                            ),
                            0.0,
                        ),
                    ),
                ):
                    arrays[key].append(value)
                if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                    raise RuntimeError("nonfinite integrated state")
                if abs(float(data.time) - start_time - physics.timestep_s) > 1e-10:
                    raise RuntimeError("engine integration time discontinuity")
                if np.any(np.asarray([row.number for row in data.warning]) > warning_before):
                    raise RuntimeError("MuJoCo warning during integrated rollout")
            controller.buffered_robot_pelvis_q9 = current_pelvis
            controller.previous_safe_native = safe.copy()
            completed += 1
            arrays["qpos"].append(data.qpos.copy())
            arrays["qvel"].append(data.qvel.copy())
            probe.qpos[:] = data.qpos
            probe.qvel[:] = data.qvel
            module.mj_forward(model, probe)
            actual = task_points(probe.xpos[1:], probe.xquat[1:])
            reference = task_points(motion["body_pos_w"][reference_index], motion["body_quat_w"][reference_index])
            arrays["landmark_error_m"].append(np.linalg.norm(actual - reference, axis=1))
            arrays["relative_landmark_error_m"].append(
                np.linalg.norm(
                    (actual - data.qpos[:3]) - (reference - motion["body_pos_w"][reference_index, 0]), axis=1
                )
            )
            arrays["joint_tracking_rmse_rad"].append(
                float(np.sqrt(np.mean((data.qpos[7:] - motion["joint_pos"][reference_index]) ** 2)))
            )
            arrays["pelvis_error_m"].append(
                float(np.linalg.norm(data.qpos[:3] - motion["body_pos_w"][reference_index, 0]))
            )
            arrays["pelvis_orientation_error_rad"].append(
                _quaternion_error_rad(data.qpos[3:7], motion["body_quat_w"][reference_index, 0])
            )
            tilt = float(np.arccos(np.clip(-_projected_gravity(data.qpos[3:7])[2], -1.0, 1.0)))
            if data.qpos[2] < 0.12 or tilt > 2.2:
                raise RuntimeError("absolute height/tilt diagnostic stop")
    except Exception as error:
        failure = dict(
            type=type(error).__name__,
            message=str(error),
            completed_controls=completed,
            physics_steps=len(arrays["physics_time"]),
        )
    result_arrays = {key: np.asarray(value) for key, value in arrays.items()}
    for key in ("requested_torque23", "applied_torque23", "engine_actuator_force23", "torque_saturated23"):
        result_arrays[key] = result_arrays[key].reshape(-1, 23)
    result_arrays["landmark_error_m"] = result_arrays["landmark_error_m"].reshape(-1, 5)
    applied = result_arrays["applied_torque23"]
    saturation = result_arrays["torque_saturated23"]
    engine_force = result_arrays["engine_actuator_force23"]
    engine_matches = bool(np.array_equal(applied, engine_force))
    if not engine_matches and failure is None:
        failure = dict(
            type="ActuatorForceMismatch", message="integrated generalized actuator force differs from command"
        )
    report = dict(
        schema_version=1,
        kind="g1_true23_generalist_reference_reset_diagnostic_v1",
        actuator_profile=profile,
        actuator_profile_scope="gain_arrays_only_not_historical_observation_frontend",
        profile_is_simulator_hypothesis_not_hardware_rating=True,
        physical_dof=23,
        actuator_count=23,
        physics_hz=500,
        policy_hz=50,
        model_sha256=sha256_file(assets / MODEL),
        compiled_model_sha256=model_sha,
        physics_config_sha256=sha256_file(root / PHYSICS),
        initial_state_and_history_sha256=initial_state_sha,
        motion_path=str(source),
        motion_sha256=source_sha,
        source_frames=count,
        available_controls=available,
        requested_controls=requested,
        completed_controls=completed,
        diagnostic_prefix_requested=requested != available,
        failure=failure,
        initialization="reference_frame10_state_and_synthetic_reference_history0_to9",
        original_comparison_reproduced_scope="reference_reset_boundary_only",
        historical_original_frontend_reproduced=False,
        observation_phase_contract=observation_phase_contract(),
        state_pose_writes_after_reset=0,
        history_resets_during_motion=0,
        metrics_only_fk_probe_not_controller_state=True,
        standing_acquisition_tested=False,
        standing_return_tested=False,
        previous_action_semantics="transformed_normalized_requested_native23_target",
        target_slew_projection=False,
        quarter_effort_target_projection=False,
        kp_hardware=physics.kp.tolist(),
        kd_hardware=physics.kd.tolist(),
        effort_limit_hardware_nm=physics.effort.tolist(),
        actuator_contract=actuation.contract(),
        physical_bounds_qualified=False,
        joint_velocity_ratio_max=float(np.max(result_arrays["physics_velocity_ratio23"]))
        if len(applied)
        else None,
        hard_joint_limit_excess_max_rad=float(np.max(result_arrays["physics_hard_limit_excess23"]))
        if len(applied)
        else None,
        pelvis_world_position_p95_m=float(np.percentile(result_arrays["pelvis_error_m"], 95))
        if completed
        else None,
        relative_landmark_position_p95_m=dict(
            zip(
                (row[0] for row in LANDMARKS),
                np.percentile(result_arrays["relative_landmark_error_m"], 95, axis=0).tolist(),
            )
        )
        if completed
        else None,
        joint_tracking_rmse_p95_rad=float(np.percentile(result_arrays["joint_tracking_rmse_rad"], 95))
        if completed
        else None,
        actual_engine_actuator_force_matches_command=engine_matches,
        saturation_fraction_by_joint=dict(zip(HARDWARE_23_JOINT_NAMES, np.mean(saturation, axis=0).tolist()))
        if len(saturation)
        else None,
        maximum_requested_torque_abs_nm_by_joint=np.max(
            np.abs(result_arrays["requested_torque23"]), axis=0
        ).tolist()
        if len(applied)
        else None,
        maximum_applied_torque_abs_nm_by_joint=np.max(np.abs(applied), axis=0).tolist() if len(applied) else None,
        completed_physics_steps=len(applied),
        landmarks=[
            dict(name=name, native_body_index=index, local_offset_m=offset) for name, index, offset in LANDMARKS
        ],
        tracking=summarize_tracking(
            result_arrays["landmark_error_m"],
            completed=completed,
            requested=requested,
            available=available,
            failure=failure,
        ),
        **FLAGS,
    )
    if sha256_file(source) != source_sha:
        raise ValueError("source motion changed during benchmark")
    json.dumps(report, allow_nan=False)
    return report, result_arrays


def compare_original29_source(report, arrays, *, source_trace, source_model, retarget_lineage, asset_root):
    """Post-process aligned original29 planner AND recorded-policy references.

    This supports the existing unretimed, all-frame neutral-hand retarget only.
    Source-model FK is diagnostic geometry, not 29-joint evaluated dynamics.
    """
    import mujoco

    from gear_sonic.utils import g1_23dof_task_space_retarget as retarget
    from gear_sonic.utils.g1_true23_hand_frame_tasks import neutral_wrist_hand_tasks

    source_trace, source_model, retarget_lineage = (
        Path(path).resolve(strict=True) for path in (source_trace, source_model, retarget_lineage)
    )
    lineage = json.loads(retarget_lineage.read_text())
    if (
        lineage.get("output_sha256") != report["motion_sha256"]
        or lineage.get("source_trace_sha256") != sha256_file(source_trace)
        or lineage.get("frames") != report["source_frames"]
        or lineage.get("frames_dropped") != 0
        or lineage.get("time_scale") != 1.0
    ):
        raise ValueError("source comparison requires exact all-frame unretimed lineage")
    source = mujoco.MjModel.from_binary_path(str(source_model))
    target = mujoco.MjModel.from_xml_path(str(Path(asset_root) / MODEL))
    tasks, convention = neutral_wrist_hand_tasks(source, target)
    with np.load(source_trace, allow_pickle=False) as archive:
        original = {
            key: archive[key].copy() for key in ("pre_qpos", "planned_qpos50", "command_time_s", "control_dt")
        }
    count, completed = report["source_frames"], report["completed_controls"]
    if (
        not np.array_equal(original["control_dt"], [0.02])
        or not np.array_equal(original["command_time_s"], np.arange(count) * 0.02)
        or any(original[key].shape != (count, 36) for key in ("pre_qpos", "planned_qpos50"))
    ):
        raise ValueError("source task comparison lacks matching full 50 Hz timeline")
    if completed < 1:
        return dict(measured=False, reason="no completed control", **FLAGS)
    with np.load(report["motion_path"], allow_pickle=False) as archive:
        adapted = np.concatenate(
            (archive["body_pos_w"][:, 0], archive["body_quat_w"][:, 0], archive["joint_pos"]), axis=1
        )
    if arrays["qpos"].shape != (completed + 1, 30):
        raise ValueError("source comparison requires recorded post-control native states")
    data_source, data_target = mujoco.MjData(source), mujoco.MjData(target)

    def points(model, data, poses, is_source):
        output = []
        for pose in poses:
            data.qpos[:] = pose
            mujoco.mj_fwdPosition(model, data)
            output.append(retarget._task_pose_arrays(model, data, tasks, source=is_source)[0])
        return np.asarray(output)

    actual_poses = arrays["qpos"][1:]
    adapted_poses = adapted[11 : 11 + completed]
    actual_points = points(target, data_target, actual_poses, False)
    adapted_points = points(target, data_target, adapted_poses, False)
    comparisons = {}
    for source_name, field in (
        ("original29_planned_choreography", "planned_qpos50"),
        ("original29_recorded_policy", "pre_qpos"),
    ):
        poses = original[field][11 : 11 + completed]
        source_points = points(source, data_source, poses, True)
        for native_name, native_points, native_poses in (
            ("executed_native23", actual_points, actual_poses),
            ("adapted_native23_reference", adapted_points, adapted_poses),
        ):
            world = np.linalg.norm(native_points - source_points, axis=-1)
            relative = np.linalg.norm(
                (native_points - native_poses[:, None, :3]) - (source_points - poses[:, None, :3]), axis=-1
            )
            comparisons[native_name + "_vs_" + source_name] = dict(
                position_world_p95_m=dict(
                    zip((task.name for task in tasks), np.percentile(world, 95, axis=0).tolist())
                ),
                position_pelvis_relative_p95_m=dict(
                    zip((task.name for task in tasks), np.percentile(relative, 95, axis=0).tolist())
                ),
                samples=completed,
            )
    return dict(
        kind="g1_true23_aligned_original29_source_task_comparison_v1",
        measured=True,
        sources={str(path): sha256_file(path) for path in (source_trace, source_model, retarget_lineage)},
        task_convention=convention,
        legacy_encoder_landmark_convention_changed=False,
        source_sample_indices=[11, 10 + completed],
        source_model_used_only_for_metrics_fk=True,
        evaluated_source29_dynamics=False,
        comparisons=comparisons,
        whole_source_duration_compared=completed == report["available_controls"] and report["failure"] is None,
        **FLAGS,
    )


def render_recorded_rollout(report, arrays, *, asset_root, output):
    """Offline visualization of measured integrated qpos, not policy execution."""
    import mujoco

    from gear_sonic.scripts.render_g1_sonic_library_motions import render_motion

    path = Path(asset_root) / MODEL
    output = Path(output)
    if sha256_file(path) != report["model_sha256"]:
        raise ValueError("render model differs from measured rollout")
    if arrays["qpos"].shape != (report["completed_controls"] + 1, 30) or not np.isfinite(arrays["qpos"]).all():
        raise ValueError("render requires complete recorded finite state boundaries")
    if output.exists() or output.is_symlink():
        raise ValueError("render refuses to overwrite evidence")
    render_motion(mujoco.MjModel.from_xml_path(str(path)), arrays["qpos"], output, fps=50)
    return dict(
        kind="offline_visualization_of_recorded_native23_dynamics",
        path=str(output.resolve()),
        sha256=sha256_file(output),
        frames=len(arrays["qpos"]),
        fps=50,
        measured_motion_duration_s=report["completed_controls"] / 50,
        new_policy_or_dynamics_execution=False,
        renderer_pose_playback_is_not_qualification_evidence=True,
        **FLAGS,
    )
