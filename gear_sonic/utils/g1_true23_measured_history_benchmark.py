"""Versioned measured-history hook for SIM-only causal observer experiments.

The reference diagnostic body is preserved from g1_true23_generalist_benchmark;
the only controller-loop extension supplies COPIES of current qpos/qvel before
history encoding. No actual model/MjData/pose setter crosses that interface.
Existing executed benchmark sources and reports are not rewritten.
"""

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np

from gear_sonic.utils.g1_true23_generalist_benchmark import (
    PROFILES,
    MODEL,
    PHYSICS,
    LANDMARKS,
    FLAGS,
    validate_library_motion,
    CleanTrue23MujocoController,
    install_training_model_counterfactual,
    NativeModelActuationProfile,
    RELEASED_RETAINED_KP,
    RELEASED_RETAINED_KD,
    SOURCE_MJ29_KEEP_INDICES,
    compiled_model_sha256,
    _quaternion_matrix,
    _reference_policy_frame,
    motion_reference_terms,
    encoder267_from_reference,
    term_major_history,
    safe_target_transform_numpy,
    native_model_pd_numpy,
    task_points,
    _quaternion_error_rad,
    _projected_gravity,
    observation_phase_contract,
    HARDWARE_23_JOINT_NAMES,
    summarize_tracking,
    sha256_file,
)


def run_reference_diagnostic(
    *,
    root,
    asset_root,
    motion_path,
    policy,
    profile="native_model",
    maximum_controls=None,
    runtime_adapter=None,
    training_model_counterfactual=None,
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
    model_counterfactual = None
    if training_model_counterfactual is not None:
        model_counterfactual = install_training_model_counterfactual(controller, training_model_counterfactual)
    module, model, data, physics = controller.module, controller.model, controller.data, controller.physics
    actuation = NativeModelActuationProfile.from_sim_config(root / PHYSICS)
    if profile == "historical_released_gains":
        np.copyto(physics.kp, RELEASED_RETAINED_KP)
        np.copyto(physics.kd, RELEASED_RETAINED_KD)
        actuation = replace(actuation, kp=tuple(physics.kp), kd=tuple(physics.kd))
    source_gain_capture = None
    if profile == "original_cpp_gains_diagnostic":
        from gear_sonic.scripts.audit_g1_true23_source_action_codec import compile_source_oracle

        source_gain_capture = compile_source_oracle(assets)
        indices = np.asarray(SOURCE_MJ29_KEEP_INDICES)
        np.copyto(physics.kp, np.asarray(source_gain_capture["kp_hardware29"])[indices])
        np.copyto(physics.kd, np.asarray(source_gain_capture["kd_hardware29"])[indices])
        actuation = replace(actuation, kp=tuple(physics.kp), kd=tuple(physics.kd))
    body_ids = getattr(controller, "diagnostic_body_ids", np.arange(1, model.nbody))
    if (model.nq, model.nv, model.nu, len(body_ids)) != (30, 29, 23, 24):
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
    if runtime_adapter is not None:
        arrays["physics_external_force_world_n"] = []
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
            if runtime_adapter is not None and hasattr(runtime_adapter, "transform_history"):
                # Pure copied-observation codec, never a physical-state setter.
                native_history = history.copy()
                if hasattr(runtime_adapter, "transform_measured_history"):
                    history = runtime_adapter.transform_measured_history(
                        native_history.copy(),
                        measured_qpos=data.qpos.copy(),
                        measured_qvel=data.qvel.copy(),
                        control_index=transition,
                    )
                else:
                    history = runtime_adapter.transform_history(native_history.copy())
                if history.shape != (930,) or history.dtype != np.float32 or not np.isfinite(history).all():
                    raise ValueError("runtime history codec changed finite float32 history930 ABI")
                arrays.setdefault("native_precodec_history930", []).append(native_history)
            if runtime_adapter is None:
                raw, decoder = policy.infer(encoder, history)
            else:
                # Only copies cross the optional adapter boundary. No model,
                # MjData, pose setter, or integration callback is exposed.
                raw, decoder = runtime_adapter.infer(
                    policy,
                    encoder.copy(),
                    history.copy(),
                    control_index=transition,
                    desired_position_w=motion["body_pos_w"][q9 + 1, 0].copy(),
                    previous_desired_position_w=motion["body_pos_w"][q9, 0].copy(),
                    measured_qpos=data.qpos.copy(),
                    measured_qvel=data.qvel.copy(),
                )
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
                if runtime_adapter is not None:
                    force = np.asarray(runtime_adapter.external_force_world(len(arrays["physics_time"])))
                    if force.shape != (3,) or not np.isfinite(force).all():
                        raise ValueError("runtime adapter must supply a finite world-frame force3")
                    # Scheduled simulator perturbation is an external pelvis
                    # force, never a root pose/velocity rewrite or actuator.
                    data.xfrc_applied[:] = 0
                    data.xfrc_applied[body_ids[0], :3] = force
                    arrays["physics_external_force_world_n"].append(force.copy())
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
            actual = task_points(probe.xpos[body_ids], probe.xquat[body_ids])
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
        **({"training_model_counterfactual": model_counterfactual} if model_counterfactual is not None else {}),
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
    if runtime_adapter is not None:
        report["runtime_adapter"] = runtime_adapter.contract()
        report["current_measurement_history_hook"] = hasattr(runtime_adapter, "transform_measured_history")
        codec = report["runtime_adapter"].get("source_action_codec")
        if codec is not None:
            report["stored_controller_previous_action_semantics"] = report["previous_action_semantics"]
            report["previous_action_semantics"] = codec["previous_action"]
            report["decoder_history_is_explicit_source_codec_output"] = True
    if source_gain_capture is not None:
        report["source_gain_capture"] = source_gain_capture
        report["original_cpp_gain_counterfactual_not_nominal_qualification"] = True
    if sha256_file(source) != source_sha:
        raise ValueError("source motion changed during benchmark")
    json.dumps(report, allow_nan=False)
    return report, result_arrays
