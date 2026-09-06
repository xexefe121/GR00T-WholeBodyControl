"""Standing-only simulator labels inside SONIC's existing output boundary.

This is a prerequisite diagnostic, not a replacement for full-body SONIC
motion, a qualified behavior bank, or a hardware controller. The compatibility
actor's requested targets are not necessarily representable by the native23
safe-target transform. Make that projection explicit and retest closed loop.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from gear_sonic.utils.g1_23dof_contract import (
    HARDWARE_23_ACTION_SCALE,
    MUJOCO_TO_ISAACLAB_DOF,
)
from gear_sonic.utils.g1_23dof_safe_target_transform import (
    SAFE_TARGET_DEFAULT_Q_HARDWARE,
    SAFE_TARGET_NEGATIVE_CAPACITY_HARDWARE,
    SAFE_TARGET_POSITIVE_CAPACITY_HARDWARE,
    SAFE_TARGET_RAW_ACTION_CLIP,
    safe_target_transform_numpy,
)
from gear_sonic.utils.g1_sonic_cpp_observation_trace import audit_engine_trace
from gear_sonic.utils.g1_true23_actuation_profile import NativeSupportActuationProfile
from gear_sonic.utils.g1_true23_clean_mujoco_teleop import (
    CleanTrue23MujocoController,
    UnitreeZeroVelocityFallbackPolicy,
    encoder267_from_reference,
    motion_reference_terms,
)
from gear_sonic.utils.g1_true23_reference_floor import compiled_model_sha256
from gear_sonic.utils.g1_true23_sim_acquisition import simulate_balance_transition
from gear_sonic.utils.g1_true23_step1b_mujoco import term_major_history


def representable_standing_request(target, *, allow_projection=False):
    """Invert the existing safe-target transform; never silently clip labels.

    With explicit projection enabled, choose the nearest target in the finite
    raw-action [-10, 10] image, then apply the original float32 transform once.
    Return the original and projected difference in the report. This operation
    is on the *request*, before the unchanged 500 Hz effort/slew projection.
    """
    target = np.asarray(target, dtype=np.float64)
    if target.shape != (23,) or not np.isfinite(target).all():
        raise ValueError("standing request requires finite 23 hardware-order targets")
    if not isinstance(allow_projection, bool):
        raise ValueError("standing request projection must be explicit boolean")
    default = np.asarray(SAFE_TARGET_DEFAULT_Q_HARDWARE)
    scale = np.asarray(HARDWARE_23_ACTION_SCALE)
    delta = target - default
    capacity = np.where(delta >= 0, SAFE_TARGET_POSITIVE_CAPACITY_HARDWARE, SAFE_TARGET_NEGATIVE_CAPACITY_HARDWARE)
    maximum = capacity * np.tanh(SAFE_TARGET_RAW_ACTION_CLIP * scale / capacity)
    outside = np.abs(delta) > maximum
    if np.any(outside) and not allow_projection:
        raise ValueError("standing request is not representable by SONIC's safe-target transform")
    limited = np.clip(delta, -maximum, maximum)
    # Very narrow joints saturate tanh in float64. The correct endpoint raw
    # label is +/-10; evaluate atanh only for strictly interior requests.
    raw_hw = np.sign(limited) * SAFE_TARGET_RAW_ACTION_CLIP
    interior = np.abs(limited) < maximum
    raw_hw[interior] = capacity[interior] * np.arctanh(limited[interior] / capacity[interior]) / scale[interior]
    raw = np.clip(raw_hw, -SAFE_TARGET_RAW_ACTION_CLIP, SAFE_TARGET_RAW_ACTION_CLIP)[
        np.asarray(MUJOCO_TO_ISAACLAB_DOF)
    ].astype(np.float32)
    _, represented = safe_target_transform_numpy(raw)
    return (
        raw,
        represented,
        {
            "projection_explicitly_enabled": allow_projection,
            "unrepresentable_joint_indices": np.flatnonzero(outside).tolist(),
            "maximum_request_change_rad": float(np.max(np.abs(represented - target))),
            "raw_action_limit": SAFE_TARGET_RAW_ACTION_CLIP,
            "output_boundary_changed": False,
            "hardware_authorized": False,
        },
    )


def collect_standing_teacher(
    *, root: Path, asset_root: Path, policy_path: Path, motion, project_request: bool, initial_joint_delta=None
):
    """Record one continuous 10-second standing-only native23 teacher rollout.

    Uses exact same configured force limits as the paired SONIC evaluator.
    Captures causal 267/930 inputs, teacher request, representable raw label,
    applied physics effort/state, clock and warnings. Perturbed starts are
    synthetic, not evidence that hardware can acquire them.
    """
    import onnxruntime as ort

    from gear_sonic.scripts import evaluate_g1_true23_deployment_envelope as envelope

    if not isinstance(project_request, bool):
        raise ValueError("standing teacher request projection must be explicit boolean")
    profile = NativeSupportActuationProfile.from_sim_config(root / envelope.PHYSICS)
    controller = CleanTrue23MujocoController(
        model_path=asset_root / envelope.MODEL, physics_path=root / envelope.PHYSICS, policy=None
    )
    np.copyto(controller.physics.kp, profile.kp)
    np.copyto(controller.physics.kd, profile.kd)
    np.copyto(controller.physics.effort, profile.effort)
    controller.model.actuator_forcerange[:, 0] = -controller.physics.effort
    controller.model.actuator_forcerange[:, 1] = controller.physics.effort
    controller.model.jnt_actfrcrange[1:, 0] = -controller.physics.effort
    controller.model.jnt_actfrcrange[1:, 1] = controller.physics.effort
    q = np.asarray(SAFE_TARGET_DEFAULT_Q_HARDWARE).copy()
    if initial_joint_delta is not None:
        delta = np.asarray(initial_joint_delta, dtype=float)
        if delta.shape != (23,) or not np.isfinite(delta).all() or np.max(np.abs(delta)) > 0.02:
            raise ValueError("synthetic standing perturbation must be finite 23 joints within 0.02 rad")
        q += delta
    quat = np.array([1.0, 0.0, 0.0, 0.0])
    height, placement = envelope.measured_ground_contact_height(controller, q, quat)
    controller.reset(
        base_position=np.array([0.0, 0.0, height]), base_quaternion_wxyz=quat, joint_position_hardware=q
    )
    controller.history = [controller._policy_frame().copy() for _ in range(10)]
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    policy = UnitreeZeroVelocityFallbackPolicy(policy_path, session_options=options)
    physics = {
        key: []
        for key in (
            "pre_qpos",
            "post_qpos",
            "pre_qvel",
            "post_qvel",
            "effort",
            "generalized_actuator_force",
            "engine_pre_time_s",
            "engine_post_time_s",
            "engine_warning_counts",
        )
    }
    observations = {
        key: []
        for key in (
            "encoder267",
            "history930",
            "original_target_hardware23",
            "target_hardware23",
            "raw_native23",
            "kp",
            "kd",
        )
    }
    decisions = []
    module = controller.module

    class Recorder:
        def __getattr__(self, name):
            return getattr(module, name)

        def mj_step(self, model, data):
            if model is not controller.model or data is not controller.data:
                raise ValueError("standing recorder received another simulation")
            for key, value in (("pre_qpos", data.qpos), ("pre_qvel", data.qvel), ("effort", data.ctrl)):
                physics[key].append(value.copy())
            physics["engine_pre_time_s"].append(float(data.time))
            module.mj_step(model, data)
            physics["engine_post_time_s"].append(float(data.time))
            physics["engine_warning_counts"].append([int(item.number) for item in data.warning])
            for key, value in (
                ("post_qpos", data.qpos),
                ("post_qvel", data.qvel),
                ("generalized_actuator_force", data.qfrc_actuator[6:]),
            ):
                physics[key].append(value.copy())

    class Teacher:
        def reset(self):
            policy.reset()

        def activate(self, current_q):
            policy.activate(current_q)

        def infer(self, **kwargs):
            original, kp, kd = policy.infer(**kwargs)
            raw, target, decision = representable_standing_request(original, allow_projection=True)
            decisions.append(decision)
            encoder = encoder267_from_reference(
                motion_reference_terms(motion, 9), controller.buffered_robot_pelvis_q9
            )
            for key, value in (
                ("encoder267", encoder),
                ("history930", term_major_history(controller.history)),
                ("original_target_hardware23", original),
                ("target_hardware23", target),
                ("raw_native23", raw),
                ("kp", kp),
                ("kd", kd),
            ):
                observations[key].append(value.copy())
            return (target if project_request else original), kp, kd

    controller.module = Recorder()
    result, states, last_target = simulate_balance_transition(
        controller, Teacher(), duration_s=10.0, slew_rate=5.0, previous_target=q, project_effort=True
    )
    arrays = {"physics_" + key: np.asarray(value) for key, value in physics.items()}
    arrays.update({key: np.asarray(value) for key, value in observations.items()})
    arrays.update(qpos=states, final_target=last_target, physics_dt=np.array([0.002]))
    engine = audit_engine_trace(arrays)
    np.testing.assert_array_equal(arrays["physics_pre_qpos"][1:], arrays["physics_post_qpos"][:-1])
    np.testing.assert_array_equal(arrays["physics_pre_qvel"][1:], arrays["physics_post_qvel"][:-1])
    np.testing.assert_array_equal(arrays["physics_effort"], arrays["physics_generalized_actuator_force"])
    per_step_kp = np.repeat(arrays["kp"], 10, axis=0)[: len(arrays["physics_effort"])]
    per_step_kd = np.repeat(arrays["kd"], 10, axis=0)[: len(arrays["physics_effort"])]
    applied = (
        arrays["physics_pre_qpos"][:, 7:]
        + (arrays["physics_effort"] + per_step_kd * arrays["physics_pre_qvel"][:, 6:]) / per_step_kp
    )
    arrays["physics_reconstructed_applied_target"] = applied
    limits = np.asarray(profile.effort) * 0.95 * 0.25
    if np.any(np.abs(arrays["physics_effort"]) > limits + 1e-10):
        raise ValueError("standing teacher exceeded unchanged effort envelope")
    max_slew = float(np.max(np.abs(np.diff(np.vstack((q, applied)), axis=0))) / 0.002)
    if max_slew > 5 + 1e-9:
        raise ValueError("standing teacher exceeded unchanged slew envelope")
    result.update(
        kind="g1_true23_representable_standing_teacher_diagnostic_v1",
        actual_engine_audit=engine,
        compiled_native_model_sha256=compiled_model_sha256(controller.model),
        placement=placement,
        original_request_projected_before_actuation=project_request,
        synthetic_initial_joint_delta=(q - np.asarray(SAFE_TARGET_DEFAULT_Q_HARDWARE)).tolist(),
        unrepresentable_requested_joint_controls=sum(
            len(row["unrepresentable_joint_indices"]) for row in decisions
        ),
        maximum_request_change_rad=max(row["maximum_request_change_rad"] for row in decisions),
        maximum_reconstructed_applied_target_slew_rad_s=max_slew,
        standing_only_labels_usable=bool(
            project_request and engine["passed"] and result["existing_guard_screen_passed"]
        ),
        full_motion_suite_not_replaced=True,
        full_motion_teacher_accepted=False,
        sonic_policy_used=False,
        hardware_authorized=False,
        deployment_ready=False,
    )
    return result, arrays
