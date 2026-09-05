"""Offline factorial diagnosis of true23 deployment gains, targets and limits.

This integrates the existing native23 MuJoCo model with measured policy
feedback. It never imports Unitree SDK, opens DDS, or authorizes deployment.
The isolated dynamics test records hardware guard violations without emulating
ownership, networking, watchdogs, or recovery. A completed case is therefore
not a hardware qualification.
"""

from __future__ import annotations

import argparse
from itertools import product
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import onnxruntime as ort

from gear_sonic.utils.g1_23dof_contract import HARDWARE_JOINT_IDS
from gear_sonic.utils.g1_23dof_safe_target_transform import (
    SAFE_TARGET_DEFAULT_Q_HARDWARE,
    safe_target_transform_numpy,
)
from gear_sonic.utils.g1_true23_actuation_profile import read_joint_amplitude_scale
from gear_sonic.utils.g1_true23_clean_mujoco_teleop import (
    ENCODER_SHA256,
    CleanTrue23MujocoController,
    UnitreeZeroVelocityFallbackPolicy,
    encoder267_from_reference,
    motion_reference_terms,
    sha256_file,
)
from gear_sonic.utils.g1_true23_diagnostic_pair import load_diagnostic_pair, load_residual_diagnostic_pair
from gear_sonic.utils.g1_true23_motion_fidelity import assess_motion_fidelity
from gear_sonic.utils.g1_true23_sim_acquisition import (
    align_reference_xy_yaw,
    audit_reference_kinematics,
    effort_feasible_target,
    simulate_balance_transition,
)
from gear_sonic.utils.g1_true23_sonic_library_replay import (
    RELEASED_RETAINED_KD,
    RELEASED_RETAINED_KP,
    TRACKED_BODY_INDICES,
    ExactHashSonicPolicy,
    _quaternion_error_rad,
    _reference_policy_frame,
    validate_library_motion,
)
from gear_sonic.utils.g1_true23_step1b_mujoco import (
    _projected_gravity,
    _quaternion_matrix,
    term_major_history,
)

HEADER = "gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/true23_active_gantry_core.hpp"
MODEL = "gear_sonic/data/robots/g1/g1_23dof_rev_1_0.xml"
PHYSICS = "gear_sonic/config/sim_validation/g1_23dof_mujoco_sim2sim.json"
ENCODER = "artifacts/g1_true23/causal_model_250_20260803/causal_model_250.encoder.onnx"
ANKLES = np.asarray([4, 5, 10, 11])
DEFAULT_Q = np.asarray(SAFE_TARGET_DEFAULT_Q_HARDWARE, dtype=np.float64)


def load_measured_initial_state(path: Path) -> dict[str, Any]:
    """Use a historical healthy LowState snapshot, not a live robot connection."""
    source = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(source, dict):
        raise ValueError("measured initial state must be a JSON object")
    rows = source.get("motors")
    if (
        source.get("kind") != "g1_true23_motor_health_readonly_v1"
        or source.get("read_only") is not True
        or source.get("robot_commands_published") is not False
        or type(source.get("invalid_samples")) is not int
        or source["invalid_samples"] != 0
        or type(source.get("valid_crc_samples")) is not int
        or source["valid_crc_samples"] < 2
        or type(source.get("advancing_samples")) is not int
        or source["advancing_samples"] < 1
        or type(source.get("mode_machine")) is not int
        or source["mode_machine"] != 4
        or not isinstance(rows, list)
        or len(rows) != 23
    ):
        raise ValueError("measured initial state lacks valid read-only motor evidence")
    for compact, (row, slot) in enumerate(zip(rows, HARDWARE_JOINT_IDS, strict=True)):
        if (
            not isinstance(row, dict)
            or type(row.get("compact_index")) is not int
            or row["compact_index"] != compact
            or type(row.get("motor_slot")) is not int
            or row["motor_slot"] != slot
            or type(row.get("mode")) is not int
            or row["mode"] != 1
            or type(row.get("motorstate")) is not int
            or row["motorstate"] != 0
            or any(
                type(row.get(key)) not in (float, int) or not np.isfinite(row[key])
                for key in ("q_rad", "dq_rad_s", "tau_est_nm")
            )
        ):
            raise ValueError("measured motor order, health or numeric fields invalid")
    raw_quaternion = source.get("imu_quaternion_wxyz")
    if not isinstance(raw_quaternion, list) or any(type(value) not in (float, int) for value in raw_quaternion):
        raise ValueError("measured IMU quaternion invalid")
    quaternion = np.asarray(raw_quaternion, dtype=np.float64)
    if (
        quaternion.shape != (4,)
        or not np.isfinite(quaternion).all()
        or abs(np.linalg.norm(quaternion) - 1.0) > 1e-3
    ):
        raise ValueError("measured IMU quaternion invalid")
    return {
        "q": np.asarray([row["q_rad"] for row in rows]),
        "dq": np.asarray([row["dq_rad_s"] for row in rows]),
        "quaternion": quaternion / np.linalg.norm(quaternion),
        "source": str(path.resolve()),
        "source_sha256": sha256_file(path),
    }


def simulate_sampled_posture_hold(
    controller: CleanTrue23MujocoController,
    *,
    kp: np.ndarray,
    kd: np.ndarray,
    duration_s: float,
) -> tuple[dict[str, Any], np.ndarray]:
    """Isolate positive-kp hold dynamics; no Unitree handoff is simulated."""
    dt = controller.physics.timestep_s
    if not np.isfinite(duration_s) or not 0 <= duration_s <= 2:
        raise ValueError("hold duration must be 0..2 seconds in exact physics steps")
    steps = round(duration_s / dt)
    if abs(steps * dt - duration_s) > 1e-9:
        raise ValueError("hold duration must be 0..2 seconds in exact physics steps")
    if any(values.shape != (23,) or not np.isfinite(values).all() or np.any(values < 0) for values in (kp, kd)):
        raise ValueError("hold gains must be 23 finite nonnegative values")
    data = controller.data
    initial = data.qpos.copy()
    target = initial[7:].copy()
    states = [initial.copy()]
    peak_effort = 0.0
    first_guard = None
    for index in range(steps):
        predicted = kp * (target - data.qpos[7:]) - kd * data.qvel[6:]
        ratio = float(np.max(np.abs(predicted) / (0.10 * controller.physics.effort)))
        if ratio > 1 and first_guard is None:
            first_guard = {"physics_step": index, "ratio": ratio}
        torque = np.clip(predicted, -controller.physics.effort, controller.physics.effort)
        peak_effort = max(peak_effort, float(np.max(np.abs(torque))))
        data.ctrl[:] = torque
        controller.module.mj_step(controller.model, data)
        states.append(data.qpos.copy())
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            break
        if (index + 1) % controller.physics.decimation == 0:
            controller.history = [*controller.history[1:], controller._policy_frame().copy()]
            controller.buffered_robot_pelvis_q9 = data.qpos[3:7].copy()
    trajectory = np.asarray(states)
    return {
        "duration_s": duration_s,
        "completed_physics_steps": len(states) - 1,
        "minimum_height_m": float(np.min(trajectory[:, 2])),
        "maximum_knee_flexion_delta_rad": float(np.max(trajectory[:, [10, 16]] - initial[[10, 16]])),
        "maximum_abs_effort_nm": peak_effort,
        "first_tenth_effort_guard_violation": first_guard,
        "finite": bool(np.isfinite(trajectory).all()),
        "mode_transfer_simulated": False,
    }, trajectory


def measured_ground_contact_height(controller, q, quaternion) -> tuple[float, dict[str, Any]]:
    """Place the lowest modeled foot sphere on the ground, not the ankle origin."""
    module, model = controller.module, controller.model
    probe = module.MjData(model)
    probe.qpos[:3] = 0.0
    probe.qpos[3:7] = quaternion
    probe.qpos[7:] = q
    module.mj_forward(model, probe)
    ankles = [
        module.mj_name2id(model, module.mjtObj.mjOBJ_BODY, name)
        for name in ("left_ankle_roll_link", "right_ankle_roll_link")
    ]
    if min(ankles) < 1:
        raise ValueError("native23 ankle body missing")
    geom_ids = [
        index
        for index in range(model.ngeom)
        if model.geom_bodyid[index] in ankles and (model.geom_contype[index] or model.geom_conaffinity[index])
    ]
    if len(geom_ids) != 8 or any(model.geom_type[index] != module.mjtGeom.mjGEOM_SPHERE for index in geom_ids):
        raise ValueError("measured contact placement requires eight native23 foot spheres")
    floor = module.mj_name2id(model, module.mjtObj.mjOBJ_GEOM, "floor")
    if floor < 0 or model.geom_type[floor] != module.mjtGeom.mjGEOM_PLANE:
        raise ValueError("measured contact placement requires ground plane")
    if not np.allclose(probe.geom_xmat[floor].reshape(3, 3)[:, 2], [0.0, 0.0, 1.0]):
        raise ValueError("measured contact placement requires horizontal ground")
    bottoms = np.asarray([probe.geom_xpos[index, 2] - model.geom_size[index, 0] for index in geom_ids])
    floor_z = float(probe.geom_xpos[floor, 2])
    height = floor_z - float(bottoms.min())
    if not 0.2 <= height <= 0.95:
        raise ValueError("measured root height outside native23 diagnostic envelope")
    return height, {
        "method": "lowest_native23_foot_collision_sphere_tangent_to_ground",
        "root_height_m": height,
        "foot_sphere_clearances_m": (bottoms + height - floor_z).tolist(),
        "gantry_forces_modeled": False,
        "root_position_measured": False,
    }


def read_cpp_array(source: str, name: str) -> np.ndarray:
    match = re.search(r"\b" + re.escape(name) + r"\s*=\s*\{([^}]+)\}", source)
    if match is None:
        raise ValueError(f"missing C++ array: {name}")
    body = re.sub(r"//[^\n]*", "", match.group(1))
    values = np.asarray([float(item.strip()) for item in body.split(",") if item.strip()])
    if values.shape != (23,) or not np.isfinite(values).all():
        raise ValueError(f"invalid C++ array: {name}")
    return values


def apply_target_envelope(
    full_target: np.ndarray,
    previous_target: np.ndarray,
    *,
    fraction: float,
    joint_scale: np.ndarray,
    slew_rate: float | None,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Match C++ default-relative action scaling, followed by 500 Hz slew."""
    if not 0.0 < fraction <= 1.0 or dt <= 0.0:
        raise ValueError("invalid target envelope")
    if slew_rate is not None and (not np.isfinite(slew_rate) or slew_rate <= 0.0):
        raise ValueError("invalid slew rate")
    # Preserve the exact baseline target bits: subtracting/re-adding default
    # can introduce an ulp that later changes discrete FSQ tokens.
    if fraction == 1.0 and np.array_equal(joint_scale, np.ones(23)):
        requested = full_target.astype(np.float64)
    else:
        requested = DEFAULT_Q + (full_target.astype(np.float64) - DEFAULT_Q) * fraction * joint_scale
    target = (
        requested
        if slew_rate is None
        else np.clip(requested, previous_target - slew_rate * dt, previous_target + slew_rate * dt)
    )
    return target, requested


def _policy(
    encoder: Path, decoder: Path, decoder_hash: str, *, encoder_hash: str = ENCODER_SHA256
) -> ExactHashSonicPolicy:
    if sha256_file(encoder) != encoder_hash or sha256_file(decoder) != decoder_hash:
        raise ValueError("encoder or decoder hash mismatch")
    # Reuse exact inference implementation while bounding CPU threading.
    result = ExactHashSonicPolicy.__new__(ExactHashSonicPolicy)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    result.encoder = ort.InferenceSession(str(encoder), options, providers=["CPUExecutionProvider"])
    result.decoder = ort.InferenceSession(str(decoder), options, providers=["CPUExecutionProvider"])
    for session, source, target in ((result.encoder, [1, 267], [1, 64]), (result.decoder, [1, 994], [1, 23])):
        if len(session.get_inputs()) != 1 or len(session.get_outputs()) != 1:
            raise ValueError("unexpected ONNX boundary count")
        if session.get_inputs()[0].shape != source or session.get_outputs()[0].shape != target:
            raise ValueError("unexpected ONNX shape")
    result.encoder_input = result.encoder.get_inputs()[0].name
    result.decoder_input = result.decoder.get_inputs()[0].name
    return result


def run_case(
    *,
    root: Path,
    asset_root: Path,
    policy: ExactHashSonicPolicy,
    motion: dict[str, np.ndarray],
    kp: np.ndarray,
    kd: np.ndarray,
    fraction: float,
    joint_scale: np.ndarray,
    ankle_effort: float,
    slew_rate: float | None,
    initial_state: str,
    maximum_steps: int | None = None,
    measured_state: dict[str, Any] | None = None,
    startup_hold_s: float = 0.0,
    return_hold_s: float = 0.0,
    transition_policy: UnitreeZeroVelocityFallbackPolicy | None = None,
    align_reference_start: bool = False,
    project_transition_effort: bool = False,
    project_active_effort: bool = False,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    count = validate_library_motion(motion)
    steps = count - 11 if maximum_steps is None else min(count - 11, maximum_steps)
    if steps <= 0:
        raise ValueError("maximum_steps must be positive")
    controller = CleanTrue23MujocoController(
        model_path=asset_root / MODEL, physics_path=root / PHYSICS, policy=policy
    )
    np.copyto(controller.physics.kp, kp)
    np.copyto(controller.physics.kd, kd)
    controller.physics.effort[ANKLES] = ankle_effort
    controller.model.actuator_forcerange[:, 0] = -controller.physics.effort
    controller.model.actuator_forcerange[:, 1] = controller.physics.effort
    # Joint-level limits must change too; otherwise the 50 Nm cell is mislabeled.
    controller.model.jnt_actfrcrange[1:, 0] = -controller.physics.effort
    controller.model.jnt_actfrcrange[1:, 1] = controller.physics.effort
    measured_contact = None
    if initial_state == "reference":
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
    elif initial_state == "neutral":
        controller.reset(
            base_position=np.array([0.0, 0.0, controller.reference_root_height(DEFAULT_Q)]),
            base_quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
            joint_position_hardware=DEFAULT_Q,
        )
        controller.history = [controller._policy_frame().copy() for _ in range(10)]
    elif initial_state == "measured":
        if measured_state is None:
            raise ValueError("measured initial state requires a validated motor-health snapshot")
        q, quaternion = measured_state["q"], measured_state["quaternion"]
        height, measured_contact = measured_ground_contact_height(controller, q, quaternion)
        controller.reset(
            base_position=np.array([0.0, 0.0, height]),
            base_quaternion_wxyz=quaternion,
            joint_position_hardware=q,
            joint_velocity_hardware=measured_state["dq"],
        )
        controller.history = [controller._policy_frame().copy() for _ in range(10)]
    else:
        raise ValueError("unknown initial state")
    data, physics, module = controller.data, controller.physics, controller.module
    previous_target = np.asarray(data.qpos[7:], dtype=np.float64).copy()
    kinematic_audit = audit_reference_kinematics(controller, motion)
    if transition_policy is None:
        startup_hold, startup_qpos = simulate_sampled_posture_hold(
            controller,
            kp=0.25 * kp,
            kd=kd,
            duration_s=startup_hold_s,
        )
    else:
        startup_hold, startup_qpos, previous_target = simulate_balance_transition(
            controller,
            transition_policy,
            duration_s=startup_hold_s,
            slew_rate=slew_rate,
            previous_target=previous_target,
            project_effort=project_transition_effort,
        )
    reference_alignment = None
    if align_reference_start:
        motion, reference_alignment = align_reference_xy_yaw(motion, data.qpos)
    initial_qpos = data.qpos.copy()
    qpos = [initial_qpos.copy()]
    records: list[dict[str, float]] = []
    failure = (
        {"transition": 0, "gates": ["acquisition_standing_screen"]}
        if transition_policy is not None and not startup_hold["standing_screen_passed"]
        else None
    )
    first_predicted_gate = None
    first_target_gate = None
    clipped_joint_steps = 0
    joint_steps = 0
    projected_joint_substeps = active_physics_steps = 0
    for transition in range(steps if failure is None else 0):
        try:
            index = transition + 11
            packet = motion_reference_terms(motion, transition + 9)
            encoder_input = encoder267_from_reference(packet, controller.buffered_robot_pelvis_q9)
            current_pelvis = data.qpos[3:7].copy()
            controller.history = [*controller.history[1:], controller._policy_frame()]
            raw, _ = policy.infer(encoder_input, term_major_history(controller.history))
            safe_native, full_target = safe_target_transform_numpy(raw)
            peak_ankle = 0.0
            for substep in range(physics.decimation):
                target, requested = apply_target_envelope(
                    full_target,
                    previous_target,
                    fraction=fraction,
                    joint_scale=joint_scale,
                    slew_rate=slew_rate,
                    dt=physics.timestep_s,
                )
                clipped_joint_steps += int(np.count_nonzero(np.abs(target - requested) > 1e-10))
                joint_steps += 23
                if project_active_effort:
                    projected = effort_feasible_target(
                        requested,
                        previous_target,
                        data.qpos[7:].copy(),
                        data.qvel[6:].copy(),
                        kp,
                        kd,
                        physics.effort,
                        dt=physics.timestep_s,
                        slew_rate=slew_rate,
                    )
                    projected_joint_substeps += int(np.count_nonzero(np.abs(projected - target) > 1e-10))
                    target = projected
                guarded_target = target if project_active_effort else requested
                if first_target_gate is None and np.any(
                    (guarded_target < controller.model.jnt_range[1:, 0] + 0.05)
                    | (guarded_target > controller.model.jnt_range[1:, 1] - 0.05)
                ):
                    first_target_gate = {"transition": transition, "substep": substep}
                predicted = kp * (target - data.qpos[7:]) - kd * data.qvel[6:]
                if first_predicted_gate is None and np.any(np.abs(predicted) > 0.25 * physics.effort):
                    first_predicted_gate = {
                        "transition": transition,
                        "substep": substep,
                        "maximum_ratio": float(np.max(np.abs(predicted) / (0.25 * physics.effort))),
                    }
                torque = np.clip(predicted, -physics.effort, physics.effort)
                peak_ankle = max(peak_ankle, float(np.max(np.abs(torque[ANKLES]))))
                data.ctrl[:] = torque
                module.mj_step(controller.model, data)
                previous_target = target.copy()
                active_physics_steps += 1
            controller.buffered_robot_pelvis_q9 = current_pelvis
            # The deployed observation also feeds back the unscaled safe action.
            controller.previous_safe_native = safe_native.astype(np.float32, copy=True)
            controller.completed += 1
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                raise RuntimeError("nonfinite simulated state")
            gravity = _projected_gravity(data.qpos[3:7])
            tilt = float(np.arccos(np.clip(-gravity[2], -1, 1)))
            actual = data.xpos[1:][np.asarray(TRACKED_BODY_INDICES)]
            reference = motion["body_pos_w"][index][np.asarray(TRACKED_BODY_INDICES)]
            relative_error = float(
                np.max(np.linalg.norm((actual - actual[0]) - (reference - reference[0]), axis=1))
            )
            orientation_error = _quaternion_error_rad(data.qpos[3:7], motion["body_quat_w"][index, 0])
            joint_rmse = float(np.sqrt(np.mean((data.qpos[7:] - motion["joint_pos"][index]) ** 2)))
            record = {
                "height_m": float(data.qpos[2]),
                "tilt_rad": tilt,
                "joint_rmse_rad": joint_rmse,
                "relative_body_error_m": relative_error,
                "pelvis_orientation_error_rad": orientation_error,
                "pelvis_position_error_m": float(np.linalg.norm(actual[0] - reference[0])),
                "maximum_ankle_torque_nm": peak_ankle,
                "knee_flexion_delta_rad": float(np.max(data.qpos[[10, 16]] - initial_qpos[[10, 16]])),
            }
            records.append(record)
            qpos.append(data.qpos.copy())
            gates = [
                name
                for name, condition in (
                    ("absolute_height", data.qpos[2] < 0.12),
                    ("absolute_tilt", tilt > 2.2),
                    ("pelvis_orientation_tracking", orientation_error > 1.5),
                    ("relative_tracked_body_position", relative_error > 1.0),
                    ("joint_tracking", joint_rmse > 0.75),
                )
                if condition
            ]
            if gates:
                failure = {"transition": transition, "gates": gates}
                break
        except Exception as error:
            failure = {"transition": transition, "type": type(error).__name__, "message": str(error)}
            break
    series = {key: np.asarray([r[key] for r in records]) for key in records[0]} if records else {}
    report = {
        "completed_transitions": len(records),
        "completed_active_physics_steps": active_physics_steps,
        "active_partial_transition_substeps": active_physics_steps % physics.decimation,
        "active_elapsed_simulation_s": active_physics_steps * physics.timestep_s,
        "active_effort_target_projection": project_active_effort,
        "projected_active_joint_substeps": projected_joint_substeps,
        "requested_transitions": steps,
        "library_completion_passed": failure is None and len(records) == steps,
        "upright_physical_bounds_passed": failure is None
        and len(records) == count - 11
        and all(r["height_m"] >= 0.45 and r["tilt_rad"] <= 1.0 for r in records),
        "failure": failure,
        "first_predicted_quarter_effort_guard_violation": first_predicted_gate,
        "first_target_margin_guard_violation": first_target_gate,
        "slew_clipped_joint_step_fraction": clipped_joint_steps / max(joint_steps, 1),
        "maximums": {key: float(np.max(value)) for key, value in series.items()},
        "minimum_base_height_m": min((r["height_m"] for r in records), default=None),
        "horizontal_displacement_m": float(np.linalg.norm(data.qpos[:2] - initial_qpos[:2])),
        "gain_kp_hardware": kp.tolist(),
        "gain_kd_hardware": kd.tolist(),
        "effort_limit_hardware_nm": physics.effort.tolist(),
    }
    report["motion_fidelity"] = assess_motion_fidelity(
        metrics={
            "maximum_pelvis_position_error_m": report["maximums"].get("pelvis_position_error_m"),
            "maximum_pelvis_orientation_error_rad": report["maximums"].get("pelvis_orientation_error_rad"),
            "maximum_relative_tracked_body_position_error_m": report["maximums"].get("relative_body_error_m"),
            "maximum_joint_tracking_rmse_rad": report["maximums"].get("joint_rmse_rad"),
        },
        completed=len(records),
        requested=steps,
        available=count - 11,
        failure=failure,
    )
    terminal_active_qpos, terminal_active_qvel = data.qpos.copy(), data.qvel.copy()
    if transition_policy is None:
        return_hold, return_qpos = simulate_sampled_posture_hold(
            controller,
            kp=0.25 * kp,
            kd=kd,
            duration_s=return_hold_s,
        )
    elif not records:
        return_hold, return_qpos = {"not_run_reason": "acquisition_failed"}, data.qpos[None].copy()
    else:
        return_hold, return_qpos, _ = simulate_balance_transition(
            controller,
            transition_policy,
            duration_s=return_hold_s,
            slew_rate=slew_rate,
            previous_target=previous_target,
            project_effort=project_transition_effort,
        )
    report["startup_hold"] = startup_hold
    report["return_hold"] = return_hold
    report["measured_contact_estimate"] = measured_contact
    report["reference_kinematic_audit"] = kinematic_audit
    report["reference_alignment"] = reference_alignment
    report["lifecycle_simulator_screen_passed"] = (
        report["motion_fidelity"]["passed"]
        and startup_hold.get("existing_guard_screen_passed") is True
        and return_hold.get("existing_guard_screen_passed") is True
        and first_predicted_gate is None
        and first_target_gate is None
    )
    report["measured_initial_state"] = (
        {key: measured_state[key] for key in ("source", "source_sha256")}
        if initial_state == "measured" and measured_state is not None
        else None
    )
    return report, {
        "qpos": np.asarray(qpos),
        "terminal_active_qpos": terminal_active_qpos,
        "terminal_active_qvel": terminal_active_qvel,
        "startup_hold_qpos": startup_qpos,
        "return_hold_qpos": return_qpos,
        **series,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--decoder", type=Path, required=True)
    parser.add_argument("--expected-decoder-sha256", required=True)
    parser.add_argument("--encoder", type=Path, help="Explicit diagnostic encoder; requires its expected SHA256")
    parser.add_argument("--expected-encoder-sha256")
    parser.add_argument("--encoder-report", type=Path)
    parser.add_argument("--decoder-report", type=Path)
    parser.add_argument(
        "--residual-manifest",
        type=Path,
        help="Validate a newly fitted residual and its exact fitting encoder/base pair",
    )
    parser.add_argument(
        "--allow-unpaired-diagnostic",
        action="store_true",
        help="Explicit historical/mismatched-pair experiment only; never qualifies a candidate",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--gain-profiles",
        nargs="+",
        choices=["stage_one", "released_retained", "configured_sim"],
        default=["stage_one", "released_retained"],
    )
    parser.add_argument("--fractions", nargs="+", type=float, default=[0.6, 1.0])
    parser.add_argument("--ankle-efforts", nargs="+", type=float, default=[35.0, 50.0])
    parser.add_argument("--slew-rates", nargs="+", default=["5", "none"])
    parser.add_argument(
        "--initial-states", nargs="+", choices=["reference", "neutral", "measured"], default=["reference"]
    )
    parser.add_argument("--motor-health-snapshot", type=Path)
    parser.add_argument("--startup-hold-s", type=float, default=0.0)
    parser.add_argument("--return-hold-s", type=float, default=0.0)
    parser.add_argument(
        "--transition-balance-model",
        type=Path,
        help="Pinned compatibility actor for startup/return only; never replaces SONIC dance",
    )
    parser.add_argument(
        "--align-reference-start",
        action="store_true",
        help="Apply one rigid XY/yaw reference alignment after acquisition",
    )
    parser.add_argument(
        "--project-transition-effort",
        action="store_true",
        help="Project balance-phase targets inside existing effort/position/slew guards; simulator only",
    )
    parser.add_argument(
        "--project-active-effort",
        action="store_true",
        help="Simulator-only SONIC target projection inside existing effort/position/slew bounds",
    )
    parser.add_argument("--maximum-steps", type=int)
    args = parser.parse_args()
    if (args.encoder_report is None) != (args.decoder_report is None):
        parser.error("--encoder-report and --decoder-report must be supplied together")
    if args.encoder_report is None and args.residual_manifest is None and not args.allow_unpaired_diagnostic:
        parser.error(
            "paired export reports required; use --allow-unpaired-diagnostic only for explicit diagnostic comparisons"
        )
    if (
        sum((args.encoder_report is not None, args.residual_manifest is not None, args.allow_unpaired_diagnostic))
        != 1
    ):
        parser.error("choose exactly one: paired reports, residual manifest, or explicit unpaired diagnostic")
    if args.project_transition_effort and args.transition_balance_model is None:
        parser.error("--project-transition-effort requires --transition-balance-model")
    root, assets, output = args.repository_root.resolve(), args.asset_root.resolve(), args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    motion_path, decoder = args.motion.resolve(), args.decoder.resolve()
    measured_state = (
        load_measured_initial_state(args.motor_health_snapshot) if args.motor_health_snapshot is not None else None
    )
    if "measured" in args.initial_states and measured_state is None:
        parser.error("measured initial state requires --motor-health-snapshot")
    header = (root / HEADER).read_text()
    gains = {
        "stage_one": (read_cpp_array(header, "kStageOneKp"), read_cpp_array(header, "kStageOneKd")),
        "released_retained": (RELEASED_RETAINED_KP, RELEASED_RETAINED_KD),
    }
    config = json.loads((root / PHYSICS).read_text())["physics"]
    gains["configured_sim"] = (np.asarray(config["kp_hardware"]), np.asarray(config["kd_hardware"]))
    scale = np.asarray(read_joint_amplitude_scale(header))
    with np.load(motion_path, allow_pickle=False) as archive:
        motion = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
    validate_library_motion(motion)
    if (args.encoder is None) != (args.expected_encoder_sha256 is None):
        parser.error("--encoder and --expected-encoder-sha256 must be supplied together")
    encoder_path = assets / ENCODER if args.encoder is None else args.encoder.resolve()
    encoder_hash = ENCODER_SHA256 if args.expected_encoder_sha256 is None else args.expected_encoder_sha256
    pair = None
    if args.encoder_report is not None:
        pair = load_diagnostic_pair(args.encoder_report, args.decoder_report)
    elif args.residual_manifest is not None:
        pair = load_residual_diagnostic_pair(args.residual_manifest, decoder)
    if pair is not None:
        for component, path, digest in (
            ("encoder", encoder_path, encoder_hash),
            ("decoder", decoder, args.expected_decoder_sha256),
        ):
            if Path(pair[component]["path"]) != path.resolve() or pair[component]["sha256"] != digest:
                raise ValueError(f"requested {component} does not match paired export")
    policy = _policy(encoder_path, decoder, args.expected_decoder_sha256, encoder_hash=encoder_hash)
    transition_policy = None
    if args.transition_balance_model is not None:
        if args.startup_hold_s <= 0 or args.return_hold_s <= 0:
            parser.error("balance transitions require positive startup/return durations")
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        transition_policy = UnitreeZeroVelocityFallbackPolicy(
            args.transition_balance_model, session_options=options
        )
    output.mkdir(parents=True)
    cases = []
    for gain, fraction, effort, slew, initial in product(
        args.gain_profiles, args.fractions, args.ankle_efforts, args.slew_rates, args.initial_states
    ):
        rate = None if slew == "none" else float(slew)
        label = f"{gain}_fraction{fraction:g}_ankle{effort:g}_slew{slew}_{initial}"
        report, arrays = run_case(
            root=root,
            asset_root=assets,
            policy=policy,
            motion=motion,
            kp=gains[gain][0],
            kd=gains[gain][1],
            fraction=fraction,
            joint_scale=scale,
            ankle_effort=effort,
            slew_rate=rate,
            initial_state=initial,
            maximum_steps=args.maximum_steps,
            measured_state=measured_state,
            startup_hold_s=args.startup_hold_s,
            return_hold_s=args.return_hold_s,
            transition_policy=transition_policy,
            align_reference_start=args.align_reference_start,
            project_transition_effort=args.project_transition_effort,
            project_active_effort=args.project_active_effort,
        )
        report.update(
            encoder_decoder_pair_validated=pair is not None,
            paired_lifecycle_simulator_screen_passed=(
                pair is not None and report["lifecycle_simulator_screen_passed"]
            ),
            label=label,
            gain_profile=gain,
            action_fraction=fraction,
            ankle_effort_nm=effort,
            target_slew_rad_s=rate,
            initial_state=initial,
        )
        with (output / f"{label}.npz").open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        (output / f"{label}.json").write_text(json.dumps(report, indent=2) + "\n")
        cases.append(report)
        print(
            json.dumps(
                {"case": label, "completed": report["completed_transitions"], "failure": report["failure"]}
            ),
            flush=True,
        )
    summary = {
        "kind": "g1_true23_deployment_envelope_diagnostic_v1",
        "diagnostic_pair": pair,
        "unpaired_diagnostic_only": pair is None,
        "cases": cases,
        "sources": {
            str(path): sha256_file(path)
            for path in (
                root / HEADER,
                assets / MODEL,
                root / PHYSICS,
                encoder_path,
                decoder,
                motion_path,
                Path(__file__).resolve(),
                root / "gear_sonic/utils/g1_true23_clean_mujoco_teleop.py",
                root / "gear_sonic/utils/g1_true23_sonic_library_replay.py",
                root / "gear_sonic/utils/g1_true23_step1b_mujoco.py",
                root / "gear_sonic/utils/g1_true23_motion_fidelity.py",
                root / "gear_sonic/utils/g1_23dof_safe_target_transform.py",
                root / "gear_sonic/utils/g1_true23_actuation_profile.py",
                root / "gear_sonic/utils/g1_true23_sim_acquisition.py",
            )
        },
        "runtime": {
            "onnxruntime": ort.__version__,
            "numpy": np.__version__,
            "intra_op_threads": 1,
            "inter_op_threads": 1,
        },
        "transition_balance_model": (
            {
                "path": str(args.transition_balance_model.resolve()),
                "sha256": sha256_file(args.transition_balance_model),
            }
            if args.transition_balance_model is not None
            else None
        ),
        "authorization": {
            "simulator_only": True,
            "robot_commands_published": False,
            "dds_opened": False,
            "network_used": False,
            "hardware_authorized": False,
            "deployment_ready": False,
        },
        "limitations": [
            "Ideal 500 Hz actuation and 50 Hz inference; no transport or ownership lifecycle.",
            "Optional holds model only explicit sampled-posture PD, not Unitree mode transfer.",
            "Optional standing actor is a 29-to-23 compatibility diagnostic, not native Unitree FSM ownership or verified hardware recovery.",
            "Measured start uses recorded joints/IMU; root position, zero base velocity and ground contact are estimated, gantry forces absent.",
            "Hardware guard crossings recorded diagnostically, not used to stop physics.",
            "Ankle limits are hypothetical model ablations, not verified hardware ratings.",
            "Neutral initial state is a simulated default pose, not measured FSM801 acquisition.",
            "Legacy library completion tolerates large world drift and tracking error; not dance fidelity.",
        ],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
