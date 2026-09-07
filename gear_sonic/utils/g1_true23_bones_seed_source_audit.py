"""Challenge a retimed reference at every original BONES-SEED timestamp.

This is an additional FK-only gate, not a replacement for the stored control
grid's trajectory/serialization audits or a claim about continuous-time physics.
Linear joint/translation interpolation and root SLERP are explicit assumptions.
No cached achieved task positions are used as evidence.
"""

from __future__ import annotations

from dataclasses import asdict

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from gear_sonic.utils import g1_23dof_task_space_retarget as ik
from gear_sonic.utils.g1_true23_generalist_retarget import (
    AdaptationLimits,
    _reduce_excursion,
    _resample,
    _source_task_poses,
    task_space_fidelity,
    validate_named_motion,
)


def _array(value, shape, label):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{label} must have finite shape {shape}")
    return result


def _quat(value, shape, label):
    result = _array(value, shape, label)
    if not np.allclose(np.linalg.norm(result, axis=-1), 1, rtol=0, atol=1e-6):
        raise ValueError(f"{label} must contain normalized quaternions")
    return result


def _interpolate(values, source_times, target_times):
    return np.column_stack(
        [np.interp(target_times, source_times, values[:, column]) for column in range(values.shape[1])]
    )


def audit_original_timestamps(source_model, target_model, source_arrays, adapted_arrays, retarget_report):
    """Recompute original/direct/adapted FK on the entire original sample grid."""
    if (source_model.nq, target_model.nq, target_model.nu) != (36, 30, 23):
        raise ValueError("original-time audit requires named29 and physical native23 models")
    if (
        retarget_report.get("accepted") is not True
        or retarget_report.get("kind") != "g1_true23_generalist_bounded_offline_retarget"
        or retarget_report.get("source_role") != "requested_choreography"
    ):
        raise ValueError("original-time audit requires an accepted requested-choreography control-grid fit")
    declared = dict(retarget_report["limits"])
    for key in ("duration_scales", "excursion_scales"):
        declared[key] = tuple(declared[key])
    limits = AdaptationLimits(**declared)
    defaults = asdict(AdaptationLimits())
    if any(
        value != defaults[key]
        for key, value in asdict(limits).items()
        if key not in {"duration_scales", "excursion_scales", "maximum_output_frames"}
    ):
        raise ValueError("original-time audit cannot weaken the existing task/trajectory limits")
    config = ik.RetargetConfig(**retarget_report["ik_config"])
    if config != ik.RetargetConfig(optimize_lower_body=True, allow_acceleration_constraint_relaxation=False):
        raise ValueError("original-time audit requires the unchanged protected-task configuration")
    selected = retarget_report["selected_attempt"]
    if type(selected) is not int or not 0 <= selected < len(retarget_report["attempts"]):
        raise ValueError("invalid selected retarget attempt")
    attempt = retarget_report["attempts"][selected]
    if attempt.get("accepted") is not True or attempt.get("failures") != []:
        raise ValueError("selected control-grid candidate was not accepted")
    duration, excursion = attempt["requested_duration_scale"], attempt["requested_excursion_scale"]
    if duration not in limits.duration_scales or excursion not in limits.excursion_scales:
        raise ValueError("selected candidate differs from declared adaptation scales")
    source = validate_named_motion(source_arrays, source_model, allow_source_limit_excess=True)
    count, fps = len(source["joint_pos"]), float(source["fps"][0])
    if retarget_report["source_frame_count"] != count or retarget_report["source_fps"] != fps:
        raise ValueError("retarget report source frame count/fps differs")
    original_control, time_map, scale = _resample(source, duration, limits)
    control_count = len(time_map)
    stored_time = _array(adapted_arrays["source_time_map_s"], (control_count,), "source-time map")
    if not np.array_equal(stored_time, time_map) or not np.isclose(
        attempt["actual_duration_scale"], scale, rtol=0, atol=1e-12
    ):
        raise ValueError("source-time map must cover every original endpoint without cropping or reanchoring")
    times = _array(adapted_arrays["timestamps_s"], (control_count,), "control timestamps")
    if not np.array_equal(times, np.arange(control_count) / limits.target_fps):
        raise ValueError("control timestamps must preserve the exact declared control grid")
    stored_fps = np.asarray(adapted_arrays["fps"])
    if stored_fps.size != 1 or float(stored_fps.reshape(-1)[0]) != limits.target_fps:
        raise ValueError("adapted fps differs from declared control grid")
    source_names = tuple(source["joint_names"].tolist())
    layout = ik._model_layout(target_model)
    if (
        tuple(adapted_arrays["joint_names"].tolist()) != layout.joint_names
        or tuple(adapted_arrays["source_joint_names"].tolist()) != source_names
    ):
        raise ValueError("reference joint names/order differ from source or target model")
    for key in ("joint_pos", "root_pos_w", "root_quat_wxyz"):
        expected = original_control[key]
        stored = _array(adapted_arrays[f"source_{key}_resampled"], expected.shape, f"stored source {key}")
        if not np.array_equal(stored, expected):
            raise ValueError("stored resampled source differs from complete original source")
    joints = _array(adapted_arrays["joint_pos"], (control_count, 23), "native23 joints")
    body_pos = _array(
        adapted_arrays["body_pos_w"], (control_count, target_model.nbody - 1, 3), "native body positions"
    )
    body_quat = _quat(
        adapted_arrays["body_quat_w"], (control_count, target_model.nbody - 1, 4), "native body quaternions"
    )
    original_times = source["timestamps_s"]
    achieved_joint = _interpolate(joints, time_map, original_times)
    achieved_root = _interpolate(body_pos[:, 0], time_map, original_times)
    achieved_quat = Slerp(time_map, Rotation.from_quat(body_quat[:, 0, [1, 2, 3, 0]]))(original_times).as_quat()[
        :, [3, 0, 1, 2]
    ]
    candidate = _reduce_excursion(source, excursion)
    contacts = source.get("contact_flags")
    if contacts is None:
        feet = ik._source_foot_positions(
            source_model,
            ik._model_layout(source_model),
            source["root_pos_w"],
            source["root_quat_wxyz"],
            source["joint_pos"],
        )
        contacts = ik.infer_foot_contacts(
            feet,
            fps=fps,
            height_tolerance_m=config.contact_height_tolerance_m,
            speed_tolerance_m_s=config.contact_speed_tolerance_m_s,
        )
    direct_layout = ik._safe_target_layout(layout, config.safe_limit_guard_rad, config.native_action_clip)
    direct = np.clip(
        candidate["joint_pos"][:, [source_names.index(name) for name in layout.joint_names]],
        direct_layout.lower,
        direct_layout.upper,
    )
    source_layout, source_data, target_data = (
        ik._model_layout(source_model),
        mujoco.MjData(source_model),
        mujoco.MjData(target_model),
    )
    original_positions, _ = _source_task_poses(source_model, source)
    desired_positions, _ = _source_task_poses(source_model, candidate)
    achieved_positions = np.empty_like(desired_positions)
    measurements = {
        side: {
            "cost": np.empty(count),
            "position": np.empty((count, len(ik.DEFAULT_TASKS))),
            "orientation": np.empty((count, len(ik.DEFAULT_TASKS))),
        }
        for side in ("before", "after")
    }
    for frame in range(count):
        ik._set_configuration(
            source_model,
            source_data,
            source_layout,
            candidate["root_pos_w"][frame],
            candidate["root_quat_wxyz"][frame],
            candidate["joint_pos"][frame],
        )
        targets = ik._task_targets(source_model, source_data, ik.DEFAULT_TASKS)
        for side, root, quat, joint in (
            ("before", candidate["root_pos_w"], candidate["root_quat_wxyz"], direct),
            ("after", achieved_root, achieved_quat, achieved_joint),
        ):
            ik._set_configuration(target_model, target_data, layout, root[frame], quat[frame], joint[frame])
            jac, residual, position, orientation, _, _ = ik._task_linearization(
                target_model,
                target_data,
                layout,
                ik.DEFAULT_TASKS,
                targets,
                tuple(bool(x) for x in contacts[frame]),
                config.contact_weight_multiplier,
                np.arange(23),
            )
            measurements[side]["cost"][frame] = ik._weighted_task_error(jac, residual)
            measurements[side]["position"][frame] = position
            measurements[side]["orientation"][frame] = orientation
        achieved_positions[frame], _ = ik._task_pose_arrays(
            target_model, target_data, ik.DEFAULT_TASKS, source=False
        )
    before, after = measurements["before"], measurements["after"]
    task_indices = {task.name: index for index, task in enumerate(ik.DEFAULT_TASKS)}
    com = task_indices["whole_robot_com"]
    low, high = ik.safe_target_joint_bounds(target_model, native_action_clip=9.5, safe_limit_guard_rad=0.05)
    # Match _refined_result_from_qpos's existing float32 serialization check;
    # this is not an additional task-error allowance.
    joint_serialization_tolerance = 1e-7
    joint_excess = np.maximum(np.maximum(low - achieved_joint, achieved_joint - high), 0)
    masks = {
        "weighted_task_cost_regression": after["cost"]
        > before["cost"] + config.priority_relative_tolerance * np.maximum(1, before["cost"]),
        "com_position_regression": after["position"][:, com]
        > before["position"][:, com] + config.valid_max_com_regression_m,
        "guarded_joint_position": np.any(joint_excess > joint_serialization_tolerance, axis=1),
        "native_action_clip": np.any(
            np.abs(ik._hardware_targets_to_raw_native(achieved_joint)) > config.native_action_clip, axis=1
        ),
        "root_offset": np.max(np.abs(achieved_root - candidate["root_pos_w"]), axis=1) > config.max_root_offset_m,
    }
    for foot in ("left_foot", "right_foot"):
        index = task_indices[foot]
        masks[f"{foot}_position"] = after["position"][:, index] > config.valid_max_foot_position_error_m
        masks[f"{foot}_orientation_regression"] = (
            after["orientation"][:, index]
            > before["orientation"][:, index] + config.valid_max_foot_orientation_regression_rad
        )
    fidelity = task_space_fidelity(original_positions, desired_positions, achieved_positions)
    failures = [name for name, mask in masks.items() if mask.any()]
    for name, metrics in fidelity.items():
        if metrics["adaptation_distortion_fraction"] > limits.max_excursion_reduction + 1e-6:
            failures.append(f"{name}: adaptation exceeds 20% original task-space excursion")
        threshold = (
            limits.foot_p95_m
            if name in ("left_foot", "right_foot")
            else limits.hand_head_p95_m
            if name in ("left_hand", "right_hand", "head_proxy")
            else None
        )
        if threshold is not None and metrics["achieved_to_adapted_m"]["p95"] > threshold:
            failures.append(f"{name}: original-time adapted-reference p95 tolerance exceeded")
    union = np.logical_or.reduce(list(masks.values()))
    return {
        "kind": "g1_true23_bones_seed_all_original_timestamps_fk_audit_v1",
        "original_timestamp_fidelity_passed": not failures,
        "failures": failures,
        "original_frames_evaluated": count,
        "original_fps": fps,
        "control_frames": control_count,
        "control_fps": limits.target_fps,
        "actual_duration_scale": scale,
        "all_original_timestamps_and_endpoints_evaluated": True,
        "source_root_reanchored_or_joint_targets_clipped": False,
        "original_time_sample_indices": list(range(count)),
        "interpolation": "linear_joint_and_world_translation_root_slerp_in_source_time",
        "cached_achieved_task_positions_used": False,
        "joint_serialization_tolerance_rad": joint_serialization_tolerance,
        "guarded_joint_position_raw_excess_max_rad": float(joint_excess.max()),
        "foot_orientation_regression_excess_max_rad": {
            foot: float(
                np.maximum(
                    after["orientation"][:, task_indices[foot]]
                    - before["orientation"][:, task_indices[foot]]
                    - config.valid_max_foot_orientation_regression_rad,
                    0,
                ).max()
            )
            for foot in ("left_foot", "right_foot")
        },
        "fidelity": fidelity,
        "invalid_original_frame_count": int(union.sum()),
        "categories": {
            name: {"count": int(mask.sum()), "frame_indices": np.flatnonzero(mask).tolist()}
            for name, mask in masks.items()
        },
        "contact_flags_are_physical_contact_evidence": False,
        "control_grid_trajectory_audit_replaced": False,
        "continuous_time_fidelity_proven": False,
        "dynamic_feasibility_verified": False,
        "training_corpus_ready": False,
        "deployment_ready": False,
        "hardware_authorized": False,
    }
