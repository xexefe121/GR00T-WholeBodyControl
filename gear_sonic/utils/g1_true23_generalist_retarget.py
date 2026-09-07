"""Bounded offline named-29 to physical-23 task-space adaptation.

This is motion preparation, not a dynamics controller or live-input adapter.
The existing MuJoCo hierarchical IK supplies actual feet/COM/head/hand fitting.
Full-clip preprocessing is deliberately rejected at the live-input boundary.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from copy import deepcopy
import math
from typing import Any

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from gear_sonic.utils import g1_23dof_task_space_retarget as ik


@dataclass(frozen=True)
class AdaptationLimits:
    target_fps: float = 50.0
    duration_scales: tuple[float, ...] = (1.0, 1.25, 1.5, 2.0)
    excursion_scales: tuple[float, ...] = (1.0, 0.9, 0.8)
    max_duration_scale: float = 2.0
    max_excursion_reduction: float = 0.2
    maximum_output_frames: int = 30000
    foot_p95_m: float = 0.05
    hand_head_p95_m: float = 0.10
    max_joint_velocity_rad_s: float = 8.0
    max_joint_acceleration_rad_s2: float = 80.0
    ik_iterations: int = 16

    def __post_init__(self) -> None:
        positive = (
            self.target_fps,
            self.max_duration_scale,
            self.foot_p95_m,
            self.hand_head_p95_m,
            self.max_joint_velocity_rad_s,
            self.max_joint_acceleration_rad_s2,
        )
        if any(not math.isfinite(value) or value <= 0 for value in positive):
            raise ValueError("adaptation limits must be finite and positive")
        if not 1 <= self.max_duration_scale <= 2 or not 0 <= self.max_excursion_reduction <= 0.2:
            raise ValueError("adaptation cannot exceed 2x duration or 20% excursion reduction")
        if not self.duration_scales or not self.excursion_scales:
            raise ValueError("adaptation requires bounded candidate scales")
        if any(
            not math.isfinite(value) or not 1 <= value <= self.max_duration_scale for value in self.duration_scales
        ):
            raise ValueError("duration candidate exceeds adaptation bounds")
        if any(
            not math.isfinite(value) or not 1 - self.max_excursion_reduction <= value <= 1
            for value in self.excursion_scales
        ):
            raise ValueError("excursion candidate exceeds adaptation bounds")
        if len(self.duration_scales) * len(self.excursion_scales) > 24:
            raise ValueError("at most 24 adaptation candidates permitted")
        if (
            isinstance(self.maximum_output_frames, bool)
            or not isinstance(self.maximum_output_frames, int)
            or not 2 <= self.maximum_output_frames <= 60000
        ):
            raise ValueError("maximum_output_frames must be an integer in [2, 60000]")
        if (
            isinstance(self.ik_iterations, bool)
            or not isinstance(self.ik_iterations, int)
            or not 1 <= self.ik_iterations <= 128
        ):
            raise ValueError("ik_iterations must be an integer in [1, 128]")


@dataclass
class AdaptationResult:
    accepted: bool
    arrays: dict[str, np.ndarray] | None
    report: dict[str, Any]
    diagnostic_arrays: dict[str, np.ndarray] | None = None


def validate_named_motion(
    arrays: dict[str, np.ndarray], source_model: mujoco.MjModel, *, allow_source_limit_excess: bool = False
) -> dict[str, np.ndarray]:
    names_array = np.asarray(arrays["joint_names"])
    if names_array.ndim != 1 or names_array.dtype.kind not in "US":
        raise ValueError("joint_names must be a one-dimensional string array")
    names = tuple(str(name) for name in names_array.tolist())
    expected = ik._model_layout(source_model).joint_names
    if len(expected) != 29 or len(names) != 29 or len(set(names)) != 29 or set(names) != set(expected):
        raise ValueError("named source motion must contain exactly all 29 source-model joints")
    fps_array = np.asarray(arrays["fps"])
    if (
        fps_array.size != 1
        or not math.isfinite(float(fps_array.reshape(-1)[0]))
        or float(fps_array.reshape(-1)[0]) <= 0
    ):
        raise ValueError("source fps must be one finite positive value")
    fps = float(fps_array.reshape(-1)[0])
    joints = np.asarray(arrays["joint_pos"], dtype=np.float64)
    if joints.ndim != 2 or joints.shape[1] != 29 or not 2 <= len(joints) <= 60000:
        raise ValueError("source joint_pos must be [2..60000 frames, 29]")
    root = np.asarray(arrays["root_pos_w"], dtype=np.float64)
    quat = np.asarray(arrays["root_quat_wxyz"], dtype=np.float64)
    if root.shape != (len(joints), 3) or quat.shape != (len(joints), 4):
        raise ValueError("source root position/quaternion shapes disagree with joint trajectory")
    if not all(np.isfinite(value).all() for value in (joints, root, quat)):
        raise ValueError("source motion contains nonfinite values")
    norms = np.linalg.norm(quat, axis=1)
    if not np.allclose(norms, 1, atol=1e-4, rtol=0):
        raise ValueError("source root quaternions must be normalized")
    ordered = joints[:, [names.index(name) for name in expected]].copy()
    layout = ik._model_layout(source_model)
    if not allow_source_limit_excess and (
        np.any(ordered < layout.lower - 1e-6) or np.any(ordered > layout.upper + 1e-6)
    ):
        raise ValueError("source joint trajectory exceeds source-model joint limits")
    timestamps = np.asarray(arrays.get("timestamps_s", np.arange(len(joints)) / fps), dtype=np.float64)
    if (
        timestamps.shape != (len(joints),)
        or not np.isfinite(timestamps).all()
        or not np.allclose(np.diff(timestamps), 1 / fps, rtol=1e-6, atol=1e-9)
    ):
        raise ValueError("source timestamps must be uniform and agree with fps")
    contacts = arrays.get("contact_flags")
    if contacts is not None:
        contacts = np.asarray(contacts)
        if contacts.shape != (len(joints), 2) or contacts.dtype.kind != "b":
            raise ValueError("source contact_flags must be boolean [frames, 2]")
    return {
        "joint_names": np.asarray(expected),
        "joint_pos": ordered,
        "root_pos_w": root.copy(),
        "root_quat_wxyz": quat.copy(),
        "fps": np.asarray([fps]),
        "timestamps_s": timestamps.copy(),
        **({"contact_flags": contacts.copy()} if contacts is not None else {}),
    }


def _resample(
    source: dict[str, np.ndarray], duration_scale: float, limits: AdaptationLimits
) -> tuple[dict[str, np.ndarray], np.ndarray, float]:
    original_times = source["timestamps_s"]
    duration = float(original_times[-1] - original_times[0])
    intervals = int(math.floor(duration * duration_scale * limits.target_fps + 1e-9))
    if intervals < 1 or intervals + 1 > limits.maximum_output_frames:
        raise ValueError("candidate duration is outside bounded output frame count")
    actual_scale = intervals / limits.target_fps / duration
    if actual_scale < 1 - 1e-9 or actual_scale > limits.max_duration_scale + 1e-9:
        raise ValueError("exact-rate full-clip sampling would violate duration bounds")
    source_times = np.linspace(original_times[0], original_times[-1], intervals + 1)
    resampled = {}
    for name in ("root_pos_w", "joint_pos"):
        values = source[name]
        resampled[name] = np.stack(
            [np.interp(source_times, original_times, values[:, index]) for index in range(values.shape[1])], axis=1
        )
    rotations = Rotation.from_quat(source["root_quat_wxyz"][:, [1, 2, 3, 0]])
    resampled["root_quat_wxyz"] = Slerp(original_times, rotations)(source_times).as_quat()[:, [3, 0, 1, 2]]
    if "contact_flags" in source:
        indices = np.clip(
            np.searchsorted(original_times, source_times, side="right") - 1, 0, len(original_times) - 1
        )
        resampled["contact_flags"] = source["contact_flags"][indices].copy()
    return resampled, source_times, actual_scale


def _source_task_poses(
    source_model: mujoco.MjModel, sample: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    layout, data = ik._model_layout(source_model), mujoco.MjData(source_model)
    positions, quaternions = [], []
    for index in range(len(sample["joint_pos"])):
        ik._set_configuration(
            source_model,
            data,
            layout,
            sample["root_pos_w"][index],
            sample["root_quat_wxyz"][index],
            sample["joint_pos"][index],
        )
        position, quaternion = ik._task_pose_arrays(source_model, data, ik.DEFAULT_TASKS, source=True)
        positions.append(position)
        quaternions.append(quaternion)
    return np.asarray(positions), np.asarray(quaternions)


def _reduce_excursion(sample: dict[str, np.ndarray], scale: float) -> dict[str, np.ndarray]:
    candidate = {name: values.copy() for name, values in sample.items()}
    for name in ("root_pos_w", "joint_pos"):
        candidate[name] = sample[name][0] + scale * (sample[name] - sample[name][0])
    rotations = Rotation.from_quat(sample["root_quat_wxyz"][:, [1, 2, 3, 0]])
    relative = rotations[0].inv() * rotations
    candidate["root_quat_wxyz"] = (rotations[0] * Rotation.from_rotvec(relative.as_rotvec() * scale)).as_quat()[
        :, [3, 0, 1, 2]
    ]
    return candidate


def _error_statistics(errors: np.ndarray) -> dict[str, float]:
    return {"mean": float(np.mean(errors)), "p95": float(np.percentile(errors, 95)), "max": float(np.max(errors))}


def task_space_fidelity(original: np.ndarray, adapted: np.ndarray, achieved: np.ndarray) -> dict[str, Any]:
    """Position errors and source-relative adaptation; no resettable drift baseline."""
    if (
        original.shape != adapted.shape
        or original.shape != achieved.shape
        or original.ndim != 3
        or original.shape[1:] != (len(ik.DEFAULT_TASKS), 3)
    ):
        raise ValueError("task arrays must share [frames, declared tasks, 3]")
    if not all(np.isfinite(value).all() for value in (original, adapted, achieved)):
        raise ValueError("task fidelity requires finite poses")
    results = {}
    for index, task in enumerate(ik.DEFAULT_TASKS):
        excursion = np.linalg.norm(original[:, index] - original[0, index], axis=1)
        source_amplitude = float(np.max(excursion))
        distortion = np.linalg.norm(adapted[:, index] - original[:, index], axis=1)
        # Static tasks cannot acquire arbitrary motion under a percentage bound.
        allowed_denominator = max(source_amplitude, 1e-8)
        results[task.name] = {
            "achieved_to_original_m": _error_statistics(
                np.linalg.norm(achieved[:, index] - original[:, index], axis=1)
            ),
            "achieved_to_adapted_m": _error_statistics(
                np.linalg.norm(achieved[:, index] - adapted[:, index], axis=1)
            ),
            "adapted_to_original_m": _error_statistics(distortion),
            "source_excursion_max_m": source_amplitude,
            "adaptation_distortion_fraction": float(np.max(distortion)) / allowed_denominator,
        }
    return results


def _retarget_candidate(
    *,
    source_model: mujoco.MjModel,
    target_model: mujoco.MjModel,
    candidate: dict[str, np.ndarray],
    fps: float,
    config: ik.RetargetConfig,
    solver_attempts: list[dict[str, Any]],
) -> ik.RetargetResult:
    """Try fixed-root all23 IK only after the specific whole-path seed failure.

    This changes the numerical strategy, not trajectory acceptance bounds. The
    alternate solver does not create or freeze a separate lower-body path; all
    23 joints are solved sequentially under the same derivative constraints.
    Every call gets copies of the exact same full candidate and contacts.
    """

    def solve(strategy: str, solver_config: ik.RetargetConfig) -> ik.RetargetResult:
        receipt: dict[str, Any] = {
            "strategy": strategy,
            "config": asdict(solver_config),
            "requested_frames": len(candidate["joint_pos"]),
            "completed": False,
        }
        solver_attempts.append(receipt)
        try:
            result = ik.retarget_trajectory(
                source_model=source_model,
                target_model=target_model,
                root_pos_w=candidate["root_pos_w"].copy(),
                root_quat_wxyz=candidate["root_quat_wxyz"].copy(),
                source_joint_pos_hardware=candidate["joint_pos"].copy(),
                fps=fps,
                contact_flags=candidate["contact_flags"].copy(),
                config=solver_config,
            )
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            receipt["error"] = f"{type(exc).__name__}: {exc}"
            raise
        receipt["completed"] = True
        receipt["completed_frames"] = len(result.joint_pos_hardware)
        return result

    try:
        return solve("whole_horizon_lower_root_then_upper", config)
    except RuntimeError as exc:
        if not str(exc).startswith("whole-horizon lower/root seed creates new invalid frames;"):
            raise
    return solve("sequential_all23_fixed_root", replace(config, enable_lower_root_feasibility=False))


def protected_frame_failure_categories(result):
    """Exact existing expert-mask categories; overlapping failures are explicit."""
    d, cfg = result.diagnostics, result.config
    before = d["weighted_task_error_before"]
    masks = {
        "weighted_task_cost_regression": d["weighted_task_error_after"]
        > before + cfg.priority_relative_tolerance * np.maximum(1.0, before),
        "com_position_regression": d["task_whole_robot_com_position_error_after_m"]
        > d["task_whole_robot_com_position_error_before_m"] + cfg.valid_max_com_regression_m,
        "trajectory_constraint_relaxation": d["constraint_relaxation_count"] != 0,
        "native_action_clip": np.any(np.abs(result.action_target_native) > cfg.native_action_clip, axis=1),
    }
    for foot in ("left_foot", "right_foot"):
        masks[f"{foot}_position"] = d[f"task_{foot}_position_error_after_m"] > cfg.valid_max_foot_position_error_m
        masks[f"{foot}_orientation_regression"] = (
            d[f"task_{foot}_orientation_error_after_rad"]
            > d[f"task_{foot}_orientation_error_before_rad"] + cfg.valid_max_foot_orientation_regression_rad
        )
    union = np.logical_or.reduce(list(masks.values()))
    if not np.array_equal(~union, result.expert_valid_mask()):
        raise ValueError("per-frame protected failure categories disagree with original expert gate")
    return {
        "frame_count": len(union),
        "invalid_frame_count": int(union.sum()),
        "valid_frame_count": int((~union).sum()),
        "category_counts_overlap": True,
        "categories": {
            name: {"count": int(mask.sum()), "frame_indices": np.flatnonzero(mask).tolist()}
            for name, mask in masks.items()
        },
    }


def _candidate_assessment(result, original_positions, limits):
    """Shared acceptance gate, identical for fixed-root and SE(3) strategies."""
    summary = result.summary()
    summary["protected_frame_failure_categories"] = protected_frame_failure_categories(result)
    fidelity = task_space_fidelity(original_positions, result.desired_task_pos_w, result.achieved_task_pos_w)
    failures = [
        failure for failure in summary["kinematic_gate_failures"] if failure != "no mean task-space improvement"
    ]
    if not np.all(result.expert_valid_mask()):
        failures.append("not every requested frame passes protected-task and trajectory bounds")
    for key, maximum in (
        ("measured_velocity_abs_max_rad_s", limits.max_joint_velocity_rad_s),
        ("measured_acceleration_abs_max_rad_s2", limits.max_joint_acceleration_rad_s2),
    ):
        if summary["constraints"][key] > maximum + 1e-6:
            failures.append(f"{key}: configured kinematic limit exceeded")
    for name, metrics in fidelity.items():
        if metrics["adaptation_distortion_fraction"] > limits.max_excursion_reduction + 1e-6:
            failures.append(f"{name}: adaptation exceeds 20% source task-space excursion bound")
        tolerance = (
            limits.foot_p95_m
            if name in ("left_foot", "right_foot")
            else limits.hand_head_p95_m
            if name in ("left_hand", "right_hand", "head_proxy")
            else None
        )
        if tolerance is not None and metrics["achieved_to_adapted_m"]["p95"] > tolerance:
            failures.append(f"{name}: adapted-reference p95 position tolerance exceeded")
    return summary, fidelity, failures


def _refined_result_from_qpos(source_model, target_model, candidate, previous, qpos, fit_report):
    """Recompute every after-error using FK; retain the original direct baseline.

    A change of root pose must not reset the protected-task baseline. Solver
    diagnostics are recomputed or explicitly replaced, never copied as proof.
    """
    layout, data = ik._model_layout(target_model), mujoco.MjData(target_model)
    source_layout, source_data = ik._model_layout(source_model), mujoco.MjData(source_model)
    config = replace(previous.config, max_velocity_rad_s=min(previous.config.max_velocity_rad_s, 5.0))
    diagnostics = {key: value.copy() for key, value in previous.diagnostics.items()}
    positions, quaternions, targets_by_frame = [], [], []
    for frame, pose in enumerate(qpos):
        ik._set_configuration(
            source_model,
            source_data,
            source_layout,
            candidate["root_pos_w"][frame],
            candidate["root_quat_wxyz"][frame],
            candidate["joint_pos"][frame],
        )
        targets = ik._task_targets(source_model, source_data, ik.DEFAULT_TASKS)
        targets_by_frame.append(targets)
        ik._set_configuration(target_model, data, layout, pose[:3], pose[3:7], pose[7:])
        jacobian, residual, position, orientation, priority, _ = ik._task_linearization(
            target_model,
            data,
            layout,
            ik.DEFAULT_TASKS,
            targets,
            tuple(bool(value) for value in candidate["contact_flags"][frame]),
            config.contact_weight_multiplier,
            np.arange(23),
        )
        diagnostics["weighted_task_error_after"][frame] = ik._weighted_task_error(jacobian, residual)
        diagnostics["position_error_after_mean"][frame] = np.mean(position)
        diagnostics["orientation_error_after_mean"][frame] = np.mean(orientation)
        for tier, value in enumerate(priority):
            diagnostics[f"priority_{tier}_error_after"][frame] = value
        for index, task in enumerate(ik.DEFAULT_TASKS):
            diagnostics[f"task_{task.name}_position_error_after_m"][frame] = position[index]
            diagnostics[f"task_{task.name}_orientation_error_after_rad"][frame] = orientation[index]
        pos, quat = ik._task_pose_arrays(target_model, data, ik.DEFAULT_TASKS, source=False)
        positions.append(pos)
        quaternions.append(quat)
    offset = qpos[:, :3] - candidate["root_pos_w"]
    for path, velocity_name, acceleration_name in (
        (qpos[:, 7:], "trajectory_velocity_abs_max", "trajectory_acceleration_abs_max"),
        (offset, "root_offset_velocity_abs_max", "root_offset_acceleration_abs_max"),
    ):
        velocity, acceleration = ik.trajectory_derivatives(path, 1 / previous.fps)
        diagnostics[velocity_name] = np.abs(velocity).max(axis=1)
        diagnostics[acceleration_name] = np.abs(acceleration).max(axis=1)
    lower, upper = ik.safe_target_joint_bounds(
        target_model, native_action_clip=config.native_action_clip, safe_limit_guard_rad=0.05
    )
    if np.any(qpos[:, 7:] < lower - 1e-7) or np.any(qpos[:, 7:] > upper + 1e-7):
        raise ValueError("serialized root refinement exceeds guarded native23 position bounds")
    iterations = len(fit_report["iterations"])
    accepted_steps = sum(row["accepted"] for row in fit_report["iterations"])
    for key, value in (
        ("solver_iterations", iterations),
        ("solver_accepted_step_count", accepted_steps),
        ("solver_used_feasible_seed_fallback", accepted_steps == 0),
        ("lower_root_solver_iterations", iterations),
        ("lower_root_accepted_step_count", accepted_steps),
        ("lower_root_projection_iteration_count", fit_report["seed_projection"]["iterations"]),
        ("constraint_relaxation_count", 0),
    ):
        diagnostics[key] = np.full(len(qpos), value, dtype=float)
    diagnostics["position_limit_hit_count"] = np.sum(
        (qpos[:, 7:] <= lower + 1e-7) | (qpos[:, 7:] >= upper - 1e-7), axis=1
    ).astype(float)
    groups = ik._joint_groups(layout)
    before = ik._lower_root_metrics(
        target_model,
        data,
        layout,
        ik.DEFAULT_TASKS,
        targets_by_frame,
        candidate["root_pos_w"],
        candidate["root_quat_wxyz"],
        previous.direct_joint_pos_hardware,
        previous.contact_flags,
        groups.lower_body,
        config,
    )
    after = ik._lower_root_metrics(
        target_model,
        data,
        layout,
        ik.DEFAULT_TASKS,
        targets_by_frame,
        qpos[:, :3],
        qpos[:, 3:7],
        qpos[:, 7:],
        previous.contact_flags,
        groups.lower_body,
        config,
    )
    for name, metrics in (("before", before), ("after", after)):
        invalid, excess = ik._lower_root_validity_excess(metrics, before, config)
        diagnostics[f"lower_root_invalid_{name}"] = invalid.astype(float)
        diagnostics[f"lower_root_validity_excess_{name}"] = excess
    return replace(
        previous,
        root_pos_w=qpos[:, :3].copy(),
        root_quat_wxyz=qpos[:, 3:7].copy(),
        root_offset_w=offset,
        joint_pos_hardware=qpos[:, 7:].copy(),
        action_target_native=ik._hardware_targets_to_raw_native(qpos[:, 7:]),
        achieved_task_pos_w=np.asarray(positions),
        achieved_task_quat_wxyz=np.asarray(quaternions),
        diagnostics=diagnostics,
        config=config,
    )


def _refine_root_reference(source_model, target_model, candidate, previous, receipt):
    """One bounded offline SE(3)+23 fit, not additional actuators or a live solver.

    Fixed-root native23 head-position joint Jacobian is zero. Root attitude is
    therefore a necessary reference variable for missing waist pitch/roll.
    Reuse the existing constrained whole-path solver and its frozen bounds.
    The seed is only numerically feasible; all original source goals, contacts,
    frame endpoints, and protected-task baselines survive unchanged.
    """
    from gear_sonic.utils.g1_23dof_trajectory_projection import project_nearest_trajectory
    from gear_sonic.utils.g1_true23_original_task_trajectory import (
        OriginalTaskConfig,
        OriginalTaskPath,
        fit_original_task_path,
    )

    if previous.fps != 50 or len(candidate["joint_pos"]) < 3:
        raise ValueError("root-reference refinement requires at least three complete 50-Hz frames")
    cfg = OriginalTaskConfig()
    receipt["config"] = asdict(cfg)
    low, high = ik.safe_target_joint_bounds(target_model, native_action_clip=9.5, safe_limit_guard_rad=0.05)
    desired = previous.joint_pos_hardware
    projection = project_nearest_trajectory(
        desired,
        lower_bounds=low,
        upper_bounds=high,
        dt=0.02,
        max_velocity=min(cfg.joint_velocity_rad_s, previous.config.max_velocity_rad_s)
        * cfg.serialization_margin_fraction,
        max_acceleration=min(cfg.joint_acceleration_rad_s2, previous.config.max_acceleration_rad_s2)
        * cfg.serialization_margin_fraction,
        initial_velocity=np.clip(
            (desired[1] - desired[0]) / 0.02, -cfg.joint_velocity_rad_s, cfg.joint_velocity_rad_s
        ),
    )
    source_qpos = np.column_stack((candidate["root_pos_w"], candidate["root_quat_wxyz"], candidate["joint_pos"]))
    problem = OriginalTaskPath(source_model, target_model, source_qpos, projection.projected_path, config=cfg)
    variables, report = fit_original_task_path(problem)
    report["seed_projection"] = {
        "iterations": projection.iterations,
        "audit": asdict(projection.audit),
        "joint_adjustment_max_rad": float(np.abs(projection.projected_path - desired).max()),
    }
    motion = problem.serialize(variables)
    serialized = problem.serialized_variables(motion)
    report["serialized_path_constraints"] = problem.audit(serialized)
    report["source_goals_replaced"] = False
    report["contacts_reinferred"] = False
    report["root_reference_variables_are_actuators"] = False
    receipt["fit_report"] = report
    if not report["path_constraints"]["passed"] or not report["serialized_path_constraints"]["passed"]:
        raise ValueError("root-reference refinement failed independent full-path or serialization audit")
    # Audit the actual serialized root/joints, not an unrounded optimizer path.
    qpos = np.column_stack((motion["body_pos_w"][:, 0], motion["body_quat_w"][:, 0], motion["joint_pos"]))
    return _refined_result_from_qpos(source_model, target_model, candidate, previous, qpos, report)


def _restore_retained_diagnostic(source_model, target_model, candidate, stored, config):
    """Revalidate saved source/baseline with actual FK before using a reject as a seed."""
    requested = np.column_stack((candidate["root_pos_w"], candidate["root_quat_wxyz"], candidate["joint_pos"]))
    if (
        not np.array_equal(stored["diagnostic_only_not_accepted_motion"], [True])
        or not np.array_equal(stored["diagnostic_requested_qpos29"], requested)
        or not np.array_equal(stored["diagnostic_contact_flags"], candidate["contact_flags"])
    ):
        raise ValueError("retained diagnostic source goals or contacts differ from this full candidate")
    count = len(requested)
    pose = np.asarray(stored["diagnostic_qpos_native23"], dtype=float)
    prior = np.asarray(stored["diagnostic_fixed_root_qpos_native23"], dtype=float)
    direct = np.asarray(stored["diagnostic_direct_joints_native23"], dtype=float)
    if pose.shape != (count, 30) or prior.shape != pose.shape or direct.shape != (count, 23):
        raise ValueError("retained full-path diagnostic shapes differ")
    if not all(np.isfinite(x).all() for x in (pose, prior, direct)):
        raise ValueError("retained diagnostic contains nonfinite state")
    source_layout = ik._model_layout(source_model)
    layout = ik._safe_target_layout(
        ik._model_layout(target_model), config.safe_limit_guard_rad, config.native_action_clip
    )
    mapped = candidate["joint_pos"][:, [source_layout.joint_names.index(name) for name in layout.joint_names]]
    if not np.array_equal(direct, np.clip(mapped, layout.lower, layout.upper)):
        raise ValueError("retained direct baseline is not the original source-mapped guarded native23 baseline")
    reserved = {
        "diagnostic_only_not_accepted_motion",
        "diagnostic_qpos_native23",
        "diagnostic_fixed_root_qpos_native23",
        "diagnostic_direct_joints_native23",
        "diagnostic_requested_qpos29",
        "diagnostic_contact_flags",
    }
    diagnostics = {
        key.removeprefix("diagnostic_"): np.asarray(value, dtype=float).copy()
        for key, value in stored.items()
        if key.startswith("diagnostic_") and key not in reserved
    }
    if any(value.shape != (count,) or not np.isfinite(value).all() for value in diagnostics.values()):
        raise ValueError("retained per-frame diagnostic schema differs")
    source_data, data = mujoco.MjData(source_model), mujoco.MjData(target_model)
    for frame in range(count):
        ik._set_configuration(
            source_model,
            source_data,
            source_layout,
            requested[frame, :3],
            requested[frame, 3:7],
            requested[frame, 7:],
        )
        targets = ik._task_targets(source_model, source_data, ik.DEFAULT_TASKS)
        ik._set_configuration(
            target_model, data, layout, requested[frame, :3], requested[frame, 3:7], direct[frame]
        )
        jacobian, residual, position, orientation, _, _ = ik._task_linearization(
            target_model,
            data,
            layout,
            ik.DEFAULT_TASKS,
            targets,
            tuple(candidate["contact_flags"][frame]),
            config.contact_weight_multiplier,
            np.arange(23),
        )
        comparisons = [("weighted_task_error_before", ik._weighted_task_error(jacobian, residual))]
        for index, task in enumerate(ik.DEFAULT_TASKS):
            comparisons.extend(
                (
                    (f"task_{task.name}_position_error_before_m", position[index]),
                    (f"task_{task.name}_orientation_error_before_rad", orientation[index]),
                )
            )
        if any(
            not np.isclose(diagnostics[key][frame], value, atol=1e-12, rtol=1e-10) for key, value in comparisons
        ):
            raise ValueError("retained protected baseline fails independent original-model FK audit")
    desired_positions, desired_quats = _source_task_poses(source_model, candidate)
    result = ik.RetargetResult(
        source_root_pos_w=candidate["root_pos_w"].copy(),
        root_pos_w=pose[:, :3],
        root_quat_wxyz=pose[:, 3:7],
        root_offset_w=pose[:, :3] - candidate["root_pos_w"],
        direct_joint_pos_hardware=direct,
        joint_pos_hardware=pose[:, 7:],
        direct_action_native=ik._hardware_targets_to_raw_native(direct),
        action_target_native=ik._hardware_targets_to_raw_native(pose[:, 7:]),
        contact_flags=candidate["contact_flags"].copy(),
        desired_task_pos_w=desired_positions,
        desired_task_quat_wxyz=desired_quats,
        achieved_task_pos_w=desired_positions.copy(),
        achieved_task_quat_wxyz=desired_quats.copy(),
        task_has_orientation=np.array([t.orientation_weight > 0 for t in ik.DEFAULT_TASKS]),
        task_names=tuple(t.name for t in ik.DEFAULT_TASKS),
        diagnostics=diagnostics,
        fps=50.0,
        config=config,
    )
    # Refresh the retained after-errors too, not only its immutable before-baseline.
    refreshed = _refined_result_from_qpos(
        source_model,
        target_model,
        candidate,
        result,
        pose,
        {"iterations": [], "seed_projection": {"iterations": 0}},
    )
    for key in (
        "weighted_task_error_after",
        "task_whole_robot_com_position_error_after_m",
        "task_left_foot_orientation_error_after_rad",
        "task_right_foot_orientation_error_after_rad",
    ):
        if not np.allclose(refreshed.diagnostics[key], diagnostics[key], atol=1e-12, rtol=1e-10):
            raise ValueError("retained rejected pose fails independent after-error FK audit")
    return refreshed


def retained_candidate_limits(forensic_report):
    """Recover one bounded candidate without inheriting weakened task gates."""
    if forensic_report.get("accepted") is not False or len(forensic_report.get("attempts", [])) != 1:
        raise ValueError("hard refinement requires one explicitly rejected full candidate")
    if forensic_report.get("source_role") != "requested_choreography":
        raise ValueError("retained source role must be requested choreography")
    declared = dict(forensic_report["limits"])
    for key in ("duration_scales", "excursion_scales"):
        declared[key] = tuple(declared[key])
    limits = AdaptationLimits(**declared)
    defaults = asdict(AdaptationLimits())
    variable_fields = {"duration_scales", "excursion_scales", "maximum_output_frames"}
    if any(value != defaults[key] for key, value in asdict(limits).items() if key not in variable_fields):
        raise ValueError("retained candidate cannot change protected task or trajectory limits")
    if len(limits.duration_scales) != 1 or len(limits.excursion_scales) != 1:
        raise ValueError("hard refinement requires exactly one retained scale candidate")
    parent = forensic_report["attempts"][0]
    actual = parent.get("actual_duration_scale")
    if (
        parent.get("requested_duration_scale") != limits.duration_scales[0]
        or parent.get("requested_excursion_scale") != limits.excursion_scales[0]
        or isinstance(actual, bool)
        or not isinstance(actual, (int, float))
        or not math.isfinite(actual)
        or not 1 <= actual <= limits.max_duration_scale
    ):
        raise ValueError("retained candidate adaptation bounds differ from declared limits")
    return limits


def refine_retained_protected_motion(
    *, source_model, target_model, arrays, stored, forensic_report, feasibility_restoration=False
):
    """One hard-protected refinement of a fully bound rejected scale candidate."""
    from gear_sonic.utils.g1_23dof_trajectory_projection import project_nearest_trajectory
    from gear_sonic.utils.g1_true23_original_task_trajectory import OriginalTaskConfig, OriginalTaskPath
    from gear_sonic.utils.g1_true23_generalist_protected_root import fit_protected_task_path

    if type(feasibility_restoration) is not bool:
        raise ValueError("feasibility restoration must be an explicit boolean")
    limits = retained_candidate_limits(forensic_report)
    parent = forensic_report["attempts"][0]
    source = validate_named_motion(arrays, source_model, allow_source_limit_excess=True)
    if "contact_flags" not in source:
        feet = ik._source_foot_positions(
            source_model,
            ik._model_layout(source_model),
            source["root_pos_w"],
            source["root_quat_wxyz"],
            source["joint_pos"],
        )
        source["contact_flags"] = ik.infer_foot_contacts(
            feet, fps=float(source["fps"][0]), height_tolerance_m=0.035, speed_tolerance_m_s=0.45
        )
    original, source_times, actual_scale = _resample(source, limits.duration_scales[0], limits)
    if not math.isclose(actual_scale, parent["actual_duration_scale"], rel_tol=0, abs_tol=1e-12):
        raise ValueError("retained duration differs from complete source resampling")
    candidate = _reduce_excursion(original, limits.excursion_scales[0])
    if not np.array_equal(source_times, stored["source_time_map_s"]):
        raise ValueError("retained source-time map differs or omits original endpoints")
    config = ik.RetargetConfig(**forensic_report["ik_config"])
    if config != ik.RetargetConfig(optimize_lower_body=True, allow_acceleration_constraint_relaxation=False):
        raise ValueError("retained diagnostic changes the unchanged protected/trajectory configuration")
    baseline = _restore_retained_diagnostic(source_model, target_model, candidate, stored, config)
    original_positions, original_quats = _source_task_poses(source_model, original)
    summary, fidelity, failures = _candidate_assessment(baseline, original_positions, limits)
    if any("adaptation exceeds 20%" in failure for failure in failures):
        raise ValueError("retained source adaptation already exceeds original task-space distortion gate")
    cfg = OriginalTaskConfig()
    low, high = ik.safe_target_joint_bounds(target_model, native_action_clip=9.5, safe_limit_guard_rad=0.05)
    prior = stored["diagnostic_fixed_root_qpos_native23"][:, 7:]
    projection = project_nearest_trajectory(
        prior,
        lower_bounds=low,
        upper_bounds=high,
        dt=0.02,
        max_velocity=cfg.joint_velocity_rad_s * cfg.serialization_margin_fraction,
        max_acceleration=cfg.joint_acceleration_rad_s2 * cfg.serialization_margin_fraction,
        initial_velocity=np.clip(
            (prior[1] - prior[0]) / 0.02, -cfg.joint_velocity_rad_s, cfg.joint_velocity_rad_s
        ),
    )
    requested = stored["diagnostic_requested_qpos29"]
    problem = OriginalTaskPath(source_model, target_model, requested, projection.projected_path, config=cfg)
    pose = stored["diagnostic_qpos_native23"]
    initial = np.column_stack(
        (
            pose[:, :3] - requested[:, :3],
            (Rotation.from_quat(pose[:, [4, 5, 6, 3]]) * problem.source_rotation.inv()).as_rotvec(),
            pose[:, 7:],
        )
    )
    if feasibility_restoration:
        from gear_sonic.utils.g1_true23_generalist_feasibility_restore import fit_with_feasibility_restoration

        variables, fit_report = fit_with_feasibility_restoration(problem, initial, baseline)
    else:
        variables, fit_report = fit_protected_task_path(problem, initial, baseline)
    fit_report["seed_projection"] = {"iterations": projection.iterations, "audit": asdict(projection.audit)}
    motion = problem.serialize(variables)
    fit_report["serialized_path_constraints"] = problem.audit(problem.serialized_variables(motion))
    receipt = {
        "strategy": "hard_soc_protected_root_se3_and_all23"
        + ("_with_intermediate_restoration" if feasibility_restoration else ""),
        "requested_frames": len(initial),
        "completed": False,
        "fit_report": fit_report,
    }
    if not fit_report["serialized_path_constraints"]["passed"]:
        failures = [*failures, "hard protected refinement failed independent serialization audit"]
        result = baseline
    else:
        qpos = np.column_stack((motion["body_pos_w"][:, 0], motion["body_quat_w"][:, 0], motion["joint_pos"]))
        result = _refined_result_from_qpos(source_model, target_model, candidate, baseline, qpos, fit_report)
        summary, fidelity, failures = _candidate_assessment(result, original_positions, limits)
        receipt.update(completed=True, completed_frames=len(qpos))
    report = deepcopy(forensic_report)
    if "hard_refinement_error" in report:
        report["retained_parent_hard_refinement_error"] = report.pop("hard_refinement_error")
    attempt = report["attempts"][0]
    attempt["before_hard_protected_refinement"] = {
        "ik_summary": parent["ik_summary"],
        "failures": parent["failures"],
    }
    attempt["solver_attempts"].append(receipt)
    attempt.update(
        accepted=not failures,
        failures=failures,
        fidelity=fidelity,
        ik_summary=summary,
        solver_strategy=receipt["strategy"],
    )
    report.update(
        accepted=not failures,
        selected_attempt=0 if not failures else None,
        hard_protected_refinement_requested=True,
        intermediate_feasibility_restoration_requested=feasibility_restoration,
        hardware_authorized=False,
        deployment_ready=False,
    )
    diagnostics = {key: value.copy() for key, value in stored.items()}
    diagnostics["diagnostic_qpos_native23"] = np.column_stack(
        (result.root_pos_w, result.root_quat_wxyz, result.joint_pos_hardware)
    )
    diagnostics.update({f"diagnostic_{key}": value for key, value in result.diagnostics.items()})
    accepted_arrays = None
    if not failures:
        accepted_arrays = {
            **ik.build_mjlab_motion_arrays(target_model, result),
            "joint_names": np.asarray(ik._model_layout(target_model).joint_names),
            "source_joint_names": source["joint_names"],
            "timestamps_s": np.arange(len(source_times)) / 50,
            "source_time_map_s": source_times,
            "source_joint_pos_resampled": original["joint_pos"],
            "source_root_pos_w_resampled": original["root_pos_w"],
            "source_root_quat_wxyz_resampled": original["root_quat_wxyz"],
            "source_task_pos_w": original_positions,
            "source_task_quat_wxyz": original_quats,
            "adapted_task_pos_w": result.desired_task_pos_w,
            "achieved_task_pos_w": result.achieved_task_pos_w,
            "contact_flags": result.contact_flags,
            "task_names": np.asarray(result.task_names),
        }
    return AdaptationResult(not failures, accepted_arrays, report, diagnostics)


def adapt_offline_motion(
    *,
    source_model: mujoco.MjModel,
    target_model: mujoco.MjModel,
    arrays: dict[str, np.ndarray],
    limits: AdaptationLimits = AdaptationLimits(),
    input_mode: str = "offline",
    source_role: str = "robot_state",
    root_reference_refinement: bool = False,
) -> AdaptationResult:
    if input_mode != "offline":
        raise ValueError("offline full-clip retargeting cannot be used as a causal live adapter")
    if type(root_reference_refinement) is not bool:
        raise ValueError("root_reference_refinement must be an explicit boolean")
    if root_reference_refinement and (len(limits.duration_scales) != 1 or len(limits.excursion_scales) != 1):
        raise ValueError("root-reference refinement requires exactly one bounded candidate, not a solver sweep")
    if (source_model.nq, target_model.nq, target_model.nu) != (36, 30, 23):
        raise ValueError("adaptation requires true 29-source and physical 23-actuator target models")
    if source_role not in {"robot_state", "requested_choreography"}:
        raise ValueError("source_role must distinguish measured robot state from requested choreography")
    source = validate_named_motion(
        arrays, source_model, allow_source_limit_excess=source_role == "requested_choreography"
    )
    source_layout = ik._model_layout(source_model)
    source_excess = np.maximum(
        np.maximum(source_layout.lower - source["joint_pos"], source["joint_pos"] - source_layout.upper), 0
    )
    contacts_supplied = "contact_flags" in source
    if not contacts_supplied:
        feet = ik._source_foot_positions(
            source_model,
            ik._model_layout(source_model),
            source["root_pos_w"],
            source["root_quat_wxyz"],
            source["joint_pos"],
        )
        # Infer once on the unmodified full source. Slowing/reducing candidates
        # must not silently change the requested contact schedule to pass IK.
        source["contact_flags"] = ik.infer_foot_contacts(
            feet, fps=float(source["fps"][0]), height_tolerance_m=0.035, speed_tolerance_m_s=0.45
        )
    attempts = []
    selected_arrays = None
    diagnostic_arrays = None
    selected = None
    config = ik.RetargetConfig(
        max_iterations=limits.ik_iterations,
        max_velocity_rad_s=limits.max_joint_velocity_rad_s,
        max_acceleration_rad_s2=limits.max_joint_acceleration_rad_s2,
        optimize_lower_body=True,
        allow_acceleration_constraint_relaxation=False,
    )
    for excursion_scale in sorted(set(limits.excursion_scales), reverse=True):
        for duration_scale in sorted(set(limits.duration_scales)):
            attempt: dict[str, Any] = {
                "requested_duration_scale": duration_scale,
                "requested_excursion_scale": excursion_scale,
                "accepted": False,
                "failures": [],
                "solver_attempts": [],
            }
            attempts.append(attempt)
            try:
                original, source_times, actual_scale = _resample(source, duration_scale, limits)
                candidate = _reduce_excursion(original, excursion_scale)
                original_positions, original_quaternions = _source_task_poses(source_model, original)
                result = _retarget_candidate(
                    source_model=source_model,
                    target_model=target_model,
                    candidate=candidate,
                    fps=limits.target_fps,
                    config=config,
                    solver_attempts=attempt["solver_attempts"],
                )
                summary, fidelity, failures = _candidate_assessment(result, original_positions, limits)
                fixed_root_qpos = np.column_stack(
                    (result.root_pos_w, result.root_quat_wxyz, result.joint_pos_hardware)
                )
                if root_reference_refinement and failures:
                    # Keep completed rejected evidence even if refinement fails
                    # before yielding a pose. Do not refine a forbidden target.
                    attempt["before_root_refinement"] = {
                        "ik_summary": summary,
                        "fidelity": fidelity,
                        "failures": failures.copy(),
                    }
                    if any("adaptation exceeds 20%" in failure for failure in failures):
                        attempt["root_refinement_skipped"] = "source task-space adaptation already exceeds bound"
                    else:
                        receipt = {
                            "strategy": "bounded_whole_path_root_se3_and_all23",
                            "requested_frames": len(candidate["joint_pos"]),
                            "completed": False,
                        }
                        attempt["solver_attempts"].append(receipt)
                        try:
                            refined = _refine_root_reference(
                                source_model, target_model, candidate, result, receipt
                            )
                            summary, fidelity, failures = _candidate_assessment(
                                refined, original_positions, limits
                            )
                            result = refined
                            receipt.update(completed=True, completed_frames=len(result.joint_pos_hardware))
                        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                            receipt["error"] = f"{type(exc).__name__}: {exc}"
                            failures = [*failures, f"root-reference refinement rejected: {receipt['error']}"]
                attempt.update(
                    actual_duration_scale=actual_scale,
                    output_frames=len(source_times),
                    fidelity=fidelity,
                    ik_summary=summary,
                    failures=failures,
                    accepted=not failures,
                    no_mean_improvement_does_not_disqualify_already_feasible_motion=True,
                    solver_strategy=attempt["solver_attempts"][-1]["strategy"],
                )
                if root_reference_refinement:
                    # Deliberately not an MJLab motion schema: rejected states
                    # remain inspectable without resembling accepted experts.
                    diagnostic_arrays = {
                        "diagnostic_only_not_accepted_motion": np.array([True]),
                        "diagnostic_qpos_native23": np.column_stack(
                            (result.root_pos_w, result.root_quat_wxyz, result.joint_pos_hardware)
                        ),
                        "diagnostic_fixed_root_qpos_native23": fixed_root_qpos,
                        "diagnostic_direct_joints_native23": result.direct_joint_pos_hardware,
                        "diagnostic_requested_qpos29": np.column_stack(
                            (candidate["root_pos_w"], candidate["root_quat_wxyz"], candidate["joint_pos"])
                        ),
                        "source_time_map_s": source_times,
                        "diagnostic_contact_flags": result.contact_flags,
                        **{f"diagnostic_{key}": value for key, value in result.diagnostics.items()},
                    }
                if not failures:
                    selected = len(attempts) - 1
                    selected_arrays = {
                        **ik.build_mjlab_motion_arrays(target_model, result),
                        "joint_names": np.asarray(ik._model_layout(target_model).joint_names),
                        "source_joint_names": source["joint_names"],
                        "timestamps_s": np.arange(len(source_times)) / limits.target_fps,
                        "source_time_map_s": source_times,
                        "source_joint_pos_resampled": original["joint_pos"],
                        "source_root_pos_w_resampled": original["root_pos_w"],
                        "source_root_quat_wxyz_resampled": original["root_quat_wxyz"],
                        "source_task_pos_w": original_positions,
                        "source_task_quat_wxyz": original_quaternions,
                        "adapted_task_pos_w": result.desired_task_pos_w,
                        "achieved_task_pos_w": result.achieved_task_pos_w,
                        "contact_flags": result.contact_flags,
                        "task_names": np.asarray(result.task_names),
                    }
                    break
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                attempt["failures"].append(f"{type(exc).__name__}: {exc}")
        if selected is not None:
            break
    report = {
        "schema_version": 1,
        "kind": "g1_true23_generalist_bounded_offline_retarget",
        "accepted": selected is not None,
        "selected_attempt": selected,
        "attempts": attempts,
        "limits": asdict(limits),
        "ik_config": asdict(config),
        "root_reference_refinement_requested": root_reference_refinement,
        "source_frame_count": len(source["joint_pos"]),
        "source_role": source_role,
        "source_joint_targets_clipped": False,
        "source_joint_limit_excess_max_rad": dict(
            zip(source_layout.joint_names, source_excess.max(axis=0).tolist())
        ),
        "source_joint_limit_excess_frame_count": int(np.any(source_excess > 1e-6, axis=1).sum()),
        "source_fps": float(source["fps"][0]),
        "source_duration_s": float(source["timestamps_s"][-1] - source["timestamps_s"][0]),
        "task_priority_order": ["feet_contact", "whole_robot_com", "torso_head", "hands", "elbows"],
        "task_frame_convention": "existing_sonic_18cm_source_yaw_target_roll_proxy",
        "contact_flags_supplied": contacts_supplied,
        "contact_schedule_source": "supplied"
        if contacts_supplied
        else "original29_height_velocity_heuristic_before_adaptation",
        "contact_schedule_reinferred_after_adaptation": False,
        "contact_flags_are_physical_contact_evidence": False,
        "full_clip_future_access": True,
        "causal_live_adapter": False,
        "dynamic_feasibility_verified": False,
        "controller_qualified": False,
        "simulator_qualification_complete": False,
        "hardware_authorized": False,
    }
    return AdaptationResult(selected is not None, selected_arrays, report, diagnostic_arrays)
