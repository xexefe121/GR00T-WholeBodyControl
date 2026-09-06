"""Single-frame, stateful native23 task-space reference adaptation.

No lookahead, timeline resampling, contact inference, transport or motor writes.
The caller supplies contact estimates and a same-clock arrival timestamp. A
rejection emits no replacement pose and latches explicit reinitialization.
Root pose follows the current input without optimization; dynamic root/support
feasibility and downstream controller readiness are not established here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from collections import deque
import math
import time
from typing import Any, Sequence

import mujoco
import numpy as np

from gear_sonic.utils import g1_23dof_task_space_retarget as ik
from gear_sonic.utils.g1_23dof_contract import HARDWARE_23_JOINT_NAMES


@dataclass(frozen=True)
class CausalRetargetConfig:
    cadence_s: float = 0.02
    cadence_tolerance_s: float = 1e-6
    maximum_source_age_s: float = 0.04
    maximum_future_clock_error_s: float = 0.002
    maximum_joint_velocity_rad_s: float = 8.0
    maximum_joint_acceleration_rad_s2: float = 80.0
    maximum_foot_error_m: float = 0.05
    maximum_hand_head_error_m: float = 0.10
    maximum_ik_iterations: int = 16

    def __post_init__(self) -> None:
        if self.cadence_s != 0.02:
            raise ValueError("causal core requires the 50-Hz reference contract")
        for name, value in asdict(self).items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (float, int))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")
        if self.cadence_tolerance_s > 0.001:
            raise ValueError("cadence tolerance cannot hide missed reference frames")
        if self.maximum_joint_velocity_rad_s > 8 or self.maximum_joint_acceleration_rad_s2 > 80:
            raise ValueError("causal core cannot relax 8-rad/s and 80-rad/s2 limits")
        if self.maximum_foot_error_m > 0.05 or self.maximum_hand_head_error_m > 0.10:
            raise ValueError("causal core cannot relax task-position tolerances")
        if not isinstance(self.maximum_ik_iterations, int) or self.maximum_ik_iterations > 128:
            raise ValueError("maximum_ik_iterations must be an integer in [1, 128]")


@dataclass(frozen=True)
class CausalReference:
    timestamp_s: float
    joint_names: tuple[str, ...]
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    root_pos_w: np.ndarray
    root_quat_wxyz: np.ndarray
    contact_flags: np.ndarray
    task_names: tuple[str, ...]
    source_task_pos_w: np.ndarray
    achieved_task_pos_w: np.ndarray
    source_task_quat_wxyz: np.ndarray
    achieved_task_quat_wxyz: np.ndarray


@dataclass(frozen=True)
class CausalStepResult:
    accepted: bool
    status: str
    reference: CausalReference | None
    diagnostics: dict[str, Any]


def _timestamp(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite same-clock timestamp")
    return float(value)


def _named_joints(names: Sequence[str], value: np.ndarray, layout: ik._ModelLayout, label: str) -> np.ndarray:
    if isinstance(names, str):
        raise ValueError(f"{label} requires explicit ordered joint names")
    names = tuple(names)
    if any(not isinstance(name, str) or not name or name != name.strip() for name in names):
        raise ValueError(f"{label} joint names must be nonempty strings")
    if (
        len(names) != len(layout.joint_names)
        or len(set(names)) != len(names)
        or set(names) != set(layout.joint_names)
    ):
        raise ValueError(f"{label} must name exactly the model's physical joints")
    values = np.asarray(value, dtype=np.float64)
    if values.shape != (len(names),) or not np.isfinite(values).all():
        raise ValueError(f"{label} must be a finite single joint vector")
    return values[[names.index(name) for name in layout.joint_names]].copy()


class CausalNative23Retargeter:
    """History is accepted joint pose/velocity only; no future-frame interface.

    ``initialize`` is an explicit reference-state contract, not robot arming.
    The first input is exactly one 20-ms tick after that initialization. After
    any rejected input, the consumer must explicitly establish a new reference
    state; this object never inserts a hold, zero action or a recovery command.
    """

    def __init__(
        self,
        source_model: mujoco.MjModel,
        target_model: mujoco.MjModel,
        *,
        config: CausalRetargetConfig = CausalRetargetConfig(),
    ):
        if (source_model.nq, target_model.nq, target_model.nu) != (36, 30, 23):
            raise ValueError("causal retargeting requires original29 and physical23 models")
        self.source_model, self.target_model, self.config = source_model, target_model, config
        self.source_layout = ik._model_layout(source_model)
        raw_target = ik._model_layout(target_model)
        if raw_target.joint_names != tuple(HARDWARE_23_JOINT_NAMES):
            raise ValueError("target model must use the exact native23 joint contract")
        self.ik_config = ik.RetargetConfig(
            max_iterations=config.maximum_ik_iterations,
            max_velocity_rad_s=config.maximum_joint_velocity_rad_s,
            max_acceleration_rad_s2=config.maximum_joint_acceleration_rad_s2,
            optimize_lower_body=True,
            enable_lower_root_feasibility=False,
            allow_acceleration_constraint_relaxation=False,
        )
        self.target_layout = ik._safe_target_layout(
            raw_target, self.ik_config.safe_limit_guard_rad, self.ik_config.native_action_clip
        )
        self.tasks = ik.validate_tasks(source_model, target_model, ik.DEFAULT_TASKS)
        self.source_data, self.target_data = mujoco.MjData(source_model), mujoco.MjData(target_model)
        self._source_to_target = np.asarray(
            [self.source_layout.joint_names.index(name) for name in self.target_layout.joint_names]
        )
        self._active_indices = np.arange(23, dtype=np.int64)
        self._previous: np.ndarray | None = None
        self._previous_velocity: np.ndarray | None = None
        self._last_timestamp_s: float | None = None
        self._requires_reinitialize = True
        self._elapsed_s: deque[float] = deque(maxlen=4096)
        self._deadline_misses = 0
        self._maximum_elapsed_s = 0.0
        self._accepted_count = 0
        self._rejected_count = 0

    def initialize(
        self, *, joint_names: Sequence[str], joint_pos: np.ndarray, joint_vel: np.ndarray, timestamp_s: float
    ) -> None:
        self._requires_reinitialize = True
        timestamp = _timestamp(timestamp_s, "initial timestamp_s")
        position = _named_joints(joint_names, joint_pos, self.target_layout, "initial23 pose")
        velocity = _named_joints(joint_names, joint_vel, self.target_layout, "initial23 velocity")
        if np.any(position < self.target_layout.lower) or np.any(position > self.target_layout.upper):
            raise ValueError("initial23 pose exceeds bounded physical joint range")
        if np.any(np.abs(velocity) > self.config.maximum_joint_velocity_rad_s):
            raise ValueError("initial23 velocity exceeds configured limit")
        ik._trajectory_bounds(self.target_layout, position, velocity, self.config.cadence_s, self.ik_config)
        self._previous, self._previous_velocity = position.copy(), velocity.copy()
        self._last_timestamp_s = timestamp
        self._requires_reinitialize = False

    def state_snapshot(self) -> dict[str, Any]:
        return {
            "requires_reinitialize": self._requires_reinitialize,
            "last_accepted_timestamp_s": self._last_timestamp_s,
            "joint_pos": None if self._previous is None else self._previous.copy(),
            "joint_vel": None if self._previous_velocity is None else self._previous_velocity.copy(),
            "accepted_count": self._accepted_count,
            "rejected_count": self._rejected_count,
        }

    def _finish(
        self,
        *,
        started_s: float,
        status: str,
        reference: CausalReference | None = None,
        diagnostics: dict[str, Any] | None = None,
    ) -> CausalStepResult:
        elapsed = time.perf_counter() - started_s
        self._elapsed_s.append(elapsed)
        self._deadline_misses += elapsed > self.config.cadence_s
        self._maximum_elapsed_s = max(self._maximum_elapsed_s, elapsed)
        accepted = reference is not None
        if accepted:
            self._accepted_count += 1
        else:
            self._rejected_count += 1
            self._requires_reinitialize = True
        detail = {
            "elapsed_s": elapsed,
            "nominal_cadence_s": self.config.cadence_s,
            "requires_reinitialize": self._requires_reinitialize,
            "full_path_solver_used": False,
            "future_samples_consumed": 0,
            "root_pose_optimized": False,
            "root_motion_dynamic_feasibility_verified": False,
            "contact_estimates_supplied_by_caller": True,
            "task_frame_convention": "existing_sonic_18cm_source_yaw_target_roll_proxy",
            "controller_qualified": False,
            "hardware_authorized": False,
            **(diagnostics or {}),
        }
        return CausalStepResult(accepted, status, reference, detail)

    def step(
        self,
        *,
        joint_names: Sequence[str],
        joint_pos: np.ndarray,
        root_pos_w: np.ndarray,
        root_quat_wxyz: np.ndarray,
        contact_flags: np.ndarray,
        timestamp_s: float,
        now_s: float,
    ) -> CausalStepResult:
        started = time.perf_counter()
        if self._requires_reinitialize:
            return self._finish(started_s=started, status="reinitialize_required")
        try:
            timestamp = _timestamp(timestamp_s, "timestamp_s")
            now = _timestamp(now_s, "now_s")
            age = now - timestamp
            if age > self.config.maximum_source_age_s or age < -self.config.maximum_future_clock_error_s:
                return self._finish(
                    started_s=started, status="stale_or_future_input", diagnostics={"source_age_s": age}
                )
            delta = timestamp - self._last_timestamp_s
            if delta <= 0 or abs(delta - self.config.cadence_s) > self.config.cadence_tolerance_s:
                return self._finish(
                    started_s=started, status="cadence_violation", diagnostics={"source_delta_s": delta}
                )
            source_joints = _named_joints(joint_names, joint_pos, self.source_layout, "source29 pose")
            if np.any(source_joints < self.source_layout.lower - 1e-6) or np.any(
                source_joints > self.source_layout.upper + 1e-6
            ):
                raise ValueError("source29 pose exceeds source-model joint limits")
            root = np.asarray(root_pos_w, dtype=np.float64)
            quat = np.asarray(root_quat_wxyz, dtype=np.float64)
            if (
                root.shape != (3,)
                or quat.shape != (4,)
                or not np.isfinite(root).all()
                or not np.isfinite(quat).all()
            ):
                raise ValueError("root pose must be finite single XYZ and WXYZ vectors")
            if not math.isclose(float(np.linalg.norm(quat)), 1.0, abs_tol=1e-4, rel_tol=0):
                raise ValueError("root quaternion must be normalized")
            contacts = np.asarray(contact_flags)
            if contacts.shape != (2,) or contacts.dtype.kind != "b":
                raise ValueError("explicit boolean left/right contact estimates are required")
            ik._set_configuration(
                self.source_model, self.source_data, self.source_layout, root, quat, source_joints
            )
            targets = ik._task_targets(self.source_model, self.source_data, self.tasks)
            desired_pos, desired_quat = ik._task_pose_arrays(
                self.source_model, self.source_data, self.tasks, source=True
            )
            lower, upper, relaxations = ik._trajectory_bounds(
                self.target_layout, self._previous, self._previous_velocity, self.config.cadence_s, self.ik_config
            )
            direct = source_joints[self._source_to_target]
            # Re-anchor to this frame's requested source posture, within the
            # accepted-history derivative bounds. Extrapolating old velocity
            # as the seed can perpetuate drift when IK accepts no new step.
            seed = np.clip(direct, lower, upper)
            candidate, solver = ik._solve_frame(
                self.target_model,
                self.target_data,
                self.target_layout,
                self.tasks,
                targets,
                root,
                quat,
                direct,
                self._previous,
                (bool(contacts[0]), bool(contacts[1])),
                lower,
                upper,
                self.ik_config,
                seed=seed,
                fallback_seed=seed,
                active_joint_indices=self._active_indices,
            )
            if not np.isfinite(list(solver.values())).all():
                raise ValueError("IK returned nonfinite solver diagnostics")
            velocity = (candidate - self._previous) / self.config.cadence_s
            acceleration = (velocity - self._previous_velocity) / self.config.cadence_s
            if not all(np.isfinite(value).all() for value in (candidate, velocity, acceleration)):
                raise ValueError("IK returned nonfinite reference values")
            failures = []
            if relaxations or np.any(candidate < lower - 1e-9) or np.any(candidate > upper + 1e-9):
                failures.append("joint trajectory bound violation")
            if np.max(np.abs(velocity)) > self.config.maximum_joint_velocity_rad_s + 1e-7:
                failures.append("joint velocity bound violation")
            if np.max(np.abs(acceleration)) > self.config.maximum_joint_acceleration_rad_s2 + 1e-6:
                failures.append("joint acceleration bound violation")
            ik._set_configuration(self.target_model, self.target_data, self.target_layout, root, quat, candidate)
            achieved_pos, achieved_quat = ik._task_pose_arrays(
                self.target_model, self.target_data, self.tasks, source=False
            )
            errors = np.linalg.norm(achieved_pos - desired_pos, axis=1)
            if not np.isfinite(errors).all():
                failures.append("nonfinite task-space error")
            per_task = {task.name: float(errors[index]) for index, task in enumerate(self.tasks)}
            for task in self.tasks:
                tolerance = (
                    self.config.maximum_foot_error_m
                    if task.name in {"left_foot", "right_foot"}
                    else self.config.maximum_hand_head_error_m
                    if task.name in {"left_hand", "right_hand", "head_proxy"}
                    else None
                )
                if tolerance is not None and per_task[task.name] > tolerance:
                    failures.append(f"{task.name} exceeds absolute position tolerance")
            for priority in range(self.ik_config.protected_priority_tiers):
                if solver[f"priority_{priority}_error_after"] > solver[
                    f"priority_{priority}_error_feasible_seed"
                ] + self.ik_config.priority_relative_tolerance * max(
                    1.0, solver[f"priority_{priority}_error_feasible_seed"]
                ):
                    failures.append(f"protected task priority {priority} regressed")
            metrics = {
                "source_age_s": age,
                "source_delta_s": delta,
                "task_position_error_m": per_task,
                "joint_velocity_abs_max_rad_s": float(np.max(np.abs(velocity))),
                "joint_acceleration_abs_max_rad_s2": float(np.max(np.abs(acceleration))),
                "constraint_relaxation_count": relaxations,
                "active_joint_count": len(self._active_indices),
                "solver": solver,
                "failures": failures,
            }
            if failures:
                return self._finish(started_s=started, status="infeasible_reference", diagnostics=metrics)
            reference = CausalReference(
                timestamp,
                self.target_layout.joint_names,
                candidate.copy(),
                velocity.copy(),
                root.copy(),
                quat.copy(),
                contacts.copy(),
                tuple(task.name for task in self.tasks),
                desired_pos.copy(),
                achieved_pos.copy(),
                desired_quat.copy(),
                achieved_quat.copy(),
            )
            self._previous, self._previous_velocity = candidate.copy(), velocity.copy()
            self._last_timestamp_s = timestamp
            return self._finish(started_s=started, status="accepted", reference=reference, diagnostics=metrics)
        except (ValueError, TypeError, KeyError) as exc:
            return self._finish(
                started_s=started, status="invalid_input", diagnostics={"error": f"{type(exc).__name__}: {exc}"}
            )
        except (RuntimeError, np.linalg.LinAlgError) as exc:
            return self._finish(
                started_s=started, status="solver_rejected", diagnostics={"error": f"{type(exc).__name__}: {exc}"}
            )

    def timing_summary(self) -> dict[str, Any]:
        elapsed = np.asarray(self._elapsed_s, dtype=np.float64)
        return {
            "calls": self._accepted_count + self._rejected_count,
            "percentile_window": "most_recent_4096_calls",
            "percentile_window_calls": len(elapsed),
            "accepted_calls": self._accepted_count,
            "rejected_calls": self._rejected_count,
            "median_s": float(np.median(elapsed)) if len(elapsed) else None,
            "p95_s": float(np.percentile(elapsed, 95)) if len(elapsed) else None,
            "maximum_s": self._maximum_elapsed_s if len(elapsed) else None,
            "calls_exceeding_20ms": self._deadline_misses,
            "all_observed_calls_within_20ms": bool(len(elapsed) and not self._deadline_misses),
            "scheduled_50hz_execution_verified": False,
            "end_to_end_teleop_latency_verified": False,
            "hardware_authorized": False,
        }
