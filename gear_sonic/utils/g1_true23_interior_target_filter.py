"""Offline effort-headroom experiment inside the existing native23 envelope.

The outer position, effort and slew intersection is unchanged and still fails
explicitly when empty. A preferred inner effort band can keep a greedy target
away from that boundary. An unavailable inner band uses the minimum-effort
point of the outer interval, with an explicit relaxation record. This is not
recursive feasibility, a physical controller or permission to change limits.
"""

from __future__ import annotations

from types import FunctionType

import numpy as np

from gear_sonic.utils.g1_sonic_cpp_observation_trace import audit_engine_trace
from gear_sonic.utils.g1_true23_reference_floor import compiled_model_sha256
from gear_sonic.utils.g1_true23_sim_acquisition import effort_feasible_target, effort_target_interval


def interior_effort_target(requested, previous, q, dq, kp, kd, effort, *, dt, slew_rate, fraction):
    if (
        isinstance(fraction, bool)
        or not np.isscalar(fraction)
        or not np.isfinite(fraction)
        or not 0 <= fraction <= 1
    ):
        raise ValueError("interior effort fraction must be finite within 0..1")
    values = [np.asarray(value, dtype=float) for value in (requested, previous, q, dq, kp, kd, effort)]
    requested, previous, q, dq, kp, kd, effort = values
    # Preserve the exact outer admission and its joint-level rejection data.
    greedy = effort_feasible_target(*values, dt=dt, slew_rate=slew_rate)
    low, high = effort_target_interval(previous, q, dq, kp, kd, effort, dt=dt, slew_rate=slew_rate)
    cap = 0.95 * 0.25 * effort
    inner_low = np.maximum(low, q + (kd * dq - fraction * cap) / kp)
    inner_high = np.minimum(high, q + (kd * dq + fraction * cap) / kp)
    inner_exists = inner_low <= inner_high
    minimum_effort = np.clip(q + kd * dq / kp, low, high)
    target = np.where(inner_exists, np.clip(requested, inner_low, inner_high), minimum_effort)
    # At fraction one this is exactly the old target, including its rounding.
    if fraction == 1:
        target = greedy.copy()
    predicted = kp * (target - q) - kd * dq
    if np.any(target < low) or np.any(target > high) or np.any(np.abs(predicted) > cap + 1e-10):
        raise ValueError("interior experiment escaped the unchanged outer envelope")
    return target, {
        "inner_effort_fraction": float(fraction),
        "inner_band_unavailable_joints": np.flatnonzero(~inner_exists).tolist(),
        "changed_joints": int(np.count_nonzero(target != greedy)),
        "maximum_change_from_greedy_rad": float(np.max(np.abs(target - greedy))),
        "maximum_outer_effort_ratio": float(np.max(np.abs(predicted) / cap)),
        "minimum_outer_target_margin_rad": float(np.min(np.minimum(target - low, high - target))),
        "hard_limits_relaxed": False,
        "recursive_feasibility_proven": False,
        "hardware_authorized": False,
    }


def run_interior_case(*, fraction, **kwargs):
    """Reuse the complete paired evaluator with one private active-only filter.

    Standing acquisition/return retain their existing policy, projection and
    gains. All source motion, observations, applied-target feedback and stopping
    criteria are unchanged. Never patch the shared evaluator or robot modules.
    """
    from gear_sonic.scripts import evaluate_g1_true23_deployment_envelope as envelope

    if (
        kwargs.get("predictive_active_effort")
        or not kwargs.get("project_active_effort")
        or not kwargs.get("stateful_native_controller")
    ):
        raise ValueError("interior experiment needs stateful active projection without another filter")
    if not kwargs.get("trace_active_actuation"):
        raise ValueError("interior experiment requires every active actuation substep")
    records, controllers = [], []
    phase, balance_calls = [0], [0]
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
            "engine_warning_lastinfo",
            "phase",
        )
    }

    class RecordedModule:
        def __init__(self, controller):
            self.original, self.controller = controller.module, controller

        def __getattr__(self, name):
            return getattr(self.original, name)

        def mj_step(self, model, data):
            if model is not self.controller.model or data is not self.controller.data:
                raise ValueError("interior physics recorder cannot accept another simulation")
            physics["pre_qpos"].append(data.qpos.copy())
            physics["pre_qvel"].append(data.qvel.copy())
            physics["effort"].append(data.ctrl.copy())
            physics["engine_pre_time_s"].append(float(data.time))
            physics["phase"].append(phase[0])
            self.original.mj_step(model, data)
            physics["post_qpos"].append(data.qpos.copy())
            physics["post_qvel"].append(data.qvel.copy())
            physics["generalized_actuator_force"].append(data.qfrc_actuator[6:].copy())
            physics["engine_post_time_s"].append(float(data.time))
            physics["engine_warning_counts"].append([int(item.number) for item in data.warning])
            physics["engine_warning_lastinfo"].append([int(item.lastinfo) for item in data.warning])

    def create_controller(**options):
        controller = envelope.CleanTrue23MujocoController(**options)
        controller.module = RecordedModule(controller)
        controllers.append(controller)
        return controller

    def balance(*args, **options):
        phase[0] = 0 if balance_calls[0] == 0 else 2
        balance_calls[0] += 1
        return envelope.simulate_balance_transition(*args, **options)

    def filtered(*args, **options):
        phase[0] = 1
        target, details = interior_effort_target(*args, **options, fraction=fraction)
        records.append(details)
        return target

    original = envelope.run_case
    function = FunctionType(
        original.__code__,
        {
            **original.__globals__,
            "effort_feasible_target": filtered,
            "CleanTrue23MujocoController": create_controller,
            "simulate_balance_transition": balance,
        },
        name=original.__name__,
        argdefs=original.__defaults__,
        closure=original.__closure__,
    )
    function.__kwdefaults__ = original.__kwdefaults__
    report, arrays = function(fraction=1.0, **kwargs)
    if len(controllers) != 1 or len(records) != report["completed_active_physics_steps"]:
        raise ValueError("interior decision trace lost an active physics step")
    arrays.update({"physics_" + key: np.asarray(value) for key, value in physics.items()})
    arrays["physics_dt"] = np.array([controllers[0].physics.timestep_s])
    if physics["phase"]:
        engine = audit_engine_trace(arrays)
        np.testing.assert_array_equal(arrays["physics_pre_qpos"][1:], arrays["physics_post_qpos"][:-1])
        np.testing.assert_array_equal(arrays["physics_pre_qvel"][1:], arrays["physics_post_qvel"][:-1])
    else:
        engine = {"passed": False, "reason": "no physics steps completed"}
    if int(np.sum(arrays["physics_phase"] == 1)) != len(records):
        raise ValueError("active physics and filter boundaries disagree")
    report["actual_engine_audit"] = engine
    report["compiled_native_model_sha256"] = compiled_model_sha256(controllers[0].model)
    for flag in (
        "library_completion_passed",
        "upright_physical_bounds_passed",
        "lifecycle_simulator_screen_passed",
    ):
        report[flag] = bool(report[flag] and engine["passed"])
    arrays["interior_changed_joints"] = np.asarray([row["changed_joints"] for row in records], dtype=np.int64)
    arrays["interior_relaxed_joints"] = np.asarray(
        [len(row["inner_band_unavailable_joints"]) for row in records], dtype=np.int64
    )
    arrays["interior_outer_effort_ratio"] = np.asarray([row["maximum_outer_effort_ratio"] for row in records])
    report["interior_effort_experiment"] = {
        "kind": "g1_true23_active_interior_effort_diagnostic_v1",
        "fraction_of_existing_outer_effort_cap": float(fraction),
        "completed_filter_calls": len(records),
        "changed_joint_substeps": sum(row["changed_joints"] for row in records),
        "inner_band_unavailable_joint_substeps": sum(len(row["inner_band_unavailable_joints"]) for row in records),
        "maximum_change_from_greedy_rad": max(
            (row["maximum_change_from_greedy_rad"] for row in records), default=0
        ),
        "maximum_outer_effort_ratio": max((row["maximum_outer_effort_ratio"] for row in records), default=0),
        "full_requested_motion_retained": True,
        "active_sonic_controlled_joint_count": 23,
        "outer_limits_relaxed": False,
        "standing_transition_controller_changed": False,
        "shared_evaluator_modified": False,
        "recursive_feasibility_proven": False,
        "teacher_accepted": False,
        "hardware_authorized": False,
        "deployment_ready": False,
    }
    return report, arrays
