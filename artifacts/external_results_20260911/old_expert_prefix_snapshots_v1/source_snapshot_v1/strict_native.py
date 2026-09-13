"""Unchanged selected native2ms oracle, AST extracted without evaluator import."""
import numpy as np

def assess(data, c, expected):
    limits = np.asarray(c['joint_limits'])
    (q, dq) = (data.qpos, data.qvel)
    finite = np.isfinite(q).all() and np.isfinite(dq).all()
    excess = float(np.maximum(0, np.maximum(limits[:, 0] - q[7:], q[7:] - limits[:, 1])).max()) if finite else float('inf')
    velocity = float(np.max(np.abs(dq[6:]) / np.asarray(c['native_velocity']))) if finite else float('inf')
    force = data.qfrc_actuator[6:]
    effort = float(np.max(np.abs(force) / np.asarray(c['native_effort']))) if np.isfinite(force).all() else float('inf')
    tilt = float(np.arccos(np.clip(1 - 2 * np.sum(q[4:6] ** 2), -1, 1))) if finite else float('inf')
    reasons = []
    if not finite or not np.isfinite(force).all():
        reasons.append('nonfinite_state_or_force')
    if finite and abs(np.linalg.norm(q[3:7]) - 1) > 1e-10:
        reasons.append('invalid_root_quaternion')
    if excess > 1e-06:
        reasons.append('native_joint_bound')
    if velocity > 1:
        reasons.append('native_joint_speed')
    if effort > 1 + 1e-09:
        reasons.append('native_actuator_force')
    if q[2] < 0.25 or tilt > 1.2:
        reasons.append('fall')
    if np.any(data.warning.number):
        reasons.append('engine_warning')
    clock = float(abs(data.time - expected))
    if not np.isfinite(data.time) or clock > 1e-10:
        reasons.append('physics_clock')
    return (reasons, dict(range_excess=excess, velocity_ratio=velocity, effort_ratio=effort, tilt=tilt, clock_error=clock))
