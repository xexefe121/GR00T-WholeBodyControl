"""Discrete reference endpoint extendability under existing joint/rate limits.

These are necessary kinematic braking-room constraints, not motor braking or
contact-dynamic guarantees. They add a stricter boundary condition without
changing original source targets, timestamps or physical limits.
"""

import numpy as np
from scipy import sparse


def terminal_braking_rows(frame_count, lower, upper, *, dt=0.02, max_velocity=5.0, max_acceleration=80.0):
    lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    if (
        type(frame_count) is not int
        or frame_count < 3
        or lower.shape != (23,)
        or upper.shape != (23,)
        or not np.isfinite(lower).all()
        or not np.isfinite(upper).all()
        or np.any(lower >= upper)
        or not all(np.isfinite(x) and x > 0 for x in (dt, max_velocity, max_acceleration))
    ):
        raise ValueError("terminal braking requires finite exact native23 bounds and positive cadence/rates")
    steps = int(np.ceil(max_velocity / (max_acceleration * dt)))
    if steps > 100:
        raise ValueError("terminal reference braking horizon exceeds bounded local audit")
    ks = np.repeat(np.arange(1, steps + 1), 23)
    joints = np.tile(np.arange(23), steps)
    rows = np.arange(len(ks))
    matrix = sparse.coo_matrix(
        (
            np.r_[-ks, ks + 1],
            (np.r_[rows, rows], np.r_[(frame_count - 2) * 29 + 6 + joints, (frame_count - 1) * 29 + 6 + joints]),
        ),
        shape=(len(ks), frame_count * 29),
        dtype=float,
    ).tocsc()
    deceleration = max_acceleration * dt**2 * ks * (ks + 1) / 2
    return matrix, np.tile(lower, steps) - deceleration, np.tile(upper, steps) + deceleration


def audit_terminal_braking(joints, lower, upper, *, dt=0.02, max_velocity=5.0, max_acceleration=80.0):
    joints = np.asarray(joints, dtype=float)
    if joints.ndim != 2 or joints.shape[1] != 23 or not np.isfinite(joints).all():
        raise ValueError("terminal braking audit requires complete finite native23 joint positions")
    matrix, lo, hi = terminal_braking_rows(
        len(joints),
        lower,
        upper,
        dt=dt,
        max_velocity=max_velocity,
        max_acceleration=max_acceleration,
    )
    variables = np.column_stack((np.zeros((len(joints), 6)), joints))
    values = matrix @ variables.ravel()
    violation = np.maximum(np.maximum(lo - values, values - hi), 0)
    return {
        "kind": "g1_true23_discrete_terminal_reference_braking_room_v1",
        "passed": bool(violation.max(initial=0) <= 2e-7),
        "maximum_joint_position_violation_rad": float(violation.max(initial=0)),
        "failed_rows": [
            {"braking_step": int(i // 23 + 1), "joint_index": int(i % 23), "violation_rad": float(violation[i])}
            for i in np.flatnonzero(violation > 2e-7)
        ],
        "dt_s": dt,
        "velocity_limit_rad_s": max_velocity,
        "acceleration_limit_rad_s2": max_acceleration,
        "terminal_velocity_rad_s": ((joints[-1] - joints[-2]) / dt).tolist(),
        "reference_constraints_only_not_actuator_braking_guarantee": True,
        "dynamic_or_contact_feasibility_proven": False,
        "hardware_authorized": False,
        "deployment_ready": False,
    }


class TerminalBrakingTaskExtension:
    """Append scalar SOC rows to existing all-original-time constraints."""

    def __init__(self, original, reference_variables, lower, upper):
        self.original = original
        self.original_times = original.original_times
        residual, _ = original.evaluate(reference_variables, derivatives=False)
        self.offset = len(residual)
        self.matrix, lo, hi = terminal_braking_rows(len(reference_variables), lower, upper)
        self.center, self.radius = (lo + hi) / 2, (hi - lo) / 2
        self.groups = list(original.groups) + [
            {
                "name": "terminal_braking_room",
                "frame": len(reference_variables) - 1,
                "indices": np.array([self.offset + i]),
                "scales": np.ones(1),
                "radius": float(radius),
            }
            for i, radius in enumerate(self.radius)
        ]

    def evaluate(self, variables, *, derivatives=True):
        residual, jacobian = self.original.evaluate(variables, derivatives=derivatives)
        if len(residual) != self.offset:
            raise ValueError("original-time residual layout changed during terminal-bound fitting")
        result = np.r_[residual, self.matrix @ np.asarray(variables).ravel() - self.center]
        return result, sparse.vstack((jacobian, self.matrix), format="csc") if derivatives else None
