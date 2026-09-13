"""Declared simulation-only ankle repulsion; this is not a safety guarantee."""

from __future__ import annotations

import numpy as np


class AnkleRollRepulsion:
    """Inward spring near each limit, with damping only against outward motion.

    Add this measured-joint torque to BFM PD torque, then apply native motor
    effort caps. It does not alter desired motion, BFM goals, or action history.
    """

    def __init__(self, joint_names, limits, *, margin=0.05, stiffness=150.0, damping=2.0):
        if not np.isfinite([margin, stiffness, damping]).all() or margin <= 0 or min(stiffness, damping) < 0:
            raise ValueError("invalid ankle repulsion parameters")
        if len(joint_names) != 23 or np.asarray(limits).shape != (23, 2):
            raise ValueError("ankle repulsion requires exact native23 joint layout")
        self.indices = np.array(
            [joint_names.index(name) for name in ("left_ankle_roll_joint", "right_ankle_roll_joint")]
        )
        self.limits = np.asarray(limits, dtype=np.float64)[self.indices].copy()
        if np.any(2 * margin >= self.limits[:, 1] - self.limits[:, 0]):
            raise ValueError("ankle margin overlaps the opposite limit")
        self.margin, self.stiffness, self.damping = margin, stiffness, damping

    def torque(self, measured_q, measured_dq):
        q, dq = np.asarray(measured_q), np.asarray(measured_dq)
        if q.shape != (23,) or dq.shape != (23,) or not np.isfinite(q).all() or not np.isfinite(dq).all():
            raise ValueError("invalid measured native23 joint state")
        position, velocity = q[self.indices], dq[self.indices]
        lower = np.maximum(self.limits[:, 0] + self.margin - position, 0.0)
        upper = np.maximum(position - (self.limits[:, 1] - self.margin), 0.0)
        inward = self.stiffness * (lower - upper)
        inward -= self.damping * (
            (lower > 0) * np.minimum(velocity, 0.0) + (upper > 0) * np.maximum(velocity, 0.0)
        )
        result = np.zeros(23, dtype=np.float64)
        result[self.indices] = inward
        return result

    def report(self):
        return {
            "kind": "measured_ankle_roll_inward_spring_outward_damping",
            "joint_indices": self.indices.tolist(),
            "margin_rad": self.margin,
            "stiffness_Nm_per_rad": self.stiffness,
            "damping_Nm_per_rad_per_second": self.damping,
            "effort_clip_order": "clip(BFM_PD_torque + declared_ankle_torque, native_effort)",
            "zero_other_joint_torque": True,
            "formal_safety_guarantee": False,
        }
