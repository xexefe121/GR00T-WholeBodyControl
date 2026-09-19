"""Local branch guidance for an offline path objective, not replacement motion.

A feasible isolated pose may guide numerical search near its source frame.
Only the objective target changes. Every full-path task, derivative, collision
and final serialized/source-time gate still applies to all proposed trajectories.
The guidance array is deliberately not exported as a motion or used by a policy.
"""

import numpy as np


def pose_escape_objective_target(initial, frame, escaped_variables, *, radius_controls=100):
    values, pose = np.asarray(initial, dtype=float), np.asarray(escaped_variables, dtype=float)
    if (
        values.ndim != 2
        or values.shape[1] != 29
        or len(values) < 3
        or pose.shape != (29,)
        or not np.isfinite(values).all()
        or not np.isfinite(pose).all()
        or type(frame) is not int
        or not 0 <= frame < len(values)
        or type(radius_controls) is not int
        or not 25 <= radius_controls <= 150
    ):
        raise ValueError("pose-escape guidance requires a finite full native23 path and bounded local radius")
    separation = np.abs(np.arange(len(values)) - frame) / radius_controls
    weight = np.where(separation < 1, 0.5 * (1 + np.cos(np.pi * np.minimum(separation, 1))), 0.0)
    result = values + weight[:, None] * (pose - values[frame])[None]
    return result, {
        "kind": "g1_true23_single_pose_local_objective_guidance_v1",
        "source_control_frame": frame,
        "radius_controls": radius_controls,
        "affected_objective_frames": np.flatnonzero(weight > 0).tolist(),
        "numerical_objective_target_only": True,
        "original29_task_targets_replaced": False,
        "guidance_is_an_accepted_motion": False,
        "full_path_constraints_or_acceptance_changed": False,
    }
