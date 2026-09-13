"""Original native joint/speed bounds with stricter offline planning insets."""

import numpy as np

JOINT_PLANNING_INSET_RAD = 1e-5
SPEED_PLANNING_INSET_RAD_S = 1e-5


def substep_bound_margins(plant, poses, velocities):
    poses, velocities = np.asarray(poses), np.asarray(velocities)
    if poses.shape != (10, 30) or velocities.shape != (10, 29):
        raise ValueError("bound prediction requires ten native23 physics substeps")
    if not np.isfinite(poses).all() or not np.isfinite(velocities).all():
        raise ValueError("bound prediction requires finite physical states")
    lower = plant.model.jnt_range[1:, 0] + JOINT_PLANNING_INSET_RAD
    upper = plant.model.jnt_range[1:, 1] - JOINT_PLANNING_INSET_RAD
    velocity = np.asarray(plant.profile.velocity) - SPEED_PLANNING_INSET_RAD_S
    if lower.shape != (23,) or upper.shape != (23,) or velocity.shape != (23,):
        raise ValueError("bound prediction requires original native23 limits")
    if np.any(lower >= upper) or np.any(velocity <= 0):
        raise ValueError("invalid original limits for stricter planning insets")
    return np.concatenate(
        (
            (poses[:, 7:] - lower).ravel(),
            (upper - poses[:, 7:]).ravel(),
            (velocity - velocities[:, 6:]).ravel(),
            (velocity + velocities[:, 6:]).ravel(),
        )
    )


def substep_bound_jacobian(position_derivatives, velocity_derivatives):
    position, velocity = np.asarray(position_derivatives), np.asarray(velocity_derivatives)
    if position.shape != (10, 29, 23) or velocity.shape != (10, 29, 23):
        raise ValueError("bound derivatives require ten native physical-to-command Jacobians")
    if not np.isfinite(position).all() or not np.isfinite(velocity).all():
        raise ValueError("bound derivatives must be finite")
    q, v = position[:, 6:].reshape(230, 23), velocity[:, 6:].reshape(230, 23)
    return np.vstack((q, -q, -v, v))
