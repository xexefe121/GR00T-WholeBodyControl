"""Constructive stopping-path constraints for reference rotation coordinates.

Separate coordinate boxes do not guarantee stopping inside an L1 ball. Check
one common braking schedule against all eight halfspaces at every stopping
sample. This proves a feasible stop for the static correction-coordinate L1
budget only; it proves neither future moving-source feasibility, base tilt,
foot fidelity, contact dynamics nor physical motor safety.
"""

import numpy as np


def l1_stopping_path(candidate, previous, *, maximum_velocity, maximum_acceleration, limit_rad, dt=0.02):
    """Return current+future coordinates for one synchronized componentwise stop.

    The caller must independently bound the current candidate's velocity and
    acceleration. The horizon is fixed by those declared velocity caps, not by
    looking at a future source sample. Derivatives are exact away from braking
    active-set changes and select the inactive derivative at a threshold.
    """
    candidate, previous, vmax, amax = (
        np.asarray(value, dtype=float) for value in (candidate, previous, maximum_velocity, maximum_acceleration)
    )
    if (
        any(value.shape != (3,) or not np.isfinite(value).all() for value in (candidate, previous, vmax, amax))
        or np.any(vmax <= 0)
        or np.any(vmax > 1.5)
        or np.any(amax <= 0)
        or np.any(amax > 12)
        or isinstance(limit_rad, bool)
        or not np.isfinite(limit_rad)
        or not 0 < limit_rad <= 0.45
        or not np.isfinite(dt)
        or dt <= 0
    ):
        raise ValueError("invalid or expanded reference stopping-path limits")
    horizon = int(np.max(np.ceil(vmax / (amax * dt))))
    if horizon > 128:
        raise ValueError("reference stopping horizon exceeds bounded computational budget")
    velocity = (candidate - previous) / dt
    if np.any(np.abs(velocity) > vmax + 1e-7):
        raise ValueError("candidate velocity outside separately required reference bounds")
    times = np.arange(1, horizon + 1, dtype=float)[:, None]
    remaining = np.maximum(np.abs(velocity)[None] - times * amax[None] * dt, 0)
    future_velocity = np.sign(velocity)[None] * remaining
    positions = np.vstack((candidate, candidate + dt * np.cumsum(future_velocity, axis=0)))
    active = np.abs(velocity)[None] > times * amax[None] * dt
    diagonal = np.vstack((np.ones(3), 1 + np.cumsum(active, axis=0)))
    signs = np.asarray([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=float)
    margins = (limit_rad - positions @ signs.T).reshape(-1)
    jacobian = (-signs[None] * diagonal[:, None]).reshape(-1, 3)
    return (
        margins,
        jacobian,
        dict(
            reference_coordinate_path=positions,
            reference_coordinate_velocity=np.vstack((velocity, future_velocity)),
            stopping_horizon_steps=horizon,
            shared_braking_schedule=True,
            guarantee_scope="static reference rotation-correction L1 only",
            deployment_ready=False,
        ),
    )


class ReferenceL1BrakingMixin:
    """Add synchronized reference-coordinate stopping to an existing root gate."""

    def root_constraints(self, lower_variables):
        original_margin, original_jacobian = super().root_constraints(lower_variables)
        c = self.temporal_config
        margins, jacobian, _ = l1_stopping_path(
            lower_variables[3:6],
            self.previous[3:6],
            maximum_velocity=self.maximum_velocity[3:6],
            maximum_acceleration=self.maximum_acceleration[3:6],
            limit_rad=c.root_rotation_l1_limit_rad * c.reserve_fraction,
        )
        expanded = np.zeros((len(margins), 19))
        expanded[:, 3:6] = jacobian
        return np.r_[original_margin, margins], np.vstack((original_jacobian, expanded))
