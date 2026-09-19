"""Optional world-axis pelvis-relative foot objective for native23 MPC.

Uses the six-body ordering documented by Native23Tracker.TRACKED: pelvis,
torso, left ankle, right ankle, left hand, right hand. No heading alignment,
time alignment, reference translation or physical state mutation is performed.
"""
import numpy as np


def relative_foot_residual(positions, reference_positions, *, weight=400.0):
    positions = np.asarray(positions)
    reference_positions = np.asarray(reference_positions)
    if positions.shape[-2:] != (6, 3) or reference_positions.shape[-2:] != (6, 3):
        raise ValueError("relative foot objective requires declared six-body positions")
    if not np.isfinite(weight) or weight < 0:
        raise ValueError("relative foot objective weight must be finite and nonnegative")
    error = ((positions[..., 2:4, :] - positions[..., :1, :])
             - (reference_positions[..., 2:4, :] - reference_positions[..., :1, :]))
    return np.sqrt(weight) * error.reshape(*error.shape[:-2], 6)
