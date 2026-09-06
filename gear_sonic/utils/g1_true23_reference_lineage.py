"""Original-choreography path checks, independent of policy survival.

A bounded repair around a controller rollout need not be a bounded repair
around the original planner motion. The lower bounds here detect that error
without running physics, changing a pose or assuming a particular optimizer.
"""

from __future__ import annotations

import numpy as np


def root_path_repair_bounds(original_xyz, candidate_xyz, *, maximum_offset_m):
    """Conservative errors after ANY per-frame +/-offset XYZ correction.

    Both paths are translated to their first root position, without time warp.
    A correction relative to the first frame can be at most twice the supplied
    per-coordinate bound. The horizontal radial bound additionally allows ANY
    constant yaw rotation, so it cannot mistake a coordinate heading for drift.
    Temporal/contact/force constraints could only make these lower bounds worse.
    """
    original, candidate = (np.asarray(value, dtype=float) for value in (original_xyz, candidate_xyz))
    if (
        original.ndim != 2
        or original.shape[1] != 3
        or len(original) < 2
        or candidate.shape != original.shape
        or not np.isfinite(original).all()
        or not np.isfinite(candidate).all()
        or isinstance(maximum_offset_m, bool)
        or not np.isfinite(maximum_offset_m)
        or maximum_offset_m < 0
    ):
        raise ValueError("root-path bound needs equal finite [frames,3] arrays and a nonnegative offset")
    source_delta, target_delta = original - original[0], candidate - candidate[0]
    error = target_delta - source_delta
    xyz_bound = np.linalg.norm(np.maximum(np.abs(error) - 2 * maximum_offset_m, 0), axis=1)
    source_radius = np.linalg.norm(source_delta[:, :2], axis=1)
    target_radius = np.linalg.norm(target_delta[:, :2], axis=1)
    radial_bound = np.maximum(np.abs(source_radius - target_radius) - 2 * np.sqrt(2) * maximum_offset_m, 0)
    worst_xyz, worst_radial = int(np.argmax(xyz_bound)), int(np.argmax(radial_bound))
    return {
        "frames_checked": len(original),
        "frames_dropped": 0,
        "time_warp_applied": False,
        "comparison_origin": "each_path_first_root_position",
        "candidate_per_frame_xyz_correction_bound_m": float(maximum_offset_m),
        "source_horizontal_displacement_m": float(source_radius[-1]),
        "candidate_horizontal_displacement_m": float(target_radius[-1]),
        "maximum_translation_aligned_root_error_m": float(np.linalg.norm(error, axis=1).max()),
        "minimum_possible_maximum_error_under_xyz_box_m": float(xyz_bound[worst_xyz]),
        "xyz_lower_bound_worst_frame": worst_xyz,
        "xyz_error_at_lower_bound_worst_frame_m": error[worst_xyz].tolist(),
        "minimum_possible_maximum_horizontal_error_allowing_yaw_m": float(radial_bound[worst_radial]),
        "yaw_invariant_lower_bound_worst_frame": worst_radial,
        "lower_bounds_ignore_derivative_contact_and_force_constraints": True,
        "original_choreography_parity_proven": False,
        "hardware_authorized": False,
        "deployment_ready": False,
    }
