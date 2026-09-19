"""Bounded generated-ramp rate projection; all source/standing poses stay exact.

This corrects finite-difference joins rather than changing source velocities,
slowing choreography, or weakening joint/velocity/acceleration limits. Geometry
must be checked separately; this module does not accept training references.
"""

from dataclasses import asdict

import numpy as np

from gear_sonic.utils.g1_23dof_trajectory_projection import (
    audit_trajectory_constraints,
    project_nearest_trajectory,
)


def project_generated_ramp_rates(poses, timeline, lower, upper, *, maximum_joint_change_rad=0.1):
    poses = np.asarray(poses, dtype=np.float64)
    if poses.ndim != 2 or poses.shape[1] != 30 or not np.isfinite(poses).all():
        raise ValueError("ramp projection requires complete finite native23 poses")
    if not np.isfinite(maximum_joint_change_rad) or not 0 < maximum_joint_change_rad <= 0.1:
        raise ValueError("generated ramp rate correction is bounded to 0.1 rad")
    editable = np.zeros(len(poses), dtype=bool)
    phases = [p for p in timeline["phases"] if p["name"] in {"acquisition_ramp", "return_ramp"}]
    if [p["name"] for p in phases] != ["acquisition_ramp", "return_ramp"]:
        raise ValueError("ramp projection requires exactly both existing generated phases")
    for phase in phases:
        start, stop = phase["frame_start"], phase["frame_stop"]
        if type(start) is not int or type(stop) is not int or not 2 <= start < stop <= len(poses) - 2:
            raise ValueError("ramp projection requires bounded interior phase indices")
        editable[start:stop] = True
    source = next(p for p in timeline["phases"] if p["name"] == "source_motion")
    if np.any(editable[source["frame_start"] : source["frame_stop"]]):
        raise ValueError("ramp projection may never include source motion")
    from gear_sonic.utils.g1_true23_terminal_braking import audit_terminal_braking

    boundary = audit_terminal_braking(poses[source["frame_start"] : source["frame_stop"], 7:], lower, upper)
    if not boundary["passed"]:
        raise ValueError(
            "source terminal velocity cannot brake inside unchanged joint limits; retarget endpoint first: "
            + str(boundary["failed_rows"])
        )
    desired = poses[:, 7:]
    lo, hi = desired.copy(), desired.copy()
    lo[editable] = np.maximum(lower, desired[editable] - maximum_joint_change_rad)
    hi[editable] = np.minimum(upper, desired[editable] + maximum_joint_change_rad)
    projected = project_nearest_trajectory(
        desired,
        lower_bounds=lo,
        upper_bounds=hi,
        dt=0.02,
        max_velocity=5.0,
        max_acceleration=80.0,
    )
    result = poses.copy()
    result[editable, 7:] = projected.projected_path[editable]
    # Only the two ramp interiors are copied. Equality tolerances in the numeric
    # solve may NEVER edit even the last bit of source or standing poses.
    audit = audit_trajectory_constraints(
        result[:, 7:],
        lower_bounds=lower,
        upper_bounds=upper,
        dt=0.02,
        max_velocity=5.0,
        max_acceleration=80.0,
        tolerance=2e-7,
    )
    delta = float(np.max(np.abs(result[:, 7:] - desired)))
    if not audit.passed or delta > maximum_joint_change_rad + 2e-7:
        raise ValueError("exact frozen-source ramp projection failed unchanged physical rate bounds")
    return result, {
        "kind": "g1_true23_generated_ramp_only_nearest_rate_projection_v1",
        "maximum_allowed_joint_change_rad": maximum_joint_change_rad,
        "maximum_actual_joint_change_rad": delta,
        "projection_iterations": projected.iterations,
        "path_bounds": asdict(audit),
        "source_and_standing_poses_bit_exact": np.array_equal(result[~editable], poses[~editable]),
        "root_trajectory_bit_exact": np.array_equal(result[:, :7], poses[:, :7]),
        "source_timing_or_frame_count_changed": False,
        "geometry_or_contact_qualification": False,
        "training_reference_accepted": False,
        "hardware_authorized": False,
        "deployment_ready": False,
    }
