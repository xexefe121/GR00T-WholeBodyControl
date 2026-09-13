"""Bounded independent interpolation repairs; no changed source or safety gates.

Used only through the explicit --grouped-protected-repair offline CLI. Each
small group checks both FK grids and the full trajectory derivative limits.
The parent CLI independently audits the final serialized complete trajectory.
"""

from __future__ import annotations

import json

import numpy as np

from gear_sonic.scripts.refine_g1_true23_original_time_foot_orientation import (
    LEG_JOINTS,
    MAX_CORRECTION_RAD,
    LocalOrientationFit,
)

MAX_TOTAL_COORDINATES = 512
MAX_GROUP_COORDINATES = 128


def coordinate_groups(coordinates):
    """Separate groups only beyond the second-difference coupling stencil."""
    coords = np.asarray(coordinates, dtype=int)
    if coords.ndim != 2 or coords.shape[1] != 2 or not len(coords):
        raise ValueError("repair requires nonempty frame/joint coordinates")
    if len(coords) > MAX_TOTAL_COORDINATES or len(np.unique(coords, axis=0)) != len(coords):
        raise ValueError("repair exceeds bounded unique coordinate budget")
    knots = np.unique(coords[:, 0])
    sections = np.split(knots, np.flatnonzero(np.diff(knots) > 2) + 1)
    groups = [coords[np.isin(coords[:, 0], section)] for section in sections]
    if any(len(group) > MAX_GROUP_COORDINATES for group in groups):
        raise ValueError("coupled repair exceeds 128 control variables")
    return groups


def grouped_repair_coordinates(audit, source_times, control_source_times, joint_names):
    allowed = {"com_position_regression", "left_foot_orientation_regression", "right_foot_orientation_regression"}
    failures = set(audit["failures"])
    if not failures or not failures <= allowed:
        raise ValueError("grouped repair permits only COM and foot-orientation interpolation failures")
    times, controls = np.asarray(source_times), np.asarray(control_source_times)
    if (
        len(times) < 2
        or len(controls) < 2
        or not np.isfinite(times).all()
        or not np.isfinite(controls).all()
        or np.any(np.diff(times) <= 0)
        or np.any(np.diff(controls) <= 0)
        or times[0] < controls[0]
        or times[-1] > controls[-1]
        or len(joint_names) != 23
        or len(set(joint_names)) != 23
    ):
        raise ValueError("repair requires covered strictly ordered times and 23 unique joints")
    coordinates = set()
    for category in sorted(failures):
        frames = audit["categories"][category]["frame_indices"]
        if not 1 <= len(frames) <= 32 or any(type(i) is not int or not 0 <= i < len(times) for i in frames):
            raise ValueError("local repair requires 1..32 valid original frames per category")
        if category == "com_position_regression":
            joints = range(23)
        else:
            side = category.split("_")[0]
            joints = [joint_names.index(f"{side}_{joint}_joint") for joint in LEG_JOINTS]
        for frame in frames:
            upper = int(np.searchsorted(controls, times[frame], side="right"))
            for knot in range(max(0, upper - 2), min(len(controls), upper + 2)):
                coordinates.update((knot, joint) for joint in joints)
    result = np.asarray(sorted(coordinates), dtype=int)
    coordinate_groups(result)
    return result


class GroupedProtectedFit:
    def __init__(self, source_model, target_model, source, candidate, pose, time_map, coordinates, config):
        self.args = (source_model, target_model, source, candidate)
        self.pose, self.time_map = pose.copy(), time_map
        self.coordinates, self.config = coordinates.copy(), config

    def solve(self):
        current = self.pose.copy()
        reports = []
        groups = coordinate_groups(self.coordinates)
        for index, group in enumerate(groups):
            problem = LocalOrientationFit(
                *self.args, current, self.time_map, group, self.config, fit_com_margin_m=1e-7
            )
            current, report = problem.solve(stop_when_serialized_feasible=True)
            reports.append(report)
            print(
                json.dumps(
                    {
                        "group": index + 1,
                        "groups": len(groups),
                        "variables": len(group),
                        "serialized_constraint_min": report["serialized_selected_constraint_min"],
                        "max_correction_rad": report["serialized_max_correction_rad"],
                        "iterations": report["optimizer_iterations"],
                    }
                ),
                flush=True,
            )
        delta = (
            current[self.coordinates[:, 0], 7 + self.coordinates[:, 1]]
            - self.pose[self.coordinates[:, 0], 7 + self.coordinates[:, 1]]
        )
        return current, {
            "method": "bounded_grouped_protected_FK_original_and_control_grid_v1",
            "groups": reports,
            "optimizer_success": all(report["optimizer_success"] for report in reports),
            "minimum_norm_optimality_proven": False,
            "max_correction_bound_rad": MAX_CORRECTION_RAD,
            "serialized_max_correction_rad": float(np.max(np.abs(delta))),
            "coordinates_frame_joint": self.coordinates.tolist(),
            "source_samples_or_root_changed": False,
            "thresholds_changed": False,
            "iterations": [{"accepted": report["serialized_selected_constraint_min"] >= 0} for report in reports],
            "seed_projection": {"iterations": 0},
        }
