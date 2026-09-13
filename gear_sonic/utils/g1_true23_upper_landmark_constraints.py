"""Fixed-reference-rank sufficient constraints for the existing upper p95 gate.

The outlier identities come only from a separately accepted full reference, not
the changing numerical iterate. At least both order statistics used by NumPy's
linear p95 must remain below the unchanged 10-cm gate. Other frames keep bounded
original-reference errors. This is a stricter fitting surrogate, not a new or
relaxed acceptance test. All frames, targets and final independent gates remain.
"""

import math

import numpy as np

from gear_sonic.utils.g1_true23_original_task_trajectory import TASK_SCALES


def fixed_rank_position_budgets(reference_errors, maximum_position_error_m=0.1):
    errors = np.asarray(reference_errors, dtype=float)
    limit = float(maximum_position_error_m)
    if (
        errors.ndim != 1
        or len(errors) < 3
        or not np.isfinite(errors).all()
        or np.any(errors < 0)
        or not np.isfinite(limit)
        or not 0 < limit <= 0.1
    ):
        raise ValueError("upper landmark budgets require a finite complete path and unchanged <=10-cm gate")
    if np.percentile(errors, 95) > limit:
        raise ValueError("upper landmark budget anchor must already pass the full original p95 gate")
    protected_count = math.ceil(0.95 * (len(errors) - 1)) + 1
    protected = np.argsort(errors, kind="stable")[:protected_count]
    radius = np.maximum(errors, limit * 0.995)
    radius[protected] = limit * 0.995
    if np.percentile(radius, 95) > limit * 0.995 + 1e-15:
        raise ValueError("fixed-rank surrogate does not imply the unchanged p95 gate")
    return radius, protected


def upper_landmark_groups(problem, reference_variables, maximum_position_error_m=0.1):
    residual = problem.evaluate(reference_variables, derivatives=False)[0]
    offsets, width = {}, 0
    for task in problem.tasks:
        for kind, scale in zip(("position", "orientation"), TASK_SCALES[task.name], strict=True):
            if scale:
                offsets[task.name, kind] = (np.arange(width, width + 3), scale)
                width += 3
    count = len(problem.initial)
    if residual.shape != (count * width,):
        raise ValueError("upper landmark residual layout must match the complete original task path")
    groups, detail = [], {}
    for name in ("left_hand", "right_hand", "head_proxy"):
        indices, scale = offsets[name, "position"]
        error = np.linalg.norm(residual.reshape(count, width)[:, indices] / scale, axis=1)
        radii, protected = fixed_rank_position_budgets(error, maximum_position_error_m)
        for frame, radius in enumerate(radii):
            groups.append(
                {
                    "name": name + "_fixed_rank_p95_position",
                    "frame": frame,
                    "indices": frame * width + indices,
                    "scales": np.full(3, 1 / scale),
                    "radius": float(radius),
                }
            )
        detail[name] = {
            "reference_position_p95_m": float(np.percentile(error, 95)),
            "reference_position_max_m": float(error.max()),
            "fixed_protected_frame_indices": protected.tolist(),
            "per_frame_radius_m": radii.tolist(),
            "radius_p95_m": float(np.percentile(radii, 95)),
        }
    return groups, {
        "kind": "g1_true23_fixed_accepted_reference_rank_upper_position_soc_v1",
        "frames": count,
        "unchanged_final_position_p95_limit_m": float(maximum_position_error_m),
        "surrogate_stricter_than_final_gate": True,
        "reference_targets_or_outlier_identities_changed_during_fit": False,
        "per_task": detail,
        "deployment_ready": False,
    }
