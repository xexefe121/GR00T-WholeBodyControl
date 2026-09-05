"""Reference-relative simulator screening, distinct from legacy survival.

Thresholds are provisional engineering acceptance criteria, not manufacturer
limits or demonstrated human perceptual parity. No absolute upright-height
gate is used here: intentional crawling should be judged against its reference.
Passing never authorizes hardware. Use the same frozen criteria for both
original and candidate policies; do not fit thresholds to a failing candidate.
"""

from __future__ import annotations

import math
from typing import Any, Mapping


MAXIMUM_ERRORS = {
    "maximum_pelvis_position_error_m": 0.25,
    "maximum_pelvis_orientation_error_rad": 0.5,
    "maximum_relative_tracked_body_position_error_m": 0.20,
    "maximum_joint_tracking_rmse_rad": 0.35,
}


def assess_motion_fidelity(
    *, metrics: Mapping[str, Any], completed: int, requested: int,
    available: int, failure: object,
) -> dict[str, Any]:
    """Fail closed on partial clips, absent/nonfinite metrics or failed dynamics."""
    failures = []
    full_clip = (
        all(type(count) is int and count > 0 for count in (completed, requested, available))
        and completed == requested == available and failure is None
    )
    if not full_clip:
        failures.append("full_clip_completion")
    for name, limit in MAXIMUM_ERRORS.items():
        value = metrics.get(name)
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= limit:
            failures.append(name)
    return {
        "kind": "g1_true23_reference_motion_fidelity_v1",
        "passed": not failures,
        "full_clip_completed": full_clip,
        "failed_checks": failures,
        "maximum_errors": dict(MAXIMUM_ERRORS),
        "threshold_status": "provisional_engineering_screen_not_hardware_limits",
        "hardware_authorized": False,
        "deployment_ready": False,
    }
