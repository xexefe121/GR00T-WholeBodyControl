"""Offline anticipatory ankle range check; not a live safety certificate.

Retain every-joint checks for the next 20 ms. Also check all four ankle axes
under a constant candidate PD target through 100 ms. This counterfactual uses
only copied current state; it is not a prediction of future policy actions.
"""

import time

import numpy as np

from gear_sonic.utils.g1_23dof_contract import MUJOCO_TO_ISAACLAB_DOF
from gear_sonic.utils.g1_23dof_safe_target_transform import (
    SAFE_TARGET_DEFAULT_Q_HARDWARE,
    SAFE_TARGET_HARD_LOWER_HARDWARE,
    SAFE_TARGET_HARD_UPPER_HARDWARE,
    safe_target_transform_numpy,
)
from gear_sonic.utils.g1_true23_range_preview import Native23RangePreview, RANGE_RESERVE_RAD, target_to_raw

ANKLES = (4, 5, 10, 11)
HORIZON_SUBSTEPS = 50
CONTROL_SUBSTEPS = 10
MAX_PREDICTIONS = 16
FAILURE = "no verified next-control and100ms ankle target found by bounded inward search"


def choose_ankle_horizon_target(requested_raw, predict):
    """Bounded directional search with exact unchanged-action preservation.

    Only coordinates violating a checked range may change. Duplicate decoded
    targets are never simulated twice. An already stronger nominal target is
    never weakened by the numerical inset on the search endpoint.
    """
    raw = np.asarray(requested_raw)
    if raw.shape != (23,) or raw.dtype != np.float32 or not np.isfinite(raw).all():
        raise ValueError("ankle horizon requires finite float32 native action23")
    if np.any(np.abs(raw) >= 10):
        raise ValueError("ankle horizon rejects raw action bound violations")
    low = np.asarray(SAFE_TARGET_HARD_LOWER_HARDWARE) + RANGE_RESERVE_RAD
    high = np.asarray(SAFE_TARGET_HARD_UPPER_HARDWARE) - RANGE_RESERVE_RAD
    _, requested = safe_target_transform_numpy(raw)
    _, endpoint_low = safe_target_transform_numpy(np.full(23, -9.99, np.float32))
    _, endpoint_high = safe_target_transform_numpy(np.full(23, 9.99, np.float32))
    endpoint_low = np.minimum(requested, endpoint_low.astype(float) + 1e-5)
    endpoint_high = np.maximum(requested, endpoint_high.astype(float) - 1e-5)
    cache = {}

    def check(candidate_raw, target):
        key = target.tobytes()
        if key in cache:
            return cache[key]
        if len(cache) >= MAX_PREDICTIONS:
            raise RuntimeError("ankle horizon prediction budget exhausted")
        poses = np.asarray(predict(target.copy()))
        if poses.shape != (HORIZON_SUBSTEPS, 30) or not np.isfinite(poses).all():
            raise ValueError("ankle horizon callback requires finite fifty native23 qpos samples")
        next_lower = np.maximum(low - poses[:CONTROL_SUBSTEPS, 7:], 0).max(0)
        next_upper = np.maximum(poses[:CONTROL_SUBSTEPS, 7:] - high, 0).max(0)
        lower, upper = next_lower.copy(), next_upper.copy()
        for j in ANKLES:
            lower[j] = max(lower[j], float(np.maximum(low[j] - poses[:, 7 + j], 0).max()))
            upper[j] = max(upper[j], float(np.maximum(poses[:, 7 + j] - high[j], 0).max()))
        row = dict(
            raw=candidate_raw.copy(),
            target=target.copy(),
            predicted=poses.copy(),
            lower=lower,
            upper=upper,
            next_lower=next_lower,
            next_upper=next_upper,
        )
        cache[key] = row
        return row

    def good(row):
        return not np.any(row["lower"] > 0) and not np.any(row["upper"] > 0)

    nominal = check(raw, requested)
    if good(nominal):
        return nominal, dict(intervened=False, preview_calls=1, changed_joints=[], alpha=0.0)
    details = dict(
        nominal_next_control_violation_rad=float(max(nominal["next_lower"].max(), nominal["next_upper"].max())),
        nominal_checked_violation_rad=float(max(nominal["lower"].max(), nominal["upper"].max())),
    )

    def fail(reason):
        error = ValueError(FAILURE)
        error.preview_details = dict(
            **details,
            reason=reason,
            preview_calls=len(cache),
            candidate_maximum_violations_rad=[
                float(max(r["lower"].max(), r["upper"].max())) for r in cache.values()
            ],
        )
        raise error

    if np.any((nominal["lower"] > 0) & (nominal["upper"] > 0)):
        fail("same joint crosses both checked bounds")
    endpoint = requested.astype(float).copy()
    endpoint[nominal["lower"] > 0] = endpoint_high[nominal["lower"] > 0]
    endpoint[nominal["upper"] > 0] = endpoint_low[nominal["upper"] > 0]
    changed = endpoint != requested
    if not np.any(changed):
        fail("no additional inward target authority")
    changed_il = changed[list(MUJOCO_TO_ISAACLAB_DOF)]

    def at(alpha):
        desired = requested.astype(float) + alpha * (endpoint - requested)
        # Invert only changed coordinates. Untouched saturated coordinates must
        # not be round-tripped or lose their original raw-action bytes.
        inverse_input = np.asarray(SAFE_TARGET_DEFAULT_Q_HARDWARE, dtype=np.float32).astype(float)
        inverse_input[changed] = desired[changed]
        converted, _ = target_to_raw(inverse_input)
        candidate_raw = raw.copy()
        candidate_raw[changed_il] = converted[changed_il]
        _, decoded = safe_target_transform_numpy(candidate_raw)
        np.testing.assert_array_equal(decoded[~changed], requested[~changed])
        return check(candidate_raw, decoded)

    previous_alpha, accepted, accepted_alpha = 0.0, None, None
    for alpha in (1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0):
        candidate = at(alpha)
        if good(candidate):
            accepted, accepted_alpha = candidate, alpha
            break
        previous_alpha = alpha
    if accepted is None:
        fail("bounded inward candidates exhausted")
    for _ in range(8):
        alpha = (previous_alpha + accepted_alpha) / 2
        candidate = at(alpha)
        if good(candidate):
            accepted, accepted_alpha = candidate, alpha
        else:
            previous_alpha = alpha
    return accepted, dict(
        **details,
        intervened=True,
        preview_calls=len(cache),
        alpha=accepted_alpha,
        changed_joints=np.flatnonzero(np.abs(accepted["target"] - requested) > 5e-7).tolist(),
        maximum_target_change_rad=float(np.max(np.abs(accepted["target"] - requested))),
    )


class AnkleHorizonNative23RangePreview(Native23RangePreview):
    """Simulation-only replacement; no access to the actual integrator."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.failed_search = None

    def filter(self, raw, measured_qpos, measured_qvel):
        if self.failed_search is not None:
            raise RuntimeError("failed ankle horizon cannot restart implicitly")
        q, v = np.asarray(measured_qpos).copy(), np.asarray(measured_qvel).copy()
        if q.shape != (30,) or v.shape != (29,) or not np.isfinite(q).all() or not np.isfinite(v).all():
            raise ValueError("ankle horizon requires copied finite native23 measured state")
        if np.any(q[7:] < SAFE_TARGET_HARD_LOWER_HARDWARE) or np.any(q[7:] > SAFE_TARGET_HARD_UPPER_HARDWARE):
            raise ValueError("ankle horizon cannot accept an already violated physical joint range")
        c = self.probe
        started = time.perf_counter()

        def predict(target):
            c.module.mj_resetData(c.model, c.data)
            c.reset(
                base_position=q[:3],
                base_quaternion_wxyz=q[3:7],
                joint_position_hardware=q[7:],
                root_velocity=v[:6],
                joint_velocity_hardware=v[6:],
            )
            poses = []
            for _ in range(HORIZON_SUBSTEPS):
                request = c.physics.kp * (target.astype(float) - c.data.qpos[7:]) - c.physics.kd * c.data.qvel[6:]
                c.data.ctrl[:] = np.clip(request, -c.physics.effort, c.physics.effort)
                c.module.mj_step(c.model, c.data)
                poses.append(c.data.qpos.copy())
            return np.asarray(poses)

        try:
            chosen, details = choose_ankle_horizon_target(raw, predict)
        except Exception as error:
            self.failed_search = dict(
                message=str(error),
                details=getattr(error, "preview_details", None),
                elapsed_s=time.perf_counter() - started,
            )
            raise
        self.records.append(dict(**details, elapsed_s=time.perf_counter() - started))
        self.predictions.append(chosen["predicted"])
        return chosen["raw"].copy()

    def contract(self):
        result = super().contract()
        result.update(
            kind="native23_offline_20ms_all_joint_and100ms_ankle_preview_v1",
            predicted_physics_substeps=HORIZON_SUBSTEPS,
            all_joint_checked_substeps=CONTROL_SUBSTEPS,
            extended_checked_hardware_joints=list(ANKLES),
            constant_candidate_target_horizon_s=0.1,
            future_policy_actions_predicted=False,
            source_lookahead_added=False,
            duplicate_predictions_skipped=True,
            numerical_inset_never_weakens_nominal_braking=True,
            maximum_predictions=MAX_PREDICTIONS,
            rejected_attempt=self.failed_search,
            actual_clock_qualified=False,
        )
        return result
