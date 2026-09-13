"""Bounded offline coordinated-leg target search; no deployment guarantee.

Used only after the unchanged one-joint inward search fails. All23 joints at
all ten2ms substeps must pass the same physical-range reserve. The search may
be slow or fail and is explicitly not a live50Hz safety controller.
"""

import time

import numpy as np
from scipy.optimize import minimize

from gear_sonic.utils.g1_23dof_safe_target_transform import (
    SAFE_TARGET_HARD_LOWER_HARDWARE,
    SAFE_TARGET_HARD_UPPER_HARDWARE,
    safe_target_transform_numpy,
)
from gear_sonic.utils.g1_true23_range_preview import Native23RangePreview, RANGE_RESERVE_RAD, target_to_raw

INWARD_FAILURE = "no verified next-control joint-range target found by bounded inward search"
MAX_PREDICTIONS = 512


def choose_coordinated_target(requested_raw, predict):
    raw = np.asarray(requested_raw)
    if raw.shape != (23,) or raw.dtype != np.float32 or not np.isfinite(raw).all() or np.any(np.abs(raw) >= 10):
        raise ValueError("coordinated preview requires bounded finite float32 native action23")
    _, requested = safe_target_transform_numpy(raw)
    _, target_low = safe_target_transform_numpy(np.full(23, -9.99, np.float32))
    _, target_high = safe_target_transform_numpy(np.full(23, 9.99, np.float32))
    target_low, target_high = target_low.astype(float) + 1e-5, target_high.astype(float) - 1e-5
    hard_low = np.asarray(SAFE_TARGET_HARD_LOWER_HARDWARE) + RANGE_RESERVE_RAD
    hard_high = np.asarray(SAFE_TARGET_HARD_UPPER_HARDWARE) - RANGE_RESERVE_RAD
    span = target_high[:13] - target_low[:13]
    cache, candidates = {}, []

    def evaluate(target, original_raw=None):
        if original_raw is None:
            candidate_raw, decoded = target_to_raw(target)
        else:
            candidate_raw, decoded = original_raw.copy(), requested.copy()
        key = decoded.tobytes()
        if key not in cache:
            if len(cache) >= MAX_PREDICTIONS:
                raise RuntimeError("coordinated range prediction budget exhausted")
            poses = np.asarray(predict(decoded.copy()))
            if poses.shape != (10, 30) or not np.isfinite(poses).all():
                raise ValueError("coordinated callback requires finite ten native23 qpos")
            margins = np.r_[np.min(poses[:, 7:] - hard_low, axis=0), np.min(hard_high - poses[:, 7:], axis=0)]
            row = dict(
                raw=candidate_raw.copy(),
                target=decoded.copy(),
                predicted=poses.copy(),
                margins=margins,
                maximum_violation_rad=float(max(0, -margins.min())),
                normalized_squared_target_change=float(np.sum(((decoded[:13] - requested[:13]) / span) ** 2)),
            )
            cache[key] = row
            candidates.append(row)
        return cache[key]

    first = evaluate(requested, raw)
    if first["maximum_violation_rad"] == 0:
        return first, dict(intervened=False, coordinated_predictions=1, changed_joints=[], optimizer_success=None)
    # Deterministic physical envelope probes escape saturated-PD flat gradients.
    # Only leg12/waist-yaw targets vary; arm10 targets remain exactly requested.
    for j in range(13):
        for endpoint in (target_low[j], target_high[j]):
            target = requested.astype(float).copy()
            target[j] = endpoint
            evaluate(target)
    seed = min(candidates, key=lambda r: (r["maximum_violation_rad"], r["normalized_squared_target_change"]))

    def whole_target(x):
        target = requested.astype(float).copy()
        target[:13] = target_low[:13] + np.clip(x, 0, 1) * span
        return target

    nominal_x = (requested[:13] - target_low[:13]) / span
    seed_x = (seed["target"][:13] - target_low[:13]) / span
    optimizer_success, optimizer_message = False, "not completed"
    try:
        solved = minimize(
            lambda x: float(np.sum((x - nominal_x) ** 2)),
            seed_x,
            method="SLSQP",
            bounds=[(0, 1)] * 13,
            constraints=[dict(type="ineq", fun=lambda x: evaluate(whole_target(x))["margins"])],
            options=dict(maxiter=25, ftol=1e-10, eps=1e-4),
        )
        evaluate(whole_target(solved.x))
        optimizer_success, optimizer_message = bool(solved.success), str(solved.message)
    except RuntimeError as error:
        if str(error) != "coordinated range prediction budget exhausted":
            raise
        optimizer_message = str(error)
    feasible = [r for r in candidates if r["maximum_violation_rad"] == 0]
    details = dict(
        intervened=True,
        coordinated_predictions=len(cache),
        optimizer_success=optimizer_success,
        optimizer_message=optimizer_message,
        maximum_predictions=MAX_PREDICTIONS,
        best_unverified_violation_rad=min(r["maximum_violation_rad"] for r in candidates),
    )
    if not feasible:
        error = ValueError("no verified coordinated-leg next-control target found")
        error.coordinated_details = details
        raise error
    chosen = min(feasible, key=lambda r: r["normalized_squared_target_change"])
    details.update(
        changed_joints=np.flatnonzero(np.abs(chosen["target"] - requested) > 5e-7).tolist(),
        maximum_target_change_rad=float(np.max(np.abs(chosen["target"] - requested))),
    )
    return chosen, details


class CoordinatedNative23RangePreview(Native23RangePreview):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.coordinated_records = []
        self.failed_search = None

    def filter(self, raw, measured_qpos, measured_qvel):
        if self.failed_search is not None:
            raise RuntimeError("failed coordinated preview cannot restart implicitly")
        try:
            return super().filter(raw, measured_qpos, measured_qvel)
        except ValueError as error:
            if str(error) != INWARD_FAILURE:
                raise
        q, v = measured_qpos.copy(), measured_qvel.copy()
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
            for _ in range(10):
                request = c.physics.kp * (target.astype(float) - c.data.qpos[7:]) - c.physics.kd * c.data.qvel[6:]
                c.data.ctrl[:] = np.clip(request, -c.physics.effort, c.physics.effort)
                c.module.mj_step(c.model, c.data)
                poses.append(c.data.qpos.copy())
            return np.asarray(poses)

        try:
            chosen, details = choose_coordinated_target(raw, predict)
        except Exception as error:
            self.failed_search = dict(message=str(error), details=getattr(error, "coordinated_details", None))
            raise
        self.coordinated_records.append(dict(**details, elapsed_s=time.perf_counter() - started))
        return chosen["raw"].copy()

    def contract(self):
        result = super().contract()
        result.update(
            kind="native23_offline_inward_then_coordinated_leg_range_preview_v1",
            original_inward_contract=super().contract(),
            coordinated_records=self.coordinated_records,
            coordinated_failure=self.failed_search,
            arms_changed_by_coordinated_target_search=False,
            maximum_coordinated_predictions=MAX_PREDICTIONS,
            closed_loop_native_hardware_or_realtime_qualified=False,
            preview_duration_change=False,
            joint_range_reserve_change=False,
        )
        return result
