"""Offline one-control joint-range preview, not a hardware safety guarantee.

Consumes copied measured state only. The independent nominal model predicts all
ten 2 ms substeps. No source sample, motor limit, gain or real plant state is
changed. A bounded search may fail; its failure never authorizes unsafe output.
"""

from pathlib import Path
import time

import numpy as np

from gear_sonic.utils.g1_23dof_contract import HARDWARE_23_ACTION_SCALE, MUJOCO_TO_ISAACLAB_DOF
from gear_sonic.utils.g1_23dof_safe_target_transform import (
    SAFE_TARGET_DEFAULT_Q_HARDWARE,
    SAFE_TARGET_HARD_LOWER_HARDWARE,
    SAFE_TARGET_HARD_UPPER_HARDWARE,
    SAFE_TARGET_NEGATIVE_CAPACITY_HARDWARE,
    SAFE_TARGET_POSITIVE_CAPACITY_HARDWARE,
    safe_target_transform_numpy,
)

RANGE_RESERVE_RAD = 0.0019


def target_to_raw(target):
    target = np.asarray(target)
    if target.shape != (23,) or target.dtype.kind not in "fi" or not np.isfinite(target).all():
        raise ValueError("target inversion requires finite hardware23")
    default = np.asarray(SAFE_TARGET_DEFAULT_Q_HARDWARE, dtype=np.float32).astype(np.float64)
    delta = target.astype(np.float64) - default
    capacity = np.where(
        delta >= 0,
        np.asarray(SAFE_TARGET_POSITIVE_CAPACITY_HARDWARE, dtype=np.float32),
        np.asarray(SAFE_TARGET_NEGATIVE_CAPACITY_HARDWARE, dtype=np.float32),
    ).astype(np.float64)
    scale = np.asarray(HARDWARE_23_ACTION_SCALE, dtype=np.float32).astype(np.float64)
    ratio = delta / capacity
    if np.any(np.abs(ratio) >= 1):
        raise ValueError("target is outside invertible unchanged envelope")
    raw = (capacity * np.arctanh(ratio) / scale)[list(MUJOCO_TO_ISAACLAB_DOF)].astype(np.float32)
    if not np.isfinite(raw).all() or np.any(np.abs(raw) >= 10):
        raise ValueError("target inversion exceeds unchanged raw-action bound")
    _, decoded = safe_target_transform_numpy(raw)
    if np.max(np.abs(decoded.astype(np.float64) - target)) > 5e-7:
        raise ValueError("target inversion failed physical round-trip")
    return raw, decoded


def choose_range_target(requested_raw, predict):
    """Find a checked inward correction; no claim of global/recursive feasibility.

    Callback consumes a decoded hardware target and returns ten predicted qpos.
    Unchanged-safe requests retain the exact original action bytes. Inward search
    frees only coordinates violating predicted limits, checking every joint at
    every substep for every candidate. Failed search raises, never falls through.
    """
    requested_raw = np.asarray(requested_raw)
    if requested_raw.shape != (23,) or requested_raw.dtype != np.float32 or not np.isfinite(requested_raw).all():
        raise ValueError("range preview requires finite float32 native action23")
    if np.any(np.abs(requested_raw) >= 10):
        raise ValueError("range preview rejects raw action bound violations")
    hard_low = np.asarray(SAFE_TARGET_HARD_LOWER_HARDWARE) + RANGE_RESERVE_RAD
    hard_high = np.asarray(SAFE_TARGET_HARD_UPPER_HARDWARE) - RANGE_RESERVE_RAD
    _, requested = safe_target_transform_numpy(requested_raw)
    _, low_target = safe_target_transform_numpy(np.full(23, -9.99, dtype=np.float32))
    _, high_target = safe_target_transform_numpy(np.full(23, 9.99, dtype=np.float32))
    # Stay strictly inside the round-trip envelope; these are targets, not gains.
    low_target = low_target.astype(np.float64) + 1e-5
    high_target = high_target.astype(np.float64) - 1e-5
    calls = 0

    def check(raw, target):
        nonlocal calls
        calls += 1
        predicted = np.asarray(predict(target.copy()))
        if predicted.shape != (10, 30) or not np.isfinite(predicted).all():
            raise ValueError("preview must return finite ten native23 qpos samples")
        lower = np.maximum(hard_low - predicted[:, 7:], 0).max(0)
        upper = np.maximum(predicted[:, 7:] - hard_high, 0).max(0)
        return dict(raw=raw, target=target, predicted=predicted, lower=lower, upper=upper)

    def good(row):
        return not np.any(row["lower"] > 0) and not np.any(row["upper"] > 0)

    nominal = check(requested_raw.copy(), requested.copy())
    if good(nominal):
        return nominal, dict(intervened=False, preview_calls=calls, changed_joints=[], alpha=0.0)
    if np.any((nominal["lower"] > 0) & (nominal["upper"] > 0)):
        raise ValueError("preview crosses both bounds of a joint; no bounded inward correction")
    endpoint = requested.astype(np.float64).copy()
    endpoint[nominal["lower"] > 0] = high_target[nominal["lower"] > 0]
    endpoint[nominal["upper"] > 0] = low_target[nominal["upper"] > 0]
    previous_alpha, accepted, accepted_alpha = 0.0, None, None
    for alpha in (1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0):
        raw, target = target_to_raw(requested + alpha * (endpoint - requested))
        candidate = check(raw, target)
        if good(candidate):
            accepted, accepted_alpha = candidate, alpha
            break
        previous_alpha = alpha
    if accepted is None:
        raise ValueError("no verified next-control joint-range target found by bounded inward search")
    # Maintain a checked feasible endpoint; no assumption that dynamics are
    # globally monotone. This only refines the first found local bracket.
    for _ in range(8):
        alpha = (previous_alpha + accepted_alpha) / 2
        raw, target = target_to_raw(requested + alpha * (endpoint - requested))
        candidate = check(raw, target)
        if good(candidate):
            accepted, accepted_alpha = candidate, alpha
        else:
            previous_alpha = alpha
    return accepted, dict(
        intervened=True,
        preview_calls=calls,
        changed_joints=np.flatnonzero(np.abs(accepted["target"] - requested) > 5e-7).tolist(),
        alpha=accepted_alpha,
        nominal_maximum_reserve_violation_rad=float(max(nominal["lower"].max(), nominal["upper"].max())),
        maximum_target_change_rad=float(np.max(np.abs(accepted["target"] - requested))),
    )


class Native23RangePreview:
    """Independent probe state, no access to the real integrator or warmstart."""

    def __init__(self, *, model_path: Path, physics_path: Path):
        from gear_sonic.utils.g1_true23_clean_mujoco_teleop import CleanTrue23MujocoController
        from gear_sonic.utils.g1_true23_reference_floor import compiled_model_sha256

        self.probe = CleanTrue23MujocoController(model_path=model_path, physics_path=physics_path, policy=None)
        self.model_sha256 = compiled_model_sha256(self.probe.model)
        self.records = []
        self.predictions = []

    def filter(self, raw, measured_qpos, measured_qvel):
        q, v = np.asarray(measured_qpos).copy(), np.asarray(measured_qvel).copy()
        if q.shape != (30,) or v.shape != (29,) or not np.isfinite(q).all() or not np.isfinite(v).all():
            raise ValueError("range preview requires copied finite native23 measured state")
        if np.any(q[7:] < SAFE_TARGET_HARD_LOWER_HARDWARE) or np.any(q[7:] > SAFE_TARGET_HARD_UPPER_HARDWARE):
            raise ValueError("range preview cannot accept an already violated physical joint range")
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
                requested = (
                    c.physics.kp * (target.astype(np.float64) - c.data.qpos[7:]) - c.physics.kd * c.data.qvel[6:]
                )
                c.data.ctrl[:] = np.clip(requested, -c.physics.effort, c.physics.effort)
                c.module.mj_step(c.model, c.data)
                poses.append(c.data.qpos.copy())
            return np.asarray(poses)

        row, details = choose_range_target(raw, predict)
        self.records.append(dict(**details, elapsed_s=time.perf_counter() - started))
        self.predictions.append(row["predicted"])
        return row["raw"].copy()

    def contract(self):
        return dict(
            kind="native23_offline_ten_substep_joint_range_preview_v1",
            compiled_model_sha256=self.model_sha256,
            reserve_rad=RANGE_RESERVE_RAD,
            predicted_physics_substeps=10,
            source_lookahead_added=False,
            measured_state="copied_qpos_qvel_with_independent_zero_warmstart",
            prediction_is_actual_hardware_state=False,
            limits_or_gains_changed=False,
            actual_previous_action_tracks_corrected_target=True,
            target_interval_changed=False,
            checked_controls=len(self.records),
            interventions=sum(r["intervened"] for r in self.records),
            maximum_filter_elapsed_s=max((r["elapsed_s"] for r in self.records), default=0),
            real_time_or_recursive_safety_proven=False,
            failure_fallback="none_reject_trial",
            hardware_authorized=False,
            deployment_ready=False,
        )
