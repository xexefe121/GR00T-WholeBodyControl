"""Versioned SIM-only ankle torque fallback within original total effort limits.

Position-only preview remains first choice and retains its original behavior.
Only after its bounded search fails, test up to three inward ankle torque
corrections at the unchanged nominal position target. No unchecked fallback.
"""

import time

import numpy as np

from gear_sonic.utils.g1_23dof_safe_target_transform import safe_target_transform_numpy
from gear_sonic.utils.g1_true23_native_model_actuation import NativeModelActuationProfile, native_model_pd_numpy
from gear_sonic.utils.g1_true23_range_preview import Native23RangePreview, RANGE_RESERVE_RAD

ANKLES = (4, 5, 10, 11)
CORRECTIONS_NM = (2.5, 5.0, 10.0)
INWARD_FAILURE = "no verified next-control joint-range target found by bounded inward search"
FAILURE = "no verified bounded ankle feedforward target after position-only range rejection"


def validate_feedforward(feedforward):
    value = np.asarray(feedforward)
    if value.shape != (23,) or value.dtype != np.float64 or not np.isfinite(value).all():
        raise ValueError("feedforward requires finite float64 hardware23")
    other = np.ones(23, bool)
    other[list(ANKLES)] = False
    if np.any(value[other] != 0) or np.any(np.abs(value) > CORRECTIONS_NM[-1]):
        raise ValueError("feedforward is limited to four ankle axes and10Nm")
    return value.copy()


def native_pd_feedforward_numpy(target, q, dq, profile, feedforward):
    """Clip the TOTAL actuator request, never add torque after saturation."""
    correction = validate_feedforward(feedforward)
    original = native_model_pd_numpy(target, q, dq, profile)
    if not np.any(correction):
        return original
    pd, _, invalid, _ = original
    if pd.shape != (23,):
        raise ValueError("feedforward diagnostic expects one unbatched native actuator state")
    requested = pd + correction
    invalid = invalid | ~np.isfinite(requested).all()
    effort = np.asarray(profile.effort)
    applied = np.where(invalid, 0.0, np.clip(requested, -effort, effort))
    excess = np.maximum(np.abs(requested) / effort - 1.0, 0.0)
    cost = np.square(np.nan_to_num(excess, nan=10, posinf=10, neginf=10).clip(0, 10)).mean()
    return requested, applied, invalid, cost


def choose_ankle_feedforward(target, predict, lower, upper, velocity_limits):
    """All10 substeps/all23 joints checked for each candidate, including coupling."""
    target, lower, upper, velocity_limits = (np.asarray(x) for x in (target, lower, upper, velocity_limits))
    if any(x.shape != (23,) or not np.isfinite(x).all() for x in (target, lower, upper, velocity_limits)):
        raise ValueError("feedforward search requires finite hardware23 vectors")
    if np.any(lower >= upper) or np.any(velocity_limits <= 0):
        raise ValueError("feedforward search requires ordered ranges and positive velocity limits")
    low, high = lower + RANGE_RESERVE_RAD, upper - RANGE_RESERVE_RAD
    attempts = []

    def check(feedforward):
        correction = validate_feedforward(feedforward)
        q, v = (np.asarray(x) for x in predict(target.copy(), correction.copy()))
        if q.shape != (10, 30) or v.shape != (10, 29) or not np.isfinite(q).all() or not np.isfinite(v).all():
            raise ValueError("feedforward predictor requires finite ten native23 qpos/qvel states")
        below, above = np.maximum(low - q[:, 7:], 0).max(0), np.maximum(q[:, 7:] - high, 0).max(0)
        row = dict(
            feedforward23=correction,
            predicted=q.copy(),
            predicted_velocity=v.copy(),
            lower=below,
            upper=above,
            range_excess_rad=float(max(below.max(), above.max())),
            velocity_ratio=float((np.abs(v[:, 6:]) / velocity_limits).max()),
        )
        attempts.append(row)
        return row

    nominal = check(np.zeros(23, np.float64))
    details = dict(
        position_only_search_failed=True,
        nominal_range_excess_rad=nominal["range_excess_rad"],
        nominal_velocity_ratio=nominal["velocity_ratio"],
    )

    def fail(reason):
        error = ValueError(FAILURE)
        error.feedforward_details = dict(
            **details,
            reason=reason,
            predictions=len(attempts),
            candidate_range_excess_rad=[row["range_excess_rad"] for row in attempts],
            candidate_velocity_ratios=[row["velocity_ratio"] for row in attempts],
        )
        raise error

    if nominal["range_excess_rad"] == 0:
        fail("position-only failure did not reproduce")
    violating = (nominal["lower"] > 0) | (nominal["upper"] > 0)
    if any(j not in ANKLES for j in np.flatnonzero(violating)):
        fail("nominal range violation includes a non-ankle joint")
    if np.any((nominal["lower"] > 0) & (nominal["upper"] > 0)):
        fail("nominal prediction crosses both bounds of an ankle")
    direction = np.zeros(23, np.float64)
    direction[nominal["lower"] > 0], direction[nominal["upper"] > 0] = 1, -1
    for magnitude in CORRECTIONS_NM:
        candidate = check(direction * magnitude)
        if candidate["range_excess_rad"] == 0 and candidate["velocity_ratio"] <= 1:
            return candidate, dict(
                **details,
                predictions=len(attempts),
                magnitude_nm=magnitude,
                changed_torque_joints=np.flatnonzero(direction).tolist(),
            )
    fail("bounded torque corrections exhausted")


class AnkleFeedforwardRangePreview:
    """No actual integrator access; failed search latches until explicit recreation."""

    def __init__(self, *, model_path, physics_path):
        self.position_preview = Native23RangePreview(model_path=model_path, physics_path=physics_path)
        self.profile = NativeModelActuationProfile.from_sim_config(physics_path)
        self.records, self.predictions, self.feedforwards = [], [], []
        self.failed_search = None
        self.feedforward23 = None

    def filter(self, raw, measured_qpos, measured_qvel):
        if self.failed_search is not None:
            raise RuntimeError("failed ankle feedforward cannot restart implicitly")
        self.feedforward23 = None
        raw = np.asarray(raw).copy()
        q, v = np.asarray(measured_qpos).copy(), np.asarray(measured_qvel).copy()
        started = time.perf_counter()
        try:
            try:
                safe = self.position_preview.filter(raw, q, v)
            except ValueError as error:
                if str(error) != INWARD_FAILURE:
                    raise
            else:
                self.feedforward23 = np.zeros(23, np.float64)
                self.records.append(
                    dict(
                        kind="original_position_only_preview",
                        feedforward_used=False,
                        position_preview=self.position_preview.records[-1],
                        elapsed_s=time.perf_counter() - started,
                    )
                )
                self.predictions.append(self.position_preview.predictions[-1].copy())
                self.feedforwards.append(self.feedforward23.copy())
                return safe
            c = self.position_preview.probe
            _, target = safe_target_transform_numpy(raw)

            def predict(position_target, correction):
                c.module.mj_resetData(c.model, c.data)
                c.reset(
                    base_position=q[:3],
                    base_quaternion_wxyz=q[3:7],
                    joint_position_hardware=q[7:],
                    root_velocity=v[:6],
                    joint_velocity_hardware=v[6:],
                )
                poses, velocities = [], []
                for _ in range(10):
                    _, applied, invalid, _ = native_pd_feedforward_numpy(
                        position_target.astype(float), c.data.qpos[7:], c.data.qvel[6:], self.profile, correction
                    )
                    if invalid:
                        raise ValueError("invalid total actuator request during ankle preview")
                    c.data.ctrl[:] = applied
                    c.module.mj_step(c.model, c.data)
                    poses.append(c.data.qpos.copy())
                    velocities.append(c.data.qvel.copy())
                return np.asarray(poses), np.asarray(velocities)

            chosen, details = choose_ankle_feedforward(
                target,
                predict,
                c.model.jnt_range[1:, 0],
                c.model.jnt_range[1:, 1],
                np.asarray(self.profile.velocity),
            )
            self.feedforward23 = chosen["feedforward23"].copy()
            self.records.append(
                dict(
                    kind="verified_bounded_ankle_feedforward",
                    feedforward_used=True,
                    details=details,
                    elapsed_s=time.perf_counter() - started,
                )
            )
            self.predictions.append(chosen["predicted"].copy())
            self.feedforwards.append(self.feedforward23.copy())
            return raw.copy()
        except Exception as error:
            self.failed_search = dict(
                message=str(error),
                details=getattr(error, "feedforward_details", None),
                elapsed_s=time.perf_counter() - started,
            )
            raise

    def contract(self):
        return dict(
            kind="native23_offline_position_preview_then_bounded_ankle_feedforward_v1",
            original_position_preview=self.position_preview.contract(),
            checked_controls=len(self.records),
            feedforward_controls=sum(row["feedforward_used"] for row in self.records),
            maximum_correction_per_ankle_nm=10.0,
            correction_candidates_nm=list(CORRECTIONS_NM),
            total_actuator_effort_limits_unchanged=True,
            target_limits_or_gains_changed=False,
            all23_range_substeps_checked=10,
            reserve_rad=RANGE_RESERVE_RAD,
            no_unchecked_output=True,
            failed_search=self.failed_search,
            maximum_filter_elapsed_s=max((row["elapsed_s"] for row in self.records), default=0),
            previous_action_semantics="commanded bounded position target; extra actuator torque logged separately",
            feedforward_exposed_to_existing_policy_as_extra_observation=False,
            full_motion_realtime_recursive_safety_or_hardware_qualified=False,
            deployment_ready=False,
            hardware_authorized=False,
        )
