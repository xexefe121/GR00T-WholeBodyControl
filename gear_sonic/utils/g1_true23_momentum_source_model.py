"""SIM-only momentum-consistent assimilation into a hypothetical source model.

No physical controller, motor limit, measured native state, or missing position
is changed. A separate internal observer updates its missing velocities so the
measurement overwrite does not silently impart generalized momentum to absent
axes. This is a model hypothesis, not an identified hardware state estimator.
"""

import mujoco
import numpy as np

from gear_sonic.utils.g1_true23_virtual_source_model import KEEP_HW, MISSING_HW, VirtualSourceModel

MISSING_V = 6 + MISSING_HW
OBSERVED_V = np.r_[np.arange(6), 6 + KEEP_HW]


def validate_native_state(qpos, qvel, raw):
    for value, size in ((qpos, 30), (qvel, 29), (raw, 29)):
        if not isinstance(value, np.ndarray) or value.shape != (size,) or not np.isfinite(value).all():
            raise ValueError("momentum assimilation requires finite native23 state and source29 proposal")
    if abs(np.linalg.norm(qpos[3:7]) - 1) > 1e-5:
        raise ValueError("momentum assimilation requires normalized measured root orientation")


class MissingMomentumAssimilation:
    """Read-only proposal; scratch FK/mass data are not the prediction MjData."""

    def __init__(self, model):
        if (model.nq, model.nv, model.nu) != (36, 35, 29):
            raise ValueError("momentum proposal requires the separate source29 model")
        self.model = model
        self.probe = mujoco.MjData(model)

    def mass(self, qpos):
        self.probe.qpos[:] = qpos
        mujoco.mj_fwdPosition(self.model, self.probe)
        matrix = np.empty((35, 35), np.float64)
        mujoco.mj_fullM(self.model, matrix, self.probe.qM)
        if not np.isfinite(matrix).all() or not np.allclose(matrix, matrix.T, rtol=0, atol=1e-12):
            raise RuntimeError("invalid source inertia matrix")
        return matrix

    def proposal(self, old_qpos, old_qvel, native_qpos, native_qvel):
        next_q, next_v = old_qpos.copy(), old_qvel.copy()
        next_q[:7], next_q[7 + KEEP_HW] = native_qpos[:7], native_qpos[7:]
        next_v[:6], next_v[6 + KEEP_HW] = native_qvel[:6], native_qvel[6:]
        if np.array_equal(next_q, old_qpos) and np.array_equal(next_v, old_qvel):
            return old_qvel[MISSING_V].copy(), dict(
                exact_identity=True,
                missing_momentum_jump=np.zeros(6),
                missing_velocity_correction=np.zeros(6),
                momentum_residual=np.zeros(6),
                conditional_correction_energy_j=0.0,
            )
        old_mass, new_mass = self.mass(old_qpos), self.mass(next_q)
        old_p = old_mass[MISSING_V] @ old_qvel
        observed_p = new_mass[np.ix_(MISSING_V, OBSERVED_V)] @ next_v[OBSERVED_V]
        block = new_mass[np.ix_(MISSING_V, MISSING_V)]
        np.linalg.cholesky(block)
        missing_v = np.linalg.solve(block, old_p - observed_p)
        delta = missing_v - old_qvel[MISSING_V]
        residual = block @ missing_v + observed_p - old_p
        if not np.isfinite(missing_v).all() or np.max(np.abs(residual)) > 1e-10:
            raise RuntimeError("missing-axis momentum assimilation residual invalid")
        return missing_v, dict(
            exact_identity=False,
            missing_momentum_jump=new_mass[MISSING_V] @ next_v - old_p,
            missing_velocity_correction=delta,
            momentum_residual=residual,
            conditional_correction_energy_j=float(0.5 * delta @ block @ delta),
        )


class MomentumSourceModel(VirtualSourceModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.assimilator = MissingMomentumAssimilation(self.model)
        self.assimilation_records = []

    def advance(self, measured_qpos23, measured_qvel23, source_raw29):
        if self.failed:
            raise RuntimeError("failed momentum source cannot restart implicitly")
        try:
            validate_native_state(measured_qpos23, measured_qvel23, source_raw29)
            if self.initialized:
                velocity, record = self.assimilator.proposal(
                    self.data.qpos.copy(), self.data.qvel.copy(), measured_qpos23, measured_qvel23
                )
                self.data.qvel[MISSING_V] = velocity
            else:
                record = dict(
                    exact_identity=True,
                    missing_momentum_jump=np.zeros(6),
                    missing_velocity_correction=np.zeros(6),
                    momentum_residual=np.zeros(6),
                    conditional_correction_energy_j=0.0,
                )
            self.assimilation_records.append(record)
            return super().advance(measured_qpos23, measured_qvel23, source_raw29)
        except Exception:
            self.failed = True
            raise

    def assimilation_arrays(self):
        keys = (
            "exact_identity",
            "missing_momentum_jump",
            "missing_velocity_correction",
            "momentum_residual",
            "conditional_correction_energy_j",
        )
        return {key: np.asarray([r[key] for r in self.assimilation_records]) for key in keys}

    def descriptor(self):
        result = super().descriptor()
        arrays = self.assimilation_arrays()
        result.update(
            kind="internal_source29_missing_momentum_preserving_update_v1",
            base_prediction_kind="internal_hypothetical_source29_mujoco_prediction_v1",
            update_phase="after_current_policy_inference_before_next_internal_prediction",
            current_policy_virtual_feedback="previous_prediction_not_post_assimilation_state",
            updated_coordinates="six_internal_missing_velocities_only",
            equation="M_new_mm*v_new_m = (M_old*v_old)_m - M_new_mo*v_measured_o",
            missing_positions_clamped_or_reset=False,
            native_gains_or_effort_caps_changed=False,
            internal_measurement_updates=len(self.assimilation_records),
            maximum_missing_velocity_correction_rad_s=float(np.abs(arrays["missing_velocity_correction"]).max())
            if self.assimilation_records
            else 0.0,
            maximum_momentum_residual=float(np.abs(arrays["momentum_residual"]).max())
            if self.assimilation_records
            else 0.0,
        )
        return result
