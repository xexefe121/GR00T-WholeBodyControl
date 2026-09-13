"""SIM-only current-measurement update before frozen SONIC inference.

The earlier momentum observer deliberately updated after inference. This
version exposes its updated missing velocities in the CURRENT history row,
without changing real23 measurements, older history, physical state or limits.
"""

import numpy as np

from gear_sonic.utils.g1_true23_momentum_source_model import (
    MISSING_V,
    MomentumSourceModel,
    validate_native_state,
)
from gear_sonic.utils.g1_true23_virtual_source_history import VirtualSourceHistoryAdapter
from gear_sonic.utils.g1_true23_virtual_source_model import KEEP_HW, VirtualSourceModel


class CurrentMomentumSourceModel(MomentumSourceModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prepared = False
        self.prepared_q = self.prepared_v = None

    def prepare_current(self, measured_qpos, measured_qvel):
        if self.failed or self.prepared:
            raise RuntimeError("current observer requires one fresh measured-state update per control")
        try:
            validate_native_state(measured_qpos, measured_qvel, np.zeros(29, np.float32))
            if self.initialized:
                velocity, record = self.assimilator.proposal(
                    self.data.qpos.copy(), self.data.qvel.copy(), measured_qpos, measured_qvel
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
            # Only the separate internal source model is written.
            self.data.qpos[:7] = measured_qpos[:7]
            self.data.qpos[7 + KEEP_HW] = measured_qpos[7:]
            self.data.qvel[:6] = measured_qvel[:6]
            self.data.qvel[6 + KEEP_HW] = measured_qvel[6:]
            self.prepared_q, self.prepared_v = measured_qpos.copy(), measured_qvel.copy()
            self.assimilation_records.append(record)
            self.prepared = True
            return self.internal_state12()
        except Exception:
            self.failed = True
            raise

    def advance(self, measured_qpos23, measured_qvel23, source_raw29):
        if self.failed:
            raise RuntimeError("failed current observer cannot restart implicitly")
        try:
            validate_native_state(measured_qpos23, measured_qvel23, source_raw29)
            # Source-only verification can invoke advance without a policy hook.
            if not self.prepared:
                self.prepare_current(measured_qpos23, measured_qvel23)
            if not np.array_equal(measured_qpos23, self.prepared_q) or not np.array_equal(
                measured_qvel23, self.prepared_v
            ):
                raise ValueError("current observer inference and integration measurements differ")
            result = VirtualSourceModel.advance(self, measured_qpos23, measured_qvel23, source_raw29)
            self.prepared = False
            self.prepared_q = self.prepared_v = None
            return result
        except Exception:
            self.failed = True
            raise

    def descriptor(self):
        result = super().descriptor()
        result.update(
            kind="internal_source29_current_momentum_observer_v1",
            update_phase="before_current_policy_inference",
            current_policy_virtual_feedback="current_measured_state_assimilated_prediction",
            pending_measurement_update=self.prepared,
            observation_history_rewritten="latest_virtual_q_dq_row_only_not_past_or_native_rows",
        )
        return result


class CurrentMomentumHistoryAdapter(VirtualSourceHistoryAdapter):
    def __init__(self, *args, predictor, **kwargs):
        if not isinstance(predictor, CurrentMomentumSourceModel):
            raise ValueError("current history requires current-momentum source model")
        super().__init__(*args, predictor=predictor, **kwargs)
        self.current_virtual_state = None

    def transform_history(self, history):
        raise RuntimeError("current observer requires explicit copied current measurements")

    def transform_measured_history(self, history, *, measured_qpos, measured_qvel, control_index):
        if self.memory.failed or self.memory.pending or control_index != self.memory.accepted:
            raise RuntimeError("current observer history cannot repeat or skip a control")
        try:
            state = self.predictor.prepare_current(measured_qpos, measured_qvel)
            current = self.memory.rows[-1].copy()
            current[:12] = state
            self.memory.rows[-1] = current
            self.current_virtual_state = state.copy()
            return self.memory.encode(history)
        except Exception:
            self.memory.fail()
            self.predictor.failed = True
            raise

    def infer(self, *args, **kwargs):
        previous = len(self.attempts)
        try:
            return super().infer(*args, **kwargs)
        finally:
            if len(self.attempts) > previous:
                self.attempts[-1]["current_assimilated_missing_state12"] = self.current_virtual_state.copy()

    def contract(self):
        result = super().contract()
        result.update(
            kind="native23_released_core_current_momentum_observer_v1",
            measured_history_hook_required=True,
            native_latest_measurements_altered=False,
            missing_model_velocity_assimilated_before_inference=True,
            past_virtual_history_rewritten=False,
        )
        return result
