"""SIM-only causal internal-source history; never invented native sensors.

The physical controller still has exactly23 actuators. Its measured retained
state, applied-action history, source references and guards are unchanged.
Only the six explicitly hypothetical source axes use a separate model state.
"""

from collections import deque

import numpy as np
import torch

from gear_sonic.utils.g1_true23_discarded_action_memory import KEEP, MISSING, validate_history
from gear_sonic.utils.g1_true23_generalist_corpus import canonical_digest
from gear_sonic.utils.g1_true23_released_core_comparison import ReleasedCoreAdapter, ReleasedCorePolicy
from gear_sonic.utils.g1_true23_source_action_codec import source_action_history_numpy
from gear_sonic.utils.g1_true23_virtual_source_model import VirtualSourceModel


class VirtualSourceHistory:
    def __init__(self):
        self.rows = deque((np.zeros(18, np.float32) for _ in range(10)), maxlen=10)
        self.pending = False
        self.failed = False
        self.accepted = 0

    def encode(self, native_history):
        if self.failed or self.pending:
            raise RuntimeError("virtual source history requires sequential accepted controls")
        try:
            validate_history(native_history, require_zero_action_slots=True)
            result = source_action_history_numpy(native_history)
            rows = np.asarray(self.rows)
            for offset, start in enumerate((30, 320, 610)):
                result[start : start + 290].reshape(10, 29)[:, MISSING] = rows[:, offset * 6 : offset * 6 + 6]
        except Exception:
            self.fail()
            raise
        self.pending = True
        return result

    def accept(self, predicted_state12, raw29):
        if self.failed or not self.pending:
            raise RuntimeError("virtual source history has no pending control")
        for value, size in ((predicted_state12, 12), (raw29, 29)):
            if (
                not isinstance(value, np.ndarray)
                or value.shape != (size,)
                or value.dtype != np.float32
                or not np.isfinite(value).all()
            ):
                self.fail()
                raise ValueError("virtual source history requires finite float32 model state12 and raw29")
        self.rows.append(np.r_[predicted_state12, raw29[MISSING]].copy())
        self.accepted += 1
        self.pending = False

    def fail(self):
        self.failed = True

    def contract(self):
        return dict(
            kind="internal_model_source29_missing_q_dq_action_h10_v1",
            source_indices=MISSING.tolist(),
            missing_q_dq="internal_hypothetical_source_model_predictions_not_native_measurements",
            missing_action="previous_unapplied_source_proposal",
            retained_q_dq="unchanged_measured_native23",
            retained_action="applied_native23_target_in_source_action_units",
            initial_virtual_history="ten_zero_frames",
            accepted_forecasts=self.accepted,
            can_reset_during_trial=False,
            failed=self.failed,
            hardware_authorized=False,
            deployment_ready=False,
        )


class VirtualSourceHistoryPolicy(ReleasedCorePolicy):
    def infer(self, encoder267, history930):
        for value, size in ((encoder267, 267), (history930, 930)):
            if (
                not isinstance(value, np.ndarray)
                or value.shape != (size,)
                or value.dtype != np.float32
                or not np.isfinite(value).all()
            ):
                raise ValueError("internal-model policy requires finite float32 encoder267/history930")
        with torch.inference_mode():
            token = self.core.encode(torch.from_numpy(encoder267[None]))
            decoder = torch.cat((token, torch.from_numpy(history930[None])), dim=-1)
            raw29 = self.core.decoder(decoder)
        if raw29.shape != (1, 29) or raw29.dtype != torch.float32 or not torch.isfinite(raw29).all():
            raise ValueError("released source core emitted invalid raw29")
        self.latest_raw29 = raw29[0].numpy().copy()
        return self.latest_raw29[KEEP].copy(), decoder[0].numpy().copy()

    def identity(self):
        return dict(
            kind="untouched_normal_sonic_with_internal_source_model_history_v1",
            frozen_parent=super().identity(),
            retained_measured_proprioception_changed=False,
            missing_axes_are_explicit_model_predictions=True,
            history_contract="internal_model_source29_missing_q_dq_action_h10_v1",
            training_updates=0,
            hardware_authorized=False,
            deployment_ready=False,
        )


class VirtualSourceHistoryAdapter(ReleasedCoreAdapter):
    def __init__(self, *args, predictor, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(predictor, VirtualSourceModel) or predictor.completed or predictor.initialized:
            raise ValueError("internal-source adapter requires a fresh explicit source model")
        self.predictor = predictor
        self.memory = VirtualSourceHistory()

    def transform_history(self, history):
        return self.memory.encode(history)

    def infer(self, policy, encoder, history, *, control_index, **state):
        if not isinstance(policy, VirtualSourceHistoryPolicy):
            raise ValueError("internal-source adapter requires its explicit source policy")
        if self.memory.failed or not self.memory.pending or self.memory.accepted != control_index:
            raise RuntimeError("virtual source adapter cannot restart or skip a forecast")
        previous = len(self.attempts)
        try:
            result = super().infer(policy, encoder, history, control_index=control_index, **state)
            predicted = self.predictor.advance(state["measured_qpos"], state["measured_qvel"], policy.latest_raw29)
            if self.predictor.maximum_virtual_joint_excess > 1e-6:
                raise RuntimeError("internal source prediction crossed missing-joint range; candidate rejected")
            self.memory.accept(predicted, policy.latest_raw29)
            self.attempts[-1]["predicted_next_missing_state12"] = predicted.copy()
            self.attempts[-1]["virtual_forecast_accepted"] = True
            return result
        except Exception:
            self.memory.fail()
            self.predictor.failed = True
            raise
        finally:
            if len(self.attempts) > previous:
                self.attempts[-1]["released_raw29"] = policy.latest_raw29.copy()
                self.attempts[-1].setdefault("virtual_forecast_accepted", False)

    def contract(self):
        result = super().contract()
        codec = dict(
            kind="released29_retained_measured_plus_internal_predicted_history_v1",
            retained_action_codec=result["source_action_codec"],
            previous_action="retained23_applied_source_equivalents_plus_six_unapplied_source_proposals",
            virtual_history=self.memory.contract(),
            source_predictor=self.predictor.descriptor(),
            native_physics_and_limits_unchanged=True,
            native_state_writes_by_predictor=0,
            hardware_authorized=False,
            deployment_ready=False,
        )
        result["source_action_codec"] = {**codec, "contract_sha256": canonical_digest(codec)}
        result["kind"] = "native23_released_core_internal_source_model_v1"
        return result
