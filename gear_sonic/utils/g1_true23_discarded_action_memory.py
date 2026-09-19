"""Offline hypothesis: retain discarded source actions as internal memory.

Only the six previous-action channels change. Missing position/velocity slots
stay zero, actual retained history still describes applied physical targets,
and the existing source-target codec and native range preview stay unchanged.
Virtual outputs are NOT measured states, applied torques, or physical commands.
This is a separately identified runtime experiment, not a deployment adapter.
"""

from collections import deque

import numpy as np
import torch

from gear_sonic.utils.g1_23dof_contract import (
    NATIVE_IL23_TO_CANONICAL_IL29,
    SOURCE_IL29_EXCLUDED_INDICES,
)
from gear_sonic.utils.g1_true23_generalist_corpus import canonical_digest
from gear_sonic.utils.g1_true23_released_core_comparison import ReleasedCoreAdapter, ReleasedCorePolicy
from gear_sonic.utils.g1_true23_source_action_codec import source_action_history_numpy

MISSING = np.asarray(SOURCE_IL29_EXCLUDED_INDICES)
KEEP = np.asarray(NATIVE_IL23_TO_CANONICAL_IL29)
ACTION_SLOTS = 610 + np.arange(10)[:, None] * 29 + MISSING[None, :]


def validate_history(history, *, require_zero_action_slots):
    if not isinstance(history, np.ndarray) or history.shape != (930,) or history.dtype != np.float32:
        raise ValueError("requires float32 history930")
    if not np.isfinite(history).all():
        raise ValueError("history930 must be finite")
    starts = (30, 320, 610) if require_zero_action_slots else (30, 320)
    for start in starts:
        if np.any(history[start : start + 290].reshape(10, 29)[:, MISSING]):
            raise ValueError("missing measured joint slots must remain zero")


class DiscardedActionMemory:
    """Ten previous source proposals, advanced only after accepted control."""

    def __init__(self):
        self._rows = deque((np.zeros(6, np.float32) for _ in range(10)), maxlen=10)
        self._pending = False
        self._failed = False
        self.accepted_controls = 0

    def encode(self, native_history):
        if self._failed or self._pending:
            raise RuntimeError("discarded-action memory requires sequential accepted controls")
        validate_history(native_history, require_zero_action_slots=True)
        result = source_action_history_numpy(native_history)
        result[ACTION_SLOTS] = np.asarray(self._rows)
        self._pending = True
        return result

    def accept(self, raw29):
        if self._failed or not self._pending:
            raise RuntimeError("discarded-action memory has no pending accepted control")
        if (
            not isinstance(raw29, np.ndarray)
            or raw29.shape != (29,)
            or raw29.dtype != np.float32
            or not np.isfinite(raw29).all()
        ):
            raise ValueError("virtual action memory requires finite float32 raw29")
        self._rows.append(raw29[MISSING].copy())
        self.accepted_controls += 1
        self._pending = False

    def fail(self):
        self._failed = True

    def contract(self):
        return dict(
            kind="discarded_source_action_memory_h10_v1",
            source_indices=MISSING.tolist(),
            retained_previous_actions="applied_native_target_in_released_source_units",
            discarded_previous_actions="unapplied_raw_source_outputs_internal_memory_only",
            absent_measured_position_and_velocity="fixed_zero_no_invented_measurements",
            initial_virtual_history="ten_zero_frames",
            samples=10,
            accepted_controls=self.accepted_controls,
            virtual_outputs_sent_to_physical_motors=False,
            can_reset_during_trial=False,
            failed=self._failed,
            hardware_authorized=False,
            deployment_ready=False,
        )


class DiscardedActionMemoryPolicy(ReleasedCorePolicy):
    """Same frozen tensors, allowing only explicitly virtual action slots."""

    def infer(self, encoder267, history930):
        validate_history(history930, require_zero_action_slots=False)
        if (
            not isinstance(encoder267, np.ndarray)
            or encoder267.shape != (267,)
            or encoder267.dtype != np.float32
            or not np.isfinite(encoder267).all()
        ):
            raise ValueError("requires finite float32 encoder267")
        with torch.inference_mode():
            token = self.core.encode(torch.from_numpy(encoder267[None]))
            decoder = torch.cat((token, torch.from_numpy(history930[None])), dim=-1)
            raw29 = self.core.decoder(decoder)
        if raw29.shape != (1, 29) or raw29.dtype != torch.float32 or not torch.isfinite(raw29).all():
            raise ValueError("released core emitted invalid raw29")
        self.latest_raw29 = raw29[0].numpy().copy()
        return self.latest_raw29[KEEP].copy(), decoder[0].numpy().copy()

    def identity(self):
        return dict(
            kind="untouched_normal_sonic_with_virtual_discarded_action_memory_v1",
            frozen_parent=super().identity(),
            measured_proprioception_changed=False,
            action_history_contract="discarded_source_action_memory_h10_v1",
            training_updates=0,
            hardware_authorized=False,
            deployment_ready=False,
        )


class DiscardedActionMemoryAdapter(ReleasedCoreAdapter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.memory = DiscardedActionMemory()

    def transform_history(self, history):
        return self.memory.encode(history)

    def infer(self, policy, encoder, history, *, control_index, **state):
        if not isinstance(policy, DiscardedActionMemoryPolicy):
            raise ValueError("virtual-memory adapter requires its explicit policy contract")
        previous_count = len(self.attempts)
        try:
            result = super().infer(policy, encoder, history, control_index=control_index, **state)
        except Exception:
            self.memory.fail()
            raise
        finally:
            if len(self.attempts) > previous_count:
                self.attempts[-1]["released_raw29"] = policy.latest_raw29.copy()
        self.memory.accept(policy.latest_raw29)
        return result

    def contract(self):
        result = super().contract()
        original_codec = result["source_action_codec"]
        codec = dict(
            kind="released29_retained_applied_plus_discarded_raw_history_v1",
            retained_action_codec=original_codec,
            previous_action="retained23_applied_source_equivalents_plus_six_unapplied_raw_proposals",
            discarded_action_memory=self.memory.contract(),
            native_physics_and_limits_unchanged=True,
            hardware_authorized=False,
            deployment_ready=False,
        )
        result["source_action_codec"] = {**codec, "contract_sha256": canonical_digest(codec)}
        result["kind"] = "native23_released_core_discarded_action_memory_v1"
        return result
