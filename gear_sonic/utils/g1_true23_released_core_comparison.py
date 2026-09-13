"""No-training, simulation-only comparison of the two complete SONIC cores.

No root-feedback head, transport, fallback, weight update or hardware export.
The complete released encoder/FSQ/decoder is retained. Only the six absent
action rows are discarded; missing proprioception slots remain fixed zero.
"""

from pathlib import Path

import numpy as np
import torch

from gear_sonic.teleop.buffered_source_horizon import ReceivedSourceHorizon, buffered_horizon_contract
from gear_sonic.teleop.normal_source_horizon import ReceivedNormalSourceHorizon, normal_horizon_contract
from gear_sonic.trl.mjlab.frozen_platform_lora_actor import FrozenPlatformTrue23Core
from gear_sonic.utils.g1_23dof_contract import (
    HARDWARE_23_JOINT_NAMES,
    REFERENCE_PROFILE_LOW_LATENCY,
    REFERENCE_PROFILE_NORMAL,
)
from gear_sonic.utils.g1_true23_buffered_reference import continued_standing_source
from gear_sonic.utils.g1_true23_generalist_benchmark import MODEL, PHYSICS
from gear_sonic.utils.g1_true23_range_preview import Native23RangePreview
from gear_sonic.utils.g1_true23_source_action_codec import (
    source_action_codec_contract,
    source_action_history_numpy,
    source_scaled_precompensation,
)

PROFILES = (REFERENCE_PROFILE_NORMAL, REFERENCE_PROFILE_LOW_LATENCY)


class ReleasedCorePolicy:
    def __init__(self, *, warm_start_path, source_checkpoint_path, profile):
        if profile not in PROFILES:
            raise ValueError("comparison requires an exact released reference profile")
        # This constructor verifies the entire merged state against the exact
        # untouched row-selected warm start. Rank1 has exactly zero effect.
        self.core = (
            FrozenPlatformTrue23Core(
                warm_start_path=warm_start_path,
                source_checkpoint_path=source_checkpoint_path,
                lora_rank=1,
                lora_alpha=1,
            )
            .eval()
            .requires_grad_(False)
        )
        if self.core.reference_profile != profile:
            raise ValueError("released weights and reference timing differ")
        for layer in self.core.decoder.layers:
            if torch.count_nonzero(layer.lora_b):
                raise ValueError("comparison refuses trained decoder adapters")
        self.profile = profile

    def infer(self, encoder267, history930):
        for value, width in ((encoder267, 267), (history930, 930)):
            if value.shape != (width,) or value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError("released core requires exact finite float32 input vectors")
        with torch.inference_mode():
            history = torch.from_numpy(history930[None])
            self.core.codec.validate_padded_proprioception(history)
            token = self.core.encode(torch.from_numpy(encoder267[None]))
            decoder = torch.cat((token, history), dim=-1)
            raw29 = self.core.decoder(decoder)
            raw23 = self.core.codec.decode_action(raw29)
        if not torch.isfinite(raw23).all():
            raise ValueError("released core emitted nonfinite native23 action")
        return raw23[0].numpy().copy(), decoder[0].numpy().copy()

    def identity(self):
        return dict(
            kind="untouched_released_sonic_core_row_selected_native23_v1",
            source_checkpoint_sha256=self.core.source_checkpoint_sha256,
            reference_profile=self.profile,
            initial_policy_state_sha256=self.core._initial_policy_sha256,
            frozen_state_sha256=self.core.frozen_state_sha256(),
            root_feedback_head=False,
            training_updates=0,
            decoder_adaptation="zero_effect_rank1_only_to_reuse_verified_release_constructor",
            deployment_ready=False,
            hardware_authorized=False,
        )


class ReleasedCoreAdapter:
    def __init__(self, motion, virtual_vr, *, profile, root, assets):
        if profile not in PROFILES:
            raise ValueError("unsupported released core profile")
        self.profile = profile
        self.source = continued_standing_source(motion, virtual_vr)
        normal = profile == REFERENCE_PROFILE_NORMAL
        self.extra_tail = 36 if normal else 0
        # Explicit simulated standing transmitter extension, not captured motion
        # or automatic EOF padding in either received-only buffer.
        if self.extra_tail:
            self.source = {
                key: np.concatenate((value, np.repeat(value[-1:], self.extra_tail, axis=0)))
                for key, value in self.source.items()
            }
        self.buffer = ReceivedNormalSourceHorizon() if normal else ReceivedSourceHorizon()
        self.emission_start = 55 if normal else 19
        self.preview = Native23RangePreview(
            model_path=Path(assets) / MODEL, physics_path=Path(root) / PHYSICS
        )
        self.next_control = 0
        self.attempts = []
        for index in range(self.emission_start):
            self._push(index)

    def _push(self, index):
        if index >= len(self.source["joint_pos"]):
            raise ValueError("received source exhausted; no automatic padding")
        return self.buffer.push(
            source_timestamp_s=index * 0.02,
            arrival_timestamp_s=index * 0.02,
            joint_names=HARDWARE_23_JOINT_NAMES,
            joint_position23=self.source["joint_pos"][index],
            root_position_w=self.source["root_position_w"][index],
            root_quaternion_wxyz=self.source["root_quaternion_wxyz"][index],
            virtual_source_vr21=self.source["virtual_vr21"][index],
        )

    def transform_history(self, history):
        return source_action_history_numpy(history)

    def infer(self, policy, encoder, history, *, control_index, **state):
        del encoder
        if policy.profile != self.profile or control_index != self.next_control:
            raise ValueError("policy profile or sequential control mismatch")
        window = self._push(self.emission_start + control_index)
        self.next_control += 1
        np.testing.assert_array_equal(
            window.anchor_root_position_w, state["previous_desired_position_w"].astype(np.float32)
        )
        np.testing.assert_array_equal(
            window.next_root_position_w, state["desired_position_w"].astype(np.float32)
        )
        actual = window.encoder267(state["measured_qpos"][3:7].astype(np.float32))
        raw, decoder = policy.infer(actual, history)
        # Save even the output that the unchanged raw bound or preview rejects.
        row = dict(
            encoder267=actual.copy(),
            history930=history.copy(),
            decoder994=decoder.copy(),
            released_raw23=raw.copy(),
            measured_qpos=state["measured_qpos"].copy(),
            measured_qvel=state["measured_qvel"].copy(),
            timestamps=np.asarray(
                [
                    window.emission_source_timestamp_s,
                    window.encoder_anchor_timestamp_s,
                    window.root_setpoint_timestamp_s,
                ],
                dtype=np.float64,
            ),
        )
        self.attempts.append(row)
        inverse, projection = source_scaled_precompensation(raw)
        row.update(inverse23=inverse.copy(), projection23=projection.copy())
        safe = self.preview.filter(inverse, state["measured_qpos"], state["measured_qvel"])
        row["accepted23"] = safe.copy()
        return safe, decoder

    def arrays(self):
        # Optional stage arrays are shorter on rejection. No padding or trimming
        # of attempted inference evidence to make an unsuccessful trial pass.
        keys = set().union(*(row.keys() for row in self.attempts))
        return {key: np.asarray([row[key] for row in self.attempts if key in row]) for key in sorted(keys)}

    def external_force_world(self, step):
        return np.zeros(3)

    def contract(self):
        return dict(
            kind="native23_released_core_matched_cpu_adapter_v1",
            reference_profile=self.profile,
            buffer=normal_horizon_contract() if self.extra_tail else buffered_horizon_contract(),
            source_action_codec=source_action_codec_contract(),
            generated_terminal_standing_samples=10 + self.extra_tail,
            captured_source_changed=False,
            scored_timeline_changed=False,
            root_feedback_used=False,
            joint_range_preview=self.preview.contract(),
            source_timestamp_jitter_tested=False,
            live_teleoperation_qualified=False,
            hardware_authorized=False,
            deployment_ready=False,
        )
