"""Received-only adapter around the unchanged CPU physics/metric referee.

Input source runs 200 ms ahead of its delayed encoder anchor. This is explicit
source buffering, not access to samples later than simulated emission time.
The scripted standing source keeps transmitting ten additional unscored frames.
No robot transport, state rewriting, EOF draining or fallback is implemented.
"""

import numpy as np

from gear_sonic.teleop.buffered_source_horizon import ReceivedSourceHorizon
from gear_sonic.utils.g1_23dof_contract import HARDWARE_23_JOINT_NAMES
from gear_sonic.utils.g1_true23_buffered_reference import (
    BUFFERED_TIMING,
    continued_standing_source,
    reference_profile_contract,
)
from gear_sonic.utils.g1_true23_root_feedback import root_feedback_contract
from gear_sonic.utils.g1_true23_root_feedback_benchmark import RootFeedbackRuntimeAdapter
from gear_sonic.utils.g1_true23_source_action_codec import (
    SOURCE_ACTION_CONVENTION,
    source_action_codec_contract,
    source_action_history_numpy,
    source_scaled_precompensation,
)


class BufferedSourceSimulationAdapter:
    def __init__(self, motion, virtual_vr, pulses=()):
        self.source = continued_standing_source(motion, virtual_vr)
        self.buffer = ReceivedSourceHorizon()
        self.root_adapter = RootFeedbackRuntimeAdapter(pulses)
        self.actual_encoder_inputs, self.model_outputs, self.projection_delta_rad = [], [], []
        self.timestamps = []
        self.next_control = 0
        # Feed genuine chronological input; discard warmup emissions until the
        # referee's unchanged initial anchor (absolute reference frame 9).
        for frame in range(19):
            self._push(frame)

    def _push(self, frame):
        if frame >= len(self.source["joint_pos"]):
            raise ValueError("simulated source exhausted; no automatic terminal padding")
        return self.buffer.push(
            source_timestamp_s=frame * 0.02,
            arrival_timestamp_s=frame * 0.02,
            joint_names=HARDWARE_23_JOINT_NAMES,
            joint_position23=self.source["joint_pos"][frame],
            root_position_w=self.source["root_position_w"][frame],
            root_quaternion_wxyz=self.source["root_quaternion_wxyz"][frame],
            virtual_source_vr21=self.source["virtual_vr21"][frame],
        )

    def transform_history(self, history):
        return source_action_history_numpy(history)

    def infer(self, policy, encoder, history, *, control_index, **state):
        del encoder  # Referee's old causal input is diagnostic-only, not consumed.
        if control_index != self.next_control:
            raise ValueError("buffered source controls must be sequential with no skipped frames")
        window = self._push(19 + control_index)
        self.next_control += 1
        current_quat = state["measured_qpos"][3:7].astype(np.float32)
        encoder = window.encoder267(current_quat)
        np.testing.assert_array_equal(window.next_root_position_w, state["desired_position_w"].astype(np.float32))
        np.testing.assert_array_equal(
            window.anchor_root_position_w, state["previous_desired_position_w"].astype(np.float32)
        )
        self.actual_encoder_inputs.append(encoder.copy())
        self.timestamps.append(
            (
                window.emission_source_timestamp_s,
                window.encoder_anchor_timestamp_s,
                window.root_setpoint_timestamp_s,
            )
        )
        raw, decoder = self.root_adapter.infer(policy, encoder, history, control_index=control_index, **state)
        expected_feedback = window.root_feedback9(
            state["measured_qpos"][:3].astype(np.float32),
            state["measured_qvel"][:3].astype(np.float32),
            current_quat,
        )
        np.testing.assert_array_equal(expected_feedback, self.root_adapter.records[-1]["root_feedback9"])
        self.model_outputs.append(raw.copy())
        compensated, projection = source_scaled_precompensation(raw)
        self.projection_delta_rad.append(projection)
        return compensated, decoder

    def external_force_world(self, step):
        return self.root_adapter.external_force_world(step)

    def contract(self):
        root = self.root_adapter.contract()
        root.update(
            root_feedback_contract=root_feedback_contract(BUFFERED_TIMING),
            desired_position_frame="received_horizon_q1_age_180ms",
            desired_velocity="float32(received_root_q1-received_root_q0)/float32(0.02)",
            measured_state_frame="current_control_boundary_not_delayed",
        )
        return dict(
            kind="g1_true23_received_source_horizon_cpu_runtime_v1",
            reference_timing=BUFFERED_TIMING,
            semantic_profile=reference_profile_contract(BUFFERED_TIMING),
            root_feedback=root,
            source_action_codec=source_action_codec_contract(),
            action_convention=SOURCE_ACTION_CONVENTION,
            actual_encoder_trace="actual_policy_encoder267",
            future_reference_consumed=False,
            future_relative_to_delayed_anchor_but_already_received=True,
            source_timestamps_trace="source_emission_anchor_setpoint_timestamps_s",
            source_clock_latency_s=0.2,
            scored_reference_timing_or_physics_changed=False,
            live_clock_dropout_and_transport_qualified=False,
            hardware_authorized=False,
            deployment_ready=False,
            simulator_qualified=False,
        )
