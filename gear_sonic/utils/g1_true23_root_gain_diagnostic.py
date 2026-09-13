"""Explicit offline root-position input intervention, never a deployed actor.

Weights, measured/reference telemetry and velocity inputs stay unchanged. Only
the three position-error features supplied to the strict CPU reader are scaled.
Record both inputs so a gain experiment cannot masquerade as nominal inference.
"""

import numpy as np


class RootPositionGainDiagnostic:
    def __init__(self, policy, gain):
        if type(gain) not in (int, float) or gain not in (0, 2):
            raise ValueError("bounded offline diagnostic permits root-position gains0 or2 only")
        if not callable(getattr(policy, "infer", None)):
            raise TypeError("root-position diagnostic requires a strict CPU inference reader")
        self.policy = policy
        self.gain = float(gain)
        self.actual_feedback = []
        self.used_feedback = []

    def infer(self, encoder267, history930, root_feedback9):
        if (
            not isinstance(root_feedback9, np.ndarray)
            or root_feedback9.shape != (9,)
            or root_feedback9.dtype != np.float32
            or not np.isfinite(root_feedback9).all()
        ):
            raise ValueError("root-position diagnostic requires finite float32 feedback9")
        actual = root_feedback9.copy()
        used = actual.copy()
        with np.errstate(over="ignore"):
            used[:3] *= np.float32(self.gain)
        if not np.isfinite(used).all():
            raise ValueError("root-position scaling overflow")
        result = self.policy.infer(encoder267, history930, used.copy())
        self.actual_feedback.append(actual)
        self.used_feedback.append(used)
        return result

    def contract(self):
        return dict(
            kind="native23_offline_root_position_gain_intervention_v1",
            gain=self.gain,
            changed_decoder_features=[0, 1, 2],
            original_telemetry_array="root_feedback9",
            actual_network_input_array="root_gain_used_feedback9",
            feature_order="position_error_xyz_desired_velocity_xyz_measured_velocity_xyz",
            original_world_reference_or_measurements_changed=False,
            frozen_source_token_pose_adapter_and_velocity_feedback_unchanged=True,
            checkpoint_or_optimizer_changed=False,
            nominal_training_contract_claimed=False,
            hardware_authorized=False,
            deployment_ready=False,
        )
