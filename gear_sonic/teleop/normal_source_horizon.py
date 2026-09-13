"""Received-only original SONIC step5 horizon; research, not live approval.

Positions sample q0,q5,...,q45. Velocities use adjacent 20-ms differences
at those samples, not differences across the 100-ms position spacing.
Thus 47 received samples are required and the reference anchor is 920 ms old.
The existing low-latency buffer and its evidence remain unchanged.
"""

from collections import deque

import numpy as np

from gear_sonic.teleop.buffered_source_horizon import BufferedReference, ReceivedSourceHorizon
from gear_sonic.utils.g1_23dof_contract import REFERENCE_PROFILE_NORMAL, reference_profile_contract

SAMPLE_COUNT = 47
POSITION_INDICES = np.arange(10) * 5


def normal_horizon_contract():
    return dict(
        kind="sonic_received_normal_step5_horizon_920ms_v1",
        source_profile=reference_profile_contract(REFERENCE_PROFILE_NORMAL),
        received_samples_required=SAMPLE_COUNT,
        reference_anchor_age_s=0.92,
        position_sample_indices=POSITION_INDICES.tolist(),
        velocity_definition="float32(q[i+1]-q[i])/float32(0.02), i=0,5,...,45",
        vr_reference="received_q0_original29_geometry",
        measured_robot_state="current_control_boundary_not_delayed",
        prediction_or_unreceived_samples=False,
        automatic_terminal_padding=False,
        deployment_ready=False,
        hardware_authorized=False,
    )


class ReceivedNormalSourceHorizon:
    """Reuse strict input admission; retain a distinct, longer reference window."""

    def __init__(self):
        self._validator = ReceivedSourceHorizon()
        self._samples = deque(maxlen=SAMPLE_COUNT)
        self._failed = False

    def reset(self):
        self._validator.reset()
        self._samples.clear()
        self._failed = False

    def push(self, **sample):
        if self._failed:
            raise RuntimeError("normal reference buffer requires explicit reset after rejection")
        try:
            # The old buffer validates clocks, names, finite float32 arrays and
            # quaternions. Its short-horizon output is deliberately not consumed.
            self._validator.push(**sample)
            self._samples.append(
                (
                    float(sample["source_timestamp_s"]),
                    sample["joint_position23"].copy(),
                    sample["root_position_w"].copy(),
                    sample["root_quaternion_wxyz"].copy(),
                    sample["virtual_source_vr21"].copy(),
                )
            )
            if len(self._samples) < SAMPLE_COUNT:
                return None
            rows = tuple(self._samples)
            joints = np.stack([row[1][:12] for row in rows])
            selected = joints[POSITION_INDICES]
            velocity = (joints[POSITION_INDICES + 1] - selected) / np.float32(0.02)
            lower = np.concatenate((selected.reshape(-1), velocity.reshape(-1)))
            if not np.isfinite(lower).all():
                raise ValueError("normal reference joint differences overflowed")
            return BufferedReference(
                emission_source_timestamp_s=rows[-1][0],
                encoder_anchor_timestamp_s=rows[0][0],
                root_setpoint_timestamp_s=rows[1][0],
                lower_body240=lower,
                virtual_vr21=rows[0][4].copy(),
                anchor_root_position_w=rows[0][2].copy(),
                anchor_root_quaternion_wxyz=rows[0][3].copy(),
                next_root_position_w=rows[1][2].copy(),
            )
        except (TypeError, ValueError, RuntimeError, KeyError):
            self._failed = True
            self._samples.clear()
            raise
