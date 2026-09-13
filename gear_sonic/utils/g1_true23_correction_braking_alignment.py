"""Include current source-correction bounds in reference braking distance.

The preserved v1 converter bounded each final correction but computed stopping
distance only to physical-reference joint limits. That allowed a reference to
approach its correction limit with no acceleration-feasible following sample.
This additive repair uses the same current correction box in the braking
calculation. It does not predict future source motion, alter any acceptance
limit, or guarantee feasibility for arbitrary unseen next frames.
"""

import numpy as np

from gear_sonic.utils.g1_true23_continuous_reference_alignment import ContinuousReferenceAlignment
from gear_sonic.utils.g1_true23_torso_reference_alignment import _pose


class CorrectionBrakingAlignment(ContinuousReferenceAlignment):
    """Same solver and limits; current correction box participates in braking."""

    def _step(self, source_pose, frame_index):
        source_pose = _pose(source_pose, 36)
        low, high = self.static_lower, self.static_upper
        correction = self.temporal_config.joint_correction_limit_rad * self.temporal_config.reserve_fraction
        current_low, current_high = low.copy(), high.copy()
        current_low[6:] = np.maximum(current_low[6:], source_pose[7 + self.keep] - correction)
        current_high[6:] = np.minimum(current_high[6:], source_pose[7 + self.keep] + correction)
        if np.any(current_low >= current_high):
            raise ValueError("source-correction joint envelope is empty")
        # This is private reference-optimizer state, not physical model limits.
        # Parent computes braking and acceleration intersection from these
        # current bounds. Restore static bounds even after a latched failure.
        self.static_lower, self.static_upper = current_low, current_high
        try:
            pose, evidence = super()._step(source_pose, frame_index)
        finally:
            self.static_lower, self.static_upper = low, high
        return pose, {**evidence, "current_source_correction_in_braking_distance": True}

    def contract(self):
        return {
            **super().contract(),
            "kind": "native23_causal_correction_aware_braking_reference_v2",
            "current_source_correction_in_braking_distance": True,
            "unseen_future_source_motion_bounded_or_predicted": False,
            "temporal_feasibility_for_arbitrary_next_source_guaranteed": False,
            "weights_limits_iterations_or_source_timing_changed_from_v1": False,
        }
