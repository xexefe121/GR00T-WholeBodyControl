from pathlib import Path

import numpy as np
import pytest

from gear_sonic.scripts.audit_g1_true23_actuation_reset_feasibility import reset_intersections
from gear_sonic.utils.g1_23dof_safe_target_transform import SAFE_TARGET_DEFAULT_Q_HARDWARE
from gear_sonic.utils.g1_true23_actuation_profile import SIM_CONFIG, NativeSupportActuationProfile


def test_reset_audit_distinguishes_bad_controller_seed_from_impossible_joint_bounds():
    profile = NativeSupportActuationProfile.from_sim_config(Path(__file__).resolve().parents[2] / SIM_CONFIG)
    q = np.tile(SAFE_TARGET_DEFAULT_Q_HARDWARE, (3, 1))
    dq = np.zeros((3, 23))
    dq[1, 16] = 10.0  # Feasible PD target exists, but not within one step of q.
    dq[2, 16] = 1000.0  # No target inside joint bounds can satisfy this guard.
    q_seed, any_seed = reset_intersections(q, dq, profile)
    assert np.any(q_seed, axis=1).tolist() == [False, True, True]
    assert np.any(any_seed, axis=1).tolist() == [False, False, True]


@pytest.mark.parametrize(
    "q,dq",
    [
        (np.zeros(23), np.zeros(23)),
        (np.zeros((1, 23)), np.zeros((2, 23))),
        (np.full((1, 23), np.nan), np.zeros((1, 23))),
    ],
)
def test_reset_audit_rejects_invalid_state(q, dq):
    with pytest.raises(ValueError, match="finite"):
        reset_intersections(q, dq, None)
