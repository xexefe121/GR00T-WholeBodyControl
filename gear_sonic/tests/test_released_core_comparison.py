from types import SimpleNamespace

import numpy as np
import pytest

from gear_sonic.utils import g1_true23_released_core_comparison as comparison


@pytest.mark.parametrize("profile", comparison.PROFILES)
def test_adapter_matches_float32_source_to_float64_referee_without_moving_state(monkeypatch, profile):
    monkeypatch.setattr(
        comparison,
        "Native23RangePreview",
        lambda **kwargs: SimpleNamespace(filter=lambda raw, q, v: raw.copy(), contract=lambda: {}),
    )
    count = 80
    root = np.asarray([0.123456789, 0.234567891, 0.789123456], dtype=np.float64)
    motion = dict(
        joint_pos=np.zeros((count, 23)),
        joint_vel=np.zeros((count, 23)),
        body_pos_w=np.tile(root, (count, 1, 1)),
        body_quat_w=np.tile([1.0, 0.0, 0.0, 0.0], (count, 1, 1)),
        body_lin_vel_w=np.zeros((count, 1, 3)),
        body_ang_vel_w=np.zeros((count, 1, 3)),
    )
    vr = np.tile(np.asarray([0] * 9 + [1, 0, 0, 0] * 3, np.float32), (count, 1))
    adapter = comparison.ReleasedCoreAdapter(motion, vr, profile=profile, root=".", assets=".")
    policy = SimpleNamespace(
        profile=profile,
        infer=lambda enc, hist: (np.zeros(23, np.float32), np.concatenate((np.zeros(64, np.float32), hist))),
    )
    q = np.concatenate((root, [1.0, 0.0, 0.0, 0.0], np.zeros(23)))
    state = dict(
        previous_desired_position_w=root.copy(),
        desired_position_w=root.copy(),
        measured_qpos=q.copy(),
        measured_qvel=np.zeros(29),
    )
    before = {key: value.copy() for key, value in state.items()}
    raw, decoder = adapter.infer(
        policy, np.zeros(267, np.float32), np.zeros(930, np.float32), control_index=0, **state
    )
    assert raw.shape == (23,) and decoder.shape == (994,)
    for key in state:
        np.testing.assert_array_equal(state[key], before[key])
    normal = profile == comparison.REFERENCE_PROFILE_NORMAL
    assert len(adapter.source["joint_pos"]) == count + (46 if normal else 10)
    assert adapter.attempts[0]["timestamps"][1] == 0.18
    assert adapter.attempts[0]["timestamps"][0] == (1.1 if normal else 0.38)
    with pytest.raises(ValueError, match="sequential control"):
        adapter.infer(policy, np.zeros(267, np.float32), np.zeros(930, np.float32), control_index=2, **state)


def test_unknown_profile_rejected_before_loading_weights():
    with pytest.raises(ValueError, match="exact released reference profile"):
        comparison.ReleasedCorePolicy(
            warm_start_path="missing", source_checkpoint_path="missing", profile="guessed"
        )
