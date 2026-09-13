"""Synthetic source-effort mapping tests, not motion qualification."""

from types import SimpleNamespace

import numpy as np
import pytest

from gear_sonic.utils.g1_29dof_source_sim_effort import (
    BODY_ACTUATORS,
    SourceSimulationEffortParameters,
    source_sim_body_effort,
)


def fixture43():
    names = list(BODY_ACTUATORS[:22]) + [f"left_hand_{i}" for i in range(7)]
    names += list(BODY_ACTUATORS[22:]) + [f"right_hand_{i}" for i in range(7)]
    return names, np.arange(1, 44, dtype=np.float64)


def test_name_mapping_handles_interposed_fingers():
    names, limits = fixture43()
    actual = source_sim_body_effort(names, limits)
    np.testing.assert_array_equal(actual, np.r_[limits[:22], limits[29:36]])
    order = np.random.default_rng(0).permutation(43)
    np.testing.assert_array_equal(actual, source_sim_body_effort([names[i] for i in order], limits[order]))
    assert not actual.flags.writeable
    limits[:] = 0
    assert np.all(actual > 0)


@pytest.mark.parametrize("bad", [0, -1, np.nan, np.inf])
def test_rejects_invalid_limits(bad):
    names, limits = fixture43()
    limits[2] = bad
    with pytest.raises(ValueError, match="finite and positive"):
        source_sim_body_effort(names, limits)


def test_rejects_missing_duplicate_or_wrong_extra_actuators():
    names, limits = fixture43()
    with pytest.raises(ValueError, match="43 unique"):
        source_sim_body_effort(names[:-1], limits[:-1])
    wrong = names.copy()
    wrong[0] = wrong[1]
    with pytest.raises(ValueError, match="43 unique"):
        source_sim_body_effort(wrong, limits)
    wrong[0] = "unrelated_axis"
    with pytest.raises(ValueError, match="missing"):
        source_sim_body_effort(wrong, limits)
    wrong = names.copy()
    wrong[-1] = "unrelated_axis"
    with pytest.raises(ValueError, match="14 hand"):
        source_sim_body_effort(wrong, limits)


def test_wrapper_preserves_target_and_source_arrays():
    names, limits = fixture43()
    raw = np.arange(29, dtype=np.float32) / 10
    base = SimpleNamespace(
        **{name: np.ones(29) for name in ("default_angles", "action_scale", "kps", "kds", "effort")},
        target=lambda x: x.astype(np.float64) + 0.2,
    )
    wrapper = SourceSimulationEffortParameters(base, names, limits)
    np.testing.assert_array_equal(wrapper.target(raw), base.target(raw))
    for name in ("default_angles", "action_scale", "kps", "kds"):
        assert getattr(wrapper, name) is getattr(base, name)
    np.testing.assert_array_equal(base.effort, np.ones(29))
    assert not wrapper.descriptor()["hardware_authorized"]
    assert not wrapper.descriptor()["native23_controller_or_limits_changed"]
