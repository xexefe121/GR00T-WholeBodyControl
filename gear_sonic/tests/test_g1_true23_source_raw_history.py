"""History-only counterfactual; never authorizes a live controller change."""

import numpy as np
import pytest

from gear_sonic.utils.g1_true23_source_raw_history import (
    HISTORY_START,
    HISTORY_STOP,
    RETAINED,
    replace_previous_action_history,
    source_raw_history_contract,
)


@pytest.mark.parametrize("count", [0, 1, 3, 9, 10])
def test_exact_chronological_requests_and_other_bytes_unchanged(count):
    history = np.arange(930, dtype=np.float32) / 1000
    raw = np.arange(count * 23, dtype=np.float32).reshape(count, 23) / 100
    old_history, old_raw = history.copy(), raw.copy()
    result = replace_previous_action_history(history, raw)
    expected = old_history.copy()
    # Independent per-frame/per-joint assembly, not helper's advanced indexing.
    for frame in range(10):
        for column, slot in enumerate(RETAINED):
            index = HISTORY_START + frame * 29 + slot
            expected[index] = 0 if frame < 10 - count else raw[frame - 10 + count, column]
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(history, old_history)
    np.testing.assert_array_equal(raw, old_raw)
    assert not np.shares_memory(result, history)


def test_equal_raw_and_target_history_is_identity():
    raw = np.arange(230, dtype=np.float32).reshape(10, 23) / 100
    history = np.zeros(930, dtype=np.float32)
    history[HISTORY_START:HISTORY_STOP].reshape(10, 29)[:, RETAINED] = raw
    np.testing.assert_array_equal(replace_previous_action_history(history, raw), history)


@pytest.mark.parametrize(
    "history,raw",
    [
        (np.zeros(929, np.float32), np.zeros((0, 23), np.float32)),
        (np.zeros(930, np.float64), np.zeros((0, 23), np.float32)),
        (np.full(930, np.nan, np.float32), np.zeros((0, 23), np.float32)),
        (np.zeros(930, np.float32), np.zeros((11, 23), np.float32)),
        (np.zeros(930, np.float32), np.zeros((1, 29), np.float32)),
        (np.zeros(930, np.float32), np.zeros((1, 23), np.float64)),
        (np.zeros(930, np.float32), np.full((1, 23), np.inf, np.float32)),
        (np.zeros(930, np.float32), np.full((1, 23), 10, np.float32)),
        (np.zeros(930, np.float32), np.full((1, 23), -10, np.float32)),
    ],
)
def test_bad_inputs_rejected(history, raw):
    with pytest.raises(ValueError):
        replace_previous_action_history(history, raw)


def test_contract_denies_training_equivalence_and_live_use():
    contract = source_raw_history_contract()
    assert contract["training_runtime_distribution_changed"] is True
    for field in (
        "checkpoint_compatibility_or_full_upstream_equivalence_claimed",
        "target_transform_gains_effort_physical_limits_changed",
        "live_export_or_transport_supported",
        "hardware_authorized",
        "deployment_ready",
    ):
        assert contract[field] is False
