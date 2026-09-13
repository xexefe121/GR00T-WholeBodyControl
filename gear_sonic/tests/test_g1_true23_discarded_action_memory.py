import numpy as np
import pytest

from gear_sonic.utils.g1_true23_discarded_action_memory import (
    ACTION_SLOTS,
    MISSING,
    DiscardedActionMemory,
    validate_history,
)
from gear_sonic.utils.g1_true23_source_action_codec import source_action_history_numpy


def history():
    value = np.arange(930, dtype=np.float32) / 1000
    for start in (30, 320, 610):
        value[start : start + 290].reshape(10, 29)[:, MISSING] = 0
    return value


def test_zero_memory_preserves_entire_original_source_codec():
    value = history()
    before = value.copy()
    actual = DiscardedActionMemory().encode(value)
    np.testing.assert_array_equal(actual, source_action_history_numpy(value))
    np.testing.assert_array_equal(value, before)


def test_only_six_unapplied_history_slots_change_in_chronological_order():
    memory = DiscardedActionMemory()
    reference = source_action_history_numpy(history())
    stored = []
    for i in range(13):
        actual = memory.encode(history())
        expected = np.zeros((10, 6), np.float32)
        if stored:
            used = stored[-10:]
            expected[-len(used) :] = used
        np.testing.assert_array_equal(actual[ACTION_SLOTS], expected)
        actual[ACTION_SLOTS] = 0
        np.testing.assert_array_equal(actual, reference)
        raw = np.arange(29, dtype=np.float32) + i
        stored.append(raw[MISSING].copy())
        memory.accept(raw)
        raw[:] = -999  # Stored memory owns its data.
    assert memory.accepted_controls == 13


@pytest.mark.parametrize("start", [30, 320])
def test_no_invented_missing_joint_position_or_velocity(start):
    value = history()
    value[start + MISSING[0]] = 1
    with pytest.raises(ValueError, match="measured joint slots"):
        validate_history(value, require_zero_action_slots=False)
    with pytest.raises(ValueError, match="measured joint slots"):
        DiscardedActionMemory().encode(value)


def test_incoming_physical_history_cannot_smuggle_virtual_action_values():
    value = history()
    value[ACTION_SLOTS[0, 0]] = 1
    validate_history(value, require_zero_action_slots=False)
    with pytest.raises(ValueError, match="measured joint slots"):
        DiscardedActionMemory().encode(value)


@pytest.mark.parametrize("bad", [np.zeros(29), np.zeros(23, np.float32), np.full(29, np.nan, np.float32)])
def test_invalid_virtual_output_rejected_without_advancing(bad):
    memory = DiscardedActionMemory()
    memory.encode(history())
    with pytest.raises(ValueError, match="finite float32 raw29"):
        memory.accept(bad)
    assert memory.accepted_controls == 0


def test_stale_duplicate_or_failed_control_rejected():
    memory = DiscardedActionMemory()
    with pytest.raises(RuntimeError, match="pending"):
        memory.accept(np.zeros(29, np.float32))
    memory.encode(history())
    with pytest.raises(RuntimeError, match="sequential"):
        memory.encode(history())
    memory.fail()
    with pytest.raises(RuntimeError, match="pending"):
        memory.accept(np.zeros(29, np.float32))
    with pytest.raises(RuntimeError, match="sequential"):
        memory.encode(history())
    assert memory.contract()["virtual_outputs_sent_to_physical_motors"] is False


@pytest.mark.parametrize("bad", [np.zeros(930), np.zeros(929, np.float32), np.full(930, np.inf, np.float32)])
def test_invalid_history_rejected(bad):
    with pytest.raises(ValueError):
        DiscardedActionMemory().encode(bad)
