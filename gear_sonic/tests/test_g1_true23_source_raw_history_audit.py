"""Counterexample checks for saved raw-history auditing."""

import copy

import numpy as np
import pytest

from gear_sonic.utils.g1_23dof_contract import NATIVE_IL23_TO_CANONICAL_IL29
from gear_sonic.utils.g1_true23_source_action_codec import source_action_history_numpy
from gear_sonic.utils.g1_true23_source_raw_history_audit import audit_consumed_histories


def synthetic_trace():
    count = 13
    native = np.zeros((count, 930), np.float32)
    native[:, :610] = np.arange(610, dtype=np.float32) / 1000
    effective = np.asarray([source_action_history_numpy(row) for row in native])
    raw = np.arange(count * 23, dtype=np.float32).reshape(count, 23) / 100
    frames = [np.zeros(29, np.float32) for _ in range(10)]
    history = []
    for i in range(count):
        row = effective[i].copy()
        row[610:900] = np.asarray(frames).reshape(-1)
        history.append(row)
        requested = np.zeros(29, np.float32)
        requested[list(NATIVE_IL23_TO_CANONICAL_IL29)] = raw[i]
        frames = [*frames[1:], requested]
    history = np.asarray(history)
    return dict(
        target23=np.zeros((count, 23), np.float32),
        released_model_raw23=raw,
        history930=history,
        consumed_raw_history930=history.copy(),
        effective_source_history930=effective,
        native_precodec_history930=native,
    )


def test_good_complete_history_reconstructs_without_mutation():
    trace = synthetic_trace()
    before = copy.deepcopy(trace)
    result = audit_consumed_histories(trace)
    assert result["physically_executed_history_rows_verified"] == 13
    assert result["first_changed_history_control"] == 1
    for key in trace:
        np.testing.assert_array_equal(trace[key], before[key])


@pytest.mark.parametrize("kind", ["future", "reversed", "measured", "effective", "raw", "absent"])
def test_counterexamples_fail(kind):
    trace = synthetic_trace()
    if kind == "future":
        trace["history930"][10] = trace["history930"][11]
    elif kind == "reversed":
        block = trace["history930"][12, 610:900].reshape(10, 29)
        block[:] = block[::-1].copy()
    elif kind == "measured":
        trace["history930"][5, 6] += 1
    elif kind == "effective":
        trace["effective_source_history930"][5, 610] += 1
    elif kind == "raw":
        trace["released_model_raw23"][3, 5] += 1
    else:
        absent = min(set(range(29)) - set(NATIVE_IL23_TO_CANONICAL_IL29))
        trace["history930"][5, 610 + absent] += 1
    # Preserve the two recorded consumed-input views, so rejection must come
    # from independent timing/value reconstruction, not a duplicate-array check.
    trace["consumed_raw_history930"] = trace["history930"].copy()
    with pytest.raises((ValueError, AssertionError)):
        audit_consumed_histories(trace)


def test_unapplied_native_history_retained_but_not_counted_executed():
    trace = synthetic_trace()
    trace["native_precodec_history930"] = np.concatenate(
        (trace["native_precodec_history930"], np.full((1, 930), 2, np.float32))
    )
    result = audit_consumed_histories(trace)
    assert result["unexecuted_native_history_rows_preserved"] == 1
    assert result["physically_executed_history_rows_verified"] == 13
