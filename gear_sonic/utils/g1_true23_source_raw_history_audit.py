"""Independent saved-history reconstruction, not a motion acceptance gate."""

import numpy as np

from gear_sonic.utils.g1_23dof_contract import NATIVE_IL23_TO_CANONICAL_IL29
from gear_sonic.utils.g1_true23_source_action_codec import source_action_history_numpy


def audit_consumed_histories(trace):
    n = len(trace["target23"])
    if n < 1:
        raise ValueError("history audit requires an executed physical prefix")
    widths = {
        "released_model_raw23": 23,
        "history930": 930,
        "consumed_raw_history930": 930,
        "effective_source_history930": 930,
    }
    for key, width in widths.items():
        values = trace[key]
        if values.shape != (n, width) or values.dtype != np.float32 or not np.isfinite(values).all():
            raise ValueError("invalid finite float32 history audit array: " + key)
    native = trace["native_precodec_history930"]
    if native.shape not in ((n, 930), (n + 1, 930)) or native.dtype != np.float32:
        raise ValueError("native measured history count/shape differs")
    effective = np.asarray([source_action_history_numpy(row) for row in native[:n]])
    np.testing.assert_array_equal(effective, trace["effective_source_history930"])
    np.testing.assert_array_equal(trace["history930"], trace["consumed_raw_history930"])
    raw = trace["released_model_raw23"]
    if np.max(np.abs(raw)) >= 10:
        raise ValueError("recorded request exceeds existing raw bound")
    retained = list(NATIVE_IL23_TO_CANONICAL_IL29)
    absent = sorted(set(range(29)) - set(retained))
    # Deliberately do not call the runtime replacement helper. Independently
    # enumerate each historical row and its strictly earlier source control.
    expected = effective.copy()
    for i in range(n):
        for historical_frame in range(10):
            previous_control = i - 10 + historical_frame
            offset = 610 + historical_frame * 29
            expected[i, offset + np.asarray(retained)] = raw[previous_control] if previous_control >= 0 else 0
    np.testing.assert_array_equal(trace["history930"], expected)
    np.testing.assert_array_equal(trace["history930"][:, 610:900].reshape(n, 10, 29)[:, :, absent], 0)
    difference = trace["history930"] - effective
    changed = np.any(difference != 0, axis=-1)
    return dict(
        physically_executed_history_rows_verified=n,
        strictly_previous_raw_requests_chronological=True,
        measured_and_absent_slots_unchanged=True,
        source_history_conversion_verified=True,
        number_of_controls_with_changed_history=int(changed.sum()),
        first_changed_history_control=int(np.flatnonzero(changed)[0]) if changed.any() else None,
        maximum_action_history_change_source_units=float(np.abs(difference).max()),
        unexecuted_native_history_rows_preserved=len(native) - n,
        hardware_authorized=False,
    )
