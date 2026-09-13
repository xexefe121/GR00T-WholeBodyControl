import json
from pathlib import Path

import numpy as np
import pytest

from gear_sonic.tests.test_g1_true23_virtual_source_history import predictor as predictor, physical_history
from gear_sonic.utils.g1_true23_current_momentum_observer import (
    CurrentMomentumHistoryAdapter,
    CurrentMomentumSourceModel,
)
from gear_sonic.utils.g1_true23_discarded_action_memory import MISSING
from gear_sonic.utils.g1_true23_generalist_benchmark import run_reference_diagnostic as old_benchmark
from gear_sonic.utils.g1_true23_measured_history_benchmark import run_reference_diagnostic as measured_benchmark
from gear_sonic.utils.g1_true23_momentum_source_model import MomentumSourceModel
from gear_sonic.utils.g1_true23_virtual_source_history import VirtualSourceHistory
from gear_sonic.utils.g1_true23_virtual_source_model import KEEP_HW


def native_state(model):
    return np.r_[model.data.qpos[:7], model.data.qpos[7 + KEEP_HW]], np.r_[
        model.data.qvel[:6], model.data.qvel[6 + KEEP_HW]
    ]


def current_model(predictor):
    return CurrentMomentumSourceModel(predictor.path, predictor.parameters, source_effort29=predictor.effort)


def test_prepare_before_inference_does_not_double_correct(predictor):
    current = current_model(predictor)
    previous = MomentumSourceModel(predictor.path, predictor.parameters, source_effort29=predictor.effort)
    q, v = native_state(predictor)
    q[2] = 1.1
    raw = np.zeros(29, np.float32)
    np.testing.assert_array_equal(current.advance(q, v, raw), previous.advance(q, v, raw))
    q, v = native_state(previous)
    v[3] += 0.1
    before = current.internal_state12().copy()
    state = current.prepare_current(q, v)
    assert np.max(np.abs(state[6:] - before[6:])) > 1e-4
    assert current.completed == 1 and current.prepared
    expected = previous.advance(q, v, raw)
    actual = current.advance(q, v, raw)
    np.testing.assert_array_equal(actual, expected)
    assert len(current.assimilation_records) == 2 and not current.prepared
    assert current.descriptor()["update_phase"] == "before_current_policy_inference"


def test_measurement_mismatch_poisoned_before_physics(predictor):
    model = current_model(predictor)
    q, v = native_state(model)
    model.prepare_current(q, v)
    changed = v.copy()
    changed[0] += 0.01
    with pytest.raises(ValueError, match="measurements differ"):
        model.advance(q, changed, np.zeros(29, np.float32))
    assert model.failed and model.internal_physics_steps == 0
    with pytest.raises(RuntimeError):
        model.prepare_current(q, v)


def test_adapter_changes_current_virtual_row_only(predictor):
    model = current_model(predictor)
    q, v = native_state(model)
    q[2] = 1.1
    model.advance(q, v, np.zeros(29, np.float32))
    q, v = native_state(model)
    v[3] += 0.1
    adapter = CurrentMomentumHistoryAdapter.__new__(CurrentMomentumHistoryAdapter)
    adapter.memory = VirtualSourceHistory()
    adapter.predictor = model
    adapter.current_virtual_state = None
    native = physical_history()
    original = native.copy()
    encoded = adapter.transform_measured_history(native, measured_qpos=q, measured_qvel=v, control_index=0)
    np.testing.assert_array_equal(encoded[30:320].reshape(10, 29)[-1, MISSING], model.internal_state12()[:6])
    np.testing.assert_array_equal(encoded[320:610].reshape(10, 29)[-1, MISSING], model.internal_state12()[6:])
    np.testing.assert_array_equal(encoded[320:610].reshape(10, 29)[:-1, MISSING], np.zeros((9, 6)))
    np.testing.assert_array_equal(native, original)
    with pytest.raises(RuntimeError):
        adapter.transform_history(native)
    with pytest.raises(RuntimeError):
        adapter.transform_measured_history(native, measured_qpos=q, measured_qvel=v, control_index=0)


def test_benchmark_noop_measurement_hook_exact_physics_and_copy_isolation():
    root = Path(__file__).resolve().parents[2]
    report_path = root / "artifacts/g1_true23_momentum_update_20260910_v1/native_actual_v1/report.json"
    assets = root.parent / "GR00T-WholeBodyControl"
    if not report_path.exists():
        pytest.skip("local recorded source unavailable")
    saved = json.loads(report_path.read_text())

    class Policy:
        def infer(self, encoder, history):
            return np.zeros(23, np.float32), np.r_[np.zeros(64, np.float32), history]

    class Adapter:
        def __init__(self):
            self.measurements = []

        def transform_history(self, h):
            pytest.fail("measurement hook was skipped")

        def transform_measured_history(self, h, *, measured_qpos, measured_qvel, control_index):
            self.measurements.append((measured_qpos.copy(), measured_qvel.copy(), control_index))
            measured_qpos[:] = 123  # Must not alias actual simulation state.
            measured_qvel[:] = 123
            return h

        def infer(self, policy, encoder, history, **kwargs):
            return policy.infer(encoder, history)

        def external_force_world(self, step):
            return np.zeros(3)

        def contract(self):
            return dict(kind="test_copied_measurement_identity", hardware_authorized=False)

    kwargs = dict(
        root=root,
        asset_root=assets,
        motion_path=saved["timeline"]["timeline_path"],
        policy=Policy(),
        maximum_controls=5,
    )
    old, old_trace = old_benchmark(**kwargs)
    adapter = Adapter()
    new, new_trace = measured_benchmark(**kwargs, runtime_adapter=adapter)
    assert old["failure"] is None and new["failure"] is None
    for key in ("qpos", "qvel", "physics_post_qpos", "physics_post_qvel", "target23", "raw23", "history930"):
        np.testing.assert_array_equal(old_trace[key], new_trace[key])
    assert len(adapter.measurements) == 5
    for q, v, i in adapter.measurements:
        np.testing.assert_array_equal(q, old_trace["qpos"][i])
        np.testing.assert_array_equal(v, old_trace["qvel"][i])
