import json
from pathlib import Path

import numpy as np
import pytest

from gear_sonic.utils.g1_sonic_cpp_parameters import CppParameters
from gear_sonic.utils.g1_true23_discarded_action_memory import KEEP, MISSING
from gear_sonic.utils.g1_true23_source_action_codec import source_action_history_numpy
from gear_sonic.utils.g1_true23_released_core_comparison import ReleasedCoreAdapter
from gear_sonic.utils.g1_true23_virtual_source_history import (
    VirtualSourceHistory,
    VirtualSourceHistoryAdapter,
    VirtualSourceHistoryPolicy,
)
from gear_sonic.utils.g1_true23_virtual_source_model import KEEP_HW, MISSING_HW, VirtualSourceModel


def physical_history():
    h = np.linspace(-0.1, 0.1, 930, dtype=np.float32)
    for start in (30, 320, 610):
        h[start : start + 290].reshape(10, 29)[:, MISSING] = 0
    return h


def test_virtual_history_exact_mapping_causal_alignment_and_no_native_mutation():
    memory = VirtualSourceHistory()
    native = physical_history()
    original = native.copy()
    encoded = memory.encode(native)
    np.testing.assert_array_equal(encoded, source_action_history_numpy(native))
    for control in range(13):
        predicted = np.arange(12, dtype=np.float32) + control
        raw = np.arange(29, dtype=np.float32) - control
        memory.accept(predicted, raw)
        predicted[:] = -999
        raw[:] = -999
        encoded = memory.encode(native)
        for offset, start in enumerate((30, 320, 610)):
            block = encoded[start : start + 290].reshape(10, 29)
            expected = source_action_history_numpy(native)[start : start + 290].reshape(10, 29)
            np.testing.assert_array_equal(block[:, KEEP], expected[:, KEEP])
            newest = np.arange(offset * 6, offset * 6 + 6) + control if offset < 2 else MISSING - control
            np.testing.assert_array_equal(block[-1, MISSING], newest)
        if control >= 9:
            np.testing.assert_array_equal(encoded[30:320].reshape(10, 29)[0, MISSING], np.arange(6) + control - 9)
    np.testing.assert_array_equal(native, original)
    assert not np.shares_memory(encoded, native)


@pytest.mark.parametrize("change", ["nonfinite", "missing_measurement", "dtype", "shape"])
def test_invalid_native_input_poison_history(change):
    h = physical_history()
    if change == "nonfinite":
        h[0] = np.nan
    elif change == "missing_measurement":
        h[30 + MISSING[0]] = 1
    elif change == "dtype":
        h = h.astype(np.float64)
    else:
        h = h[:-1]
    memory = VirtualSourceHistory()
    with pytest.raises(ValueError):
        memory.encode(h)
    with pytest.raises(RuntimeError):
        memory.encode(physical_history())


def test_no_double_encode_and_invalid_prediction_cannot_restart():
    memory = VirtualSourceHistory()
    memory.encode(physical_history())
    with pytest.raises(RuntimeError):
        memory.encode(physical_history())
    bad = np.zeros(12, np.float32)
    bad[1] = np.nan
    with pytest.raises(ValueError):
        memory.accept(bad, np.zeros(29, np.float32))
    with pytest.raises(RuntimeError):
        memory.accept(np.zeros(12, np.float32), np.zeros(29, np.float32))
    assert memory.contract()["accepted_forecasts"] == 0


@pytest.fixture
def predictor():
    root = Path(__file__).resolve().parents[2]
    directory = root / "artifacts/g1_true23_frozen_lora/original29_recorded_baseline_20260906_v1"
    model = directory / "original29.mjb"
    if not model.exists():
        pytest.skip("local pinned source29 model unavailable")
    p = CppParameters(json.loads((directory / "cpp_capture/parameters.json").read_text()))
    effort = [
        88,
        88,
        88,
        139,
        50,
        50,
        88,
        88,
        88,
        139,
        50,
        50,
        88,
        50,
        50,
        25,
        25,
        25,
        25,
        25,
        5,
        5,
        25,
        25,
        25,
        25,
        25,
        5,
        5,
    ]
    return VirtualSourceModel(model, p, source_effort29=effort)


def test_actual_source_model_is_separate_and_missing_axes_persist(predictor):
    q = np.r_[[0.0, 0.0, 1.1, 1.0, 0.0, 0.0, 0.0], predictor.parameters.default_angles[KEEP_HW]]
    dq, raw = np.zeros(29), np.zeros(29, np.float32)
    before = q.copy()
    raw[MISSING] = 0.2
    first = predictor.advance(q, dq, raw)
    np.testing.assert_array_equal(q, before)
    np.testing.assert_array_equal(dq, np.zeros(29))
    assert not np.shares_memory(q, predictor.data.qpos)
    assert np.max(np.abs(first)) > 0
    predictor.advance(q, dq, raw)
    assert predictor.completed == 2 and predictor.internal_physics_steps == 20
    assert not np.array_equal(first, predictor.internal_state12())
    assert set(KEEP_HW).isdisjoint(MISSING_HW)
    assert sorted([*KEEP_HW, *MISSING_HW]) == list(range(29))
    assert predictor.descriptor()["native_model_or_state_written"] is False
    predictor.failed = True
    with pytest.raises(RuntimeError):
        predictor.advance(q, dq, raw)


def test_prediction_refuses_mutated_model_identity(predictor):
    predictor.model.opt.timestep = 0.003
    with pytest.raises(ValueError, match="model changed"):
        predictor.descriptor()


@pytest.mark.parametrize("cause", ["native_preview", "prediction", "virtual_range"])
def test_adapter_failure_is_terminal_without_history_commit(predictor, monkeypatch, cause):
    adapter = VirtualSourceHistoryAdapter.__new__(VirtualSourceHistoryAdapter)
    adapter.memory = VirtualSourceHistory()
    adapter.predictor = predictor
    adapter.attempts = []
    policy = VirtualSourceHistoryPolicy.__new__(VirtualSourceHistoryPolicy)
    policy.latest_raw29 = np.zeros(29, np.float32)

    def parent_infer(self, *args, **kwargs):
        self.attempts.append({})
        if cause == "native_preview":
            raise RuntimeError("native preview rejected unchanged physical proposal")
        return np.zeros(23, np.float32), np.zeros(994, np.float32)

    calls = []

    def forecast(*args):
        calls.append(True)
        if cause == "prediction":
            raise FloatingPointError("internal prediction nonfinite")
        predictor.maximum_virtual_joint_excess = 0.001
        return np.zeros(12, np.float32)

    monkeypatch.setattr(ReleasedCoreAdapter, "infer", parent_infer)
    monkeypatch.setattr(predictor, "advance", forecast)
    h = adapter.transform_history(physical_history())
    with pytest.raises((RuntimeError, FloatingPointError)):
        adapter.infer(
            policy,
            np.zeros(267, np.float32),
            h,
            control_index=0,
            measured_qpos=np.zeros(30),
            measured_qvel=np.zeros(29),
        )
    assert calls == ([] if cause == "native_preview" else [True])
    assert adapter.memory.accepted == 0 and adapter.memory.failed and predictor.failed
    assert adapter.attempts[-1]["virtual_forecast_accepted"] is False
    with pytest.raises(RuntimeError):
        adapter.transform_history(physical_history())
