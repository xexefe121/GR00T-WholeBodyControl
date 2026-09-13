"""Regression for the preserved v1 post-integration metadata failure."""

from types import SimpleNamespace

import numpy as np
import pytest

from gear_sonic.scripts.record_g1_true23_source_raw_history_v2 import MetadataCompatibleAdapter
from gear_sonic.utils.g1_true23_root_feedback_benchmark import RootFeedbackRuntimeAdapter


def make_adapter_without_physics():
    adapter = MetadataCompatibleAdapter.__new__(MetadataCompatibleAdapter)
    adapter.root_adapter = RootFeedbackRuntimeAdapter()
    adapter.preview = SimpleNamespace(contract=lambda: {"fake_unit_test_preview": True})
    adapter.model_outputs = []
    adapter.effective_histories = []
    adapter.raw_histories = []
    return adapter


def test_actual_referee_metadata_access_retains_counterfactual_label():
    adapter = make_adapter_without_physics()
    result = adapter.contract()
    # Exact access performed AFTER integration by the unchanged referee.
    label = result["source_action_codec"]["previous_action"]
    assert label == result["previous_action_counterfactual"]["previous_action"]
    assert "raw23" in label and "before target projection" in label
    assert result["actual_training_contract_applied_without_change"] is False
    assert result["source_action_codec"]["original_combined_codec_contract_claimed"] is False
    assert result["source_action_codec"]["target_codec"]["native_physics_and_limits_unchanged"] is True


def test_adapter_rejects_duplicate_history_without_inference():
    adapter = make_adapter_without_physics()
    zeros = np.zeros(930, np.float32)
    np.testing.assert_array_equal(adapter.transform_history(zeros), zeros)
    with pytest.raises(ValueError, match="exactly one completed previous inference"):
        adapter.transform_history(zeros)


def test_adapter_uses_previous_request_without_changing_effective_history():
    from gear_sonic.utils.g1_true23_source_raw_history import RETAINED

    adapter = make_adapter_without_physics()
    zeros = np.zeros(930, np.float32)
    adapter.transform_history(zeros)
    request = np.linspace(-2, 2, 23, dtype=np.float32)
    adapter.model_outputs.append(request)
    actual = adapter.transform_history(zeros)
    block = actual[610:900].reshape(10, 29)
    np.testing.assert_array_equal(block[-1, RETAINED], request)
    np.testing.assert_array_equal(block[:-1], 0)
    np.testing.assert_array_equal(adapter.effective_histories[-1], zeros)
