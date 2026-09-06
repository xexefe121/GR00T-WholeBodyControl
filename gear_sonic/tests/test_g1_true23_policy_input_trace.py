import numpy as np
import pytest

from gear_sonic.utils import g1_true23_policy_input_trace as trace


class Policy:
    def __init__(self):
        self.calls = []
        self.raw = np.zeros(23, dtype=np.float32)
        self.decoder = np.zeros(994, dtype=np.float32)

    def infer(self, encoder, history):
        self.calls.append((encoder, history))
        self.decoder[64:] = history
        return self.raw, self.decoder


def test_recorder_preserves_arguments_results_and_call_count_with_immutable_copies():
    policy = Policy()
    recorder = trace.RecordedPolicy(policy)
    encoder, history = np.zeros(267, np.float32), np.zeros(930, np.float32)
    raw, decoder = recorder.infer(encoder, history)
    assert policy.calls[0][0] is encoder and policy.calls[0][1] is history
    assert raw is policy.raw and decoder is policy.decoder and len(policy.calls) == 1
    encoder[:] = history[:] = raw[:] = decoder[:] = 1
    for key, value in recorder.arrays().items():
        assert value.all() if key == "policy_inference_returned" else not value.any()


def test_empty_trace_has_explicit_zero_row_shapes():
    arrays = trace.RecordedPolicy(Policy()).arrays()
    assert arrays["policy_encoder267"].shape == (0, 267)
    assert arrays["policy_raw23"].shape == (0, 23)
    assert arrays["policy_inference_returned"].shape == (0,)


@pytest.mark.parametrize("change", ["dtype", "shape", "nan"])
def test_changed_input_boundary_rejects_before_inference(change):
    policy = Policy()
    encoder = np.zeros(267, np.float32)
    if change == "dtype":
        encoder = encoder.astype(np.float64)
    elif change == "shape":
        encoder = encoder.reshape(1, 267)
    else:
        encoder[0] = np.nan
    with pytest.raises(ValueError, match="finite float32"):
        trace.RecordedPolicy(policy).infer(encoder, np.zeros(930, np.float32))
    assert not policy.calls


def test_failed_inference_is_recorded_without_fabricated_outputs():
    class Failing:
        def infer(self, encoder, history):
            raise RuntimeError("original failure")

    recorder = trace.RecordedPolicy(Failing())
    with pytest.raises(RuntimeError, match="original failure"):
        recorder.infer(np.zeros(267, np.float32), np.zeros(930, np.float32))
    arrays = recorder.arrays()
    assert np.isnan(arrays["policy_raw23"]).all()
    assert not arrays["policy_inference_returned"].any()
    assert recorder.records[0]["error"] == "RuntimeError: original failure"


def test_recorded_case_leaves_original_physics_arrays_and_result_unchanged(monkeypatch):
    original_arrays = {"physics": np.array([1.0, 2.0])}
    original_result = dict(completed_transitions=1, requested_transitions=2)

    def run(**kwargs):
        kwargs["policy"].infer(np.zeros(267, np.float32), np.zeros(930, np.float32))
        return original_result, original_arrays

    monkeypatch.setattr(trace, "run_interior_case", run)
    result, arrays = trace.run_recorded_case(policy=Policy(), fraction=1.0)
    assert arrays["physics"] is original_arrays["physics"]
    assert result["policy_input_trace"]["inference_calls"] == 1
    assert list(original_arrays) == ["physics"]
    assert original_result == dict(completed_transitions=1, requested_transitions=2)
