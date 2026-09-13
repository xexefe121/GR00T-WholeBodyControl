import numpy as np
import pytest

from gear_sonic.utils.g1_true23_root_gain_diagnostic import RootPositionGainDiagnostic


class Reader:
    def infer(self, encoder, history, feedback):
        self.seen = (encoder.copy(), history.copy(), feedback.copy())
        feedback[:] = 999  # Reader cannot mutate either diagnostic record.
        return np.zeros(23, dtype=np.float32), np.zeros(994, dtype=np.float32)


@pytest.mark.parametrize("gain", [0, 2])
def test_changes_position_input_only_without_mutating_records(gain):
    reader = Reader()
    policy = RootPositionGainDiagnostic(reader, gain)
    encoder = np.arange(267, dtype=np.float32)
    history = np.arange(930, dtype=np.float32)
    root = np.arange(9, dtype=np.float32) - 2
    saved = root.copy()
    raw, combined = policy.infer(encoder, history, root)
    expected = saved.copy()
    expected[:3] *= gain
    np.testing.assert_array_equal(root, saved)
    np.testing.assert_array_equal(reader.seen[0], encoder)
    np.testing.assert_array_equal(reader.seen[1], history)
    np.testing.assert_array_equal(reader.seen[2], expected)
    np.testing.assert_array_equal(policy.actual_feedback[0], saved)
    np.testing.assert_array_equal(policy.used_feedback[0], expected)
    assert raw.shape == (23,) and combined.shape == (994,)
    assert policy.contract()["gain"] == gain
    assert not policy.contract()["nominal_training_contract_claimed"]
    assert not policy.contract()["hardware_authorized"]


@pytest.mark.parametrize("bad", [True, -1, 1, 3, np.nan, "2"])
def test_only_predeclared_non_boolean_gains_allowed(bad):
    with pytest.raises(ValueError, match="gains0 or2"):
        RootPositionGainDiagnostic(Reader(), bad)


@pytest.mark.parametrize("bad", [np.zeros(8, dtype=np.float32), np.zeros(9), np.full(9, np.nan, dtype=np.float32)])
def test_invalid_feedback_rejected_without_emission(bad):
    policy = RootPositionGainDiagnostic(Reader(), 2)
    with pytest.raises(ValueError, match="finite float32"):
        policy.infer(None, None, bad)
    assert not policy.actual_feedback and not policy.used_feedback


def test_overflow_rejected_before_reader():
    policy = RootPositionGainDiagnostic(Reader(), 2)
    with pytest.raises(ValueError, match="overflow"):
        policy.infer(None, None, np.full(9, np.finfo(np.float32).max, dtype=np.float32))
    assert not policy.actual_feedback
