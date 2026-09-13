import numpy as np
import pytest

from gear_sonic.scripts.probe_g1_true23_training_boundary import control_batches, error_summary


@pytest.mark.parametrize("count", [11, 32, 42, 43, 725, 916, 1417])
def test_every_measured_control_once_without_synthetic_prehistory(count):
    rows = list(control_batches(count))
    np.testing.assert_array_equal(
        np.concatenate([indices[:valid] for indices, valid in rows]), np.arange(10, count)
    )
    for indices, valid in rows:
        assert len(indices) == 32 and 1 <= valid <= 32
        assert indices.min() - 9 >= 1
        assert indices.max() < count
        assert np.all(indices[valid:] == indices[valid - 1])


@pytest.mark.parametrize("count", [True, 0, 10, -1, 12.0])
def test_invalid_counts_rejected(count):
    with pytest.raises(ValueError):
        list(control_batches(count))


def test_error_summary_exposes_single_outlier_and_counts_rows():
    expected = np.zeros((100, 4), dtype=np.float32)
    actual = expected.copy()
    actual[67, 2] = 0.0625
    result = error_summary(actual, expected)
    assert result["max_abs"] == 0.0625
    assert result["row_max_p95"] == 0
    assert result["different_rows"] == 1
    assert (result["worst_row"], result["worst_feature"]) == (67, 2)


@pytest.mark.parametrize("actual", [np.zeros((2, 4)), np.full((3, 4), np.nan), np.zeros((3, 4, 1))])
def test_invalid_error_inputs_rejected(actual):
    with pytest.raises(ValueError):
        error_summary(actual, np.zeros((3, 4)))
