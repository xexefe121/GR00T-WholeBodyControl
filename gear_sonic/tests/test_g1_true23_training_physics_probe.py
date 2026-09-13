import pytest

from gear_sonic.scripts.probe_g1_true23_training_physics import sample_controls


@pytest.mark.parametrize("count", [8, 16, 725, 916, 1114, 1417])
def test_samples_include_both_endpoints_and_are_unique(count):
    selected = sample_controls(count)
    assert len(set(selected)) == 8
    assert selected[0] == 0 and selected[-1] == count - 1


@pytest.mark.parametrize("count", [0, 7, True, 10.0])
def test_invalid_sampling_rejected(count):
    with pytest.raises(ValueError):
        sample_controls(count)
