import numpy as np
import pytest

from gear_sonic.utils.g1_true23_missing_axis_geometry import decompose_position_error, position_summary


def test_exact_retained_pose_has_geometry_error_but_zero_controller_error():
    source = np.zeros((20, 3, 3))
    ideal = source.copy()
    ideal[:, :, 0] = [0.1, 0.2, 0.3]
    result = decompose_position_error(ideal, ideal, source)
    np.testing.assert_allclose(result["total"]["p95_m"], [0.1, 0.2, 0.3])
    np.testing.assert_array_equal(result["controller_mse_m2"], [0, 0, 0])
    np.testing.assert_array_equal(result["signed_cross_term_m2"], [0, 0, 0])
    assert result["causal_attribution_fraction_claimed"] is False


def test_opposed_errors_cancel_and_cross_term_is_not_dropped():
    source = np.zeros((8, 3, 3))
    ideal = np.ones_like(source)
    result = decompose_position_error(source, ideal, source)
    np.testing.assert_array_equal(result["total_mse_m2"], [0, 0, 0])
    np.testing.assert_array_equal(result["controller_mse_m2"], [3, 3, 3])
    np.testing.assert_array_equal(result["geometry_mse_m2"], [3, 3, 3])
    np.testing.assert_array_equal(result["signed_cross_term_m2"], [-6, -6, -6])


def test_rotation_and_translation_preserve_decomposition_magnitudes():
    rng = np.random.default_rng(813)
    values = [rng.normal(size=(50, 3, 3)) for _ in range(3)]
    matrix = np.asarray([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])
    transformed = [v @ matrix + [8, -4, 2] for v in values]
    before, after = decompose_position_error(*values), decompose_position_error(*transformed)
    for name in ("total_mse_m2", "controller_mse_m2", "geometry_mse_m2", "signed_cross_term_m2"):
        np.testing.assert_allclose(before[name], after[name], atol=1e-12, rtol=0)


@pytest.mark.parametrize("invalid", [np.empty((0, 3, 3)), np.zeros((10, 3)), np.full((10, 3, 3), np.nan)])
def test_invalid_samples_rejected(invalid):
    with pytest.raises(ValueError):
        position_summary(invalid)


def test_phase_length_mismatch_rejected():
    with pytest.raises(ValueError, match="identically phased"):
        decompose_position_error(np.zeros((11, 3, 3)), np.zeros((10, 3, 3)), np.zeros((10, 3, 3)))


def test_structural_geometry_is_not_mislabelled_as_missing_joint_motion():
    original = np.zeros((9, 3, 3))
    zeroed = original.copy()
    zeroed[:, 0, 0] = 0.2
    native = zeroed.copy()
    native[:, 2, 2] = -0.01
    result = decompose_position_error(native, native, original, zeroed_source_centered=zeroed)
    np.testing.assert_allclose(result["pure_missing_axes"]["p95_m"], [0.2, 0, 0])
    np.testing.assert_allclose(result["structural_native_minus_zeroed_source"]["p95_m"], [0, 0, 0.01])
    np.testing.assert_allclose(result["reference_embodiment_gap"]["p95_m"], [0.2, 0, 0.01])
