import numpy as np
import pytest

from gear_sonic.utils.g1_true23_reference_lineage import root_path_repair_bounds


def test_translation_and_yaw_are_not_mistaken_for_drift():
    source = np.column_stack((np.linspace(0, 2, 16), np.zeros(16), np.ones(16)))
    candidate = source @ np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]) + [3, 4, 5]
    result = root_path_repair_bounds(source, candidate, maximum_offset_m=0)
    assert result["minimum_possible_maximum_horizontal_error_allowing_yaw_m"] == pytest.approx(0, abs=1e-12)
    assert result["maximum_translation_aligned_root_error_m"] > 2
    assert not result["original_choreography_parity_proven"]


def test_endpoint_corrections_are_counted_twice_and_cannot_hide_large_drift():
    source = np.array([[0, 0, 0], [9, 0, 0]], dtype=float)
    candidate = np.array([[0, 0, 0], [8.3, 0, 0]], dtype=float)
    result = root_path_repair_bounds(source, candidate, maximum_offset_m=0.08)
    assert result["minimum_possible_maximum_error_under_xyz_box_m"] == pytest.approx(0.54)
    assert result["minimum_possible_maximum_horizontal_error_allowing_yaw_m"] == pytest.approx(
        0.7 - 0.16 * np.sqrt(2)
    )
    assert result["xyz_lower_bound_worst_frame"] == 1


def test_every_frame_matters_even_when_endpoints_match():
    source = np.array([[0, 0, 1], [4, 0, 1], [0, 0, 1]], dtype=float)
    candidate = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=float)
    result = root_path_repair_bounds(source, candidate, maximum_offset_m=0.08)
    assert result["frames_checked"] == 3 and result["frames_dropped"] == 0
    assert result["minimum_possible_maximum_error_under_xyz_box_m"] == pytest.approx(3.84)
    assert result["xyz_lower_bound_worst_frame"] == 1


def test_bound_holds_for_random_feasible_corrections_without_mutating_inputs():
    rng = np.random.default_rng(711)
    original, candidate = rng.normal(size=(2, 20, 3))
    snapshots = original.copy(), candidate.copy()
    result = root_path_repair_bounds(original, candidate, maximum_offset_m=0.08)
    for _ in range(50):
        repaired = candidate + rng.uniform(-0.08, 0.08, candidate.shape)
        error = (repaired - repaired[0]) - (original - original[0])
        assert (
            np.linalg.norm(error, axis=1).max() >= result["minimum_possible_maximum_error_under_xyz_box_m"] - 1e-12
        )
    np.testing.assert_array_equal(original, snapshots[0])
    np.testing.assert_array_equal(candidate, snapshots[1])


@pytest.mark.parametrize("offset", [-1, np.nan, np.inf, True])
def test_invalid_bounds_rejected(offset):
    with pytest.raises(ValueError):
        root_path_repair_bounds(np.zeros((3, 3)), np.zeros((3, 3)), maximum_offset_m=offset)


@pytest.mark.parametrize("bad", [np.zeros((1, 3)), np.zeros((3, 2)), np.full((3, 3), np.nan)])
def test_partial_or_nonfinite_paths_rejected(bad):
    with pytest.raises(ValueError):
        root_path_repair_bounds(np.zeros((3, 3)), bad, maximum_offset_m=0.08)
