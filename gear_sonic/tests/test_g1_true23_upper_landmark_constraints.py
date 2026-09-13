from types import SimpleNamespace

import numpy as np
import pytest

from gear_sonic.utils.g1_true23_generalist_protected_root import audit_norms
from gear_sonic.utils.g1_true23_original_task_trajectory import TASK_SCALES
from gear_sonic.utils.g1_true23_upper_landmark_constraints import (
    fixed_rank_position_budgets,
    upper_landmark_groups,
)


@pytest.mark.parametrize("count", [3, 20, 21, 469, 546, 1091, 1408])
def test_fixed_caps_imply_original_linear_percentile_gate_even_at_fractional_rank(count):
    error = np.linspace(0.01, 0.095, count)
    error[-1] = 0.15 if count >= 21 else 0.095
    before = error.copy()
    radii, protected = fixed_rank_position_budgets(error)
    assert np.percentile(radii, 95) <= 0.0995 + 1e-15
    np.testing.assert_array_equal(error, before)
    assert len(set(protected)) == len(protected)
    assert set(np.flatnonzero(radii > 0.1)).isdisjoint(protected)
    rng = np.random.default_rng(count)
    for _ in range(10):
        feasible = radii * rng.uniform(0, 1, count)
        assert np.percentile(feasible, 95) <= 0.1


def test_naive_max_of_reference_and_threshold_would_not_preserve_fractional_p95():
    error = np.r_[np.zeros(19), 0.2]
    assert np.percentile(error, 95) < 0.1
    assert np.percentile(np.maximum(error, 0.1), 95) > 0.1
    radii, _ = fixed_rank_position_budgets(error)
    assert np.percentile(radii, 95) < 0.1


@pytest.mark.parametrize("bad", [[0.2] * 21, [0.0, np.nan, 0.0], [-0.1, 0, 0], [0, 0], [[0, 0, 0]]])
def test_bad_or_already_failed_anchor_cannot_expand_position_gate(bad):
    with pytest.raises(ValueError):
        fixed_rank_position_budgets(bad)


@pytest.mark.parametrize("limit", [0, -0.1, 0.1001, np.inf, np.nan])
def test_position_limit_cannot_be_raised(limit):
    with pytest.raises(ValueError):
        fixed_rank_position_budgets([0, 0, 0], limit)


def test_groups_use_exact_existing_position_rows_not_orientation_or_other_frames():
    tasks = [SimpleNamespace(name=name) for name in TASK_SCALES]
    width = 3 * sum(bool(scale) for scales in TASK_SCALES.values() for scale in scales)
    residual = np.zeros(3 * width)
    problem = SimpleNamespace(
        tasks=tasks, initial=np.zeros((3, 29)), evaluate=lambda *args, **kwargs: (residual.copy(), None, None)
    )
    groups, report = upper_landmark_groups(problem, problem.initial)
    assert len(groups) == 9 and report["surrogate_stricter_than_final_gate"]
    assert audit_norms(residual, groups)["passed"]
    for group in groups:
        changed = residual.copy()
        changed[group["indices"][0]] = 20 * 0.101
        failed = audit_norms(changed, groups)
        assert not failed["passed"]
        assert failed["categories"][group["name"]]["failed_frames"] == [group["frame"]]
        assert group["indices"].min() >= group["frame"] * width
        assert group["indices"].max() < (group["frame"] + 1) * width
