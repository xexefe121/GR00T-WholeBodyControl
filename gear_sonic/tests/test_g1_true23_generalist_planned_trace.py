"""Explicit bounded planned-source CLI options; no simulator or transport run."""

import pytest

from gear_sonic.scripts.retarget_g1_true23_generalist_planned_trace import planned_adaptation_options


def test_root_refinement_requests_exactly_one_approved_candidate():
    options = planned_adaptation_options(root_reference_refinement=True)
    assert options["root_reference_refinement"] is True
    assert options["limits"].duration_scales == (2.0,)
    assert options["limits"].excursion_scales == (0.9,)


def test_default_keeps_bounded_original_sweep_without_root_refinement():
    options = planned_adaptation_options(root_reference_refinement=False)
    assert options["root_reference_refinement"] is False
    assert options["limits"].duration_scales == (1.0, 1.25, 1.5, 2.0)
    assert options["limits"].excursion_scales == (1.0, 0.9, 0.8)


@pytest.mark.parametrize("value", [None, 0, 1, "true"])
def test_refinement_option_cannot_use_implicit_truthiness(value):
    with pytest.raises(ValueError, match="boolean"):
        planned_adaptation_options(root_reference_refinement=value)
