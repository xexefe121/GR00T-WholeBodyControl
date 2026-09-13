"""Explicit bounded planned-source CLI options; no simulator or transport run."""

import pytest

from gear_sonic.scripts.retarget_g1_true23_generalist_planned_trace import main, planned_adaptation_options


def test_full_excursion_probe_is_one_unchanged_bound_candidate():
    old = planned_adaptation_options(root_reference_refinement=True)
    new = planned_adaptation_options(root_reference_refinement=True, preserve_source_excursion=True)
    assert old["limits"].excursion_scales == (0.9,)
    assert new["limits"].excursion_scales == (1.0,)
    assert new["limits"].duration_scales == (2.0,)
    assert new["limits"].foot_p95_m == old["limits"].foot_p95_m
    assert new["limits"].hand_head_p95_m == old["limits"].hand_head_p95_m
    assert new["root_reference_refinement"]


def test_full_excursion_cannot_silently_enable_another_solver_sweep():
    with pytest.raises(ValueError, match="single bounded"):
        planned_adaptation_options(root_reference_refinement=False, preserve_source_excursion=True)
    with pytest.raises(ValueError, match="boolean"):
        planned_adaptation_options(root_reference_refinement=True, preserve_source_excursion=1)


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


@pytest.mark.parametrize("other_options", [[], ["--root-reference-refinement"]])
def test_restoration_requires_a_bound_rejected_diagnostic_before_file_access(tmp_path, capsys, other_options):
    output = tmp_path / "must_not_be_created"
    with pytest.raises(SystemExit) as error:
        main(
            [
                "--trace",
                str(tmp_path / "missing_trace.npz"),
                "--source-model",
                str(tmp_path / "missing_source.mjb"),
                "--output-directory",
                str(output),
                "--feasibility-restoration",
                *other_options,
            ]
        )
    assert error.value.code == 2
    assert "--feasibility-restoration requires --protected-root-refinement-from" in capsys.readouterr().err
    assert not output.exists()
