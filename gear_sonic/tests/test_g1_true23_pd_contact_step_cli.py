import pytest

from gear_sonic.scripts.materialize_g1_true23_pd_contact_step import main


def required_args():
    return [
        "--optimizer-report",
        "unused",
        "--reproduction-report",
        "unused",
        "--source-archive",
        "unused",
        "--asset-root",
        "unused",
        "--output-directory",
        "unused",
    ]


@pytest.mark.parametrize("clearance", ["-0.001", "0.006", "nan", "inf"])
def test_invalid_planning_clearance_rejected_before_loading_any_inputs(clearance, capsys):
    with pytest.raises(SystemExit) as exc:
        main(required_args() + ["--step-method", "contact_predictive", "--planning-clearance-m", clearance])
    assert exc.value.code == 2
    assert "between zero and five millimetres" in capsys.readouterr().err


def test_clearance_override_cannot_be_silently_ignored(capsys):
    with pytest.raises(SystemExit) as exc:
        main(required_args() + ["--step-method", "box", "--planning-clearance-m", "0"])
    assert exc.value.code == 2
    assert "requires a reactive or predictive" in capsys.readouterr().err


def test_orientation_guidance_requires_velocity_protection(capsys):
    with pytest.raises(SystemExit) as exc:
        main(required_args() + ["--standing-orientation-guidance"])
    assert exc.value.code == 2
    assert "requires standing velocity guidance" in capsys.readouterr().err


def test_placement_guidance_is_explicit_in_cached_help(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert "com_lift_placement" in capsys.readouterr().out
