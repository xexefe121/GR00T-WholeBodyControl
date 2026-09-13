import pytest

from gear_sonic.scripts.optimize_g1_true23_pd_trajectory import main


def required_args():
    return ["--reproduction-directory", "unused", "--asset-root", "unused", "--output-directory", "unused"]


@pytest.mark.parametrize("value", ["-0.001", "0.006", "nan", "inf"])
def test_invalid_predictive_clearance_stops_before_input_loading(value, capsys):
    with pytest.raises(SystemExit) as exc:
        main(required_args() + ["--step-method", "contact_predictive", "--planning-clearance-m", value])
    assert exc.value.code == 2
    assert "between zero and five millimetres" in capsys.readouterr().err


@pytest.mark.parametrize("value", ["0", "nan", "inf"])
def test_invalid_regularization_rejected(value, capsys):
    with pytest.raises(SystemExit) as exc:
        main(required_args() + ["--regularizations", value])
    assert exc.value.code == 2
    assert "finite and positive" in capsys.readouterr().err


def test_predictive_search_cannot_omit_protected_referee(capsys):
    with pytest.raises(SystemExit) as exc:
        main(required_args() + ["--step-method", "contact_predictive", "--objective", "original"])
    assert exc.value.code == 2
    assert "requires the protected physical referee" in capsys.readouterr().err


def test_unused_box_clearance_cannot_be_ignored(capsys):
    with pytest.raises(SystemExit) as exc:
        main(required_args() + ["--planning-clearance-m", "0"])
    assert exc.value.code == 2
    assert "requires the predictive local step" in capsys.readouterr().err


def test_standing_guidance_cannot_omit_protected_referee(capsys):
    with pytest.raises(SystemExit) as exc:
        main(required_args() + ["--standing-velocity-guidance", "--objective", "original"])
    assert exc.value.code == 2
    assert "requires the protected physical referee" in capsys.readouterr().err


def test_orientation_guidance_requires_velocity_guidance(capsys):
    with pytest.raises(SystemExit) as exc:
        main(required_args() + ["--standing-orientation-guidance"])
    assert exc.value.code == 2
    assert "requires standing velocity guidance" in capsys.readouterr().err


def test_placement_guidance_cannot_omit_protected_referee(capsys):
    with pytest.raises(SystemExit) as exc:
        main(required_args() + ["--entry-guidance", "com_lift_placement", "--objective", "original"])
    assert exc.value.code == 2
    assert "requires the protected physical referee" in capsys.readouterr().err


def test_placement_guidance_is_explicit_in_help(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert "com_lift_placement" in capsys.readouterr().out
