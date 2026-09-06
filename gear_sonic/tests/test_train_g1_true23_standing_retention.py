import pytest

from gear_sonic.scripts import train_g1_true23_standing_retention as launcher


def fixture(monkeypatch):
    captured, delegated = {}, []
    monkeypatch.setattr(launcher.standing, "read_standing_initialization", lambda path: ({}, {"checked": True}))
    monkeypatch.setattr(
        launcher, "load_training_states", lambda path, descriptor: ({"states": True}, {"bound": True})
    )
    monkeypatch.setattr(launcher, "install_hooks", lambda data, inputs, **kwargs: captured.update(kwargs))
    monkeypatch.setattr(launcher.standing, "main", lambda values: delegated.extend(values) or 0)
    return (
        ["train", "--standing-teacher-report=teacher.json", "--standing-bootstrap-report=fit.json"],
        captured,
        delegated,
    )


def test_retention_options_are_lineage_parameters_not_forwarded_unknown_flags(monkeypatch):
    args, captured, delegated = fixture(monkeypatch)
    assert (
        launcher.main(args + ["--standing-retention-weight=10", "--standing-retention-batch-size=64", "--seed=5"])
        == 0
    )
    assert captured == {"weight": 10.0, "batch_size": 64, "seed": 5}
    assert "--standing-teacher-report" not in delegated
    assert "--standing-retention-weight" not in delegated
    assert delegated == ["train", "--standing-bootstrap-report", "fit.json", "--seed", "5"]


@pytest.mark.parametrize(
    "extra",
    [
        ["--standing-retention-weight=nan"],
        ["--standing-retention-weight=-1"],
        ["--standing-retention-batch-size=0"],
        ["--standing-retention-batch-size=1501"],
        ["--seed=-1"],
    ],
)
def test_invalid_retention_parameters_never_start_training(monkeypatch, extra):
    args, _, delegated = fixture(monkeypatch)
    with pytest.raises(SystemExit, match="invalid standing retention"):
        launcher.main(args + extra)
    assert delegated == []


def test_actual_resume_is_left_for_checked_standing_wrapper(monkeypatch):
    args, _, delegated = fixture(monkeypatch)
    assert launcher.main(args + ["--resume=checkpoints/frozen_lora_model_2.pt"]) == 0
    assert delegated[-2:] == ["--resume", "checkpoints/frozen_lora_model_2.pt"]


@pytest.mark.parametrize("missing", ["--standing-teacher-report", "--standing-bootstrap-report"])
def test_both_standing_sources_are_required(monkeypatch, missing):
    args, _, delegated = fixture(monkeypatch)
    args = [arg for arg in args if not arg.startswith(missing)]
    with pytest.raises(SystemExit, match="requires"):
        launcher.main(args)
    assert delegated == []
