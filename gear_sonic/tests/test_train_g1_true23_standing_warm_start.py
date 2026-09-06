import hashlib

import pytest

from gear_sonic.scripts import train_g1_true23_standing_warm_start as launcher


def setup(tmp_path, monkeypatch):
    source = tmp_path / "source.pt"
    source.write_bytes(b"fixture")
    checkpoint = tmp_path / "standing_lora.pt"
    descriptor = {
        "adapter_path": str(checkpoint),
        "frozen_platform_contract": {
            "lora_rank": 8,
            "lora_alpha": 8.0,
            "source_checkpoint_sha256": hashlib.sha256(b"fixture").hexdigest(),
        },
    }
    monkeypatch.setattr(launcher, "read_standing_initialization", lambda path: ({}, descriptor))
    captured, delegated = {}, []
    monkeypatch.setattr(
        launcher, "install_standing_hooks", lambda payload, desc, **kwargs: captured.update(kwargs)
    )
    monkeypatch.setattr(launcher.frozen, "main", lambda values: delegated.extend(values) or 0)
    args = [
        "train",
        "--standing-bootstrap-report",
        str(tmp_path / "report.json"),
        "--run-dir",
        str(tmp_path / "new"),
        "--source-checkpoint",
        str(source),
        "--actuation-profile",
        "native_support_stateful_v2",
    ]
    return args, captured, delegated, checkpoint


def test_initialization_is_distinct_from_exact_resume(tmp_path, monkeypatch):
    args, captured, delegated, checkpoint = setup(tmp_path, monkeypatch)
    assert launcher.main(args) == 0
    assert captured["initialization_mode"] is True
    assert delegated[delegated.index("--resume") + 1] == str(checkpoint)
    assert "--standing-bootstrap-report" not in delegated


def test_actual_resume_restores_new_ppo_checkpoint_not_standing_adapter(tmp_path, monkeypatch):
    args, captured, delegated, _ = setup(tmp_path, monkeypatch)
    checkpoint = tmp_path / "frozen_lora_model_10.pt"
    assert launcher.main(args + [f"--resume={checkpoint}"]) == 0
    assert captured["initialization_mode"] is False
    assert delegated[delegated.index("--resume") + 1] == str(checkpoint)


@pytest.mark.parametrize(
    "extra", [["--phase", "polish"], ["--adapter-init", "old.pt"], ["--behavior-bank", "old.json"]]
)
def test_bootstrap_does_not_bypass_polish_bank_rules(tmp_path, monkeypatch, extra):
    args, _, delegated, _ = setup(tmp_path, monkeypatch)
    with pytest.raises(SystemExit, match="breadth initialization"):
        launcher.main(args + extra)
    assert not delegated


def test_nonempty_new_run_rejected_before_resume_argument_injection(tmp_path, monkeypatch):
    args, _, delegated, _ = setup(tmp_path, monkeypatch)
    directory = tmp_path / "new"
    directory.mkdir()
    (directory / "existing.json").write_text("preserve")
    with pytest.raises(SystemExit, match="new empty run"):
        launcher.main(args)
    assert not delegated
    assert (directory / "existing.json").read_text() == "preserve"


def test_incompatible_actuation_and_rank_rejected(tmp_path, monkeypatch):
    args, _, _, _ = setup(tmp_path, monkeypatch)
    args[-1] = "stage_one_cpp"
    with pytest.raises(SystemExit, match="unchanged"):
        launcher.main(args)
    args[-1] = "native_support_stateful_v2"
    with pytest.raises(SystemExit, match="rank/alpha"):
        launcher.main(args + ["--lora-rank", "16"])
