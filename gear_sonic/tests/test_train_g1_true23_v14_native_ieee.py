from dataclasses import replace
import json
from types import SimpleNamespace

import pytest
import torch

from gear_sonic.scripts import train_g1_true23_v14_native_ieee as launcher


@pytest.mark.parametrize("args", [["--resume", "model.pt"], ["--resume=model.pt"]])
def test_fresh_run_cannot_relabel_resume(args):
    with pytest.raises(ValueError, match="fresh run"):
        launcher.fresh_arguments(["train", *args])


@pytest.mark.parametrize(
    "args",
    [
        ["--learning-rate", "nan"],
        ["--learning-rate=0.1"],
        ["--learning-rate"],
        ["--learning-rate=5e-6", "--learning-rate", "5e-6"],
    ],
)
def test_learning_rate_binding_cannot_lie(args):
    with pytest.raises(ValueError):
        launcher.fresh_arguments(["train", *args])


def test_exact_original_initial_learning_rate_is_default():
    assert launcher.fresh_arguments(["train"]) == ["train", "--learning-rate", "5e-6"]
    args = ["train", "--learning-rate=0.000005"]
    assert launcher.fresh_arguments(args) == args


def sidecar(tmp_path, **changes):
    data = dict(
        kind="g1_true23_motion_corpus_spans_v1",
        fps=50,
        clip_count=2,
        total_frames=40,
        spans=[dict(start=0, length=20), dict(start=20, length=20)],
    )
    data.update(changes)
    path = tmp_path / "corpus.spans.json"
    path.write_text(json.dumps(data))
    return path


def profile():
    return replace(
        launcher.NativeSupportActuationProfile.from_sim_config(launcher.ROOT / launcher.SIM_CONFIG),
        consistent_controller_state=True,
    )


def test_comparison_does_not_claim_original_reproduction_or_lora_ablation(tmp_path):
    contract = launcher.comparison_contract(sidecar(tmp_path), profile())
    assert contract["original_v14_trainable_actor_tensor_count"] == 4
    assert contract["intentional_changes_from_original_v14"] == ["native_support_stateful_v2", "ieee_float32"]
    for flag in (
        "one_variable_lora_ablation",
        "historical_run_reproduced_unmodified",
        "resume_supported",
        "standing_retention_loss_used",
        "standing_lora_bootstrap_used",
        "frozen_sonic_lora_used",
        "deployment_ready",
        "hardware_authorized",
        "promotion_eligible",
    ):
        assert contract[flag] is False


@pytest.mark.parametrize(
    "changes",
    [
        dict(fps=30),
        dict(clip_count=True),
        dict(total_frames=39),
        dict(spans=[dict(start=0, length=20), dict(start=19, length=20)]),
        dict(spans=[dict(start=0, length=15), dict(start=15, length=25)]),
    ],
)
def test_changed_tempo_or_invalid_causal_spans_rejected(tmp_path, changes):
    with pytest.raises(ValueError):
        launcher.comparison_contract(sidecar(tmp_path, **changes), profile())


def test_stateless_profile_rejected(tmp_path):
    with pytest.raises(ValueError, match="stateful"):
        launcher.comparison_contract(sidecar(tmp_path), replace(profile(), consistent_controller_state=False))


def test_checkpoint_guards_precede_save_and_load_never_runs(monkeypatch, tmp_path):
    events = []

    class Parent:
        def __init__(self):
            actor = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1))
            self.alg = SimpleNamespace(get_policy=lambda: actor, learning_rate=5e-6)
            self.checkpoint_dir = tmp_path / "checkpoints"

        def assert_frozen_actor_unchanged(self):
            events.append("frozen")

        def save(self, path):
            events.append("save")
            return path

    monkeypatch.setattr(launcher, "guard_algorithm", lambda *args: None)
    monkeypatch.setattr(launcher, "backend_state", lambda: {"precision": "ieee"})
    receipt = []
    monkeypatch.setattr(launcher, "write_runtime", lambda path, doc: receipt.append(doc))
    runner = launcher.guarded_runner(Parent, {}, {}, lambda: events.append("precision"))()
    assert receipt[0]["trainable_actor_elements"] == 9
    assert receipt[0]["checkpoint_frozen_actor_guard"] is True
    events.clear()
    assert runner.save("causal_model_100.pt") == "causal_model_100.pt"
    assert events == ["precision", "frozen", "save"]
    with pytest.raises(ValueError, match="fresh-only"):
        runner.load("causal_model_100.pt")


def test_drift_cannot_be_saved(monkeypatch):
    class Parent:
        def assert_frozen_actor_unchanged(self):
            raise ValueError("frozen drift")

        def save(self, *args):
            pytest.fail("a drifted model must never be saved")

    cls = launcher.guarded_runner(Parent, {}, {}, lambda: None)
    with pytest.raises(ValueError, match="frozen drift"):
        cls.__new__(cls).save("model.pt")
