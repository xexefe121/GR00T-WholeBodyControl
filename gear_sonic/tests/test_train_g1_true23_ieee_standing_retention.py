from contextlib import contextmanager
from types import SimpleNamespace

from gear_sonic.scripts import train_g1_true23_ieee_standing_retention as launcher


def test_main_keeps_original_retention_arguments_and_context(monkeypatch):
    events = []

    @contextmanager
    def context():
        events.append("enter")
        yield {"precision": "ieee"}, lambda: None
        events.append("exit")

    monkeypatch.setattr(launcher, "ieee_training_precision", context)
    monkeypatch.setattr(launcher, "install_hooks", lambda contract, guard: events.append(contract))
    monkeypatch.setattr(launcher.retention, "main", lambda args: events.append(args) or 0)
    args = ["train", "--resume", "model_2.pt", "--standing-retention-weight", "10"]
    assert launcher.main(args) == 0
    assert events == ["enter", {"precision": "ieee"}, args, "exit"]


def test_precision_runner_wraps_already_installed_retention_and_binds_lineage(monkeypatch, tmp_path):
    events = []

    class Parent:
        def __init__(self):
            self.alg = SimpleNamespace(retention_installed=True)
            self.checkpoint_dir = tmp_path / "checkpoints"
            events.append("parent_initialized")

        def load(self, path):
            return {"loaded": path}

        def _checkpoint_payload(self):
            return {"original_checkpoint": True}

    frozen = SimpleNamespace(
        FrozenPlatformLoraRunner=Parent,
        base=SimpleNamespace(_resolved_training_config=lambda: {"old": True}, CAUSAL_SOURCE_FILES=()),
    )
    frozen._install_frozen_lora_hooks = lambda **kwargs: events.append(frozen.FrozenPlatformLoraRunner)
    monkeypatch.setattr(launcher.retention.standing, "frozen", frozen)
    monkeypatch.setattr(
        launcher, "guard_algorithm", lambda algorithm, guard: events.append(algorithm.retention_installed)
    )
    monkeypatch.setattr(launcher, "backend_state", lambda: {"actual": "ieee"})
    monkeypatch.setattr(launcher, "write_runtime", lambda path, document: events.append(document))
    launcher.install_hooks({"requested": "ieee"}, lambda: events.append("guard"))
    frozen._install_frozen_lora_hooks()
    assert issubclass(events[0], Parent)
    runner = frozen.FrozenPlatformLoraRunner()
    assert events[1:4] == ["guard", "parent_initialized", True]
    assert runner.load("model_2.pt") == {"loaded": "model_2.pt"}
    assert runner._checkpoint_payload() == {"original_checkpoint": True}
    assert frozen.base._resolved_training_config() == {"old": True, "training_precision": {"requested": "ieee"}}
    assert [path.name for path in frozen.base.CAUSAL_SOURCE_FILES] == [
        "train_g1_true23_ieee_standing_retention.py",
        "g1_true23_training_precision.py",
    ]
