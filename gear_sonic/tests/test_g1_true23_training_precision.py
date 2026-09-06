from types import SimpleNamespace

import pytest
import torch

from gear_sonic.utils import g1_true23_training_precision as precision


@pytest.fixture
def neutral_backend(monkeypatch):
    for name in precision.OVERRIDES:
        monkeypatch.delenv(name, raising=False)


def test_context_binds_actual_helper_changes_only_tf32_and_restores(neutral_backend):
    from mjlab.utils import torch as backend

    original, before = backend.configure_torch_backends, precision.backend_state()
    with precision.ieee_training_precision() as (contract, guard):
        assert contract["requested_state"] == precision.backend_state() == precision.EXPECTED_STATE
        assert contract["changed_from_inherited_backend_defaults"] == [
            "global_fp32",
            "cuda_matmul_fp32",
            "cudnn_fp32",
            "cudnn_conv_fp32",
            "cudnn_rnn_fp32",
        ]
        backend.configure_torch_backends()
        guard()
        with pytest.raises(ValueError, match="cannot enable TF32"):
            backend.configure_torch_backends(allow_tf32=True)
        with pytest.raises(ValueError, match="change determinism"):
            backend.configure_torch_backends(deterministic=True)
    assert backend.configure_torch_backends is original
    assert precision.backend_state() == before


@pytest.mark.parametrize("name", precision.OVERRIDES)
@pytest.mark.parametrize("value", ["1", "unknown"])
def test_override_rejects_before_training(monkeypatch, neutral_backend, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match="environment overrides"), precision.ieee_training_precision():
        pytest.fail("must not enter IEEE context")


def test_source_helper_change_rejects(monkeypatch, neutral_backend):
    monkeypatch.setattr(precision, "file_sha256", lambda path: "0" * 64)
    with pytest.raises(ValueError, match="helper changed"), precision.ieee_training_precision():
        pytest.fail("must not configure unverified helper")


def test_guard_detects_precision_flip_and_exception_restores(neutral_backend):
    before = precision.backend_state()
    with pytest.raises(ValueError, match="contract changed"):
        with precision.ieee_training_precision() as (_, guard):
            torch.backends.cuda.matmul.fp32_precision = "tf32"
            guard()
    assert precision.backend_state() == before


def test_guard_detects_environment_change(monkeypatch, neutral_backend):
    with precision.ieee_training_precision() as (_, guard):
        monkeypatch.setenv("NVIDIA_TF32_OVERRIDE", "0")
        with pytest.raises(ValueError, match="contract changed"):
            guard()


def test_guard_does_not_consume_random_state(neutral_backend):
    before = torch.get_rng_state().clone()
    with precision.ieee_training_precision() as (_, guard):
        guard()
    torch.testing.assert_close(before, torch.get_rng_state(), rtol=0, atol=0)


def test_model_and_update_guards_preserve_outputs_and_stop_changed_precision(neutral_backend):
    calls = []
    algorithm = SimpleNamespace(
        actor=torch.nn.Linear(3, 2),
        critic=torch.nn.Linear(3, 1),
        update=lambda: calls.append("update") or {"loss": 1.0},
    )
    data = torch.zeros(1, 3)
    expected = algorithm.actor(data).detach().clone()
    with precision.ieee_training_precision() as (_, guard):
        precision.guard_algorithm(algorithm, guard)
        torch.testing.assert_close(expected, algorithm.actor(data), rtol=0, atol=0)
        assert algorithm.update() == {"loss": 1.0} and calls == ["update"]
        torch.backends.cuda.matmul.fp32_precision = "tf32"
        for operation in (lambda: algorithm.actor(data), lambda: algorithm.critic(data), algorithm.update):
            with pytest.raises(ValueError, match="contract changed"):
                operation()
        assert calls == ["update"]


def test_non_float32_model_rejects(neutral_backend):
    algorithm = SimpleNamespace(actor=torch.nn.Linear(3, 2).double(), critic=torch.nn.Linear(3, 1))
    with precision.ieee_training_precision() as (_, guard):
        with pytest.raises(ValueError, match="float32 parameters"):
            precision.guard_algorithm(algorithm, guard)


def test_runtime_resume_requires_identical_receipt(tmp_path):
    path = tmp_path / "precision.json"
    precision.write_runtime(path, {"precision": "ieee"})
    before = path.read_bytes()
    precision.write_runtime(path, {"precision": "ieee"})
    with pytest.raises(ValueError, match="changed on resume"):
        precision.write_runtime(path, {"precision": "tf32"})
    assert path.read_bytes() == before
