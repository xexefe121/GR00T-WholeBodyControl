"""Explicit stage transfer preserves ownership, never optimizer resume."""

from types import SimpleNamespace

import pytest
import torch

from gear_sonic.scripts import train_g1_true23_generalist_curriculum as trainer


@pytest.fixture
def parent(monkeypatch, tmp_path):
    from gear_sonic.scripts import export_g1_true23_generalist as exporter
    from gear_sonic.trl.mjlab import native23_generalist_runner as runner

    path = tmp_path / "parent.pt"
    path.write_bytes(b"explicit unit fixture, not a checkpoint")
    audit = {"manifest_sha256": "manifest", "split_sha256": "split"}
    checkpoint = {
        "actor": {"state_sha256": "actor"},
        "lineage_sha256": "lineage",
        "lineage": {},
        "trainer_state": {"completed_update_count": 2},
    }
    monkeypatch.setattr(torch, "load", lambda *a, **kw: checkpoint)
    monkeypatch.setattr(
        exporter,
        "validate_export_semantics",
        lambda value: {"training_configuration": {"training_inputs": {"corpus_audit": audit}}},
    )
    monkeypatch.setattr(runner, "validate_generalist_checkpoint", lambda *a, **kw: None)
    return path, audit, checkpoint


def test_transfer_contract_distinguishes_fresh_optimizer(parent):
    path, audit, _ = parent
    contract = trainer.parent_actor_contract(path, {"smoke_only": False, "corpus_audit": audit})
    assert contract["completed_update_count"] == 2
    assert contract["actor_and_bounded_noise_transferred"]
    assert not contract["critic_optimizer_counters_or_rng_resumed"]


@pytest.mark.parametrize("audit", [None, {"manifest_sha256": "different", "split_sha256": "split"}])
def test_production_stage_transfer_rejects_different_or_unknown_split(parent, audit):
    with pytest.raises(ValueError, match="identical audited corpus"):
        trainer.parent_actor_contract(parent[0], {"smoke_only": False, "corpus_audit": audit})


def test_untrained_parent_not_a_curriculum_continuation(parent):
    parent[2]["trainer_state"]["completed_update_count"] = 0
    with pytest.raises(ValueError, match="completed training update"):
        trainer.parent_actor_contract(parent[0])


def test_parent_transfer_copies_actor_only(parent):
    path, _, checkpoint = parent
    contract = trainer.parent_actor_contract(path)
    loaded = []
    actor = SimpleNamespace(load_training_artifact=lambda value: loaded.append(value))
    optimizer = SimpleNamespace(state={})
    runner = SimpleNamespace(
        completed_update_count=0,
        alg=SimpleNamespace(optimizer=optimizer, get_policy=lambda: actor),
        _assert_boundary=lambda: None,
    )
    trainer.initialize_actor_from_parent(runner, contract)
    assert loaded == [checkpoint["actor"]]
    assert runner.completed_update_count == 0 and optimizer.state == {}


def test_parent_transfer_refuses_nonfresh_optimizer(parent):
    contract = trainer.parent_actor_contract(parent[0])
    runner = SimpleNamespace(
        completed_update_count=0, alg=SimpleNamespace(optimizer=SimpleNamespace(state={"already_trained": True}))
    )
    with pytest.raises(ValueError, match="fresh optimizer"):
        trainer.initialize_actor_from_parent(runner, contract)


def test_parent_hash_change_rejected_before_loading(parent):
    contract = trainer.parent_actor_contract(parent[0])
    parent[0].write_bytes(b"changed")
    with pytest.raises(ValueError, match="bound hash"):
        trainer.initialize_actor_from_parent(SimpleNamespace(), contract)


def test_no_parent_is_explicit_released_initialization():
    assert trainer.parent_actor_contract(None) is None
    trainer.initialize_actor_from_parent(SimpleNamespace(), None)
