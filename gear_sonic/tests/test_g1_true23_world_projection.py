"""New training boundary and passive actual-mean capture contracts."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gear_sonic.scripts.verify_g1_true23_world_projection_update import (
    audit_projection_arrays,
    independent_bounds,
)
from gear_sonic.trl.mjlab import native23_world_projection_runner as new, native23_world_quality_runner as previous
from gear_sonic.trl.mjlab.native23_projected_target_ppo import (
    PROFILE,
    projection_objective_contract,
    source_projection_loss,
)
from gear_sonic.utils import g1_true23_world_projection_checkpoint as reader


def fixture_checkpoint():
    return dict(
        header=copy.deepcopy(new.CHECKPOINT_HEADER),
        lineage=dict(
            materials=dict(
                resolved_config=dict(
                    payload=dict(
                        native23_world_projection=new.training_contract(),
                        ppo_auxiliary_objective=projection_objective_contract(PROFILE),
                        agent=dict(algorithm=dict(class_name=new.ALGORITHM)),
                    )
                ),
                source_files=dict(
                    files=[
                        dict(logical_path=name)
                        for name in (
                            "gear_sonic/trl/mjlab/native23_world_projection_runner.py",
                            "gear_sonic/trl/mjlab/native23_projected_target_ppo.py",
                            "gear_sonic/utils/g1_true23_world_projection_checkpoint.py",
                            "gear_sonic/scripts/train_g1_true23_world_projection.py",
                        )
                    ]
                ),
            )
        ),
    )


def test_header_view_never_relabels_source():
    checkpoint = fixture_checkpoint()
    before = copy.deepcopy(checkpoint)
    result = new.quality_schema_view(checkpoint)
    assert checkpoint == before
    assert result["header"] == previous.CHECKPOINT_HEADER
    assert result["lineage"] is checkpoint["lineage"]


def test_previous_reader_rejects_changed_training_objective():
    with pytest.raises(ValueError):
        previous.world_schema_view(fixture_checkpoint())


def test_new_reader_rejects_previous_snapshot():
    checkpoint = fixture_checkpoint()
    checkpoint["header"] = previous.CHECKPOINT_HEADER
    with pytest.raises(ValueError):
        new.quality_schema_view(checkpoint)


@pytest.mark.parametrize("field", ["native23_world_projection", "ppo_auxiliary_objective", "agent"])
def test_missing_objective_or_algorithm_rejected(field):
    checkpoint = fixture_checkpoint()
    del checkpoint["lineage"]["materials"]["resolved_config"]["payload"][field]
    with pytest.raises(ValueError):
        new.quality_schema_view(checkpoint)


def test_changed_coefficient_rejected():
    checkpoint = fixture_checkpoint()
    checkpoint["lineage"]["materials"]["resolved_config"]["payload"]["ppo_auxiliary_objective"]["weight"] = 0.1
    with pytest.raises(ValueError):
        new.quality_schema_view(checkpoint)


def test_source_closure_requires_actual_loss_code(monkeypatch):
    monkeypatch.setattr(reader.quality, "validate_semantics", lambda checkpoint: {})
    checkpoint = fixture_checkpoint()
    assert reader.validate_semantics(checkpoint)["world_projection"] == new.training_contract()
    checkpoint["lineage"]["materials"]["source_files"]["files"].pop(1)
    with pytest.raises(ValueError, match="source closure"):
        reader.validate_semantics(checkpoint)


def test_capture_is_detached_passive_and_distinguishes_rollout_from_minibatch():
    runner = object.__new__(new.Native23WorldProjectionRunner)
    runner._projection_learning = True
    runner._projection_updating = False
    runner._projection_rollouts, runner._projection_minibatches = [], []
    runner.env = SimpleNamespace(unwrapped=SimpleNamespace(common_step_counter=13))
    mean = torch.ones(4, 23, requires_grad=True)
    std, sample = torch.full((4, 23), 0.1), torch.zeros(4, 23)
    module = SimpleNamespace(output_distribution_params=(mean, std))
    rng = torch.get_rng_state().clone()
    assert runner._capture_mean(module, (), {"stochastic_output": True}, sample) is None
    row = runner._projection_rollouts[0]
    assert row["common_step_counter"] == 13 and not row["mean"].requires_grad
    assert row["mean"].data_ptr() != mean.data_ptr()
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    runner._projection_updating = True
    runner._capture_mean(module, (), {"stochastic_output": True}, sample)
    assert len(runner._projection_minibatches) == 1
    assert "sample" not in runner._projection_minibatches[0]
    runner._capture_mean(module, (), {}, sample)
    runner._projection_learning = False
    runner._capture_mean(module, (), {"stochastic_output": True}, sample)
    assert len(runner._projection_minibatches) == 1


def captured_fixture():
    low, high = independent_bounds()
    mean = np.tile(((low + high) / 2).astype(np.float32), (8, 4, 1))
    mean[:, 0, 0] = low[0] - 0.5
    losses, fractions = [], []
    for row in mean:
        loss, projected = source_projection_loss(torch.from_numpy(row))
        losses.append(float(loss))
        fractions.append(float(projected.float().mean()))
    arrays = dict(
        rollout_mean=mean[:, :2].copy(),
        rollout_std=np.full((8, 2, 23), 0.1, dtype=np.float32),
        rollout_sample=mean[:, :2].copy(),
        minibatch_mean=mean,
        minibatch_std=np.full((8, 4, 23), 0.1, dtype=np.float32),
        rollout_common_step_counter=np.arange(8),
    )
    metadata = dict(
        training_contract=new.training_contract(),
        training_state_poisoned=False,
        completed_updates=2,
        actual_rollout_calls=8,
        actual_minibatch_calls=8,
        algorithm_receipt=dict(completed_ppo_updates=2, completed_minibatches=8, coefficient=0.05),
        update_losses=[
            dict(
                source_projection=float(np.mean(losses[i : i + 4])),
                source_mean_projected_fraction=float(np.mean(fractions[i : i + 4])),
            )
            for i in (0, 4)
        ],
    )
    return arrays, metadata


def test_independent_actual_minibatch_loss_reconstruction():
    arrays, metadata = captured_fixture()
    report = audit_projection_arrays(arrays, metadata, 2, 4, 2, 2, 2)
    assert report["actual_transitions"] == 16
    assert report["actual_minibatch_policy_mean_rows"] == 32
    assert report["independently_reconstructed_projection_loss_max_error"] < 1e-7
    assert report["actual_training_any_joint_under_one_percent_unprojected_probability_fraction"] == 0.5


@pytest.mark.parametrize(
    "tamper", ["missing_row", "nan_mean", "duplicate_step", "coefficient", "loss", "poisoned"]
)
def test_capture_audit_fails_closed(tamper):
    arrays, metadata = captured_fixture()
    if tamper == "missing_row":
        arrays["rollout_mean"] = arrays["rollout_mean"][:-1]
    elif tamper == "nan_mean":
        arrays["minibatch_mean"][0, 0, 0] = np.nan
    elif tamper == "duplicate_step":
        arrays["rollout_common_step_counter"][1] = 0
    elif tamper == "coefficient":
        metadata["algorithm_receipt"]["coefficient"] = 0.1
    elif tamper == "loss":
        metadata["update_losses"][0]["source_projection"] += 0.1
    else:
        metadata["training_state_poisoned"] = True
    with pytest.raises(ValueError):
        audit_projection_arrays(arrays, metadata, 2, 4, 2, 2, 2)
