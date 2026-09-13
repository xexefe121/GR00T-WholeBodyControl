import copy

import pytest

from gear_sonic.trl.mjlab import native23_world_quality_runner as new, native23_world_tracking_runner as old
from gear_sonic.utils import g1_true23_world_quality_checkpoint as reader
from gear_sonic.utils.g1_true23_world_quality import quality_contract


def fixture_checkpoint():
    return dict(
        header=copy.deepcopy(new.CHECKPOINT_HEADER),
        lineage=dict(
            materials=dict(
                resolved_config=dict(payload=dict(native23_world_quality_bonus=quality_contract())),
                source_files=dict(
                    files=[
                        dict(logical_path=name)
                        for name in (
                            "gear_sonic/utils/g1_true23_world_quality.py",
                            "gear_sonic/utils/g1_true23_world_quality_checkpoint.py",
                            "gear_sonic/trl/mjlab/native23_world_quality_runner.py",
                            "gear_sonic/scripts/train_g1_true23_world_quality.py",
                        )
                    ]
                ),
            )
        ),
    )


def test_schema_view_never_relabels_saved_checkpoint():
    checkpoint = fixture_checkpoint()
    original = copy.deepcopy(checkpoint)
    common = new.world_schema_view(checkpoint)
    assert checkpoint == original
    assert common["header"] == old.CHECKPOINT_HEADER
    assert common["lineage"] is checkpoint["lineage"]


def test_previous_reader_rejects_new_training_semantics():
    with pytest.raises(ValueError):
        old.decoder_schema_view(fixture_checkpoint())


def test_new_reader_rejects_previous_snapshot():
    checkpoint = fixture_checkpoint()
    checkpoint["header"] = old.CHECKPOINT_HEADER
    with pytest.raises(ValueError):
        new.world_schema_view(checkpoint)


@pytest.mark.parametrize(
    "field,value",
    [
        ("rate", 200.0),
        ("step_dt", 0.01),
        ("done_bonus", 1.0),
        ("hardware_authorized", True),
        ("all_existing_base_rewards_and_potential_shaping_unchanged", False),
    ],
)
def test_changed_reward_contract_rejected(field, value):
    checkpoint = fixture_checkpoint()
    checkpoint["lineage"]["materials"]["resolved_config"]["payload"]["native23_world_quality_bonus"][field] = value
    with pytest.raises(ValueError):
        new.world_schema_view(checkpoint)


def test_source_closure_requires_every_new_runtime_file(monkeypatch):
    monkeypatch.setattr(reader.world, "validate_semantics", lambda checkpoint: {})
    checkpoint = fixture_checkpoint()
    assert reader.validate_semantics(checkpoint)["world_quality_bonus"] == quality_contract()
    checkpoint["lineage"]["materials"]["source_files"]["files"].pop()
    with pytest.raises(ValueError, match="source closure"):
        reader.validate_semantics(checkpoint)
