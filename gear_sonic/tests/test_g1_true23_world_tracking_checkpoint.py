import copy

import pytest

from gear_sonic.envs.mjlab.sonic_true23_world_tracking_termination import termination_contract
from gear_sonic.trl.mjlab import native23_bounded_progress_runner as old, native23_world_tracking_runner as new
from gear_sonic.utils.g1_true23_bounded_progress import GAMMA, reward_contract


def fixture_checkpoint():
    return dict(header=copy.deepcopy(new.CHECKPOINT_HEADER), lineage=dict(materials=dict(resolved_config=dict(
        payload=dict(native23_world_tracking_termination=termination_contract(),
                     native23_bounded_progress=reward_contract(), agent=dict(algorithm=dict(gamma=GAMMA)))
    ))))


def test_new_schema_view_never_relabels_saved_checkpoint():
    checkpoint = fixture_checkpoint()
    original = copy.deepcopy(checkpoint)
    common = new.decoder_schema_view(checkpoint)
    assert checkpoint == original
    assert common["header"] == new.DECODER_HEADER
    assert common["lineage"] is checkpoint["lineage"]


def test_previous_reader_rejects_new_training_semantics():
    with pytest.raises(ValueError):
        old.decoder_schema_view(fixture_checkpoint())


def test_new_reader_rejects_previous_snapshot_header():
    checkpoint = fixture_checkpoint()
    checkpoint["header"] = old.CHECKPOINT_HEADER
    with pytest.raises(ValueError):
        new.decoder_schema_view(checkpoint)


@pytest.mark.parametrize("field,value", [("threshold_m", 0.31), ("timeout", True),
                                       ("existing_termination_terms_preserved", False)])
def test_changed_failure_semantics_rejected(field, value):
    checkpoint = fixture_checkpoint()
    resolved = checkpoint["lineage"]["materials"]["resolved_config"]["payload"]
    resolved["native23_world_tracking_termination"][field] = value
    with pytest.raises(ValueError):
        new.decoder_schema_view(checkpoint)


def test_bound_discount_must_stay_unchanged():
    checkpoint = fixture_checkpoint()
    checkpoint["lineage"]["materials"]["resolved_config"]["payload"]["agent"]["algorithm"]["gamma"] = 0.9
    with pytest.raises(ValueError):
        new.decoder_schema_view(checkpoint)
