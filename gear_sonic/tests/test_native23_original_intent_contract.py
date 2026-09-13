from types import SimpleNamespace

import pytest

from gear_sonic.scripts.train_g1_true23_original_intent import validate_fresh_recipe
from gear_sonic.trl.mjlab.native23_original_intent_actor import (
    original_intent_compatibility,
    validate_original_intent_compatibility,
)
from gear_sonic.utils.g1_true23_buffered_reference import BUFFERED_TIMING
from gear_sonic.utils.g1_true23_original29_reference import REFERENCE_KIND
from gear_sonic.utils.g1_true23_release_compatibility import (
    release_compatibility_contract,
    validate_release_compatibility,
)
from gear_sonic.utils.g1_true23_source_action_codec import SOURCE_ACTION_CONVENTION


def test_new_source_kind_is_not_a_relabelled_old_contract():
    value = original_intent_compatibility("a" * 64, "b" * 64)
    assert validate_original_intent_compatibility(value) == value
    assert value["reference_geometry"] == REFERENCE_KIND
    assert value["reference_timing"] == BUFFERED_TIMING
    assert (
        value["source_action_codec"]
        == release_compatibility_contract("a" * 64, SOURCE_ACTION_CONVENTION, BUFFERED_TIMING)[
            "source_action_codec"
        ]
    )
    assert not value["deployment_ready"] and not value["hardware_authorized"]
    assert not value["old_zero_absent_checkpoint_relabelling_allowed"]
    with pytest.raises(ValueError):
        validate_release_compatibility(value)
    old = release_compatibility_contract("a" * 64, SOURCE_ACTION_CONVENTION, BUFFERED_TIMING)
    with pytest.raises(ValueError):
        validate_original_intent_compatibility(old)


@pytest.mark.parametrize(
    "field,value",
    (
        ("reference_geometry", "zero"),
        ("deployment_ready", True),
        ("native_geometry_sha256", "broken"),
        ("previous_action", "raw"),
    ),
)
def test_contract_changes_rejected(field, value):
    contract = original_intent_compatibility("a" * 64, "b" * 64)
    contract[field] = value
    with pytest.raises(ValueError):
        validate_original_intent_compatibility(contract)


def fresh():
    return SimpleNamespace(
        mode="smoke",
        resume=None,
        continue_from=None,
        initialize_actor_from=None,
        reference_timing=BUFFERED_TIMING,
        release_action_convention=SOURCE_ACTION_CONVENTION,
        release_source_geometry="source.xml",
        training_physics_profile="pinned_cpu_referee_scene_v1",
        objective_profile="root_and_upper_feet_world_v4",
        curriculum_stage="lifecycle",
    )


def test_explicit_fresh_recipe_only():
    args = fresh()
    validate_fresh_recipe(args)
    args.mode = "regression"
    validate_fresh_recipe(args)


@pytest.mark.parametrize(
    "field,value",
    (
        ("mode", "campaign"),
        ("resume", "old.pt"),
        ("initialize_actor_from", "old.pt"),
        ("reference_timing", "causal_history"),
        ("training_physics_profile", "legacy_training_asset"),
        ("objective_profile", "legacy_root_tracking"),
    ),
)
def test_implicit_transition_or_wrong_base_rejected(field, value):
    args = fresh()
    setattr(args, field, value)
    with pytest.raises(ValueError):
        validate_fresh_recipe(args)
