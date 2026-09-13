"""Distinct reader/launcher fail-closed checks; shared forward covered separately."""

import pytest
from torch import nn

from gear_sonic.scripts.train_g1_true23_frozen_decoder import validate_frozen_recipe
from gear_sonic.tests.test_native23_original_intent_contract import fresh
from gear_sonic.trl.mjlab.native23_frozen_decoder_runner import CHECKPOINT_HEADER
from gear_sonic.trl.mjlab.native23_original_intent_actor import ACTOR_KIND as OLD_KIND
from gear_sonic.utils.g1_true23_frozen_decoder_checkpoint import FrozenDecoderCPUActor, validate_semantics
from gear_sonic.utils.g1_true23_original_intent_checkpoint import validate_semantics as old_reader


@pytest.mark.parametrize("value", (None, [], {}, {"header": {}}))
def test_new_reader_rejects_unknown_header(value):
    with pytest.raises(ValueError, match="research checkpoint header"):
        validate_semantics(value)


@pytest.mark.parametrize("kind", (None, OLD_KIND, "g1_native23_root_feedback_buffered_source_actor_v3"))
def test_new_reader_rejects_old_actor_kinds(kind):
    with pytest.raises(ValueError, match="old/relabelled"):
        validate_semantics({"header": CHECKPOINT_HEADER, "actor": {"contract": {"kind": kind}}})


def test_old_reader_rejects_new_header_and_non_actor_cannot_infer():
    with pytest.raises(ValueError, match="header"):
        old_reader({"header": CHECKPOINT_HEADER})
    with pytest.raises(TypeError, match="distinct actor"):
        FrozenDecoderCPUActor(nn.Linear(2, 2))


def args_fixture():
    args = fresh()
    args.optimizer_profile = "feedback_priority"
    args.ppo_auxiliary_objective = "none"
    args.iterations = args.session_updates = 2
    return args


def test_fresh_recipe_smoke_and_capped_regression():
    args = args_fixture()
    validate_frozen_recipe(args)
    args.mode = "regression"
    args.iterations = args.session_updates = 100
    validate_frozen_recipe(args)


@pytest.mark.parametrize(
    "field,value",
    (
        ("resume", "old.pt"),
        ("initialize_actor_from", "old.pt"),
        ("mode", "campaign"),
        ("optimizer_profile", "legacy_uniform"),
        ("ppo_auxiliary_objective", "mean_projection_squared_rad_v1"),
        ("iterations", 101),
        ("iterations", 0),
        ("session_updates", 101),
        ("reference_timing", "causal_history"),
    ),
)
def test_training_change_or_budget_escape_rejected(field, value):
    args = args_fixture()
    setattr(args, field, value)
    with pytest.raises(ValueError):
        validate_frozen_recipe(args)
