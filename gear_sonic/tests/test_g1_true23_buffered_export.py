from copy import deepcopy

import pytest

from gear_sonic.scripts.export_g1_true23_root_feedback import validate_export_semantics
from gear_sonic.scripts.train_g1_23dof_mjlab_causal_history import selected_semantic_contract
from gear_sonic.tests.test_native23_generalist_export import config
from gear_sonic.tests.test_native23_generalist_runner import lineage as lineage
from gear_sonic.tests.test_native23_root_feedback_export import checkpoint
from gear_sonic.tests.test_train_g1_true23_root_feedback import arguments
from gear_sonic.trl.mjlab.native23_root_feedback_actor import ROOT_FEEDBACK_ARCHITECTURE
from gear_sonic.utils.g1_true23_buffered_reference import BUFFERED_TIMING, reference_profile_contract
from gear_sonic.utils.g1_true23_release_compatibility import release_compatibility_contract
from gear_sonic.utils.g1_true23_root_feedback import root_feedback_contract


def resolved_config():
    resolved = config()
    compatibility = release_compatibility_contract("1" * 64, "released29_scale_bounded_linear_v2", BUFFERED_TIMING)
    resolved["semantic_profile"] = reference_profile_contract(BUFFERED_TIMING)
    resolved["native23_root_feedback"] = dict(
        feature_contract=root_feedback_contract(BUFFERED_TIMING),
        architecture=ROOT_FEEDBACK_ARCHITECTURE,
        deployment_ready=False,
        reference_timing=BUFFERED_TIMING,
        release_compatibility=compatibility,
    )
    return resolved


def test_base_training_selection_and_export_preserve_new_timing(lineage):
    args = arguments("regression", "--reference-timing", BUFFERED_TIMING)
    resolved = resolved_config()
    assert selected_semantic_contract(args) == resolved["semantic_profile"]
    value = checkpoint(lineage, resolved)
    value["actor"] = {
        "contract": {"release_compatibility": resolved["native23_root_feedback"]["release_compatibility"]}
    }
    actual = validate_export_semantics(value)
    assert actual["root_feedback_contract"] == root_feedback_contract(BUFFERED_TIMING)
    assert actual["semantic_profile"] == reference_profile_contract(BUFFERED_TIMING)


@pytest.mark.parametrize("tamper", [False, True])
def test_projection_training_objective_is_recorded_not_waived_by_export(lineage, tamper):
    from gear_sonic.trl.mjlab.native23_projected_target_ppo import PROFILE, projection_objective_contract

    resolved = resolved_config()
    objective = projection_objective_contract(PROFILE)
    resolved["native23_root_feedback"]["ppo_auxiliary_objective"] = objective
    if tamper:
        objective["weight"] = 1.0
    value = checkpoint(lineage, resolved)
    value["actor"] = {
        "contract": {"release_compatibility": resolved["native23_root_feedback"]["release_compatibility"]}
    }
    if tamper:
        with pytest.raises(ValueError, match="auxiliary objective"):
            validate_export_semantics(value)
    else:
        actual = validate_export_semantics(value)
        assert actual["root_feedback_training_configuration"]["ppo_auxiliary_objective"] == objective


@pytest.mark.parametrize(
    "tamper",
    (
        "old_semantics",
        "old_root_features",
        "old_release_compatibility",
        "old_actor",
        "missing_compatibility",
        "missing_timing",
    ),
)
def test_no_relabel_with_rehashed_lineage(lineage, tamper):
    resolved = resolved_config()
    actor_compatibility = deepcopy(resolved["native23_root_feedback"]["release_compatibility"])
    root = resolved["native23_root_feedback"]
    if tamper == "old_semantics":
        resolved["semantic_profile"] = reference_profile_contract()
    elif tamper == "old_root_features":
        root["feature_contract"] = root_feedback_contract()
    elif tamper == "old_release_compatibility":
        root["release_compatibility"] = release_compatibility_contract(
            "1" * 64, "released29_scale_bounded_linear_v2"
        )
    elif tamper == "old_actor":
        actor_compatibility = release_compatibility_contract("1" * 64, "released29_scale_bounded_linear_v2")
    elif tamper == "missing_compatibility":
        root.pop("release_compatibility")
    else:
        root.pop("reference_timing")
    value = checkpoint(lineage, resolved)
    value["actor"] = {"contract": {"release_compatibility": actor_compatibility}}
    with pytest.raises(ValueError):
        validate_export_semantics(value)
