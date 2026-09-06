import ast
import copy
import inspect
import json

import pytest

from gear_sonic.scripts import evaluate_g1_true23_motion_ppo as old_evaluation
from gear_sonic.scripts import evaluate_g1_true23_v14_motion_ppo as new_evaluation
from gear_sonic.utils import g1_true23_v14_diagnostic_pair as pair


def documents():
    expected = dict(
        stage_one_actuation={"kind": "native_stateful"},
        hardware_authorized=False,
        deployment_ready=False,
        promotion_eligible=False,
    )
    resolved = {
        pair.CONTRACT_KEY: copy.deepcopy(expected),
        "stage_one_actuation": copy.deepcopy(expected["stage_one_actuation"]),
        "training_precision": {
            "kind": "g1_true23_ieee_training_precision_v1",
            "requested_state": pair.EXPECTED_STATE.copy(),
        },
    }
    metadata = dict(schema_version=1, contract={}, source=dict(reference_profile=pair.CAUSAL_HISTORY_PROFILE))
    return metadata, resolved, expected


def test_raw_pair_with_external_transform_once_is_accepted():
    pair.require_raw_v14_contract(*documents())


@pytest.mark.parametrize(
    "key,value",
    [
        ("schema_version", 2),
        ("contract", {"safe_target_transform": {}}),
        ("contract", {"decoder_output_semantics": "applied_safe_native_action"}),
    ],
)
def test_embedded_transform_cannot_be_applied_twice(key, value):
    metadata, resolved, expected = documents()
    metadata[key] = value
    with pytest.raises(ValueError, match="raw output"):
        pair.require_raw_v14_contract(metadata, resolved, expected)


def test_different_controller_is_rejected():
    metadata, resolved, expected = documents()
    resolved["stage_one_actuation"] = {"kind": "old_position_actuator"}
    with pytest.raises(ValueError, match="controller differs"):
        pair.require_raw_v14_contract(metadata, resolved, expected)


def test_precision_mismatch_is_rejected():
    metadata, resolved, expected = documents()
    resolved["training_precision"]["requested_state"]["cuda_matmul_fp32"] = "tf32"
    with pytest.raises(ValueError, match="IEEE-trained"):
        pair.require_raw_v14_contract(metadata, resolved, expected)


def test_changed_method_cannot_claim_v14():
    metadata, resolved, expected = documents()
    resolved[pair.CONTRACT_KEY]["standing_lora_bootstrap_used"] = True
    with pytest.raises(ValueError, match="original-method"):
        pair.require_raw_v14_contract(metadata, resolved, expected)


def test_future_reference_cannot_claim_causal():
    metadata, resolved, expected = documents()
    metadata["source"]["reference_profile"] = "released_low_latency_step1_0p02s"
    with pytest.raises(ValueError, match="causal"):
        pair.require_raw_v14_contract(metadata, resolved, expected)


@pytest.mark.parametrize("flag", ["hardware_authorized", "deployment_ready", "promotion_eligible"])
def test_no_simulator_comparison_authorizes_robot(flag):
    metadata, resolved, expected = documents()
    expected[flag] = True
    resolved[pair.CONTRACT_KEY] = copy.deepcopy(expected)
    with pytest.raises(ValueError, match="cannot authorize"):
        pair.require_raw_v14_contract(metadata, resolved, expected)


def test_saved_json_lists_match_live_tuple_controller():
    metadata, resolved, expected = documents()
    expected["stage_one_actuation"]["gains"] = (1.0, 2.0)
    resolved[pair.CONTRACT_KEY] = json.loads(json.dumps(expected))
    resolved["stage_one_actuation"] = json.loads(json.dumps(expected["stage_one_actuation"]))
    pair.require_raw_v14_contract(metadata, resolved, expected)


def test_both_methods_share_plan_and_exact_execution_parameters():
    assert old_evaluation.evaluation_plan is new_evaluation.evaluation_plan

    def calls(module):
        tree = ast.parse(inspect.getsource(module.main))
        return [
            ast.dump(node)
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "run_interior_case"
        ]

    assert len(calls(old_evaluation)) == 1
    assert calls(old_evaluation) == calls(new_evaluation)
