import pytest

from gear_sonic.scripts.refine_g1_true23_lifecycle_ramps import validate_ramp_input_model


def diagnostic():
    return {
        "kind": "g1_true23_registered_source_lifecycle_geometry_diagnostic_v1",
        "compiled_physics_model_sha256": "exact-model",
        "policy_evaluation_performed": False,
        "training_reference_accepted": False,
    }


def test_unevaluated_geometry_requires_explicit_reference_only_mode():
    validate_ramp_input_model(diagnostic(), "exact-model", reference_only=True)
    with pytest.raises(ValueError, match="evaluated"):
        validate_ramp_input_model(diagnostic(), "exact-model")


@pytest.mark.parametrize(
    "key,value",
    [
        ("kind", "unrecognized"),
        ("compiled_physics_model_sha256", "different"),
        ("policy_evaluation_performed", True),
        ("training_reference_accepted", True),
    ],
)
def test_wrong_model_or_fabricated_status_rejected(key, value):
    source = diagnostic()
    source[key] = value
    with pytest.raises(ValueError, match="same-model unevaluated"):
        validate_ramp_input_model(source, "exact-model", reference_only=True)


def test_old_evaluated_input_stays_separate_and_checks_every_case():
    source = {
        "kind": "g1_true23_root_feedback_single_policy_lifecycle_campaign_v1",
        "records": [{"result": {"compiled_model_sha256": "exact-model"}}],
    }
    validate_ramp_input_model(source, "exact-model")
    with pytest.raises(ValueError, match="unevaluated"):
        validate_ramp_input_model(source, "exact-model", reference_only=True)
    source["records"].append({"result": {"compiled_model_sha256": "different"}})
    with pytest.raises(ValueError, match="same physical model"):
        validate_ramp_input_model(source, "exact-model")
