import numpy as np
import pytest

from gear_sonic.scripts.diagnose_g1_true23_root_error_scale import (
    EXPERIMENT_KIND,
    RootErrorScalePolicy,
    disclosed_report,
    experiment_contract,
)
from gear_sonic.utils.g1_true23_root_feedback_campaign import validate_campaign_evaluation


class Probe:
    def infer(self, encoder, history, feedback):
        self.inputs = encoder, history, feedback
        return np.zeros(23, dtype=np.float32), np.zeros(994, dtype=np.float32)


@pytest.mark.parametrize("scale", [0, 1, 2, 4])
def test_only_position_columns_change_and_every_actual_input_recorded(scale):
    original = Probe()
    policy = RootErrorScalePolicy(original, scale)
    encoder, history = np.ones(267, dtype=np.float32), np.ones(930, dtype=np.float32)
    feedback = np.arange(9, dtype=np.float32) - 4
    saved = feedback.copy()
    policy.infer(encoder, history, feedback)
    assert original.inputs[0] is encoder and original.inputs[1] is history
    np.testing.assert_array_equal(feedback, saved)
    np.testing.assert_array_equal(original.inputs[2][:3], feedback[:3] * scale)
    np.testing.assert_array_equal(original.inputs[2][3:], feedback[3:])
    records = policy.take_records()
    np.testing.assert_array_equal(records["actual_policy_root_feedback9"][0], original.inputs[2])
    np.testing.assert_array_equal(records["policy_received_root_feedback9"][0], saved)
    assert policy.take_records()["actual_policy_root_feedback9"].shape == (0, 9)


def test_counterfactual_reports_cannot_enter_normal_training_continuation():
    original = {"kind": "g1_true23_root_feedback_single_policy_lifecycle_campaign_v1", "value": [1]}
    report = disclosed_report(original, experiment_contract(4))
    assert original["kind"] != EXPERIMENT_KIND
    assert report["kind"] == EXPERIMENT_KIND
    assert not report["eligible_parent_for_training_continuation"]
    assert not report["deployment_ready"]
    with pytest.raises(ValueError, match="CPU lifecycle campaign"):
        validate_campaign_evaluation(
            report, checkpoint_sha256="x", actor_sha256="y", lineage_sha256="z", updates=1200
        )


@pytest.mark.parametrize("scale", [-1, 8, float("nan"), True, "4"])
def test_unsupported_or_nonfinite_scale_rejected(scale):
    with pytest.raises(ValueError):
        experiment_contract(scale)
