"""Root9 replay leaves the received-reference/guard path authoritative."""

import numpy as np
import pytest

pytest.importorskip("mjlab")

from gear_sonic.utils.g1_true23_normal_lora_replay import NormalLoraAdapter, NormalLoraPolicy
from gear_sonic.utils.g1_true23_released_core_comparison import ReleasedCoreAdapter


@pytest.mark.parametrize("reject", [False, True])
def test_root9_current_state_and_failed_attempt_record(monkeypatch, reject):
    adapter = object.__new__(NormalLoraAdapter)
    adapter.attempts = []
    policy = object.__new__(NormalLoraPolicy)
    seen = []
    policy.infer = lambda encoder, history, root: (
        seen.append(root) or np.zeros(23, np.float32),
        np.zeros(994, np.float32),
    )

    def parent(self, proxy, encoder, history, **state):
        output = proxy.infer(encoder, history)
        self.attempts.append({"released_raw23": output[0]})
        if reject:
            raise RuntimeError("unchanged guard rejected")
        return output

    monkeypatch.setattr(ReleasedCoreAdapter, "infer", parent)
    qpos = np.zeros(30)
    qpos[:3] = [1, 2, 0.8]
    qpos[3] = 1
    qvel = np.zeros(29)
    qvel[:3] = [0.1, -0.2, 0]
    state = dict(
        control_index=0,
        desired_position_w=np.array([1.1, 2.2, 0.8]),
        previous_desired_position_w=np.array([1.08, 2.2, 0.8]),
        measured_qpos=qpos,
        measured_qvel=qvel,
    )

    def call():
        return adapter.infer(policy, np.zeros(267, np.float32), np.zeros(930, np.float32), **state)

    if reject:
        with pytest.raises(RuntimeError, match="guard rejected"):
            call()
    else:
        call()
    np.testing.assert_allclose(seen[0], [0.1, 0.2, 0, 1, 0, 0, 0.1, -0.2, 0], rtol=0, atol=2e-6)
    np.testing.assert_array_equal(adapter.attempts[0]["root_feedback9"], seen[0])
    np.testing.assert_array_equal(qpos[:3], [1, 2, 0.8])
    np.testing.assert_array_equal(qvel[:3], [0.1, -0.2, 0])


def test_unrelated_policy_rejected_before_guard():
    adapter = object.__new__(NormalLoraAdapter)
    with pytest.raises(ValueError, match="checkpoint family"):
        adapter.infer(object(), np.zeros(267), np.zeros(930), control_index=0)


def test_contract_overrides_parent_fields_without_duplicate_keyword_error(monkeypatch):
    inherited = dict(kind="original", root_feedback_used=False, hardware_authorized=False, deployment_ready=False)
    monkeypatch.setattr(ReleasedCoreAdapter, "contract", lambda self: inherited.copy())
    value = object.__new__(NormalLoraAdapter).contract()
    assert value["kind"] == "native23_normal_lora_root9_guarded_cpu_adapter_v1"
    assert value["root_feedback_used"] is True
    assert value["hardware_authorized"] is False and value["deployment_ready"] is False
    assert inherited["kind"] == "original" and inherited["root_feedback_used"] is False
