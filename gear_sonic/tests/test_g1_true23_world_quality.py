"""New positive task reward never pays for an episode's reset state."""

import numpy as np
import pytest
import torch

from gear_sonic.scripts.audit_g1_true23_learning_signal import classify_frames, describe
from gear_sonic.utils.g1_true23_world_quality import WorldQualityStep, quality_contract, world_quality_bonus


def test_exact_formula_monotonic_and_bounded():
    error = torch.tensor([0.0, 1.0, 9.0, 99.0])
    flags = torch.zeros(4, dtype=torch.bool)
    result = world_quality_bonus(-torch.log1p(error), flags, flags)
    torch.testing.assert_close(result, 2 / (1 + error), rtol=1e-6, atol=1e-7)
    assert (torch.diff(result) < 0).all()
    assert quality_contract()["deployment_ready"] is False
    assert quality_contract()["changes_optimization_objective"] is True


@pytest.mark.parametrize("flags", [(True, False), (False, True), (True, True)])
def test_perfect_reset_state_gets_zero_reward(flags):
    result = world_quality_bonus(torch.zeros(1), torch.tensor([flags[0]]), torch.tensor([flags[1]]))
    assert torch.equal(result, torch.zeros(1))


@pytest.mark.parametrize("bad", ["positive", "nan", "dimensions", "flags"])
def test_invalid_quality_inputs_rejected(bad):
    phi, terminated, timeouts = torch.zeros(2), torch.zeros(2, dtype=torch.bool), torch.zeros(2, dtype=torch.bool)
    if bad == "positive":
        phi[0] = 0.1
    elif bad == "nan":
        phi[0] = float("nan")
    elif bad == "dimensions":
        phi = phi[:, None]
    else:
        timeouts = timeouts.float()
    with pytest.raises(ValueError):
        world_quality_bonus(phi, terminated, timeouts)


def test_one_original_step_preserved_and_actual_reward_captured():
    rows, calls = [], []

    def step(actions):
        calls.append(actions)
        rows.append(
            dict(
                phi_after_including_reset_states=torch.tensor([-np.log(2), 0.0], dtype=torch.float32),
                returned_reward=torch.tensor([2.0, -100.0]),
            )
        )
        return (
            "obs",
            torch.tensor([2.0, -100.0]),
            torch.tensor([False, True]),
            torch.tensor([False, False]),
            {"x": 1},
        )

    class Original:
        def __call__(self, actions):
            return step(actions)

    original = Original()
    original.rows = rows
    wrapped = WorldQualityStep(original)
    obs, reward, terminated, timeout, extras = wrapped("unchanged actions")
    assert calls == ["unchanged actions"]
    assert obs == "obs" and extras == {"x": 1}
    assert terminated.tolist() == [False, True] and timeout.tolist() == [False, False]
    torch.testing.assert_close(reward, torch.tensor([3.0, -100.0]))
    torch.testing.assert_close(rows[0]["pre_quality_returned_reward"], torch.tensor([2.0, -100.0]))
    torch.testing.assert_close(rows[0]["world_quality_bonus"], torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(rows[0]["returned_reward"], reward)


def test_phase_classification_uses_held_q1_and_rejects_missing_coverage():
    spans = [
        dict(
            name="motion",
            start=0,
            timeline=dict(
                phases=[
                    dict(name="stand", frame_start=11, frame_stop=13),
                    dict(name="source", frame_start=13, frame_stop=15),
                ]
            ),
        )
    ]
    labels, names = classify_frames(np.array([[10, 11, 12, 13]]), spans)
    assert labels.tolist() == [[0, 0, 1, 1]]
    assert names == [("motion", "stand"), ("motion", "source")]
    with pytest.raises(ValueError, match="lacks"):
        classify_frames(np.array([[9]]), spans)
    assert describe([]) is None
