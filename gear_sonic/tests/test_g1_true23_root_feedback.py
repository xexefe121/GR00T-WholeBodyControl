"""Root error observability and exact frame/causality conventions."""

import numpy as np
import pytest
import torch

from gear_sonic.utils.g1_true23_root_feedback import (
    root_feedback_contract,
    root_feedback_numpy,
    root_feedback_torch,
)


def fixture():
    return [
        np.array([0.0, 0.0, 0.8]),
        np.array([0.0, 0.0, 0.8]),
        np.array([0.2, -0.1, 0.0]),
        np.array([0.1, 0.1, 0.0]),
        np.array([1.0, 0.0, 0.0, 0.0]),
    ]


def test_root_displacement_and_velocity_are_observable_without_clipping():
    values = fixture()
    base = root_feedback_numpy(*values)
    values[1] += [8, -3, 0]
    shifted = root_feedback_numpy(*values)
    np.testing.assert_array_equal(shifted[:3] - base[:3], [-8, 3, 0])
    values[3] += [1.5, -0.75, 0.2]
    moving = root_feedback_numpy(*values)
    np.testing.assert_allclose(moving[6:] - shifted[6:], [1.5, -0.75, 0.2], atol=2e-7)
    np.testing.assert_array_equal(moving[:6], shifted[:6])


def test_current_measured_yaw_and_common_world_translation():
    values = fixture()
    values[4] = np.array([np.sqrt(0.5), 0, 0, np.sqrt(0.5)])
    values[0] += [1, 0, 0]
    result = root_feedback_numpy(*values)
    np.testing.assert_allclose(result[:3], [0, -1, 0], atol=2e-7)
    values[0] += [12, 4, 3]
    values[1] += [12, 4, 3]
    np.testing.assert_allclose(result, root_feedback_numpy(*values), atol=2e-7)


def test_batched_numpy_torch_agree():
    generator = np.random.default_rng(19)
    values = [generator.normal(size=(31, 3)).astype(np.float32) for _ in range(4)]
    quaternion = generator.normal(size=(31, 4)).astype(np.float32)
    quaternion /= np.linalg.norm(quaternion, axis=-1, keepdims=True)
    values.append(quaternion)
    actual = root_feedback_torch(*(torch.from_numpy(value) for value in values)).numpy()
    np.testing.assert_allclose(actual, root_feedback_numpy(*values), atol=2e-6, rtol=2e-6)
    assert actual.shape == (31, 9) and actual.dtype == np.float32


@pytest.mark.parametrize(
    "index,bad",
    [
        (0, [1, 2]),
        (1, [[1, 2, 3]]),
        (2, [np.nan, 0, 0]),
        (3, [True, False, True]),
        (4, [2, 0, 0, 0]),
        (4, [1, 0, 0]),
    ],
)
def test_invalid_inputs_fail_closed(index, bad):
    values = fixture()
    values[index] = bad
    with pytest.raises(ValueError):
        root_feedback_numpy(*values)


def test_contract_preserves_sonic_branch_and_only_uses_current_reference():
    contract = root_feedback_contract()
    assert contract["dimension"] == 9
    assert contract["future_samples_consumed"] == 0
    assert contract["sonic_semantic_branch_reference_frame"] == "q9_unchanged"
    assert contract["desired_velocity_definition"] == "(root_q10-root_q9)/0.02"
    assert contract["separate_from_existing_994_decoder_input"]
    assert not contract["input_clipping"]
