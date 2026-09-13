import numpy as np
import pytest
import torch

from gear_sonic.scripts.record_g1_true23_training_engine_lifecycle_v2 import (
    scored_world,
    selected_inference_rows,
    training_inference_batch,
)


def test_all_four_worlds_are_scored_without_discarding_small_differences():
    rows = np.zeros((4, 30))
    rows[:, 0] = np.arange(4) * 1e-12
    for index in range(4):
        np.testing.assert_array_equal(scored_world(rows, index), rows[index])


@pytest.mark.parametrize("rows,index", [(np.zeros((32, 30)), 0), (np.zeros((4, 30)), 4)])
def test_wrong_world_count_or_index_rejected(rows, index):
    with pytest.raises(ValueError):
        scored_world(rows, index)


def test_nonfinite_scored_world_rejected():
    rows = np.zeros((4, 30))
    rows[2, 4] = np.nan
    with pytest.raises(ValueError):
        scored_world(rows, 2)


def test_inference_batch_retains_every_measured_row_without_history_or_route_edit():
    obs = {"policy": torch.arange(4 * 930).reshape(4, 930).float(),
           "tokenizer": torch.arange(4 * 267).reshape(4, 267).float(),
           "root_feedback": torch.arange(4 * 9).reshape(4, 9).float()}
    result = training_inference_batch(obs)
    for key, value in result.items():
        assert value.shape == (32, obs[key].shape[1])
        for index in range(4):
            torch.testing.assert_close(value[index * 8 : (index + 1) * 8], obs[key][index].repeat(8, 1))
        np.testing.assert_array_equal(selected_inference_rows(value.numpy()), obs[key].numpy())
        assert obs[key].shape[0] == 4


def test_selected_rows_do_not_claim_other_inference_rows_are_exact():
    rows = np.arange(32 * 23).reshape(32, 23).astype(np.float32)
    np.testing.assert_array_equal(selected_inference_rows(rows), rows[[0, 8, 16, 24]])


@pytest.mark.parametrize("rows", [np.zeros((4, 23)), np.full((32, 23), np.nan), np.zeros(32)])
def test_invalid_inference_output_rejected(rows):
    with pytest.raises(ValueError):
        selected_inference_rows(rows)


def test_wrong_measured_batch_rejected():
    with pytest.raises(ValueError):
        training_inference_batch({"policy": torch.zeros(32, 930)})
