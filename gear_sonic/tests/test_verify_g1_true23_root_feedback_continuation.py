import torch

from gear_sonic.scripts.verify_g1_true23_root_feedback_continuation import exact_tree_equal


def test_exact_continuation_tree_preserves_tensor_dtype_values_groups_and_counters():
    tree = {
        "state": {1: {"step": torch.tensor(100.0), "moment": torch.ones(3)}},
        "groups": [{"lr": 1e-4}],
        "updates": 100,
    }
    clone = {
        "state": {1: {"step": torch.tensor(100.0), "moment": torch.ones(3)}},
        "groups": [{"lr": 1e-4}],
        "updates": 100,
    }
    assert exact_tree_equal(tree, clone)
    clone["state"][1]["moment"][2] = 0
    assert not exact_tree_equal(tree, clone)
    assert not exact_tree_equal(torch.ones(3), torch.ones(3, dtype=torch.float64))
    assert not exact_tree_equal({"updates": 100}, {"updates": 101})
    assert not exact_tree_equal([1, 2], (1, 2))
    assert not exact_tree_equal({"lr": 1e-4}, {"lr": 5e-7})
