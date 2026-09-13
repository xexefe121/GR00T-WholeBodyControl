import numpy as np
import pytest

from gear_sonic.scripts.verify_g1_true23_world_quality_update import audit_quality_arrays
from gear_sonic.tests.test_g1_true23_bounded_progress_audit import capture as original_capture


def capture():
    arrays, metadata = original_capture()
    done = arrays["terminated"] | arrays["timeouts"]
    phi = arrays["phi_after_including_reset_states"]
    bonus = np.where(done, np.float32(0), np.float32(2) * np.exp(phi))
    arrays["pre_quality_returned_reward"] = arrays["returned_reward"].copy()
    arrays["world_quality_bonus"] = bonus
    arrays["returned_reward"] += bonus
    arrays["stored_reward_with_timeout_bootstrap"] = (
        arrays["returned_reward"] + 0.99 * arrays["timeouts"] * arrays["critic_value_before_bootstrap"]
    ).astype(np.float32)
    return arrays, metadata


def test_every_done_combination_and_actual_augmented_storage_reconstruct():
    arrays, metadata = capture()
    result = audit_quality_arrays(arrays, metadata, 1, 2, 4)
    assert result["zero_bonus_on_every_failure_and_timeout"]
    assert result["actual_augmented_PPO_storage_reconstructed"]
    assert result["actual_transitions"] == 8


@pytest.mark.parametrize(
    "reason", ["done_bonus", "scale", "double", "missing_bonus", "stored", "base", "dtype", "nan"]
)
def test_corrupted_actual_reward_rejected(reason):
    arrays, metadata = capture()
    if reason == "done_bonus":
        arrays["world_quality_bonus"][0, 1] = 0.1
        arrays["returned_reward"][0, 1] += 0.1
        arrays["stored_reward_with_timeout_bootstrap"][0, 1] += 0.1
    elif reason == "scale":
        arrays["world_quality_bonus"] *= 2
        arrays["returned_reward"] = arrays["pre_quality_returned_reward"] + arrays["world_quality_bonus"]
        arrays["stored_reward_with_timeout_bootstrap"] = (
            arrays["returned_reward"] + 0.99 * arrays["timeouts"] * arrays["critic_value_before_bootstrap"]
        ).astype(np.float32)
    elif reason == "double":
        arrays["returned_reward"] += arrays["world_quality_bonus"]
    elif reason == "missing_bonus":
        del arrays["world_quality_bonus"]
    elif reason == "stored":
        arrays["stored_reward_with_timeout_bootstrap"] += 0.01
    elif reason == "base":
        arrays["base_reward"] += 0.01
    elif reason == "dtype":
        arrays["world_quality_bonus"] = arrays["world_quality_bonus"].astype(np.float64)
    else:
        arrays["world_quality_bonus"][0, 0] = np.nan
    with pytest.raises(ValueError):
        audit_quality_arrays(arrays, metadata, 1, 2, 4)
