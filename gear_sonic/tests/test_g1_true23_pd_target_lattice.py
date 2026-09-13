import json

import numpy as np
import pytest

from gear_sonic.utils.g1_23dof_safe_target_transform import safe_target_transform_numpy
from gear_sonic.utils.g1_true23_pd_target_lattice import nearest_original_codec_target
from gear_sonic.utils.g1_true23_source_action_codec import source_scaled_precompensation


def emitted(raw):
    inverse, _ = source_scaled_precompensation(np.asarray(raw, np.float32))
    return safe_target_transform_numpy(inverse)[1]


@pytest.mark.parametrize("seed", range(12))
def test_original_emitted_targets_survive_exact_roundtrip_with_forward_codec_witness(seed):
    rng = np.random.default_rng(seed)
    for scale in (0.01, 0.5, 3.0, 9.9):
        target = emitted(rng.uniform(-scale, scale, 23))
        projected, witness = nearest_original_codec_target(target)
        np.testing.assert_array_equal(projected, target)
        np.testing.assert_array_equal(emitted(witness), target)
        again, _ = nearest_original_codec_target(projected)
        np.testing.assert_array_equal(again, target)


def test_bounded_nonlattice_requests_project_to_real_original_targets_and_are_idempotent():
    rng = np.random.default_rng(92081)
    for _ in range(12):
        target = emitted(rng.uniform(-0.5, 0.5, 23)).astype(np.float64) + rng.uniform(-1e-7, 1e-7, 23)
        projected, witness = nearest_original_codec_target(target)
        np.testing.assert_array_equal(projected, emitted(witness))
        again, _ = nearest_original_codec_target(projected)
        np.testing.assert_array_equal(again, projected)
        assert np.max(np.abs(projected - target)) < 2e-7


def test_original_codec_endpoints_and_invalid_requests():
    limit = np.nextafter(np.float32(10), np.float32(0))
    for sign in (-1, 1):
        target = emitted(np.full(23, sign * limit, np.float32))
        np.testing.assert_array_equal(nearest_original_codec_target(target)[0], target)
        with pytest.raises(ValueError, match="cannot expand"):
            nearest_original_codec_target(target + sign * 1e-5)
    with pytest.raises(ValueError, match="finite"):
        nearest_original_codec_target(np.full(23, np.nan))


def test_zero_update_replay_preserves_perturbed_emitted_commands_and_every_physics_array(monkeypatch):
    import os
    from pathlib import Path

    from gear_sonic.tests.test_g1_true23_pd_shooting import plant_for
    from gear_sonic.utils.g1_true23_generalist_benchmark import MODEL, PHYSICS
    import gear_sonic.utils.g1_true23_pd_trajectory_optimizer as optimizer
    from gear_sonic.utils.g1_true23_step1b_mujoco import prepare_true23_model

    root = Path(__file__).resolve().parents[2]
    assets = Path(os.environ.get("G1_TRUE23_TEST_ASSET_ROOT", str(root)))
    _, model, _ = prepare_true23_model(assets / MODEL, root / PHYSICS)
    plant = plant_for(model)
    config = json.loads((root / PHYSICS).read_text())["initial_state"]
    standing = np.asarray(
        [*config["base_position_m"], *config["base_quaternion_wxyz"], *config["joint_position_hardware_rad"]]
    )
    rng, count = np.random.default_rng(88319), 50
    commands = np.asarray([emitted(rng.uniform(-0.01, 0.01, 23)) for _ in range(count)])
    state = plant.state(plant.initial_data(standing, np.zeros(29)))
    original = plant.rollout(state, commands, physics_records=True)
    monkeypatch.setattr(optimizer, "canonical_target", lambda target: nearest_original_codec_target(target)[0])
    replay, targets = optimizer.feedback_trial(
        plant, state, original, commands, rng.normal(size=(count, 23)), rng.normal(size=(count, 23, 81)), 0.0
    )
    np.testing.assert_array_equal(targets, commands)
    for key, value in original.items():
        assert value.dtype == replay[key].dtype
        assert value.tobytes() == replay[key].tobytes()
