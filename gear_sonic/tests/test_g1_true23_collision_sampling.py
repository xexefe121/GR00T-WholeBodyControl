from copy import deepcopy

import numpy as np
import pytest
from scipy import sparse

from gear_sonic.scripts.refine_g1_true23_collision_clearance import (
    retained_diagnostic_input,
    validate_warm_start_bindings,
)
from gear_sonic.utils.g1_true23_collision_sampling import (
    CollisionPathSampler,
    interpolate_original_poses,
    interpolation_weights,
)


def test_union_contains_every_original_and_control_time_without_dropping_endpoints():
    sampler = CollisionPathSampler([0.0, 0.5, 1.0], [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    np.testing.assert_array_equal(sampler.query_times, [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0])
    weights = interpolation_weights(sampler.control_times, sampler.query_times)
    np.testing.assert_allclose(np.asarray(weights.sum(axis=1)).ravel(), 1.0)
    assert np.all(weights.data >= 0) and weights.getnnz(axis=1).max() == 2


class NonlinearContactQuery:
    """Analytic curved signed-distance fixture, independent of interpolation."""

    def path_rows(self, poses, joint_dof_addresses):
        joint = poses[:, 7]
        rows = np.arange(len(joint))
        jac = sparse.csc_matrix((2 * joint, (rows, 23 * rows)), shape=(len(joint), 23 * len(joint)))
        return jac, joint**2 - 0.25, [(int(i), 1, 2) for i in rows]


def test_clear_control_knots_do_not_hide_penetration_at_original_timestamps():
    poses = np.zeros((3, 30))
    poses[:, 3] = 1
    poses[:, 7] = [-1.0, 1.0, 2.0]
    query = NonlinearContactQuery()
    _, control, _ = query.path_rows(poses, np.arange(23))
    assert np.all(control > 0)
    sampler = CollisionPathSampler([0.0, 1.0, 2.0], [0.0, 0.5, 2.0])
    jac, distances, _ = sampler.path_rows(query, poses, np.arange(23))
    assert distances[1] == -0.25 and jac.shape == (4, 87)


def test_interpolated_distance_jacobian_maps_to_correct_adjacent_control_knots():
    poses = np.zeros((3, 30))
    poses[:, 3] = 1
    poses[:, 7] = [0.2, 0.9, -0.3]
    sampler = CollisionPathSampler([0.0, 1.0, 2.0], [0.0, 0.25, 1.75, 2.0])
    query = NonlinearContactQuery()
    jac, _, _ = sampler.path_rows(query, poses, np.arange(23))
    for frame in range(3):
        delta = np.zeros_like(poses)
        delta[frame, 7] = 1e-6
        plus = sampler.path_rows(query, poses + delta, np.arange(23))[1]
        minus = sampler.path_rows(query, poses - delta, np.arange(23))[1]
        np.testing.assert_allclose(jac[:, 29 * frame + 6].toarray().ravel(), (plus - minus) / 2e-6, atol=1e-9)
    root_columns = np.concatenate([29 * frame + np.arange(6) for frame in range(3)])
    assert jac[:, root_columns].nnz == 0


@pytest.mark.parametrize("original", [[0.0, 1.0], [0.0, 2.0, 1.0], [0.0, np.nan, 2.0], [-0.1, 2.0]])
def test_union_cannot_hide_invalid_or_cropped_original_timeline(original):
    with pytest.raises(ValueError):
        CollisionPathSampler([0.0, 1.0, 2.0], original)


def test_nonunit_quaternion_is_not_silently_normalized():
    poses = np.zeros((3, 30))
    poses[:, 3] = 2
    with pytest.raises(ValueError, match="unit root"):
        interpolate_original_poses(poses, [0.0, 1.0, 2.0], [0.0, 2.0])


def warm_evidence():
    return {
        "kind": "g1_true23_full_collision_repair_audit_v1",
        "acceptance": {
            "passed": False,
            "checks": {
                "complete_control_grid_fidelity_passed": True,
                "all_original_timestamp_fidelity_passed": True,
                "serialized_declared_path_bounds_passed": True,
            },
        },
        "compiled_models": {"source": "source-hash", "target": "target-hash"},
        "compiled_physics_model_sha256": "physics-hash",
        "rejected_candidate": {"path": "collision.rejected.npz", "sha256": "candidate-hash"},
        "input_bindings": {"parent": "parent-hash", "source": "source-file-hash"},
    }


def test_rejected_warm_start_reuses_only_bound_same_source_and_models():
    audit = warm_evidence()
    saved = deepcopy(audit)
    validate_warm_start_bindings(audit, audit["input_bindings"], audit["compiled_models"], "physics-hash")
    assert audit == saved and audit["acceptance"]["passed"] is False


@pytest.mark.parametrize("failure", ["physics", "source", "control", "original", "path", "accepted"])
def test_warm_start_cannot_change_source_or_hide_failed_old_gates(failure):
    audit = warm_evidence()
    expected = dict(audit["input_bindings"])
    if failure == "physics":
        audit["compiled_physics_model_sha256"] = "different"
    elif failure == "source":
        audit["input_bindings"]["source"] = "different"
    elif failure == "accepted":
        audit["acceptance"]["passed"] = True
    else:
        gate = {
            "control": "complete_control_grid_fidelity_passed",
            "original": "all_original_timestamp_fidelity_passed",
            "path": "serialized_declared_path_bounds_passed",
        }[failure]
        audit["acceptance"]["checks"][gate] = False
    with pytest.raises(ValueError):
        validate_warm_start_bindings(
            audit, expected, {"source": "source-hash", "target": "target-hash"}, "physics-hash"
        )


def test_original_time_repair_can_reuse_its_unique_already_bound_fixed_root_seed(tmp_path):
    path = tmp_path / "original" / "fit.diagnostic.npz"
    selected, digest = retained_diagnostic_input({"input_bindings": {str(path): "seed-hash"}}, tmp_path / "repair")
    assert selected == path and digest == "seed-hash"


@pytest.mark.parametrize("count", [0, 2])
def test_missing_or_ambiguous_ancestor_seed_cannot_be_guessed(tmp_path, count):
    parent = {"input_bindings": {str(tmp_path / str(i) / "fit.diagnostic.npz"): str(i) for i in range(count)}}
    with pytest.raises(ValueError, match="exactly one bound"):
        retained_diagnostic_input(parent, tmp_path)


def test_original_time_restoration_must_be_explicit_and_cover_every_original_sample():
    audit = warm_evidence()
    audit["original_frames"] = 3
    audit["acceptance"]["checks"]["all_original_timestamp_fidelity_passed"] = False
    audit["after_original_time"] = {
        "all_original_timestamps_and_endpoints_evaluated": True,
        "original_time_sample_indices": [0, 1, 2],
        "failures": ["right_foot_position"],
    }
    args = (audit, audit["input_bindings"], audit["compiled_models"], "physics-hash")
    with pytest.raises(ValueError, match="explicit"):
        validate_warm_start_bindings(*args)
    validate_warm_start_bindings(*args, restore_original_time_tasks=True)
    assert audit["acceptance"]["passed"] is False
    audit["after_original_time"]["original_time_sample_indices"].pop()
    with pytest.raises(ValueError):
        validate_warm_start_bindings(*args, restore_original_time_tasks=True)
