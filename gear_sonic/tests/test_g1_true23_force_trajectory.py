import copy
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
from scipy import sparse
from scipy.spatial.transform import Rotation

from gear_sonic.scripts.refine_g1_true23_reference_forces import bind_identity, validate_manifest
from gear_sonic.utils.g1_23dof_safe_target_transform import SAFE_TARGET_DEFAULT_Q_HARDWARE
from gear_sonic.utils.g1_true23_force_restoration import ForceRestorationConfig, force_restoration_step
from gear_sonic.utils.g1_true23_force_trajectory import (
    ForceLinearization,
    FrozenContactLoad,
    InverseForcePath,
    fit_contact_forces,
    pose_derivative_operator,
)
from gear_sonic.utils.g1_true23_reference_floor import compiled_model_sha256
from gear_sonic.utils.g1_true23_reference_support import floor_contact_map


@pytest.mark.parametrize("frames", [3, 4, 8, 101])
def test_sparse_derivatives_match_full_pose_difference_convention(frames):
    values = np.random.default_rng(918).normal(size=(frames, 26))
    operator = pose_derivative_operator(frames)
    first = np.gradient(values, 0.02, axis=0, edge_order=2)
    np.testing.assert_allclose(operator @ values, first, atol=1e-12, rtol=1e-14)
    np.testing.assert_allclose(
        operator @ operator @ values, np.gradient(first, 0.02, axis=0, edge_order=2), atol=1e-10
    )


@pytest.fixture
def force_path():
    root = Path(__file__).resolve().parents[2]
    model = mujoco.MjModel.from_xml_path(str(root / "gear_sonic/data/robots/g1/g1_23dof_rev_1_0.xml"))
    source = np.tile(model.qpos0, (8, 1))
    t = np.arange(8) * 0.02
    source[:, :3] = np.column_stack((0.04 * t, -0.03 * t, 0.78 + 0.01 * t**2))
    source[:, 3:7] = Rotation.from_euler("xyz", np.column_stack((0.04 * t, 0.01 * t, -0.03 * t))).as_quat()[
        :, [3, 0, 1, 2]
    ]
    source[:, 7:] = np.asarray(SAFE_TARGET_DEFAULT_Q_HARDWARE) + 0.03 * np.sin(t[:, None] * np.arange(1, 24))
    path = np.column_stack((np.zeros((8, 3)), source[:, 7:]))
    return model, InverseForcePath(model, source), path


def test_required_force_matches_independent_mass_bias_passive_with_no_model_mutation(force_path):
    model, problem, path = force_path
    before = compiled_model_sha256(model)
    state_before = problem.source.copy()
    result, _ = problem.evaluate(path)
    qpos, qvel, qacc = problem.state(path)
    data = mujoco.MjData(model)
    independent = []
    for q, v, a in zip(qpos, qvel, qacc):
        data.qpos[:], data.qvel[:] = q, v
        mujoco.mj_fwdPosition(model, data)
        mujoco.mj_fwdVelocity(model, data)
        inertia = np.zeros(model.nv)
        mujoco.mj_mulM(model, data, inertia, a)
        independent.append(inertia + data.qfrc_bias - data.qfrc_passive)
    np.testing.assert_allclose(result, independent, atol=1e-9, rtol=1e-11)
    assert compiled_model_sha256(model) == before
    np.testing.assert_array_equal(problem.source, state_before)
    assert result.shape == (8, 29)
    assert np.max(np.abs(result[:, :6])) > 100


def test_force_jacobian_couples_neighbor_frames_and_matches_independent_directions(force_path):
    _model, problem, path = force_path
    _, derivative = problem.evaluate(path)
    rng = np.random.default_rng(311)
    for frame in (0, 3, 7):
        direction = np.zeros_like(path)
        direction[frame] = rng.normal(size=26) * 0.01
        epsilon = 1e-4
        plus, _ = problem.evaluate(path + epsilon * direction, jacobian=False)
        minus, _ = problem.evaluate(path - epsilon * direction, jacobian=False)
        numeric = (plus - minus).ravel() / (2 * epsilon)
        analytical = derivative @ direction.ravel()
        np.testing.assert_allclose(analytical, numeric, atol=2e-4, rtol=2e-5)
    # A single interior pose affects adjacent velocities and second-neighbor accelerations.
    support = np.max(np.abs(derivative[:, 3 * 26 + 2].toarray().reshape(8, 29)), axis=1)
    assert np.all(support[[1, 2, 3, 4, 5]] > 0)
    assert support[7] == 0


def test_force_path_rejects_invalid_shape_and_rotation(force_path):
    model, problem, path = force_path
    with pytest.raises(ValueError, match="finite"):
        problem.evaluate(np.full_like(path, np.nan))
    with pytest.raises(ValueError, match="finite"):
        problem.evaluate(path[:, :-1])
    wrong = problem.source.copy()
    wrong[2, 3:7] = 0
    with pytest.raises(ValueError, match="nonzero"):
        InverseForcePath(model, wrong)
    with pytest.raises(ValueError, match="at least three"):
        InverseForcePath(model, problem.source[:2])


def test_force_seed_cannot_invent_base_actuation_or_exceed_joint_effort():
    required = np.zeros(29)
    required[2] = 300
    required[6] = 12
    result = fit_contact_forces(required, np.zeros((29, 0)), np.ones(23) * 5, np.zeros(23), body_weight_n=300)
    assert result["residual"][2] == 300
    assert result["joint_torque"][0] <= 5
    assert result["residual"][6] >= 7 - 1e-8
    assert result["ray_weights"].size == 0
    assert not result["force_feasibility_proven"]
    cone = np.zeros((29, 1))
    cone[2, 0] = 1
    supported = fit_contact_forces(required, cone, np.ones(23) * 15, np.zeros(23), body_weight_n=300)
    assert np.max(np.abs(supported["residual"])) < 1e-6
    assert np.all(supported["ray_weights"] >= 0)
    assert not supported["force_feasibility_proven"]


def test_attached_contact_load_matches_actual_map_and_local_derivatives(force_path):
    source_model, problem, _path = force_path
    original_hash = compiled_model_sha256(source_model)
    model = copy.copy(source_model)
    model.geom_margin[:] = np.maximum(model.geom_margin, 0.002)
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_MIDPHASE)
    plane = model.geom("floor").id
    data = mujoco.MjData(model)
    data.qpos[:] = problem.source[3]
    mujoco.mj_fwdPosition(model, data)
    lowest = min(
        mujoco.mj_geomDistance(model, data, plane, geom, 3.0, None)
        for geom in range(model.ngeom)
        if model.geom_bodyid[geom] and (model.geom_contype[geom] or model.geom_conaffinity[geom])
    )
    data.qpos[2] += 0.001 - lowest
    mujoco.mj_fwdPosition(model, data)
    force_map, contacts = floor_contact_map(model, data, plane, 0.002)
    assert contacts
    weights = np.linspace(1.0, 2.0, force_map.shape[1])
    load = FrozenContactLoad(model, data, contacts, weights)
    np.testing.assert_allclose(load.evaluate(data.qpos), force_map @ weights, atol=1e-10)
    derivative = load.derivative()
    for column, index in enumerate(np.r_[0:3, 7:30]):
        difference = np.zeros(model.nq)
        difference[index] = 1e-5
        numeric = (load.evaluate(data.qpos + difference) - load.evaluate(data.qpos - difference)) / 2e-5
        np.testing.assert_allclose(derivative[:, column], numeric, atol=3e-5, rtol=2e-5)
    assert compiled_model_sha256(source_model) == original_hash
    with pytest.raises(ValueError, match="force coefficients"):
        FrozenContactLoad(model, data, contacts, np.r_[weights, 1])


def test_actual_force_linearization_covers_every_frame_and_both_models(force_path):
    model, inverse, path = force_path
    before = compiled_model_sha256(model)
    system = ForceLinearization({"first": model, "second": model}, inverse.source, np.ones(23) * 20)
    result = system.evaluate(path)
    assert result["jacobian"].shape == (2 * 8 * 29, 8 * 26)
    assert result["force_map"].shape[0] == 2 * 8 * 29
    assert [row["frames"] for row in result["models"]] == [8, 8]
    assert result["required"].shape == (2 * 8 * 29,)
    np.testing.assert_array_equal(result["required"][: 8 * 29], result["required"][8 * 29 :])
    assert np.all(result["seed"] >= result["lower"] - 1e-8)
    assert np.all(result["seed"] <= result["upper"] + 1e-8)
    assert compiled_model_sha256(model) == before
    assert not result["force_feasibility_proven"]


def test_absolute_force_penalty_restores_exact_balance_without_fake_actuation():
    current = np.zeros((4, 26))
    required = np.zeros((4, 29))
    required[:, 2] = 1
    derivative = sparse.csc_matrix(
        (-np.ones(4) * 100, (np.arange(4) * 29 + 2, np.arange(4) * 26 + 2)), shape=(4 * 29, 4 * 26)
    )
    forces = {
        "required": required.ravel(),
        "jacobian": derivative,
        "force_map": sparse.csc_matrix((4 * 29, 0)),
        "lower": np.zeros(0),
        "upper": np.zeros(0),
        "seed": np.zeros(0),
        "scale": np.ones(4 * 29),
        "normalized_residual": required.ravel(),
    }
    result, report = force_restoration_step(
        current,
        current,
        current - 0.08,
        current + 0.08,
        np.ones(26),
        np.ones(26) * 6,
        np.zeros(26),
        np.ones(4),
        sparse.csc_matrix((4, 4 * 26)),
        forces,
        config=ForceRestorationConfig(),
    )
    assert result is not None, report
    # With the original squared penalty this achievable 1 cm correction left
    # a residual above 1e-5. An absolute penalty restores it without relaxing tolerance.
    np.testing.assert_allclose(result[:, 2], 0.01, atol=1e-9)
    assert report["maximum_temporary_generalized_force_residual"] <= 1e-8
    np.testing.assert_allclose(np.delete(result, 2, axis=1), 0, atol=1e-10)


def test_old_squared_force_penalty_has_nonzero_bias_even_when_balance_is_possible():
    weight = ForceRestorationConfig().force_residual_cost
    optimum = weight * 100 / (1 / 0.08**2 + weight * 100**2)
    assert 1 - 100 * optimum > ForceRestorationConfig().generalized_force_tolerance


def test_minimum_step_avoids_fidelity_drift_near_a_curved_force_boundary():
    # Closed-form objective-choice regression, not solver or robot acceptance.
    # OSQP also fails numerical convergence on this tiny example; that must
    # not be mistaken for an infeasibility proof. The constrained minima below
    # isolate the old-pose tangent drift from the independent solver issue.
    current = np.array([0.4, 0.840001])
    desired = np.array([0.0, 0.5])

    def nonlinear_force(path):
        return path[0] ** 2 + path[1] - 1

    gradient, residual = np.array([0.8, 1.0]), nonlinear_force(current)

    def proposal(center):
        # Equal joint-space weights, subject to gradient @ step = -residual.
        free_step = center - current
        multiplier = (gradient @ free_step + residual) / (gradient @ gradient)
        step = free_step - gradient * multiplier
        np.testing.assert_allclose(gradient @ step, -residual, atol=1e-15)
        assert abs(multiplier) / 0.6**2 < ForceRestorationConfig().force_residual_cost
        assert np.max(np.abs(step)) < ForceRestorationConfig().joint_trust_rad
        return current + step

    pulled = proposal(desired)
    before = abs(nonlinear_force(current))
    for fraction in (1, 0.5, 0.25, 0.125, 0.0625):
        assert abs(nonlinear_force(current + fraction * (pulled - current))) > before
    restored = proposal(current)
    assert abs(nonlinear_force(restored)) < 1e-9
    assert np.max(np.abs(restored - current)) < 1e-5


def test_force_refinement_cannot_replace_previous_input_pins(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("first")
    identities = {}
    bind_identity(identities, source)
    bind_identity(identities, source)
    source.write_text("changed")
    with pytest.raises(ValueError, match="previously pinned"):
        bind_identity(identities, source)


@pytest.mark.parametrize(
    "change",
    [
        {"teacher_accepted": True},
        {"hardware_authorized": None},
        {"deployment_ready": True},
        {"motions": []},
        {"motions": [7]},
        {"motions": [{"name": "../escape"}]},
        {"motions": [{"name": "same"}, {"name": "same"}]},
    ],
)
def test_force_refinement_manifest_remains_explicitly_unaccepted(change):
    manifest = {
        "kind": "g1_true23_stance_candidate_manifest_v1",
        "teacher_accepted": False,
        "hardware_authorized": False,
        "deployment_ready": False,
        "motions": [{"name": "clip"}],
    }
    assert validate_manifest(manifest) == manifest["motions"]
    manifest.update(change)
    with pytest.raises(ValueError):
        validate_manifest(manifest)


def test_force_qp_reported_success_cannot_bypass_independent_linear_audit(monkeypatch):
    import osqp

    class FalseSuccess:
        def setup(self, **kwargs):
            self.size = len(kwargs["q"])

        def warm_start(self, **_kwargs):
            pass

        def solve(self, **_kwargs):
            return SimpleNamespace(
                x=np.ones(self.size),
                info=SimpleNamespace(status="solved", status_val=1, iter=1, prim_res=0, dual_res=0, run_time=0),
            )

    monkeypatch.setattr(osqp, "OSQP", FalseSuccess)
    path = np.zeros((4, 26))
    force = {
        "required": np.ones(116),
        "jacobian": sparse.csc_matrix((116, 104)),
        "force_map": sparse.csc_matrix((116, 0)),
        "seed": np.zeros(0),
        "lower": np.zeros(0),
        "upper": np.zeros(0),
        "scale": np.ones(116),
        "normalized_residual": np.ones(116),
    }
    result, report = force_restoration_step(
        path,
        path,
        path - 0.08,
        path + 0.08,
        np.ones(26),
        np.ones(26) * 6,
        np.zeros(26),
        np.ones(4),
        sparse.csc_matrix((4, 104)),
        force,
        config=ForceRestorationConfig(),
    )
    assert result is None
    assert report["status"] == "independent_linear_constraints_failed"
