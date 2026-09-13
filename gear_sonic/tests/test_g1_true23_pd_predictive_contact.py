from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from gear_sonic.tests.test_g1_true23_contact_step_transition import native  # noqa: F401
from gear_sonic.tests.test_g1_true23_pd_shooting import plant_for
from gear_sonic.utils.g1_true23_pd_predictive_contact import SubstepContactPredictor, make_predictive_contact_law
from gear_sonic.utils.g1_true23_pd_trajectory_optimizer import canonical_target, feedback_trial


@pytest.mark.parametrize("contact", [False, True])
def test_substep_prediction_exact_without_state_mutation(native, contact):  # noqa: F811
    model, standing, _ = native
    plant = plant_for(model)
    pose = standing.copy()
    pose[2] += 0 if contact else 0.2
    data = plant.initial_data(pose, np.zeros(29))
    for _ in range(4):
        plant.integrate_control(data, standing[7:])
    state, target = plant.state(data), canonical_target(standing[7:])
    predictor = SubstepContactPredictor(plant)
    positions, velocities = predictor.poses(state, target)
    expected = plant.rollout(state, target[None], physics_records=True)
    np.testing.assert_array_equal(positions, expected["physics_post_qpos"])
    np.testing.assert_array_equal(velocities, expected["physics_post_qvel"])
    np.testing.assert_array_equal(state, plant.state(data))
    plant.assert_unchanged()


@pytest.mark.parametrize("contact", [False, True])
def test_substep_derivatives_match_independent_control_probes(native, contact):  # noqa: F811
    model, standing, _ = native
    plant = plant_for(model)
    pose = standing.copy()
    pose[2] += 0 if contact else 0.2
    state = plant.state(plant.initial_data(pose, np.zeros(29)))
    target = canonical_target(standing[7:]).astype(float)
    predictor = SubstepContactPredictor(plant)
    derivative, velocity_derivative = predictor.target_derivatives(state, target, include_velocity=True)
    direction, epsilon = np.random.default_rng(3102).normal(size=23), 2e-6
    runs = [
        plant.rollout(state, (target + sign * epsilon * direction)[None], physics_records=True) for sign in (-1, 1)
    ]
    for step in range(10):
        independent = np.empty(29)
        mujoco.mj_differentiatePos(
            model,
            independent,
            2 * epsilon,
            runs[0]["physics_post_qpos"][step],
            runs[1]["physics_post_qpos"][step],
        )
        np.testing.assert_allclose(derivative[step] @ direction, independent, atol=1e-5, rtol=3e-3)
        independent_velocity = (runs[1]["physics_post_qvel"][step] - runs[0]["physics_post_qvel"][step]) / (
            2 * epsilon
        )
        np.testing.assert_allclose(
            velocity_derivative[step] @ direction, independent_velocity, atol=1e-4, rtol=3e-3
        )
    plant.assert_unchanged()


def test_state_callback_receives_exact_warmstart_and_replays(native):  # noqa: F811
    model, standing, _ = native
    plant = plant_for(model)
    initial = plant.state(plant.initial_data(standing, np.zeros(29)))
    targets = np.tile(canonical_target(standing[7:]), (4, 1))
    original = plant.rollout(initial, targets, physics_records=True)
    seen = []

    def state_law(index, difference, state):
        np.testing.assert_array_equal(state, original["integration_state"][index])
        seen.append(index)
        return np.zeros(23)

    replay, actual = feedback_trial(
        plant,
        initial,
        original,
        targets,
        np.zeros((4, 23)),
        np.zeros((4, 23, 81)),
        0,
        state_control_law=state_law,
    )
    assert seen == list(range(4))
    np.testing.assert_array_equal(actual, targets)
    for key in original:
        np.testing.assert_array_equal(replay[key], original[key])
    with pytest.raises(ValueError, match="only one"):
        feedback_trial(plant, None, None, None, None, None, 0, control_law=state_law, state_control_law=state_law)


@pytest.mark.parametrize("airborne", [False, True])
def test_zero_predictive_step_preserves_actual_native_rollout(native, airborne):  # noqa: F811
    model, standing, _ = native
    plant = plant_for(model)
    pose = standing.copy()
    pose[2] += 0.2 if airborne else 0.0
    data = plant.initial_data(pose, np.zeros(29))
    target = canonical_target(standing[7:])
    for _ in range(4):
        plant.integrate_control(data, target)
    initial = plant.state(data)
    targets = np.tile(target, (6, 1))
    nominal = plant.rollout(initial, targets, physics_records=True)
    # Nonzero direction and state feedback must disappear at alpha=0 and
    # zero state deviation; exercise the real QP, codec and substep predictor.
    rng = np.random.default_rng(81617)
    stages = [
        dict(hessian=np.eye(23), gradient=rng.normal(size=23), gradient_state=rng.normal(size=(23, 81)))
        for _ in targets
    ]
    law, stats = make_predictive_contact_law(plant, targets, stages, 0.0, {}, clearance_m=0.0)
    replay, emitted = feedback_trial(
        plant,
        initial,
        nominal,
        targets,
        np.zeros((6, 23)),
        np.zeros((6, 23, 81)),
        0.0,
        state_control_law=law,
    )
    np.testing.assert_array_equal(emitted, targets)
    for key in nominal:
        assert replay[key].dtype == nominal[key].dtype
        assert replay[key].tobytes() == nominal[key].tobytes(), key
    np.testing.assert_array_equal(initial, plant.state(data))
    assert stats["controls_solved"] == 6
    assert stats["controls_requiring_repair"] == 0
    plant.assert_unchanged()


@pytest.mark.parametrize("repairable", [True, False])
def test_substep_violation_repaired_or_rejected(native, monkeypatch, repairable):  # noqa: F811
    import gear_sonic.utils.g1_true23_pd_predictive_contact as predictive

    model, standing, _ = native
    original = plant_for(model)
    nominal = canonical_target(standing[7:])

    class Predictor:
        def __init__(self, plant):
            pass

        def poses(self, state, target):
            np.testing.assert_array_equal(state, np.array([123.0]))
            return target, None

        def rows(self, target, limits, clearance):
            jacobian = np.zeros(29)
            jacobian[6] = 1
            return [
                dict(
                    substep=3,
                    geoms=(22, 59),
                    distance=float(target[0] - nominal[0] - 0.01),
                    floor=clearance,
                    jacobian=jacobian,
                )
            ]

        def target_derivatives(self, state, target, *, include_velocity=False):
            result = np.zeros((10, 29, 23))
            result[3, 6, 0] = float(repairable)
            return (result, np.zeros_like(result)) if include_velocity else result

    monkeypatch.setattr(predictive, "SubstepContactPredictor", Predictor)
    monkeypatch.setattr(predictive, "substep_bound_margins", lambda *args: np.ones(920))
    plant = SimpleNamespace(lower=original.lower, upper=original.upper)
    stages = [dict(hessian=np.eye(23), gradient=np.zeros(23), gradient_state=np.zeros((23, 81)))]
    law, stats = make_predictive_contact_law(plant, nominal[None], stages, 1.0, {})
    if repairable:
        correction = law(0, np.zeros(81), np.array([123.0]))
        assert correction[0] == pytest.approx(0.011002, abs=1e-8)
        assert stats["controls_requiring_repair"] == 1
        assert stats["predicted_substeps_per_control"] == 10
    else:
        with pytest.raises(ValueError):
            law(0, np.zeros(81), np.array([123.0]))
        assert stats["controls_solved"] == 0


@pytest.mark.parametrize("joint", [3, 9, 16, 21])
def test_codec_rounding_gets_outward_clearance_without_relaxing_gate(native, monkeypatch, joint):  # noqa: F811
    from gear_sonic.utils.g1_23dof_safe_target_transform import SAFE_TARGET_DEFAULT_Q_HARDWARE
    import gear_sonic.utils.g1_true23_pd_predictive_contact as predictive

    model, _, _ = native
    original = plant_for(model)
    nominal = canonical_target(np.asarray(SAFE_TARGET_DEFAULT_Q_HARDWARE))
    probe = nominal.astype(float)
    probe[joint] += 0.011
    lower = canonical_target(probe).astype(float)
    desired = lower[joint] + 2e-8
    probe = lower.copy()
    probe[joint] = desired
    assert desired - canonical_target(probe)[joint] > 1e-8

    class Predictor:
        def __init__(self, plant):
            pass

        def poses(self, state, target):
            return target, None

        def rows(self, target, limits, clearance):
            jacobian = np.zeros(29)
            jacobian[6 + joint] = 1
            return [
                dict(
                    substep=3,
                    geoms=(22, 59),
                    distance=float(target[joint] - desired + clearance),
                    floor=clearance,
                    jacobian=jacobian,
                )
            ]

        def target_derivatives(self, state, target, *, include_velocity=False):
            result = np.zeros((10, 29, 23))
            result[3, 6 + joint, joint] = 1
            return (result, np.zeros_like(result)) if include_velocity else result

    monkeypatch.setattr(predictive, "SubstepContactPredictor", Predictor)
    monkeypatch.setattr(predictive, "substep_bound_margins", lambda *args: np.ones(920))
    plant = SimpleNamespace(lower=original.lower, upper=original.upper)
    stages = [dict(hessian=np.eye(23), gradient=np.zeros(23), gradient_state=np.zeros((23, 81)))]
    law, stats = make_predictive_contact_law(plant, nominal[None], stages, 1.0, {})
    correction = law(0, np.zeros(81), np.zeros(1))
    emitted = canonical_target(nominal.astype(float) + correction)
    assert emitted[joint] >= desired - 1e-8
    assert stats["controls_requiring_repair"] == 1
