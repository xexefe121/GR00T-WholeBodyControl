import numpy as np
import pytest

from gear_sonic.scripts.prepare_g1_true23_contact_step_lifecycle import motion_from_poses
from gear_sonic.tests.test_g1_true23_contact_step_transition import native  # noqa: F401
from gear_sonic.tests.test_g1_true23_pd_shooting import plant_for
from gear_sonic.utils.g1_true23_generalist_benchmark import LANDMARKS
from gear_sonic.utils.g1_true23_pd_protected_objective import (
    ProtectedMotionObjective,
    ProtectedPdPlant,
    protected_tracking_acceptance,
)
from gear_sonic.utils.g1_true23_pd_trajectory_optimizer import canonical_target

BAD_BODY_V4_QPOS = np.asarray(
    [
        -0.0027187252099622524,
        -0.026109161021419486,
        0.7351867837878371,
        0.998651235449829,
        0.009264610904381833,
        -0.013869037949237455,
        -0.049168350650436,
        -0.23258975771394774,
        0.18357276348219834,
        0.1524770594012513,
        0.8181606321149396,
        -0.4511972434204943,
        -0.05608070897495423,
        -0.4219053407534038,
        -0.007902704566580671,
        0.03469887297441167,
        0.9458037501543971,
        -0.3991375276494056,
        -0.0069274188349978585,
        0.01183413007334155,
        -0.16913887102706313,
        0.19778277811876935,
        -0.6932096067967966,
        0.4884740215769161,
        0.3018800015010453,
        -0.1346804632581833,
        0.03352201121298441,
        0.5031269135417289,
        0.8374286117626215,
        0.05857530705020142,
    ]
)


def metrics(values):
    return dict(
        lifecycle=dict(
            source_motion_tracking=dict(
                landmark_position_p95_m=dict(zip([row[0] for row in LANDMARKS], values, strict=True))
            )
        )
    )


def test_better_feet_cannot_sacrifice_a_passing_hand_screen():
    initial = metrics([0.065, 0.047, 0.069, 0.077, 0.041])
    bad = metrics([0.0456, 0.036, 0.158, 0.087, 0.0305])
    verdict = protected_tracking_acceptance(initial, bad)
    assert not verdict["accepted"]
    assert verdict["violations"][0]["landmark"] == "left_hand_point"
    assert verdict["violations"][0]["fixed_ceiling_m"] == 0.10
    assert protected_tracking_acceptance(initial, metrics([0.060, 0.044, 0.07, 0.08, 0.040]))["accepted"]


def test_already_failing_part_cannot_worsen_under_fixed_initial_ceiling():
    initial = metrics([0.094, 0.086, 0.108, 0.067, 0.083])
    assert not protected_tracking_acceptance(initial, metrics([0.096, 0.07, 0.09, 0.06, 0.07]))["accepted"]


def test_contact_guard_observations_preserve_exact_physics_and_can_replay_their_own_envelope(native):  # noqa: F811
    model, standing, _ = native
    original = plant_for(model)
    plant = ProtectedPdPlant(model, original.profile)
    initial = original.state(original.initial_data(standing, np.zeros(29)))
    targets = np.tile(canonical_target(standing[7:]), (20, 1))
    baseline = original.rollout(initial, targets, physics_records=True)
    observed = plant.rollout(initial, targets, physics_records=True)
    for key in baseline:
        np.testing.assert_array_equal(baseline[key], observed[key])
    plant.set_research_contact_envelope(plant.self_contact_depths)
    replay = plant.rollout(initial, targets, physics_records=True)
    plant.check_terminal_geometry(replay["qpos"][-1], replay["qvel"][-1])
    for key in baseline:
        np.testing.assert_array_equal(baseline[key], replay[key])
    assert plant.contact_observations()["physics_steps"] == 200


def test_protected_landmark_gradient_matches_independent_finite_difference(native):  # noqa: F811
    import mujoco

    model, standing, _ = native
    plant = plant_for(model)
    timeline = dict(
        total_requested_controls=4, phases=[dict(name="source_motion", control_start=0, control_stop=4)]
    )
    objective = ProtectedMotionObjective(plant, motion_from_poses(model, np.tile(standing, (15, 1))), timeline)
    pose, velocity = standing.copy(), np.zeros(29)
    pose[:2] += [0.16, 0.08]  # Every world landmark outside its unchanged threshold.
    rng, epsilon = np.random.default_rng(778), 1e-6
    direction = rng.normal(size=58)
    _, gradient, hessian = objective.state_cost(pose, velocity, 2, derivatives=True)
    plus, minus = pose.copy(), pose.copy()
    mujoco.mj_integratePos(model, plus, direction[:29], epsilon)
    mujoco.mj_integratePos(model, minus, direction[:29], -epsilon)
    finite = (
        objective.state_cost(plus, velocity + epsilon * direction[29:], 2)
        - objective.state_cost(minus, velocity - epsilon * direction[29:], 2)
    ) / (2 * epsilon)
    np.testing.assert_allclose(gradient[:58] @ direction, finite, atol=1e-4, rtol=2e-7)
    assert np.linalg.eigvalsh(hessian).min() >= -1e-7
    plant.assert_unchanged()


def test_invalid_contact_envelope_is_rejected(native):  # noqa: F811
    model, _, _ = native
    plant = ProtectedPdPlant(model, plant_for(model).profile)
    for value in (0.0, 0.01, np.nan, -np.inf):
        with pytest.raises(ValueError, match="actually measured"):
            plant.set_research_contact_envelope({(1, 2): value})


@pytest.mark.parametrize("allowance", [{}, {(49, 59): -0.002}])
def test_actual_rejected_body_pose_cannot_create_new_or_deeper_self_contact(native, allowance):  # noqa: F811
    model, standing, _ = native
    plant = ProtectedPdPlant(model, plant_for(model).profile)
    plant.set_research_contact_envelope(allowance)
    data = plant.initial_data(BAD_BODY_V4_QPOS, np.zeros(29))
    with pytest.raises(ValueError, match="self-contact nonregression"):
        plant.integrate_control(data, canonical_target(standing[7:]))
    assert plant.physics_steps == 1  # Rejected at the first physical substep, not after a full clip.


def test_terminal_geometry_rejects_contact_without_integrating_or_mutating_physics(native):  # noqa: F811
    model, _, _ = native
    plant = ProtectedPdPlant(model, plant_for(model).profile)
    plant.set_research_contact_envelope({})
    with pytest.raises(ValueError, match="self-contact nonregression"):
        plant.check_terminal_geometry(BAD_BODY_V4_QPOS, np.zeros(29))
    assert plant.physics_steps == 0
    plant.assert_unchanged()


def test_actual_self_collision_penalty_gradient_matches_finite_difference(native):  # noqa: F811
    import mujoco

    model, standing, _ = native
    plant = plant_for(model)
    timeline = dict(
        total_requested_controls=4, phases=[dict(name="source_motion", control_start=0, control_stop=4)]
    )
    objective = ProtectedMotionObjective(plant, motion_from_poses(model, np.tile(standing, (15, 1))), timeline)
    pose, velocity = BAD_BODY_V4_QPOS.copy(), np.zeros(29)
    assert min(row["distance_m"] for row in objective.collision_query.pose_rows(pose, np.arange(29))) < -0.02
    direction, epsilon = np.random.default_rng(8113).normal(size=58), 1e-6
    _, gradient, hessian = objective.state_cost(pose, velocity, 2, derivatives=True)
    plus, minus = pose.copy(), pose.copy()
    mujoco.mj_integratePos(model, plus, direction[:29], epsilon)
    mujoco.mj_integratePos(model, minus, direction[:29], -epsilon)
    finite = (
        objective.state_cost(plus, velocity + epsilon * direction[29:], 2)
        - objective.state_cost(minus, velocity - epsilon * direction[29:], 2)
    ) / (2 * epsilon)
    np.testing.assert_allclose(gradient[:58] @ direction, finite, atol=1e-3, rtol=2e-6)
    assert np.linalg.eigvalsh(hessian).min() >= -1e-6
    plant.assert_unchanged()
