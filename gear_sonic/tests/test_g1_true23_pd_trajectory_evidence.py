import numpy as np
import pytest

from gear_sonic.scripts.diagnose_g1_true23_pd_feedback_step import FailureLocatedPlant
from gear_sonic.scripts.optimize_g1_true23_pd_trajectory import evaluate_trajectory
from gear_sonic.scripts.prepare_g1_true23_contact_step_lifecycle import motion_from_poses
from gear_sonic.scripts.verify_g1_true23_pd_trajectory import ContactObservedPlant, verify_protected_nonregression
from gear_sonic.tests.test_g1_true23_contact_step_transition import native  # noqa: F401
from gear_sonic.tests.test_g1_true23_pd_shooting import plant_for
from gear_sonic.utils.g1_true23_pd_trajectory_optimizer import MotionObjective, canonical_target


@pytest.mark.parametrize(
    "defect", ["hand_regression", "new_self_pair", "deeper_self_pair", "ground", "missing_end"]
)
def test_independent_protected_verifier_rejects_each_regression(defect):
    import copy

    from gear_sonic.utils.g1_true23_generalist_benchmark import LANDMARKS

    initial = dict(
        lifecycle=dict(
            source_motion_tracking=dict(
                full_source_motion_completed=True,
                landmark_position_p95_m=dict(zip([row[0] for row in LANDMARKS], [0.065, 0.047, 0.07, 0.08, 0.04])),
            )
        )
    )
    candidate = copy.deepcopy(initial)
    contacts = dict(
        all_penetrating_self_pairs=[dict(geoms=[49, 59], distance_m=-0.002)],
        nonfoot_ground_penetration_samples=0,
        terminal_geometry_checked=True,
    )
    current_contacts = copy.deepcopy(contacts)
    assert verify_protected_nonregression(initial, candidate, contacts, current_contacts)
    if defect == "hand_regression":
        candidate["lifecycle"]["source_motion_tracking"]["landmark_position_p95_m"]["left_hand_point"] = 0.158
    elif defect == "new_self_pair":
        current_contacts["all_penetrating_self_pairs"].append(dict(geoms=[35, 45], distance_m=-0.003))
    elif defect == "deeper_self_pair":
        current_contacts["all_penetrating_self_pairs"][0]["distance_m"] = -0.02
    elif defect == "ground":
        current_contacts["nonfoot_ground_penetration_samples"] = 1
    else:
        current_contacts["terminal_geometry_checked"] = False
    with pytest.raises(ValueError):
        verify_protected_nonregression(initial, candidate, contacts, current_contacts)


def fixture_trajectory(model, standing):
    plant = plant_for(model)
    timeline = dict(
        total_requested_controls=4,
        configured_standing_qpos=standing.tolist(),
        phases=[
            dict(name="initial_standing", control_start=0, control_stop=1, requested_controls=1),
            dict(name="source_motion", control_start=1, control_stop=3, requested_controls=2),
            dict(name="standing_proof_margin", control_start=3, control_stop=4, requested_controls=1),
        ],
    )
    objective = MotionObjective(plant, motion_from_poses(model, np.tile(standing, (15, 1))), timeline)
    initial = plant.state(plant.initial_data(standing, np.zeros(29)))
    controls = np.tile(canonical_target(standing[7:]), (4, 1))
    return plant, objective, timeline, plant.rollout(initial, controls, physics_records=True)


def test_completed_offline_motion_does_not_inherit_a_single_policy_or_deployment_claim(native):  # noqa: F811
    model, standing, _ = native
    plant, objective, timeline, trajectory = fixture_trajectory(model, standing)
    report, errors = evaluate_trajectory(plant, objective, timeline, trajectory)
    assert errors.shape == (4, 5)
    lifecycle = report["lifecycle"]
    assert "single_policy_full_lifecycle_integrated" not in lifecycle
    assert lifecycle["full_physical_lifecycle_integrated"]
    assert lifecycle["not_a_learned_policy"]
    assert lifecycle["every_original_source_frame_evaluated"]
    assert not report["deployment_ready"] and not report["simulator_qualified"]
    assert not report["hardware_authorized"]


def test_truncated_physical_trace_cannot_receive_complete_lifecycle_metrics(native):  # noqa: F811
    model, standing, _ = native
    plant, objective, timeline, trajectory = fixture_trajectory(model, standing)
    trajectory["qpos"] = trajectory["qpos"][:-1]
    with pytest.raises(ValueError, match="incomplete trajectory"):
        evaluate_trajectory(plant, objective, timeline, trajectory)


def test_contact_observation_does_not_change_any_integrated_state_or_force(native):  # noqa: F811
    model, standing, _ = native
    original = plant_for(model)
    observed = ContactObservedPlant(model, original.profile)
    initial = original.state(original.initial_data(standing, np.zeros(29)))
    controls = np.tile(canonical_target(standing[7:]), (20, 1))
    baseline = original.rollout(initial, controls, physics_records=True)
    replay = observed.rollout(initial, controls, physics_records=True)
    for key in baseline:
        np.testing.assert_array_equal(replay[key], baseline[key])
    contacts = observed.contact_evidence()
    assert contacts["physics_substeps_observed"] == 200
    assert contacts["nonfoot_ground_loaded_substeps"] == 0
    assert contacts["robot_self_contact_loaded_substeps"] == 0
    assert min(contacts["foot_normal_force_max_n"]) > 1
    assert contacts["collision_model_coverage_not_qualified"]


def test_failure_locator_does_not_change_original_plant_dynamics(native):  # noqa: F811
    model, standing, _ = native
    original = plant_for(model)
    located = FailureLocatedPlant(model, original.profile)
    initial = original.state(original.initial_data(standing, np.zeros(29)))
    controls = np.tile(canonical_target(standing[7:]), (20, 1))
    baseline = original.rollout(initial, controls, physics_records=True)
    replay = located.rollout(initial, controls, physics_records=True)
    for key in baseline:
        np.testing.assert_array_equal(replay[key], baseline[key])
    assert located.last["completed_controls"] == 20
    assert located.last["joint_limit_excess_rad"] < 0
    assert located.last["motor_velocity_ratio_max"] < 1
