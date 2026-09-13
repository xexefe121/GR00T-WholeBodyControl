import mujoco
import numpy as np
import pytest

from gear_sonic.scripts.prepare_g1_true23_contact_step_lifecycle import motion_from_poses
from gear_sonic.tests.test_g1_true23_contact_step_transition import native  # noqa: F401
from gear_sonic.utils.g1_true23_pd_entry_guidance import EntryGuidance


def make_guide(model, standing, **kwargs):
    poses = np.tile(standing, (18, 1))
    poses[11:13, 2] += 0.04
    motion = motion_from_poses(model, poses)
    timeline = dict(
        total_requested_controls=7,
        phases=[
            dict(name="acquisition_ramp", control_start=0, control_stop=3),
            dict(name="source_motion", control_start=3, control_stop=4),
            dict(name="return_ramp", control_start=4, control_stop=5),
            dict(name="returned_standing", control_start=5, control_stop=7),
        ],
    )
    return EntryGuidance(model, motion, timeline, **kwargs), motion


@pytest.mark.parametrize("foot_xy_weight", [0.0, 25000.0])
def test_transition_guide_gradient_matches_native_kinematic_difference(native, foot_xy_weight):  # noqa: F811
    model, standing, _ = native
    guide, _ = make_guide(model, standing, foot_xy_weight=foot_xy_weight)
    rng = np.random.default_rng(14560)
    pose = standing.copy()
    mujoco.mj_integratePos(model, pose, rng.normal(0, 0.002, 29), 1)
    value, gradient, hessian = guide.state_terms(pose, 1)
    direction, epsilon = rng.normal(size=29), 1e-6
    plus, minus = pose.copy(), pose.copy()
    mujoco.mj_integratePos(model, plus, direction, epsilon)
    mujoco.mj_integratePos(model, minus, direction, -epsilon)
    finite = (guide.state_terms(plus, 1)[0] - guide.state_terms(minus, 1)[0]) / (2 * epsilon)
    assert value > 0
    np.testing.assert_allclose(gradient[:29] @ direction, finite, atol=2e-5, rtol=1e-6)
    np.testing.assert_allclose(hessian, hessian.T, atol=1e-12)
    assert np.linalg.eigvalsh(hessian).min() >= -1e-8


@pytest.mark.parametrize("foot_xy_weight", [0.0, 25000.0])
def test_guide_preserves_cached_physics_original_arrays_and_source_nodes(native, foot_xy_weight):  # noqa: F811
    model, standing, _ = native
    guide, motion = make_guide(model, standing, foot_xy_weight=foot_xy_weight)
    source_bytes = {key: value.tobytes() for key, value in motion.items()}
    local = dict(
        a=np.zeros((7, 81, 81)),
        b=np.zeros((7, 81, 23)),
        gradient=np.zeros((7, 81)),
        hessian=np.zeros((7, 81, 81)),
        terminal_g=np.zeros(81),
        terminal_h=np.zeros((81, 81)),
    )
    result, evidence, contract = guide.augment(local, dict(qpos=np.tile(standing, (8, 1))))
    assert result["a"] is local["a"] and result["b"] is local["b"]
    assert result["terminal_h"] is local["terminal_h"]
    assert not np.any(local["gradient"]) and not np.any(local["hessian"])
    assert np.any(result["gradient"][1])
    for index in (0, 4, 6):
        assert not np.any(result["gradient"][index])
        assert not np.any(result["hessian"][index])
    assert evidence["indices"].tolist() == [1, 2, 3, 5]
    assert contract["original_acceptance_objective_and_thresholds_unchanged"]
    if foot_xy_weight:
        assert contract["kind"] == "g1_true23_original_reference_local_transition_placement_guidance_v2"
        assert contract["foot_horizontal_guided_nodes_per_foot"] == 4
        assert contract["foot_horizontal_weight"] == foot_xy_weight
        assert contract["source_motion_and_standing_nodes_not_given_placement_terms"]
        assert contract["measured_foot_offsets_not_subtracted_from_reference"]
    else:
        assert contract["kind"] == "g1_true23_original_reference_local_transition_guidance_v1"
        assert "foot_horizontal_weight" not in contract
    assert source_bytes == {key: value.tobytes() for key, value in motion.items()}


def test_lift_guidance_only_penalizes_underlift_and_never_source(native):  # noqa: F811
    model, standing, _ = native
    guide, _ = make_guide(model, standing, com_weight=0)
    above = standing.copy()
    above[2] += 0.1
    assert guide.state_terms(standing, 1)[0] > 0
    assert guide.state_terms(above, 1)[0] == 0
    assert guide.state_terms(standing, 4)[0] == 0


@pytest.mark.parametrize("weight", [-1.0, np.nan, np.inf])
@pytest.mark.parametrize("field", ["com_weight", "lift_weight", "foot_xy_weight"])
def test_invalid_guide_weight_rejected(native, weight, field):  # noqa: F811
    model, standing, _ = native
    with pytest.raises(ValueError, match="weights"):
        make_guide(model, standing, **{field: weight})


def test_placement_tracks_original_world_xy_for_both_feet(native):  # noqa: F811
    model, standing, _ = native
    guide, _ = make_guide(model, standing, com_weight=0, lift_weight=0, foot_xy_weight=25000)
    shifted = standing.copy()
    shifted[:2] += [0.03, -0.01]
    value, gradient, hessian = guide.state_terms(shifted, 1)
    # Two feet share this world translation; no reference is moved to the robot.
    assert value == pytest.approx(25.0, abs=1e-10)
    np.testing.assert_allclose(gradient[:3], [1500, -500, 0], atol=1e-9, rtol=0)
    np.testing.assert_allclose(hessian[:3, :3], np.diag([50000, 50000, 0]), atol=1e-9, rtol=0)
    assert not gradient[29:].any()
    assert not hessian[29:].any() and not hessian[:, 29:].any()
    for index in (0, 4, 6, 7):
        value, gradient, hessian = guide.state_terms(shifted, index)
        assert value == 0 and not gradient.any() and not hessian.any()


def test_placement_does_not_add_vertical_or_underlift_terms(native):  # noqa: F811
    model, standing, _ = native
    guide, _ = make_guide(model, standing, com_weight=0, lift_weight=0, foot_xy_weight=25000)
    for height in (-0.03, 0.1):
        pose = standing.copy()
        pose[2] += height
        value, gradient, _ = guide.state_terms(pose, 1)
        assert value == pytest.approx(0, abs=1e-20)
        np.testing.assert_allclose(gradient, 0, atol=1e-10, rtol=0)


def test_zero_placement_weight_preserves_default_terms_exactly(native):  # noqa: F811
    model, standing, _ = native
    default, _ = make_guide(model, standing)
    explicit, _ = make_guide(model, standing, foot_xy_weight=0)
    for index in range(8):
        for actual, expected in zip(
            default.state_terms(standing, index), explicit.state_terms(standing, index), strict=True
        ):
            np.testing.assert_array_equal(actual, expected)
