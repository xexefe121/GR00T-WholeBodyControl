import mujoco
import numpy as np
import pytest

from gear_sonic.scripts.prepare_g1_true23_contact_step_lifecycle import motion_from_poses
from gear_sonic.tests.test_g1_true23_contact_step_transition import native  # noqa: F401
from gear_sonic.utils.g1_true23_pd_standing_guidance import StandingVelocityGuidance


def guide_fixture(model, standing, **kwargs):
    motion = motion_from_poses(model, np.tile(standing, (18, 1)))
    timeline = dict(
        total_requested_controls=7,
        phases=[
            dict(name="source_motion", control_start=0, control_stop=3),
            dict(name="return_ramp", control_start=3, control_stop=4),
            dict(name="returned_standing", control_start=4, control_stop=6),
            dict(name="standing_proof_margin", control_start=6, control_stop=7),
        ],
    )
    return StandingVelocityGuidance(model, motion, timeline, **kwargs), motion


def test_native_velocity_gradient_and_hessian_match_independent_differences(native):  # noqa: F811
    model, standing, _ = native
    guide, _ = guide_fixture(model, standing)
    rng = np.random.default_rng(746)
    velocity, direction = rng.normal(0, 0.1, (2, 29))
    cost, gradient, hessian = guide.state_terms(velocity, 5)
    step = 1e-5
    plus = guide.state_terms(velocity + step * direction, 5)
    minus = guide.state_terms(velocity - step * direction, 5)
    np.testing.assert_allclose(gradient[29:58] @ direction, (plus[0] - minus[0]) / (2 * step), rtol=1e-9)
    np.testing.assert_allclose(hessian[29:58, 29:58] @ direction, (plus[1][29:58] - minus[1][29:58]) / (2 * step))
    assert cost > 0 and not gradient[:29].any() and not gradient[58:].any()
    assert not guide.state_terms(velocity, 3)[1].any()
    assert guide.state_terms(np.zeros(29), 7)[0] == 0


def test_only_original_standing_and_final_state_guide_no_source_or_cache_mutation(native):  # noqa: F811
    model, standing, _ = native
    guide, motion = guide_fixture(model, standing)
    source_bytes = {name: value.tobytes() for name, value in motion.items()}
    local = dict(
        a=np.zeros((7, 81, 81)),
        b=np.zeros((7, 81, 23)),
        gradient=np.zeros((7, 81)),
        hessian=np.zeros((7, 81, 81)),
        terminal_g=np.zeros(81),
        terminal_h=np.zeros((81, 81)),
    )
    result, evidence, contract = guide.augment(local, dict(qvel=np.full((8, 29), 0.1)))
    assert result["a"] is local["a"] and result["b"] is local["b"]
    assert not result["gradient"][:5].any() and not result["hessian"][:5].any()
    assert result["gradient"][5:].any() and result["terminal_g"].any()
    assert result["terminal_h"].any() and contract["terminal_node_included"]
    assert evidence["indices"].tolist() == [5, 6, 7]
    assert all(not values.any() for name, values in local.items() if name not in ("a", "b"))
    assert source_bytes == {name: value.tobytes() for name, value in motion.items()}


@pytest.mark.parametrize("weight", [-1, np.inf, np.nan])
def test_invalid_weights_rejected(native, weight):  # noqa: F811
    with pytest.raises(ValueError, match="weights"):
        guide_fixture(native[0], native[1], root_weight=weight)


def test_incomplete_trajectory_rejected(native):  # noqa: F811
    guide, _ = guide_fixture(native[0], native[1])
    local = dict(
        gradient=np.zeros((7, 81)),
        hessian=np.zeros((7, 81, 81)),
        terminal_g=np.zeros(81),
        terminal_h=np.zeros((81, 81)),
    )
    with pytest.raises(ValueError, match="crop"):
        guide.augment(local, dict(qvel=np.zeros((7, 29))))


def test_native_orientation_gradient_matches_independent_pose_perturbation(native):  # noqa: F811
    model, standing, _ = native
    guide, _ = guide_fixture(model, standing, orientation_weight=1000)
    pose, direction = standing.copy(), np.zeros(29)
    direction[3:6] = [0.7, -0.4, 0.2]
    mujoco.mj_integratePos(model, pose, direction, 0.23)
    velocity = np.zeros(29)
    cost, gradient, hessian = guide.state_terms(velocity, 5, pose)
    step = 1e-5
    rng = np.random.default_rng(540)
    for probe in rng.normal(size=(4, 29)):
        plus, minus = pose.copy(), pose.copy()
        mujoco.mj_integratePos(model, plus, probe, step)
        mujoco.mj_integratePos(model, minus, probe, -step)
        delta = (guide.state_terms(velocity, 5, plus)[0] - guide.state_terms(velocity, 5, minus)[0]) / (2 * step)
        np.testing.assert_allclose(gradient[:29] @ probe, delta, rtol=1e-8, atol=1e-7)
    assert cost > 0 and np.linalg.eigvalsh(hessian).min() >= -1e-9
    assert not gradient[:3].any() and not gradient[6:].any()
    opposite = pose.copy()
    opposite[3:7] *= -1
    alternate = guide.state_terms(velocity, 5, opposite)
    for actual, expected in zip(alternate, (cost, gradient, hessian), strict=True):
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)
    zero = guide.state_terms(velocity, 5, standing)
    assert zero[0] == 0 and not zero[1].any()
    np.testing.assert_allclose(zero[2][3:6, 3:6], 1000 * np.eye(3))


def test_orientation_guidance_includes_terminal_node_without_changing_velocity_or_source(native):  # noqa: F811
    model, standing, _ = native
    guide, _ = guide_fixture(model, standing, orientation_weight=1000)
    original, _ = guide_fixture(model, standing)
    local = dict(
        a=np.zeros((7, 81, 81)),
        b=np.zeros((7, 81, 23)),
        gradient=np.zeros((7, 81)),
        hessian=np.zeros((7, 81, 81)),
        terminal_g=np.zeros(81),
        terminal_h=np.zeros((81, 81)),
    )
    poses = np.tile(standing, (8, 1))
    for pose in poses:
        direction = np.zeros(29)
        direction[4] = 0.1
        mujoco.mj_integratePos(model, pose, direction, 1)
    trajectory = dict(qpos=poses, qvel=np.full((8, 29), 0.1))
    before = {key: value.tobytes() for key, value in trajectory.items()}
    result, _, contract = guide.augment(local, trajectory)
    old_result, _, _ = original.augment(local, trajectory)
    for key in ("gradient", "hessian", "terminal_g", "terminal_h"):
        assert not local[key].any()
    np.testing.assert_array_equal(result["gradient"][:, 29:], old_result["gradient"][:, 29:])
    assert not result["gradient"][:5, :29].any()
    assert result["terminal_g"][4] > 0 and contract["terminal_node_included"]
    assert contract["root_orientation_weight"] == 1000
    assert result["a"] is local["a"] and result["b"] is local["b"]
    assert before == {key: value.tobytes() for key, value in trajectory.items()}


@pytest.mark.parametrize("weight", [-1, np.inf, np.nan])
def test_invalid_orientation_weight_rejected(native, weight):  # noqa: F811
    with pytest.raises(ValueError, match="weights"):
        guide_fixture(native[0], native[1], orientation_weight=weight)


@pytest.mark.parametrize("pose", [None, np.zeros(30), np.full(30, np.nan), np.zeros(29)])
def test_missing_or_invalid_orientation_pose_rejected(native, pose):  # noqa: F811
    guide, _ = guide_fixture(native[0], native[1], orientation_weight=1000)
    with pytest.raises(ValueError, match="finite native23 pose"):
        guide.state_terms(np.zeros(29), 5, pose)
