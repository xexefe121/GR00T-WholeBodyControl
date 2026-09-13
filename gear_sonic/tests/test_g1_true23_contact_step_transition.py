import json
import os
from pathlib import Path

import mujoco
import numpy as np
import pytest

from gear_sonic.scripts.prepare_g1_true23_contact_step_lifecycle import motion_from_poses, preserve_nongenerated
from gear_sonic.utils import g1_true23_reference_support as support_audit
from gear_sonic.utils.g1_true23_contact_step_transition import (
    STAGES,
    plan_dynamic_com,
    smooth_fraction,
    step_targets,
    swing_lift,
)
from gear_sonic.utils.g1_true23_generalist_benchmark import MODEL, PHYSICS
from gear_sonic.utils.g1_true23_reference_floor import compiled_model_sha256
from gear_sonic.utils.g1_true23_stance_foot_cleanup import foot_frames
from gear_sonic.utils.g1_true23_step1b_mujoco import prepare_true23_model


@pytest.fixture(scope="module")
def native():
    root = Path(__file__).resolve().parents[2]
    assets = Path(os.environ.get("G1_TRUE23_TEST_ASSET_ROOT", str(root)))
    _, model, _ = prepare_true23_model(assets / MODEL, root / PHYSICS)
    initial = json.loads((root / PHYSICS).read_text())["initial_state"]
    standing = np.asarray(
        [*initial["base_position_m"], *initial["base_quaternion_wxyz"], *initial["joint_position_hardware_rad"]]
    )
    finish = standing.copy()
    finish[:2] += [-0.02, 0.01]
    return model, standing, finish


def test_minimum_jerk_lift_has_exact_zero_endpoints_and_declared_apex():
    np.testing.assert_array_equal(smooth_fraction([0, 1]), [0, 1])
    np.testing.assert_array_equal(swing_lift([0, 1]), [0, 0])
    assert swing_lift(0.5) == 0.04
    t = np.linspace(0, 1, 1001)
    assert np.all(np.diff(smooth_fraction(t)) >= 0)
    np.testing.assert_allclose(swing_lift(t), swing_lift(1 - t), atol=1e-16)


def test_fixed_support_xy_complete_schedule_and_exact_endpoints(native):
    model, start, finish = native
    identity = compiled_model_sha256(model)
    templates, targets, report = step_targets(model, start, finish)
    expected, rotations = foot_frames(model, [start, finish])
    assert len(targets) == 1 + sum(count for _, count in STAGES)
    np.testing.assert_array_equal(templates[[0, -1]], [start, finish])
    np.testing.assert_allclose([targets[0]["positions"], targets[-1]["positions"]], expected, atol=1e-15)
    np.testing.assert_allclose([targets[0]["rotations"], targets[-1]["rotations"]], rotations, atol=1e-15)
    for previous, current in zip(targets[:-1], targets[1:], strict=True):
        for side, support in enumerate(current["support"]):
            if support:
                np.testing.assert_array_equal(current["positions"][side, :2], previous["positions"][side, :2])
    assert not report["measured_robot_state_used"]
    assert compiled_model_sha256(model) == identity


def test_planned_sole_points_do_not_pass_through_floor(native):
    model, start, finish = native
    _, targets, _ = step_targets(model, start, finish)
    for side in (0, 1):
        body = model.body(("left", "right")[side] + "_ankle_roll_link").id
        geoms = [
            int(g)
            for g in np.flatnonzero(model.geom_bodyid == body)
            if model.geom_contype[g] or model.geom_conaffinity[g]
        ]
        assert all(model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE for g in geoms)
        for target in targets:
            points = target["positions"][side] + model.geom_pos[geoms] @ target["rotations"][side].T
            assert np.min(points[:, 2] - model.geom_size[geoms, 0]) >= -2e-7


def test_dynamic_com_plan_independently_satisfies_polygon_and_preserves_foot_targets(native):
    model, start, finish = native
    identity = compiled_model_sha256(model)
    _, targets, _ = step_targets(model, start, finish)
    feet = np.array([target["positions"] for target in targets])
    orientations = np.array([target["rotations"] for target in targets])
    endpoints = np.array([targets[0]["com"], targets[-1]["com"]])
    report = plan_dynamic_com(model, targets)
    np.testing.assert_array_equal([target["positions"] for target in targets], feet)
    np.testing.assert_array_equal([target["rotations"] for target in targets], orientations)
    np.testing.assert_allclose([targets[0]["com"], targets[-1]["com"]], endpoints, atol=1e-8)
    assert report["maximum_support_polygon_violation_m"] <= 1e-8
    assert not report["full_rigid_body_dynamics_qualified"]
    assert compiled_model_sha256(model) == identity


def test_source_and_standing_channels_are_not_recomputed_or_cropped():
    before = dict(fps=np.array([50.0]), channel=np.arange(40)[:, None])
    after = dict(fps=np.array([50.0]), channel=np.zeros((45, 1)))
    old_phases = [
        dict(name="initial_standing", frame_start=11, frame_stop=16, requested_controls=5),
        dict(name="acquisition_ramp", frame_start=16, frame_stop=20, requested_controls=4),
        dict(name="source_motion", frame_start=20, frame_stop=40, requested_controls=20),
    ]
    phases = [
        dict(old_phases[0]),
        dict(name="acquisition_ramp", frame_start=16, frame_stop=25, requested_controls=9),
        dict(name="source_motion", frame_start=25, frame_stop=45, requested_controls=20),
    ]
    preserve_nongenerated(before, dict(phases=old_phases), after, dict(phases=phases))
    np.testing.assert_array_equal(after["channel"][:16], before["channel"][:16])
    np.testing.assert_array_equal(after["channel"][25:], before["channel"][20:])
    np.testing.assert_array_equal(after["channel"][16:25], np.zeros((9, 1)))


def test_wrong_robot_or_nonfinite_endpoints_fail_before_planning(native):
    model, start, finish = native
    finish = finish.copy()
    finish[0] = np.nan
    with pytest.raises(ValueError, match="finite exact native23"):
        step_targets(model, start, finish)


def test_support_numerical_failure_is_retained_as_unknown_never_a_pass(native, monkeypatch):
    model, start, _ = native
    motion = motion_from_poses(model, np.tile(start, (12, 1)))

    def fail(*_):
        raise RuntimeError("injected numerical support failure")

    monkeypatch.setattr(support_audit, "minimum_effort_support", fail)
    result = support_audit.audit_reference_support(model, motion, np.full(23, 35.0), record_solver_failures=True)
    assert result["frames_checked"] == result["frames_with_indeterminate_support"] == 12
    assert result["frames_dropped"] == result["frames_with_conditional_solution_within_effort_limits"] == 0
    assert all(not row["infeasibility_proven"] for row in result["rows"])
    with pytest.raises(RuntimeError, match="injected numerical support failure"):
        support_audit.audit_reference_support(model, motion, np.full(23, 35.0))
