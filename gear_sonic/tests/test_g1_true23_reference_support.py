import copy
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from gear_sonic.utils.g1_23dof_safe_target_transform import SAFE_TARGET_DEFAULT_Q_HARDWARE
from gear_sonic.utils.g1_23dof_task_space_retarget import build_mjlab_motion_arrays
from gear_sonic.utils.g1_true23_reference_floor import compiled_model_sha256, condition_reference_floor
from gear_sonic.utils.g1_true23_reference_support import (
    audit_reference_support,
    contact_cone_rays,
    floor_contact_map,
    minimum_effort_support,
    pose_path_derivatives,
)


def test_analytic_support_torque_and_bounded_static_friction():
    required = np.zeros(29)
    required[2] = 100
    contact_map = np.zeros((29, 1))
    contact_map[2, 0], contact_map[6, 0] = 1, 0.2
    result = minimum_effort_support(required, contact_map, np.full(23, 10), np.ones(23))
    assert result["minimum_peak_effort_ratio"] == pytest.approx(1.9)
    assert result["joint_torques_nm"][0] == pytest.approx(-19)
    assert result["static_friction_assistance_nm"][0] == pytest.approx(-1)
    assert result["maximum_generalized_force_residual"] < 1e-8
    assert not result["within_supplied_effort_limits"]


def test_contact_cannot_pull_ground_or_invent_floating_base_actuator():
    required = np.zeros(29)
    required[2] = -100
    contact_map = np.zeros((29, 1))
    contact_map[2] = 1
    result = minimum_effort_support(required, contact_map, np.ones(23), np.zeros(23))
    assert result["minimum_peak_effort_ratio"] is None
    required[2] = 100
    result = minimum_effort_support(required, np.empty((29, 0)), np.ones(23), np.zeros(23))
    assert result["minimum_peak_effort_ratio"] is None


def test_com_outside_support_cannot_be_repaired_by_arbitrary_joint_torque():
    required = np.zeros(29)
    required[2], required[4] = 100, 20
    contact_map = np.zeros((29, 2))
    contact_map[2] = 1
    contact_map[4] = [-0.1, 0.1]
    result = minimum_effort_support(required, contact_map, np.full(23, 1e6), np.zeros(23))
    assert result["minimum_peak_effort_ratio"] is None


def test_unknown_solver_status_is_retried_not_mislabeled_infeasible(monkeypatch):
    from gear_sonic.utils import g1_true23_reference_support as support

    real_solver = support.linprog
    calls = []

    def first_unknown(*args, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return SimpleNamespace(status=4, success=False, message="Unknown")
        return real_solver(*args, **kwargs)

    monkeypatch.setattr(support, "linprog", first_unknown)
    result = minimum_effort_support(np.zeros(29), np.zeros((29, 0)), np.ones(23), np.zeros(23))
    assert result["minimum_peak_effort_ratio"] == 0
    assert result["solver_retried_without_presolve"]
    assert calls[1]["method"] == "highs-ds" and not calls[1]["options"]["presolve"]
    monkeypatch.setattr(
        support, "linprog", lambda *args, **kwargs: SimpleNamespace(status=4, success=False, message="Unknown")
    )
    with pytest.raises(RuntimeError, match="did not solve reliably"):
        minimum_effort_support(np.zeros(29), np.zeros((29, 0)), np.ones(23), np.zeros(23))


def test_degenerate_full_lp_requires_independent_root_infeasibility_status(monkeypatch):
    from gear_sonic.utils import g1_true23_reference_support as support

    real_solver = support.linprog

    def unknown_full_lp(objective, **kwargs):
        if len(objective) > 2:
            return SimpleNamespace(status=4, success=False, message="Unknown")
        return real_solver(objective, **kwargs)

    monkeypatch.setattr(support, "linprog", unknown_full_lp)
    required = np.zeros(29)
    required[2], required[4] = 100, 20
    force_map = np.zeros((29, 2))
    force_map[2], force_map[4] = 1, [-0.1, 0.1]
    result = minimum_effort_support(required, force_map, np.ones(23), np.zeros(23))
    assert result["minimum_peak_effort_ratio"] is None
    assert result["infeasibility_confirmed_by_root_wrench_lp"]


@pytest.mark.parametrize("dimension", [1, 3, 4, 6])
def test_cone_rays_respect_mujoco_pyramid(dimension):
    friction = np.array([0.6, 0.4, 0.01, 0.002, 0.003])
    rays = contact_cone_rays(dimension, friction)
    assert np.all(rays[0] == 1)
    assert np.all(rays[dimension:] == 0)
    if dimension > 1:
        np.testing.assert_allclose(np.sum(np.abs(rays[1:dimension]) / friction[: dimension - 1, None], axis=0), 1)


@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf")])
def test_invalid_effort_limits_rejected(bad):
    with pytest.raises(ValueError, match="positive torque"):
        minimum_effort_support(np.zeros(29), np.zeros((29, 0)), np.full(23, bad), np.zeros(23))


@pytest.fixture
def materials():
    root = Path(__file__).resolve().parents[2]
    model = mujoco.MjModel.from_xml_path(str(root / "gear_sonic/data/robots/g1/g1_23dof_rev_1_0.xml"))
    trajectory = SimpleNamespace(
        root_pos_w=np.tile([0, 0, 0.74], (16, 1)),
        root_quat_wxyz=np.tile([1, 0, 0, 0], (16, 1)),
        joint_pos_hardware=np.tile(SAFE_TARGET_DEFAULT_Q_HARDWARE, (16, 1)),
        fps=50.0,
    )
    source = build_mjlab_motion_arrays(model, trajectory)
    motion, _ = condition_reference_floor(source, {"mesh": model}, output_model=model)
    return model, motion


def test_native_model_standing_screen_preserves_inputs_and_false_readiness(materials):
    model, motion = materials
    model_hash = compiled_model_sha256(model)
    original = {key: value.copy() for key, value in motion.items()}
    report = audit_reference_support(model, motion, np.full(23, 100))
    assert report["frames_checked"] == 16 and report["frames_dropped"] == 0
    assert report["frames_with_no_candidate_contact"] == 0
    assert report["frames_with_no_support_solution"] == 0
    assert report["frames_with_conditional_solution_within_effort_limits"] == 16
    # The heel gap is 10 um and the toe gap is about 1 mm. The precompiled
    # BVH misses the toes if only geom_margin changes at runtime.
    assert report["rows"][0]["candidate_contact_count"] == 8
    assert max(c["distance_m"] for c in report["rows"][0]["candidate_contacts"]) > 0.001
    assert not report["dynamic_feasibility_proven"] and not report["hardware_authorized"]
    assert not report["deployment_ready"]
    assert report["velocity_and_acceleration_assumed_zero"]
    assert compiled_model_sha256(model) == model_hash
    for key, value in original.items():
        np.testing.assert_array_equal(motion[key], value)


def test_floor_gap_not_silently_bridged(materials):
    model, motion = materials
    # Conditioned floor clearance is 10 micrometres: literal zero-gap candidates
    # cannot carry weight, while the explicit default near-contact screen can.
    report = audit_reference_support(model, motion, np.full(23, 100), gap_tolerance_m=0)
    assert report["frames_with_no_candidate_contact"] == 16
    motion["body_pos_w"][:, :, 2] += 0.1
    report = audit_reference_support(model, motion, np.full(23, 100))
    assert report["frames_with_no_candidate_contact"] == 16
    assert report["frames_with_conditional_solution_within_effort_limits"] == 0


def test_pose_derivatives_respect_quaternions_and_accelerating_translation(materials):
    model, _ = materials
    time = np.arange(16) * 0.02
    positions = np.tile(model.qpos0, (16, 1))
    positions[:, 0] = 0.5 * 3 * time**2
    positions[:, 3:7] = np.column_stack((np.cos(time), np.zeros((16, 2)), np.sin(time)))
    positions[5:8, 3:7] *= -1  # Quaternion signs must not create velocity spikes.
    velocity, acceleration = pose_path_derivatives(model, positions, 0.02)
    np.testing.assert_allclose(velocity[:, 0], 3 * time, atol=1e-12)
    np.testing.assert_allclose(acceleration[:, 0], 3, atol=1e-11)
    np.testing.assert_allclose(velocity[:, 3:6], np.tile([0, 0, 2], (16, 1)), atol=1e-12)
    np.testing.assert_allclose(acceleration[:, 3:6], 0, atol=1e-10)


def test_airborne_freefall_can_balance_forces_but_floating_stand_cannot(materials):
    model, motion = materials
    # The source XML's <joint type="free"> inherits small armature/damping.
    # This analytic test needs an ideal free base; do not silently apply this
    # different physical model to production reference audits.
    model.dof_armature[:6] = 0
    model.dof_damping[:6] = 0
    model.dof_frictionloss[:6] = 0
    motion = {key: value.astype(np.float64) for key, value in motion.items()}
    time = np.arange(16) * 0.02
    relative = motion["body_pos_w"] - motion["body_pos_w"][:, :1]
    relative[:, :, 2] += (2 - 0.5 * 9.81 * time**2)[:, None]
    motion["body_pos_w"] = relative
    dynamic = audit_reference_support(model, motion, np.full(23, 100), reference_dynamics=True)
    stationary = audit_reference_support(model, motion, np.full(23, 100))
    assert dynamic["frames_with_no_candidate_contact"] == 16
    assert dynamic["frames_with_no_support_solution"] == 0
    assert dynamic["maximum_finite_minimum_effort_ratio"] < 1e-8
    assert stationary["frames_with_no_support_solution"] == 16
    assert not dynamic["dynamic_feasibility_proven"] and not dynamic["deployment_ready"]


def test_constant_pose_modes_agree_and_ignore_archived_velocity_noise(materials):
    model, motion = materials
    motion["joint_vel"][:] = 123
    stationary = audit_reference_support(model, motion, np.full(23, 100))
    dynamic = audit_reference_support(model, motion, np.full(23, 100), reference_dynamics=True)
    assert not dynamic["reference_velocity_channels_used"]
    assert dynamic["maximum_finite_minimum_effort_ratio"] == pytest.approx(
        stationary["maximum_finite_minimum_effort_ratio"]
    )


def test_contact_jacobian_force_mapping_matches_mujoco_applyft(materials):
    model, motion = materials
    candidate = copy.copy(model)
    candidate.geom_margin[:] = 0.002
    data = mujoco.MjData(candidate)
    data.qpos[:3] = motion["body_pos_w"][0, 0]
    data.qpos[7:] = motion["joint_pos"][0]
    mujoco.mj_fwdPosition(candidate, data)
    plane = int(np.flatnonzero(candidate.geom_type == mujoco.mjtGeom.mjGEOM_PLANE)[0])
    force_map, records = floor_contact_map(candidate, data, plane, 0.002)
    assert records
    contact = next(c for c in data.contact[: data.ncon] if plane in c.geom)
    sign = 1 if contact.geom[0] == plane else -1
    rotation = sign * contact.frame.reshape(3, 3).T
    rays = contact_cone_rays(int(contact.dim), contact.friction)
    expected = np.zeros(candidate.nv)
    mujoco.mj_applyFT(
        candidate,
        data,
        rotation @ rays[:3, 0],
        rotation @ rays[3:, 0],
        contact.pos,
        candidate.body(records[0]["body"]).id,
        expected,
    )
    np.testing.assert_allclose(force_map[:, 0], expected, atol=1e-12)


def test_cli_records_whole_clip_and_does_not_overwrite(materials, tmp_path, monkeypatch):
    from gear_sonic.scripts import audit_g1_true23_reference_support as cli

    model, motion = materials
    np.savez_compressed(tmp_path / "motion.npz", **motion)
    manifest = tmp_path / "motions.json"
    manifest.write_text(json.dumps({"motions": [{"name": "full_clip", "path": "motion.npz"}]}))
    monkeypatch.setattr(cli.geometry_cli, "build_training_geometry", lambda: (model, []))
    monkeypatch.setattr(
        "sys.argv",
        [
            "audit",
            "--manifest",
            str(manifest),
            "--asset-root",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "audit"),
        ],
    )
    assert cli.main() == 0
    report = json.loads((tmp_path / "audit/report.json").read_text())
    assert report["controlled_joint_count"] == 23 and not report["deployment_ready"]
    assert report["records"][0]["models"]["retarget_mesh"]["frames_checked"] == 16
    with pytest.raises(FileExistsError):
        cli.main()
