from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from gear_sonic.utils.g1_23dof_safe_target_transform import SAFE_TARGET_DEFAULT_Q_HARDWARE
from gear_sonic.utils.g1_23dof_task_space_retarget import build_mjlab_motion_arrays
from gear_sonic.utils.g1_true23_clean_mujoco_teleop import motion_reference_terms
from gear_sonic.utils.g1_true23_pico_fullbody_motion import (
    build_pico_fullbody_motion,
    collision_grounded_root_heights,
)
from gear_sonic.utils.g1_true23_reference_floor import compiled_model_sha256
from gear_sonic.utils.g1_true23_reference_support import audit_reference_support

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def model_and_bundle():
    model = mujoco.MjModel.from_xml_path(str(ROOT / "gear_sonic/data/robots/g1/g1_23dof_rev_1_0.xml"))
    trajectory = SimpleNamespace(
        root_pos_w=np.tile([0, 0, 0.8], (32, 1)),
        root_quat_wxyz=np.tile([1, 0, 0, 0], (32, 1)),
        joint_pos_hardware=np.tile(SAFE_TARGET_DEFAULT_Q_HARDWARE, (32, 1)),
        fps=50.0,
    )
    motion = build_mjlab_motion_arrays(model, trajectory)
    packets = [motion_reference_terms(motion, q9) for q9 in range(9, 30)]
    return model, {"robot_independent_reference_packets": packets, "semantic_packets": [{} for _ in packets]}


def test_collision_grounding_fixes_hover_without_touching_joint_proof_or_original_default(
    model_and_bundle, monkeypatch
):
    from gear_sonic.utils import g1_true23_pico_fullbody_motion as pico

    model, bundle = model_and_bundle
    # Geometry unit test is independent of checkout line endings. The CLI
    # integration uses the unchanged byte-pinned LF asset, not a relaxed pin.
    monkeypatch.setattr(pico, "prepare_true23_model", lambda *args: (mujoco, model, None))
    legacy, old_report = build_pico_fullbody_motion(repository_root=ROOT, packet_bundle=bundle, minimum_frames=32)
    grounded, report = build_pico_fullbody_motion(
        repository_root=ROOT,
        packet_bundle=bundle,
        minimum_frames=32,
        collision_grounding=True,
    )
    assert old_report["root_height_rule"] == "minimum_ankle_roll_body_z_equals_0p06m"
    assert report["kind"] == "g1_true23_pico_fullbody_kinematic_motion_v2"
    assert report["collision_grounding"]["minimum_foot_floor_gap_m"] == pytest.approx(1e-5, abs=1e-7)
    assert report["collision_grounding"]["frames_dropped"] == 0
    for key in ("joint_pos", "joint_vel", "body_quat_w", "body_ang_vel_w", "fps"):
        np.testing.assert_array_equal(grounded[key], legacy[key])
    np.testing.assert_array_equal(grounded["body_pos_w"][:, :, :2], legacy["body_pos_w"][:, :, :2])
    np.testing.assert_array_equal(grounded["body_lin_vel_w"][:, :, :2], legacy["body_lin_vel_w"][:, :, :2])
    assert np.all(grounded["body_pos_w"][:, 0, 2] < legacy["body_pos_w"][:, 0, 2] - 0.01)
    before = audit_reference_support(model, legacy, np.full(23, 100))
    after = audit_reference_support(model, grounded, np.full(23, 100))
    assert before["frames_with_no_candidate_contact"] == 32
    assert after["frames_with_conditional_solution_within_effort_limits"] == 32
    assert not report["authorization"]["hardware_authorized"]
    assert not report["collision_grounding"]["dynamic_feasibility_proven"]


def test_grounding_preserves_compiled_model_and_hardware_input(model_and_bundle):
    model, _ = model_and_bundle
    hardware = np.tile(SAFE_TARGET_DEFAULT_Q_HARDWARE, (16, 1))
    previous = hardware.copy()
    model_hash = compiled_model_sha256(model)
    heights, evidence = collision_grounded_root_heights(mujoco, model, hardware)
    assert heights.shape == (16,) and np.isfinite(heights).all()
    np.testing.assert_array_equal(hardware, previous)
    assert compiled_model_sha256(model) == model_hash
    assert not evidence["joint_samples_changed"] and not evidence["deployment_ready"]


def test_grounding_does_not_hide_another_body_floor_penetration(model_and_bundle):
    model, _ = model_and_bundle
    pelvis = model.body("pelvis").id
    geom = next(i for i in range(model.ngeom) if model.geom_bodyid[i] == pelvis and model.geom_contype[i])
    model.geom_pos[geom, 2] -= 1.0
    with pytest.raises(ValueError, match="lift bound|other body geometry"):
        collision_grounded_root_heights(mujoco, model, np.tile(SAFE_TARGET_DEFAULT_Q_HARDWARE, (16, 1)))


@pytest.mark.parametrize("clearance", [0, -0.001, 0.01, float("nan")])
def test_invalid_grounding_clearance_rejected(model_and_bundle, clearance):
    model, _ = model_and_bundle
    with pytest.raises(ValueError, match="clearance"):
        collision_grounded_root_heights(
            mujoco, model, np.tile(SAFE_TARGET_DEFAULT_Q_HARDWARE, (16, 1)), clearance_m=clearance
        )
