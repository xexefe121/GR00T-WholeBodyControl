from pathlib import Path

import mujoco
import numpy as np
import pytest

from gear_sonic.utils.g1_sonic_hand_collision_variant import (
    COLLISION_DERIVED_FIELDS,
    compile_hand_collision_variant,
    inspect_hand_contacts,
    numeric_differences,
)
from gear_sonic.utils.g1_true23_reference_floor import compiled_model_sha256


@pytest.fixture(scope="module")
def material():
    scene = Path(__file__).resolve().parents[2] / "gear_sonic_deploy/g1/scene_29dof.xml"
    original = mujoco.MjModel.from_xml_path(str(scene))
    original.opt.timestep = 0.002
    return compile_hand_collision_variant(scene, expected_baseline_sha256=compiled_model_sha256(original))


def test_recompiled_variant_changes_only_hand_masks_and_compiler_collision_data(material):
    baseline, variant, report = material
    assert report["baseline_compiled_sha256"] == compiled_model_sha256(baseline)
    assert report["variant_compiled_sha256"] == compiled_model_sha256(variant)
    assert report["baseline_compiled_sha256"] != report["variant_compiled_sha256"]
    assert set(numeric_differences(baseline, variant)) <= COLLISION_DERIVED_FIELDS | {
        "geom_contype",
        "geom_conaffinity",
    }
    for name in (
        "body_mass",
        "body_inertia",
        "geom_pos",
        "geom_quat",
        "mesh_vert",
        "mesh_face",
        "actuator_ctrlrange",
        "jnt_range",
        "dof_damping",
        "dof_armature",
    ):
        np.testing.assert_array_equal(getattr(baseline, name), getattr(variant, name))
    for name in ("teacher_accepted", "hardware_authorized", "deployment_ready", "native23_model_modified"):
        assert report[name] is False


def test_visible_hand_overlap_becomes_real_collision_without_changing_geometry(material):
    baseline, variant, _ = material
    pose = baseline.qpos0.copy()
    first = inspect_hand_contacts(baseline, pose[None])
    pose[2] -= first["minimum_hand_vertex_clearance_m"] + 0.01
    original = inspect_hand_contacts(baseline, pose[None])
    active = inspect_hand_contacts(variant, pose[None])
    assert original["minimum_hand_vertex_clearance_m"] == pytest.approx(-0.01)
    assert active["minimum_hand_vertex_clearance_m"] == pytest.approx(-0.01)
    assert original["frames_with_active_hand_floor_contacts"] == 0
    assert active["frames_with_active_hand_floor_contacts"] == 1
    assert active["minimum_active_hand_contact_distance_m"] == pytest.approx(-0.01, abs=1e-6)
    assert not active["physical_contact_feasibility_proven"]


def test_wrong_baseline_identity_rejected():
    scene = Path(__file__).resolve().parents[2] / "gear_sonic_deploy/g1/scene_29dof.xml"
    with pytest.raises(ValueError, match="exact recorded"):
        compile_hand_collision_variant(scene, expected_baseline_sha256="0" * 64)


@pytest.mark.parametrize("poses", [np.empty((0, 36)), np.zeros((3, 30)), np.full((3, 36), np.nan)])
def test_invalid_hand_inspection_source_rejected(material, poses):
    with pytest.raises(ValueError):
        inspect_hand_contacts(material[0], poses)
