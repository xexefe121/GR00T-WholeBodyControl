from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from gear_sonic.utils.g1_true23_reference_floor import compiled_model_sha256
from gear_sonic.utils.g1_true23_self_collision import SelfCollisionLinearizer, self_contact_rows


def model_fixture():
    return mujoco.MjModel.from_xml_string("""<mujoco><option gravity="0 0 0"/>
      <worldbody><geom name="ground" type="plane" size="3 3 .1"/>
        <body name="left" pos="0 0 .09"><joint name="slide" type="slide" axis="1 0 0"/>
          <geom name="left_ball" type="sphere" size=".1" mass="1"/>
        </body>
        <body name="right" pos=".18 0 .09"><joint name="other" type="slide" axis="1 0 0"/>
          <geom name="right_ball" type="sphere" size=".1" mass="1"/>
        </body>
      </worldbody></mujoco>""")


def test_exact_contact_filter_excludes_ground_but_keeps_signed_distance():
    model = model_fixture()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert data.ncon >= 3  # Both ground contacts are deliberately present.
    rows = self_contact_rows(model, data)
    assert len(rows) == 1
    assert rows[0]["geoms"] == (1, 2)
    assert rows[0]["distance_m"] == pytest.approx(-0.02)


def test_query_copy_does_not_mutate_physical_model_or_state():
    model = model_fixture()
    before = compiled_model_sha256(model)
    query = SelfCollisionLinearizer(model)
    pose = np.array([0.0, 0.03])
    saved = pose.copy()
    rows = query.pose_rows(pose, np.array([0, 1]))
    assert len(rows) == 1 and rows[0]["distance_m"] == pytest.approx(0.01)
    assert compiled_model_sha256(model) == before
    np.testing.assert_array_equal(pose, saved)
    data = mujoco.MjData(model)
    data.qpos[:] = pose
    mujoco.mj_forward(model, data)
    assert not self_contact_rows(model, data)


@pytest.mark.parametrize("separation", [0.0, 0.03])
def test_distance_gradient_matches_independent_finite_difference(separation):
    model = model_fixture()
    query = SelfCollisionLinearizer(model)
    pose = np.array([0.0, separation])
    row = query.pose_rows(pose, np.array([0, 1]))[0]
    numerical = []
    for column in range(2):
        delta = np.zeros(2)
        delta[column] = 1e-6
        plus = query.pose_rows(pose + delta, np.array([0, 1]))[0]["distance_m"]
        minus = query.pose_rows(pose - delta, np.array([0, 1]))[0]["distance_m"]
        numerical.append((plus - minus) / 2e-6)
    np.testing.assert_allclose(row["joint_jacobian"], numerical, atol=1e-8, rtol=0)
    np.testing.assert_allclose(numerical, [-1, 1], atol=1e-8)


def test_path_jacobian_does_not_mix_frames():
    query = SelfCollisionLinearizer(model_fixture())
    matrix, distance, ids = query.path_rows(np.array([[0.0, 0.0], [0.0, 0.03]]), np.array([0, 1]))
    np.testing.assert_allclose(matrix.toarray(), [[-1, 1, 0, 0], [0, 0, -1, 1]])
    np.testing.assert_allclose(distance, [-0.02, 0.01])
    assert ids == [(0, 1, 2), (1, 1, 2)]


def test_pair_deduplication_and_reversed_normal():
    contacts = [
        SimpleNamespace(geom1=2, geom2=1, dist=depth, frame=np.array([-1.0, 0, 0]), pos=np.zeros(3))
        for depth in [-0.01, -0.02, -0.005]
    ]
    rows = self_contact_rows(
        SimpleNamespace(geom_bodyid=np.array([0, 1, 2])), SimpleNamespace(contact=contacts, ncon=3)
    )
    assert len(rows) == 1 and rows[0]["distance_m"] == -0.02
    np.testing.assert_array_equal(rows[0]["normal_first_to_second"], [1, 0, 0])


@pytest.mark.parametrize("distance", [-1, 0, 0.051, np.nan, np.inf])
def test_invalid_query_margin_rejected(distance):
    with pytest.raises(ValueError, match="query distance"):
        SelfCollisionLinearizer(model_fixture(), near_distance_m=distance)


@pytest.fixture
def native_motion():
    from gear_sonic.utils.g1_23dof_contract import HARDWARE_23_JOINT_NAMES
    from gear_sonic.utils.g1_23dof_safe_target_transform import SAFE_TARGET_DEFAULT_Q_HARDWARE
    from gear_sonic.utils.g1_23dof_task_space_retarget import build_mjlab_motion_arrays

    model = mujoco.MjModel.from_xml_path(
        str(Path(__file__).resolve().parents[1] / "data/robots/g1/g1_23dof_rev_1_0.xml")
    )
    motion = build_mjlab_motion_arrays(
        model,
        SimpleNamespace(
            root_pos_w=np.tile([0.0, 0.0, 0.76], (12, 1)),
            root_quat_wxyz=np.tile([1.0, 0.0, 0.0, 0.0], (12, 1)),
            joint_pos_hardware=np.tile(SAFE_TARGET_DEFAULT_Q_HARDWARE, (12, 1)),
            fps=50.0,
        ),
    )
    motion["joint_names"] = np.array(HARDWARE_23_JOINT_NAMES)
    motion["extra_provenance_field"] = np.arange(12)
    return model, motion


def test_named_reference_validates_derived_body_channels_not_extra_metadata(native_motion):
    from gear_sonic.scripts.audit_g1_true23_reference_bank_self_contacts import validated_reference_qpos

    model, motion = native_motion
    poses, audit = validated_reference_qpos(model, motion)
    assert poses.shape == (12, 30) and audit["position_fk_consistent"]
    motion["body_pos_w"][:, 5, 0] += 0.1
    with pytest.raises(ValueError, match="body channels"):
        validated_reference_qpos(model, motion)


def test_reordered_reference_rejected(native_motion):
    from gear_sonic.scripts.audit_g1_true23_reference_bank_self_contacts import validated_reference_qpos

    model, motion = native_motion
    motion["joint_names"] = motion["joint_names"][::-1]
    with pytest.raises(ValueError, match="joint order"):
        validated_reference_qpos(model, motion)


def test_geometric_audit_ignores_global_translation_and_checks_root_quaternion(native_motion):
    from gear_sonic.scripts.audit_g1_true23_reference_bank_self_contacts import (
        measure_self_contacts,
        validated_reference_qpos,
    )

    model, motion = native_motion
    poses, _ = validated_reference_qpos(model, motion)
    first = measure_self_contacts(model, poses)
    poses[:, 2] += 2
    second = measure_self_contacts(model, poses)
    assert first["frames_with_robot_robot_penetration"] == second["frames_with_robot_robot_penetration"]
    assert first["maximum_penetration_m"] == pytest.approx(second["maximum_penetration_m"])
    assert not first["force_history_training_penalty_reproduced"]
    poses[:, 3:7] *= 2
    with pytest.raises(ValueError, match="normalized"):
        measure_self_contacts(model, poses)
