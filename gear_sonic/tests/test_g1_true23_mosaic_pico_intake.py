import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from gear_sonic.scripts.audit_g1_true23_mosaic_pico_intake import poses_for_order


@pytest.mark.parametrize("root_yaw", [0.0, 0.3])
def test_serial_chain_matches_analytic_parent_origin_and_joint_composition(root_yaw):
    names = [f"joint_{i}" for i in range(29)]
    joints = [
        dict(
            name=name,
            type="revolute",
            parent="pelvis" if i == 0 else f"link_{i - 1}",
            child=f"link_{i}",
            xyz=np.array([1.0, 0.0, 0.0]),
            rpy=np.zeros(3),
            axis=np.array([0.0, 0.0, 1.0]),
        )
        for i, name in enumerate(names)
    ]
    angles = np.zeros((1, 29))
    angles[0, 0] = np.pi / 2
    root_quat = Rotation.from_euler("z", root_yaw).as_quat()[[3, 0, 1, 2]]
    arrays = dict(joint_pos=angles, body_pos_w=np.array([[[2.0, 3.0, 4.0]]]), body_quat_w=root_quat[None, None])
    actual_names, positions, rotations = poses_for_order(arrays, joints, "pelvis", names)
    root_rotation = Rotation.from_euler("z", root_yaw).as_matrix()
    assert actual_names[0] == "pelvis"
    for i in range(29):
        expected = np.array([2.0, 3.0, 4.0]) + root_rotation @ np.array([1.0, float(i), 0.0])
        np.testing.assert_allclose(positions[0, i + 1], expected, atol=1e-12)
        np.testing.assert_allclose(
            rotations[0, i + 1], Rotation.from_euler("z", root_yaw + np.pi / 2).as_matrix(), atol=1e-12
        )
