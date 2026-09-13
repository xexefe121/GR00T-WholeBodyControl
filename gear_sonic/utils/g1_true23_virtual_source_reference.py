"""Simulation-only reference geometry for an unchanged released SONIC encoder.

The reference joints live on native23, whereas released SONIC learned VR points
on original29. Embed the 23 reference angles in a kinematic original29 with its
six absent axes fixed to zero. This model is never dynamically integrated and
never supplies robot feedback or evaluation truth. Native23 targets and all
physical safety limits remain unchanged. This is a new encoder-input convention,
not permission to relabel existing trained checkpoints or deploy on hardware.
"""

from pathlib import Path

import numpy as np

from gear_sonic.utils.g1_23dof_contract import HARDWARE_23_JOINT_NAMES
from gear_sonic.utils.g1_true23_step1b_mujoco import (
    _quaternion_conjugate,
    _quaternion_matrix,
    _quaternion_multiply,
)

VIRTUAL_SOURCE_REFERENCE = "native23_angles_original29_zero_absent_reference_v1"
VR_BODIES = ("left_wrist_yaw_link", "right_wrist_yaw_link", "torso_link")
VR_OFFSETS = ((0.18, -0.025, 0), (0.18, 0.025, 0), (0, 0, 0.35))


def virtual_source_vr_terms(motion, source_model_path):
    """Return [frames,21] source-shaped reference VR terms; never robot state."""
    import mujoco

    reference = np.asarray(motion["joint_pos"])
    if reference.ndim != 2 or reference.shape[1] != 23 or not len(reference):
        raise ValueError("reference requires nonempty native23 joint frames")
    if reference.dtype.kind not in "fi" or not np.isfinite(reference).all():
        raise ValueError("reference joint angles must be finite numbers")
    model = mujoco.MjModel.from_xml_path(str(Path(source_model_path).resolve(strict=True)))
    if (model.nq, model.nv, model.nu) != (36, 35, 29):
        raise ValueError("virtual source reference requires original29 geometry")
    joints = [model.joint(name) for name in HARDWARE_23_JOINT_NAMES]
    if any(int(joint.type[0]) != int(mujoco.mjtJoint.mjJNT_HINGE) for joint in joints):
        raise ValueError("expected 23 named scalar hinge joints")
    addresses = np.asarray([int(joint.qposadr[0]) for joint in joints])
    if len(set(addresses)) != 23:
        raise ValueError("native23-to-original29 reference mapping is not one-to-one")
    missing = set(range(7, 36)) - set(addresses)
    absent_names = {model.joint(index).name for index in range(1, 30) if int(model.jnt_qposadr[index]) in missing}
    expected_absent = {"waist_roll_joint", "waist_pitch_joint"} | {
        f"{side}_wrist_{axis}_joint" for side in ("left", "right") for axis in ("pitch", "yaw")
    }
    if absent_names != expected_absent:
        raise ValueError("unexpected absent-joint set")
    data = mujoco.MjData(model)
    result = []
    for q23 in reference:
        data.qpos[:] = 0
        # Removing common root translation/orientation is exact here: all VR
        # terms are expressed in that same reference pelvis frame.
        data.qpos[3] = 1
        data.qpos[addresses] = q23
        mujoco.mj_fwdPosition(model, data)
        root = data.xpos[model.body("pelvis").id]
        quat = data.xquat[model.body("pelvis").id]
        inverse = _quaternion_matrix(quat).T
        positions, quaternions = [], []
        for body, offset in zip(VR_BODIES, VR_OFFSETS, strict=True):
            body_id = model.body(body).id
            point = data.xpos[body_id] + _quaternion_matrix(data.xquat[body_id]) @ np.asarray(offset)
            positions.extend(inverse @ (point - root))
            quaternions.extend(_quaternion_multiply(_quaternion_conjugate(quat), data.xquat[body_id]))
        result.append([*positions, *quaternions])
    terms = np.asarray(result, dtype=np.float32)
    if terms.shape != (len(motion["joint_pos"]), 21) or not np.isfinite(terms).all():
        raise ValueError("virtual source produced invalid reference terms")
    return terms
