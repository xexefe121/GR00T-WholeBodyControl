"""Lossless original29 intent for a native23 research reference pipeline.

The 29 joints here belong to the recorded source, never the physical robot.
Preserve all of them when deriving the source hand/head positions and rotations.
Native23 feedback, action mappings, motor limits, and existing policy contracts
are not changed. These arrays are neither motor commands nor a qualified policy.
"""

from dataclasses import dataclass

import mujoco
import numpy as np

from gear_sonic.scripts import simulate_g1_sonic_library_motions as stock
from gear_sonic.utils.g1_23dof_contract import (
    HARDWARE_23_JOINT_NAMES,
    SOURCE_IL29_JOINT_NAMES,
    SOURCE_MJ29_KEEP_INDICES,
)
from gear_sonic.utils.g1_true23_generalist_corpus import canonical_digest
from gear_sonic.utils.g1_true23_reference_floor import compiled_model_sha256

REFERENCE_KIND = "original29_all_axes_source_intent_for_native23_v1"
SOURCE_JOINT_NAMES = tuple(SOURCE_IL29_JOINT_NAMES[i] for i in stock.MUJOCO_TO_ISAAC_INDEX)
VR_BODIES = ("left_wrist_yaw_link", "right_wrist_yaw_link", "torso_link")
VR_OFFSETS = ((0.18, -0.025, 0.0), (0.18, 0.025, 0.0), (0.0, 0.0, 0.35))
FLAGS = dict(hardware_authorized=False, deployment_ready=False, simulator_qualified=False)


def reference_contract(source_geometry_sha256):
    if (
        not isinstance(source_geometry_sha256, str)
        or len(source_geometry_sha256) != 64
        or any(c not in "0123456789abcdef" for c in source_geometry_sha256)
    ):
        raise ValueError("original29 reference requires a source geometry SHA256")
    payload = dict(
        kind=REFERENCE_KIND,
        source_geometry_sha256=source_geometry_sha256,
        source_joint_names=list(SOURCE_JOINT_NAMES),
        physical_joint_names=list(HARDWARE_23_JOINT_NAMES),
        physical_joint_count=23,
        missing_physical_axes_are_retained_source_data=True,
        source_period_s=0.02,
        task_body_names=list(VR_BODIES),
        task_body_local_offsets_m=[list(offset) for offset in VR_OFFSETS],
        virtual_vr21_layout="three_pelvis_local_xyz_then_three_pelvis_local_WXYZ",
        world_task_layout="same_three_original_source_points_and_body_WXYZ",
        receiver_lookahead_prediction_or_padding_added=False,
        robot_proprioception_or_commands_generated=False,
        old_zero_absent_reference_checkpoint_may_be_relabelled=False,
        training_reward_or_acceptance_criteria_changed=False,
        cross_robot_hand_registration_or_contact_geometry_claimed=False,
        **FLAGS,
    )
    return {**payload, "contract_sha256": canonical_digest(payload)}


def _poses(value):
    poses = np.asarray(value)
    if poses.ndim != 2 or poses.shape[1] != 36 or not len(poses):
        raise ValueError("source poses require nonempty [frames,36] original29 qpos")
    if poses.dtype.kind not in "fi" or not np.isfinite(poses).all():
        raise ValueError("source poses require finite real numbers")
    if not np.allclose(np.linalg.norm(poses[:, 3:7], axis=1), 1, rtol=0, atol=1e-6):
        raise ValueError("source poses require normalized WXYZ, never implicit repair")
    return poses.astype(np.float64, copy=True)


def _model(model):
    if (model.nq, model.nv, model.nu, model.njnt) != (36, 35, 29, 30):
        raise ValueError("reference geometry requires exactly original29")
    if tuple(model.joint(i).name for i in range(1, 30)) != SOURCE_JOINT_NAMES:
        raise ValueError("original29 joint names/order differ from the declared source")
    if (
        int(model.jnt_type[0]) != int(mujoco.mjtJoint.mjJNT_FREE)
        or np.any(model.jnt_type[1:] != int(mujoco.mjtJoint.mjJNT_HINGE))
        or not np.array_equal(model.jnt_qposadr, np.r_[0, np.arange(7, 36)])
    ):
        raise ValueError("original29 reference requires one free root and29 scalar hinges")
    return tuple(model.body(name).id for name in VR_BODIES)


@dataclass(frozen=True)
class Original29Reference:
    source_qpos29: np.ndarray
    virtual_vr21: np.ndarray
    source_task_position_w: np.ndarray
    source_task_quaternion_wxyz: np.ndarray

    def arrays(self):
        return {name: getattr(self, name).copy() for name in self.__dataclass_fields__}


def build_original29_reference(model, source_qpos29):
    """Per-frame FK only: no policy, integration, robot feedback, or zeroed axes."""
    poses, bodies = _poses(source_qpos29), _model(model)
    before = compiled_model_sha256(model)
    data = mujoco.MjData(model)
    vr, world_positions, world_quaternions = [], [], []
    for pose in poses:
        data.qpos[:] = pose
        mujoco.mj_fwdPosition(model, data)
        inverse = stock._quat_conjugate(pose[3:7])
        local_pos, local_quat, positions, quaternions = [], [], [], []
        for body, offset in zip(bodies, VR_OFFSETS, strict=True):
            point = data.xpos[body] + stock._quat_rotate(data.xquat[body], np.asarray(offset))
            positions.append(point)
            quaternions.append(data.xquat[body].copy())
            local_pos.extend(stock._quat_rotate(inverse, point - pose[:3]))
            local_quat.extend(stock._quat_multiply(inverse, data.xquat[body]))
        vr.append([*local_pos, *local_quat])
        world_positions.append(positions)
        world_quaternions.append(quaternions)
    result = Original29Reference(
        poses,
        np.asarray(vr, dtype=np.float32),
        np.asarray(world_positions, dtype=np.float64),
        np.asarray(world_quaternions, dtype=np.float64),
    )
    for array in result.arrays().values():
        if not np.isfinite(array).all():
            raise ValueError("original29 FK produced nonfinite reference data")
    for name in result.__dataclass_fields__:
        getattr(result, name).setflags(write=False)
    if compiled_model_sha256(model) != before:
        raise ValueError("reference FK unexpectedly mutated model geometry")
    return result


def verify_unmodified_native_pair(reference, motion):
    """Check the specific unretargeted 29→23 projection, not arbitrary task fits.

    A later root/joint retarget must carry its own explicit correspondence and
    correction certificate. It cannot pretend to be this lossless projection.
    """
    if not isinstance(reference, Original29Reference):
        raise ValueError("pairing requires an explicit original29 reference")
    poses = _poses(reference.source_qpos29)
    if np.asarray(motion["fps"]).shape != (1,) or float(motion["fps"][0]) != 50:
        raise ValueError("paired native source must preserve the50-Hz time grid")
    expected = {
        "joint_pos": poses[:, 7:][:, SOURCE_MJ29_KEEP_INDICES],
        "root_position": poses[:, :3],
        "root_quaternion": poses[:, 3:7],
    }
    positions, quaternions = np.asarray(motion["body_pos_w"]), np.asarray(motion["body_quat_w"])
    if positions.shape != (len(poses), 24, 3) or quaternions.shape != (len(poses), 24, 4):
        raise ValueError("paired native body channels require the complete24-body layout")
    actual = {
        "joint_pos": np.asarray(motion["joint_pos"]),
        "root_position": positions[:, 0],
        "root_quaternion": quaternions[:, 0],
    }
    for name in expected:
        a, b = actual[name], expected[name]
        if a.shape != b.shape or a.dtype.kind not in "fi" or not np.isfinite(a).all():
            raise ValueError(f"invalid paired native {name}")
        if not np.allclose(a, b, rtol=0, atol=1e-12):
            raise ValueError(f"paired native {name} changed; needs a distinct retarget contract")
    return dict(
        kind="original29_to_native23_unmodified_reference_pair_v1",
        frames=len(poses),
        source_speed_factor=1.0,
        root_and_retained_joints_preserved=True,
        source_missing_axes_preserved_for_intent=True,
        source_or_native_dynamic_feasibility_proven=False,
        **FLAGS,
    )


def task_targets_from_received_vr(root_position_w, root_quaternion_wxyz, virtual_vr21):
    """Recover intended original world hand/head targets without native re-FK.

    Consumers must explicitly select the same received reference index and use
    an independently declared native hand proxy. This helper does not install a
    reward, retargeter, transport, live controller, or new acceptance threshold.
    """
    p, q, vr = (np.asarray(value) for value in (root_position_w, root_quaternion_wxyz, virtual_vr21))
    if p.ndim != 2 or p.shape[1] != 3 or not len(p) or q.shape != (len(p), 4) or vr.shape != (len(p), 21):
        raise ValueError("received task targets require aligned [frames,3/4/21] arrays")
    if any(a.dtype.kind not in "fi" or not np.isfinite(a).all() for a in (p, q, vr)):
        raise ValueError("received task targets must be finite real numbers")
    if not np.allclose(np.linalg.norm(q, axis=1), 1, rtol=0, atol=1e-4) or not np.allclose(
        np.linalg.norm(vr[:, 9:].reshape(-1, 3, 4), axis=-1), 1, rtol=0, atol=1e-4
    ):
        raise ValueError("received task target quaternions must be normalized WXYZ")
    positions, orientations = [], []
    for root_p, root_q, value in zip(p, q, vr, strict=True):
        root_q = root_q.astype(float) / np.linalg.norm(root_q.astype(float))
        positions.append([root_p + stock._quat_rotate(root_q, local) for local in value[:9].reshape(3, 3)])
        quats = [stock._quat_multiply(root_q, local) for local in value[9:].reshape(3, 4)]
        orientations.append([x / np.linalg.norm(x) for x in quats])
    return np.asarray(positions), np.asarray(orientations)
