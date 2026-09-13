"""Prepared native23 reference FK for SIM; no dynamics or robot interface.

The legacy reference path allocates MjData and solves contacts/dynamics for a
height calculation every packet. Only body transforms are consumed. Separate
prepared data plus mj_kinematics computes those transforms without running a
second dynamics solve. Actual plant integration and packet validation stay in
their original code. The two scratch states never alias measured robot state.
"""

import copy

import numpy as np

from gear_sonic.utils.g1_true23_clean_mujoco_teleop import (
    NATIVE_TO_MJ,
    _finite_vector,
    _quaternion_conjugate,
    _quaternion_matrix,
    _quaternion_multiply,
    validate_reference_terms,
)


class PreparedNative23Reference:
    """Instance-local replacement of reference-only FK, installed before input."""

    def __init__(self, controller):
        if controller.completed != 0 or float(controller.data.time) != 0:
            raise ValueError("prepare reference FK before the first physical control")
        self.module, self.model = controller.module, controller.model
        if (self.model.nq, self.model.nv, self.model.nu) != (30, 29, 23):
            raise ValueError("prepared reference requires the true23 physical model")
        self.height_data = self.module.MjData(self.model)
        self.reference_data = self.module.MjData(self.model)
        self.ankles = tuple(
            self.module.mj_name2id(self.model, self.module.mjtObj.mjOBJ_BODY, name)
            for name in ("left_ankle_roll_link", "right_ankle_roll_link")
        )
        if min(self.ankles) < 1:
            raise ValueError("native23 ankle body missing")
        # Retain the legacy native body contract explicitly, rather than using
        # indices against an arbitrary 23-actuator morphology.
        for index, name in (
            (19, "left_wrist_roll_rubber_hand"),
            (24, "right_wrist_roll_rubber_hand"),
            (14, "torso_link"),
        ):
            if self.model.body(index).name != name:
                raise ValueError("prepared reference native23 task-body order differs")

    def root_height(self, joint_position_hardware):
        q = np.asarray(joint_position_hardware, dtype=np.float64)
        if q.shape != (23,) or not np.isfinite(q).all():
            raise ValueError("reference joint position must be finite native23 hardware order")
        probe = self.height_data
        probe.qpos[:3] = (0.0, 0.0, 0.0)
        probe.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
        probe.qpos[7:] = q
        self.module.mj_kinematics(self.model, probe)
        height = 0.06 - min(float(probe.xpos[index, 2]) for index in self.ankles)
        if not 0.20 <= height <= 0.95:
            raise ValueError("reference root height outside native23 envelope")
        return height

    def retarget(self, packet):
        validate_reference_terms(packet)
        q9_native = _finite_vector(packet["q_ref23_native"], 23, "q9 native")
        q9_hardware = q9_native[NATIVE_TO_MJ]
        probe = self.reference_data
        probe.qpos[:3] = (0.0, 0.0, self.root_height(q9_hardware))
        probe.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
        probe.qpos[7:] = q9_hardware
        probe.qvel[:] = 0.0
        self.module.mj_kinematics(self.model, probe)
        body_pos = np.asarray(probe.xpos[1:], dtype=np.float64)
        body_quat = np.asarray(probe.xquat[1:], dtype=np.float64)
        anchor_pos, anchor_quat = body_pos[0], body_quat[0]
        inverse = _quaternion_matrix(anchor_quat).T
        vr_pos, vr_quat = [], []
        for body_index, offset in (
            (18, (0.18, -0.025, 0.0)),
            (23, (0.18, 0.025, 0.0)),
            (13, (0.0, 0.0, 0.35)),
        ):
            quaternion = body_quat[body_index]
            point = body_pos[body_index] + _quaternion_matrix(quaternion) @ np.asarray(offset)
            vr_pos.extend((inverse @ (point - anchor_pos)).tolist())
            vr_quat.extend(_quaternion_multiply(_quaternion_conjugate(anchor_quat), quaternion).tolist())
        result = copy.deepcopy(dict(packet))
        result["vr_3point_local_target"] = vr_pos
        result["vr_3point_local_orn_target"] = vr_quat
        result["reference_anchor_quaternion_xyzw"] = anchor_quat[[1, 2, 3, 0]].tolist()
        validate_reference_terms(result)
        return result

    def install(self, controller):
        if controller.model is not self.model or controller.module is not self.module:
            raise ValueError("prepared FK belongs to another simulator instance")
        if controller.completed != 0 or float(controller.data.time) != 0:
            raise ValueError("install reference FK before the first physical control")
        controller.reference_root_height = self.root_height
        controller.retarget_pico_reference_packet = self.retarget
