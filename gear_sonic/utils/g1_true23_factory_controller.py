"""Received-state adapter for the recovered factory pose-conditioned actor.

Simulation candidate only. The vendor actor was trained with 29 physical joints;
this adapter represents the six absent native23 joints as fixed at zero. Success
must be established by physical rollouts, not by the network's output shape.
No motion archive, future sample, clock, robot network, or DDS dependency here.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
import yaml

NATIVE23_TO_VENDOR29 = np.array(list(range(13)) + list(range(15, 20)) + list(range(22, 27)))
ABSENT_JOINTS = np.array([13, 14, 20, 21, 27, 28])


def normalized_quaternion(wxyz):
    value = np.asarray(wxyz, np.float64)
    if value.shape != (4,) or not np.isfinite(value).all():
        raise ValueError("invalid reference quaternion")
    length = np.linalg.norm(value)
    if length < 1e-8:
        raise ValueError("zero reference quaternion")
    value = value / length
    return value if value[0] >= 0 else -value


def quaternion_product(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([aw*bw-ax*bx-ay*by-az*bz,
                     aw*bx+ax*bw+ay*bz-az*by,
                     aw*by-ax*bz+ay*bw+az*bx,
                     aw*bz+ax*by-ay*bx+az*bw])


def quaternion_matrix(wxyz):
    w,x,y,z = normalized_quaternion(wxyz)
    return np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],
                     [2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],
                     [2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])


@dataclass(frozen=True)
class ReceivedPose:
    sequence: int
    timestamp: float
    root_position: np.ndarray
    root_quaternion_wxyz: np.ndarray
    joint_position: np.ndarray


class ReceivedReference:
    """Factory target state from one received pose plus backward differences.

    Factory CSV columns: root xyz, root quaternion xyzw, joint q29, root linear
    and angular velocities in the reference BODY frame, joint dq29. This reproduces the first 71 CSV
columns copied by FsmCpyDance29dofs::Run. Missing joint position and velocity are
always zero. Supplied offline velocity/future arrays are deliberately unused.
"""
    def __init__(self):
        self.previous = None
        self.state = None

    def accept(self, pose: ReceivedPose):
        if not np.isfinite(pose.timestamp):
            raise ValueError("nonfinite packet timestamp")
        p = np.asarray(pose.root_position, np.float64)
        q = np.asarray(pose.joint_position, np.float64)
        quat = normalized_quaternion(pose.root_quaternion_wxyz)
        if p.shape != (3,) or q.shape != (23,) or not np.isfinite(p).all() or not np.isfinite(q).all():
            raise ValueError("invalid native23 pose packet")
        prior = self.previous
        if prior is not None and (pose.sequence <= prior.sequence or pose.timestamp <= prior.timestamp):
            raise ValueError("out-of-order pose packet")
        linear, angular, velocity = np.zeros(3), np.zeros(3), np.zeros(23)
        if prior is not None:
            dt = pose.timestamp - prior.timestamp
            linear = (p - prior.root_position) / dt
            velocity = (q - prior.joint_position) / dt
            # First differentiate in world frame, then rotate both velocities
            # into the reference body's frame, as in the vendor motion CSV.
            inverse = normalized_quaternion(prior.root_quaternion_wxyz) * np.array([1, -1, -1, -1])
            delta = normalized_quaternion(quaternion_product(quat, inverse))
            sine = np.linalg.norm(delta[1:])
            if sine > 1e-9:
                angular = delta[1:] / sine * (2 * np.arctan2(sine, delta[0]) / dt)
            rotation = quaternion_matrix(quat)
            linear = rotation.T @ linear
            angular = rotation.T @ angular
        q29, v29 = np.zeros(29), np.zeros(29)
        q29[NATIVE23_TO_VENDOR29] = q
        v29[NATIVE23_TO_VENDOR29] = velocity
        self.state = np.concatenate((p, quat[[1, 2, 3, 0]], q29, linear, angular, v29)).astype(np.float32)
        self.previous = ReceivedPose(pose.sequence, pose.timestamp, p.copy(), quat.copy(), q.copy())
        return self.state.copy()


class FactoryPoseController:
    """Vendor pose actor with observed fixed joints and native target limits."""
    def __init__(self, model_path: Path, config_path: Path, joint_limits, margin=.06):
        cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf8"))
        self.default29 = np.asarray(cfg["default_dof_pos"], np.float32)
        self.action_scale = float(cfg["action_scale"])
        self.output_scale = np.asarray(cfg["output_scale"], np.float32)
        self.kp = np.asarray(cfg["kp"])[NATIVE23_TO_VENDOR29]
        self.kd = np.asarray(cfg["kd"])[NATIVE23_TO_VENDOR29]
        self.default = self.default29[NATIVE23_TO_VENDOR29].copy()
        self.limits = np.asarray(joint_limits)
        self.margin = float(margin)
        if self.limits.shape != (23, 2) or margin < 0 or np.any(2*margin >= np.diff(self.limits, axis=1)):
            raise ValueError("invalid native joint bounds")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(model_path), sess_options=opts, providers=["CPUExecutionProvider"])
        self.inputs = {i.name: i.shape for i in self.session.get_inputs()}
        expected = {"memory": [1, 465], "proprioception": [1, 93], "quaternion": [1, 4], "target_state": [1, 71]}
        if self.inputs != expected:
            raise ValueError(f"unsupported factory contract: {self.inputs}")
        self.history = np.zeros((5, 93), np.float32)
        self.previous_actions = np.zeros(29, np.float32)
        self.initialized = False
        self.last_raw_target = self.default.copy()

    def step(self, q23, dq23, angular_velocity_body, gravity_body, quaternion_wxyz, target_state):
        q29, dq29 = np.zeros(29, np.float32), np.zeros(29, np.float32)
        q29[NATIVE23_TO_VENDOR29] = q23
        dq29[NATIVE23_TO_VENDOR29] = dq23
        # Factory UpdateRlState uses absolute q, unscaled dq/gyro/gravity and
        # prior raw actor output. Normalization lives inside the ONNX graph.
        proprio = np.concatenate((angular_velocity_body, gravity_body, q29, dq29, self.previous_actions)).astype(np.float32)
        quaternion = normalized_quaternion(quaternion_wxyz)[[1, 2, 3, 0]].astype(np.float32)
        target = np.asarray(target_state, np.float32)
        if target.shape != (71,) or not np.isfinite(target).all() or not np.isfinite(proprio).all():
            raise ValueError("invalid factory observation")
        if not self.initialized:
            self.history[:] = proprio
            self.initialized = True
        else:
            self.history[:-1] = self.history[1:]
            self.history[-1] = proprio
        inputs = {"memory": self.history.reshape(1, -1), "proprioception": proprio[None],
                  "quaternion": quaternion[None], "target_state": target[None]}
        actions = self.session.run(["actor_actions"], inputs)[0][0]
        if actions.shape != (29,) or not np.isfinite(actions).all():
            raise ValueError("invalid factory action")
        self.previous_actions[:] = actions
        # The absent motors cannot execute policy commands. Their previous
        # command observations must reflect their fixed zero target.
        self.previous_actions[ABSENT_JOINTS] = -self.default29[ABSENT_JOINTS] / self.action_scale
        target29 = self.default29 + actions * self.action_scale * self.output_scale
        self.last_raw_target = target29[NATIVE23_TO_VENDOR29].copy()
        return np.clip(self.last_raw_target, self.limits[:, 0]+self.margin, self.limits[:, 1]-self.margin)

    def commit_applied(self, target23):
        """Commit the command actually applied at a control boundary."""
        target = np.asarray(target23, np.float32)
        if target.shape != (23,) or not np.isfinite(target).all():
            raise ValueError("invalid applied command")
        scale = self.action_scale * self.output_scale[NATIVE23_TO_VENDOR29]
        if np.any(scale == 0):
            raise ValueError("zero native action scale")
        self.previous_actions[NATIVE23_TO_VENDOR29] = (target-self.default)/scale
