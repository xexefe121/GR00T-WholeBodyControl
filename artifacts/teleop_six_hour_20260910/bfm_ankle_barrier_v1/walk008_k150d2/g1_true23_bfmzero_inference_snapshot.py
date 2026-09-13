"""Simulation-only inference for the public, native23 BFM-Zero checkpoint.

Architecture and observation contract follow LeCAR-Lab/BFM-Zero at
318cf44a3262e5bdec5944f82f1a5f509b95d09b. Weights are a separately licensed
community checkpoint. No training object, pickle, DDS or robot API is loaded.
This module does not establish simulator or hardware qualification.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from safetensors.torch import load_file

from gear_sonic.utils.g1_23dof_contract import HARDWARE_23_JOINT_NAMES
from gear_sonic.utils.g1_true23_step1b_mujoco import _quaternion_matrix

WEIGHTS_SHA = "28a2d82a2975c37b8a1533f2e3b0224d27f8a0813283640ee7eeee348fe42c2e"
CONFIG_SHA = "f3ba93872527acba52aa567187f225aa1d2825fef82f4a8497807a654153978d"


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_contract(path):
    if _sha(path) != CONFIG_SHA:
        raise ValueError("BFM-Zero native23 configuration hash mismatch")
    cfg = yaml.safe_load(Path(path).read_text())
    robot = cfg["robot"]
    if tuple(robot["dof_names"]) != tuple(HARDWARE_23_JOINT_NAMES):
        raise ValueError("public checkpoint is not the exact native23 hardware joint order")
    if robot["actions_dim"] != 23 or robot["motion"]["nums_extend_bodies"] != 1:
        raise ValueError("unexpected public BFM-Zero topology")
    if robot["motion"]["extend_config"] != [dict(
        joint_name="head_link", parent_name="torso_link", pos=[0.0, 0.0, 0.35], rot=[1.0, 0.0, 0.0, 0.0]
    )]:
        raise ValueError("unexpected virtual head definition")

    def values(mapping):
        result = []
        for name in HARDWARE_23_JOINT_NAMES:
            matches = [value for key, value in mapping.items() if key in name]
            if len(matches) != 1:
                raise ValueError("ambiguous gain mapping for " + name)
            result.append(matches[0])
        return np.asarray(result, dtype=np.float64)

    control = robot["control"]
    if (control["control_type"], control["action_scale"], control["normalize_action_from"],
            control["normalize_action_to"], control["action_rescale"]) != ("P", 0.25, 1.0, 5.0, True):
        raise ValueError("unsupported BFM-Zero action contract")
    return dict(
        config=cfg,
        default_q=np.asarray([robot["init_state"]["default_joint_angles"][n] for n in HARDWARE_23_JOINT_NAMES]),
        kp=values(control["stiffness"]), kd=values(control["damping"]),
        training_effort=np.asarray(robot["dof_effort_limit_list"], dtype=np.float64),
        body_names=tuple(robot["body_names"]),
    )


class BFMZeroInference:
    """Strict functional reconstruction of the released inference-only tensors."""

    def __init__(self, weights):
        if _sha(weights) != WEIGHTS_SHA:
            raise ValueError("BFM-Zero inference weights hash mismatch")
        self.weights = load_file(str(weights), device="cpu")
        if len(self.weights) != 54 or self.weights["_actor.policy.4.mlp.1.weight"].shape != (23, 2048):
            raise ValueError("unexpected inference tensor schema")
        for tensor in self.weights.values():
            if tensor.is_floating_point() and not torch.isfinite(tensor).all():
                raise ValueError("nonfinite checkpoint tensor")

    def normalize(self, key, x):
        prefix = f"_obs_normalizer._normalizers.{key}._normalizer."
        return F.batch_norm(x, self.weights[prefix + "running_mean"], self.weights[prefix + "running_var"],
                            training=False, eps=1e-5)

    def linear(self, key, x):
        return F.linear(x, self.weights[key + ".weight"], self.weights[key + ".bias"])

    def layernorm(self, key, x):
        return F.layer_norm(x, (x.shape[-1],), self.weights[key + ".weight"],
                            self.weights[key + ".bias"], eps=1e-5)

    def block(self, prefix, x, *, activation=True):
        result = self.linear(prefix + ".mlp.1", self.layernorm(prefix + ".mlp.0", x))
        return F.mish(result) if activation else result

    @torch.inference_mode()
    def backward(self, state, privileged):
        x = torch.cat((self.normalize("state", state), self.normalize("privileged_state", privileged)), dim=-1)
        if x.shape[-1] != 425:
            raise ValueError("BFM goal requires state52 and privileged373")
        x = self.linear("_backward_map.net.0", x)
        x = torch.tanh(self.layernorm("_backward_map.net.1", x))
        x = self.linear("_backward_map.net.3", x)
        return 16.0 * F.normalize(x, dim=-1)

    @torch.inference_mode()
    def actor(self, state, last_action, history, z):
        x = torch.cat((self.normalize("state", state), self.normalize("last_action", last_action),
                       self.normalize("history_actor", history)), dim=-1)
        if x.shape[-1] != 375 or z.shape != (*x.shape[:-1], 256):
            raise ValueError("BFM actor requires375 plus goal256")
        zs = self.block("_actor.embed_z.1", self.block("_actor.embed_z.0", torch.cat((x, z), dim=-1)))
        ss = self.block("_actor.embed_s.1", self.block("_actor.embed_s.0", x))
        x = torch.cat((ss, zs), dim=-1)
        for i in range(4):
            x = x + self.block(f"_actor.policy.{i}", x)
        result = torch.tanh(self.block("_actor.policy.4", x, activation=False))
        if not torch.isfinite(result).all():
            raise ValueError("BFM actor emitted nonfinite output")
        return result


def state_and_terms(q, dq, root_quat, root_angular_body, last_action, default_q):
    """Actual sensor contract: gyro scaled .25; last action in post-rescale units."""
    relative = np.asarray(q) - default_q
    gyro = np.asarray(root_angular_body) * 0.25
    gravity = _quaternion_matrix(root_quat).T @ np.array([0.0, 0.0, -1.0])
    terms = dict(actions=np.asarray(last_action), base_ang_vel=gyro, dof_pos=relative,
                 dof_vel=np.asarray(dq), projected_gravity=gravity)
    state = np.r_[relative, dq, gravity, gyro].astype(np.float32)
    return state, {k: np.asarray(v, np.float32) for k, v in terms.items()}


class BFMHistory:
    """Four previous samples, newest-first within each alphabetically sorted term."""

    def __init__(self):
        self.data = {k: np.zeros((4, n), np.float32) for k, n in
                     dict(actions=23, base_ang_vel=3, dof_pos=23, dof_vel=23, projected_gravity=3).items()}

    def before_update(self, terms):
        value = np.concatenate([self.data[k].reshape(-1) for k in sorted(self.data)]).copy()
        for key in self.data:
            self.data[key][1:] = self.data[key][:-1].copy()
            self.data[key][0] = terms[key]
        return value


def reference_features(motion, contract):
    """Public backward-map input; original source reference, no simulated feedback.

    The published expert encoder uses unscaled WORLD root angular velocity in
    state52, unlike the online actor's scaled BODY gyro. Preserve and disclose
    that upstream convention instead of silently assuming they are identical.
    """
    pos = np.asarray(motion["body_pos_w"], dtype=np.float64)
    quat = np.asarray(motion["body_quat_w"], dtype=np.float64)
    vel = np.asarray(motion["body_lin_vel_w"], dtype=np.float64)
    ang = np.asarray(motion["body_ang_vel_w"], dtype=np.float64)
    if pos.shape[1:] != (24, 3) or quat.shape != (*pos.shape[:2], 4):
        raise ValueError("BFM reference requires exact native24 rigid bodies")
    n = len(pos)
    rotations = np.asarray([[_quaternion_matrix(q) for q in row] for row in quat])
    torso = contract["body_names"].index("torso_link")
    offset = rotations[:, torso] @ np.array([0.0, 0.0, 0.35])
    pos = np.concatenate((pos, (pos[:, torso] + offset)[:, None]), axis=1)
    rotations = np.concatenate((rotations, rotations[:, torso:torso + 1]), axis=1)
    vel = np.concatenate((vel, (vel[:, torso] + np.cross(ang[:, torso], offset))[:, None]), axis=1)
    ang = np.concatenate((ang, ang[:, torso:torso + 1]), axis=1)
    yaw = np.arctan2(rotations[:, 0, 1, 0], rotations[:, 0, 0, 0])
    heading = np.zeros((n, 3, 3))
    heading[:, 0, 0] = heading[:, 1, 1] = np.cos(yaw)
    heading[:, 1, 0] = np.sin(yaw)
    heading[:, 0, 1] = -np.sin(yaw)
    heading[:, 2, 2] = 1.0
    local_pos = (pos - pos[:, :1]) @ heading
    local_rot = heading.transpose(0, 2, 1)[:, None] @ rotations
    # BFM tangent/normal uses X and Z axes, not SONIC's first-two-column6D.
    rotation6 = np.concatenate((local_rot[..., 0], local_rot[..., 2]), axis=-1)
    privileged = np.concatenate((pos[:, 0, 2:3], local_pos[:, 1:].reshape(n, -1),
                                 rotation6.reshape(n, -1), (vel @ heading).reshape(n, -1),
                                 (ang @ heading).reshape(n, -1)), axis=-1).astype(np.float32)
    gravity = np.einsum("nji,j->ni", rotations[:, 0], np.array([0.0, 0.0, -1.0]))
    state = np.concatenate((motion["joint_pos"] - contract["default_q"], motion["joint_vel"],
                            gravity, ang[:, 0]), axis=-1).astype(np.float32)
    if state.shape != (n, 52) or privileged.shape != (n, 373):
        raise ValueError("BFM reference feature shape mismatch")
    if not np.isfinite(state).all() or not np.isfinite(privileged).all():
        raise ValueError("nonfinite BFM reference features")
    return state, privileged


def identity():
    return json.loads(json.dumps(dict(
        kind="public_native23_bfmzero_inference_candidate_v1", weights_sha256=WEIGHTS_SHA,
        source_revision="32e70627dfbb957ad9bddf0b81456601811b7e73", actions=23,
        sonic_checkpoint_parity=False, simulator_qualified=False, deployment_ready=False,
        hardware_authorized=False,
    )))
