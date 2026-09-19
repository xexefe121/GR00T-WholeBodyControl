"""Offline signed self-contact geometry and native-joint linearization.

The physical model is never edited or integrated. A separate query model has
larger contact margins solely to expose nearby pairs using MuJoCo's own contact
filters. Distances are surface distances, not margin-adjusted distances. Every
accepted path must still be audited using the unchanged physical model.
"""

from __future__ import annotations

from copy import copy

import mujoco
import numpy as np
from scipy import sparse


def self_contact_rows(model, data):
    """One deepest contact per robot-robot geom pair, preserving signed distance."""
    rows = {}
    for contact in data.contact[: data.ncon]:
        first, second = int(contact.geom1), int(contact.geom2)
        if first < 0 or second < 0:
            raise ValueError("flex contacts are outside the rigid native23 diagnostic")
        if model.geom_bodyid[first] == 0 or model.geom_bodyid[second] == 0:
            continue
        pair = tuple(sorted((first, second)))
        if pair in rows and rows[pair]["distance_m"] <= contact.dist:
            continue
        normal = np.array(contact.frame[:3], dtype=float, copy=True)
        if first > second:
            normal *= -1
        rows[pair] = {
            "geoms": pair,
            "distance_m": float(contact.dist),
            "position": np.array(contact.pos, dtype=float, copy=True),
            "normal_first_to_second": normal,
        }
    return [rows[pair] for pair in sorted(rows)]


class SelfCollisionLinearizer:
    def __init__(self, model, *, near_distance_m=0.03):
        if not np.isfinite(near_distance_m) or not 0 < near_distance_m <= 0.05:
            raise ValueError("near-contact query distance must be in (0, 0.05] m")
        self.physical_model = model
        self.model = copy(model)
        # Sum of two margins is the collision-query horizon. Do not use this
        # model for rollout, force inference, or final geometric acceptance.
        self.model.geom_margin[:] = np.maximum(self.model.geom_margin, near_distance_m / 2)
        self.data = mujoco.MjData(self.model)
        self.near_distance_m = float(near_distance_m)

    def pose_rows(self, qpos, joint_dof_addresses):
        pose = np.asarray(qpos, dtype=float)
        columns = np.asarray(joint_dof_addresses)
        if pose.shape != (self.model.nq,) or not np.isfinite(pose).all():
            raise ValueError("collision linearization requires one finite exact-model qpos")
        if (
            columns.ndim != 1
            or columns.dtype.kind not in "iu"
            or np.any(columns < 0)
            or np.any(columns >= self.model.nv)
        ):
            raise ValueError("collision Jacobian columns must be valid joint DOF addresses")
        self.data.qpos[:] = pose
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)
        rows = self_contact_rows(self.model, self.data)
        for row in rows:
            jacobians = []
            for geom in row["geoms"]:
                jp = np.zeros((3, self.model.nv))
                mujoco.mj_jac(self.model, self.data, jp, None, row["position"], int(self.model.geom_bodyid[geom]))
                jacobians.append(jp)
            row["joint_jacobian"] = row["normal_first_to_second"] @ (jacobians[1] - jacobians[0])[:, columns]
        return rows

    def path_rows(self, poses, joint_dof_addresses):
        rows, columns, values, distances, identities = [], [], [], [], []
        width = len(joint_dof_addresses)
        for frame, pose in enumerate(poses):
            for contact in self.pose_rows(pose, joint_dof_addresses):
                index = len(distances)
                gradient = contact["joint_jacobian"]
                nonzero = np.flatnonzero(np.abs(gradient) > 1e-14)
                rows.extend([index] * len(nonzero))
                columns.extend((frame * width + nonzero).tolist())
                values.extend(gradient[nonzero].tolist())
                distances.append(contact["distance_m"])
                identities.append((frame, *contact["geoms"]))
        matrix = sparse.csc_matrix((values, (rows, columns)), shape=(len(distances), len(poses) * width))
        return matrix, np.asarray(distances), identities
