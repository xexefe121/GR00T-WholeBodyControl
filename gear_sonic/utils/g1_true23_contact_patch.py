"""Conservative local candidate-support preservation for offline QP trials.

MuJoCo stores a contact midpoint, not the robot's material surface point:
https://mujoco.readthedocs.io/en/3.5.0/APIreference/APItypes.html#mjcontact
These body-attached surface points constrain one local trial. They are not
closest-feature derivatives, guaranteed contact persistence, no-slip constraints
or a replacement for independently recomputed geometry and force feasibility.
"""

from __future__ import annotations

import mujoco
import numpy as np

from gear_sonic.utils.g1_true23_contact_trajectory import VARIABLE_DOFS


class FrozenFloorSupportPatch:
    """Keep every current candidate surface point inside a stricter floor band.

    A QP may redistribute load to any of its candidate cone rays, so the patch
    retains every candidate point, not only points loaded by the previous seed.
    This is a conservative local hypothesis, not an immutable motion schedule.
    The source model/data are never modified.
    """

    def __init__(self, model, data, contacts, plane_id, *, gap_m=0.002, guard_m=0.00005):
        if (
            model.nq != 30
            or model.nv != 29
            or not np.isfinite([gap_m, guard_m]).all()
            or not 0 < guard_m < gap_m <= 0.01
            or not 0 <= plane_id < model.ngeom
            or model.geom_type[plane_id] != mujoco.mjtGeom.mjGEOM_PLANE
            or model.geom_bodyid[plane_id] != 0
            or not np.allclose(data.geom_xmat[plane_id].reshape(3, 3)[:, 2], [0, 0, 1], atol=1e-8, rtol=0)
        ):
            raise ValueError("support patch requires native23, a fixed upward floor and explicit guard < gap")
        self.model, self.data = model, mujoco.MjData(model)
        self.base_qpos = data.qpos.copy()
        self.ceiling = gap_m - guard_m
        self.floor_z = float(data.geom_xpos[plane_id, 2])
        self.points = []
        for contact in contacts:
            geom, distance = contact["robot_geom_id"], contact["distance_m"]
            midpoint = np.asarray(contact["position_w"], dtype=float)
            if (
                not isinstance(geom, (int, np.integer))
                or not 0 <= geom < model.ngeom
                or model.geom_bodyid[geom] == 0
                or midpoint.shape != (3,)
                or not np.isfinite(midpoint).all()
                or not np.isfinite(distance)
                or distance > gap_m + 1e-10
            ):
                raise ValueError("support patch needs finite current robot/floor candidate contacts")
            body = int(model.geom_bodyid[geom])
            if model.body(body).name != contact["body"]:
                raise ValueError("support patch contact body differs from its robot collider")
            # The normal points upward from the floor toward the robot.
            surface = midpoint + np.array([0.0, 0.0, 0.5 * distance])
            if abs(surface[2] - self.floor_z - distance) > 1e-7:
                raise ValueError("support patch midpoint/distance does not describe the fixed flat floor")
            local = data.xmat[body].reshape(3, 3).T @ (surface - data.xpos[body])
            self.points.append((body, local))

    def evaluate(self, qpos, *, jacobian=True):
        """Return nonnegative-inside band values and native26 local derivatives."""
        qpos = np.asarray(qpos, dtype=float)
        if qpos.shape != (30,) or not np.isfinite(qpos).all() or np.linalg.norm(qpos[3:7]) < 1e-12:
            raise ValueError("support patch needs a finite native23 pose with nonzero rotation")
        model, data = self.model, self.data
        data.qpos[:] = qpos
        mujoco.mj_kinematics(model, data)
        if jacobian:
            mujoco.mj_comPos(model, data)
        values = np.empty(len(self.points))
        derivative = np.empty((len(self.points), 26)) if jacobian else None
        for index, (body, local) in enumerate(self.points):
            surface = data.xpos[body] + data.xmat[body].reshape(3, 3) @ local
            values[index] = self.ceiling - (surface[2] - self.floor_z)
            if jacobian:
                jacp = np.zeros((3, model.nv))
                mujoco.mj_jac(model, data, jacp, None, surface, body)
                derivative[index] = -jacp[2, VARIABLE_DOFS]
        return values, derivative
