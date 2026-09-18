"""Minimal, path-scoped native23 model helpers for the isolated Orin benchmark."""
import json
from pathlib import Path
import mujoco
import numpy as np

def _quaternion_matrix(quaternion_wxyz):
    value = np.array(quaternion_wxyz, dtype=np.float64, copy=True)
    if value.shape != (4,) or not np.isfinite(value).all() or np.linalg.norm(value) == 0.0:
        raise ValueError("rotation requires a finite nonzero wxyz quaternion")
    value /= np.linalg.norm(value)
    w, x, y, z = value
    return np.asarray(((1 - 2 * (y*y + z*z), 2 * (x*y - w*z), 2 * (x*z + w*y)),
                       (2 * (x*y + w*z), 1 - 2 * (x*x + z*z), 2 * (y*z - w*x)),
                       (2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y))))

def _quaternion_multiply(left, right):
    w1, x1, y1, z1 = np.asarray(left, dtype=np.float64)
    w2, x2, y2, z2 = np.asarray(right, dtype=np.float64)
    return np.asarray((w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
                       w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2))

def prepare_true23_model(path, config_path):
    model = mujoco.MjModel.from_xml_path(str(Path(path)))
    if (model.nq, model.nv, model.nu) != (30, 29, 23):
        raise ValueError("native23 dimensions differ")
    cfg = json.loads(Path(config_path).read_text())
    physics = cfg["physics"]
    joint_ids = np.arange(1, model.njnt, dtype=np.int32)
    dofs = model.jnt_dofadr[joint_ids]
    model.dof_armature[dofs] = np.asarray(physics["armature_hardware"], dtype=np.float64)
    model.dof_damping[dofs] = np.asarray(physics["joint_damping_hardware"], dtype=np.float64)
    model.dof_frictionloss[dofs] = np.asarray(physics["joint_frictionloss_hardware"], dtype=np.float64)
    model.jnt_actfrclimited[joint_ids] = 1
    effort = np.asarray(physics["effort_limit_hardware_nm"], dtype=np.float64)
    model.jnt_actfrcrange[joint_ids, 0] = -effort
    model.jnt_actfrcrange[joint_ids, 1] = effort
    mujoco.mj_setConst(model, mujoco.MjData(model))
    return mujoco, model, physics
