"""Exact NumPy BFM observation contract, extracted without Torch dependencies."""
from __future__ import annotations
import numpy as np

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

def _quaternion_matrix(quaternion_wxyz: Sequence[float]) -> np.ndarray:
    # Quaternion observations must not normalize a view into simulator state
    # or an immutable reference in place.
    value = np.array(quaternion_wxyz, dtype=np.float64, copy=True)
    if value.shape != (4,) or not np.isfinite(value).all() or np.linalg.norm(value) == 0.0:
        raise ValueError("rotation requires a finite nonzero wxyz quaternion")
    value /= np.linalg.norm(value)
    w, x, y, z = value
    return np.asarray(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
            (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
            (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)),
        )
    )
