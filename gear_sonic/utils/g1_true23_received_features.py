"""Native23 reference/state features using only present and past samples.

The actor sees the same 1323-wide layout as the earlier full-range student,
but its eight reference slots are causal. Derivatives are backward differences.
No reference identifier, clip phase, future pose, or prepared feedback gain is
an actor input. Simulation root pose/velocity remain privileged in this pilot.
"""
import numpy as np
from scipy.spatial.transform import Rotation

OFFSETS = np.array([0, -1, -2, -4, -8, -16, -24, -37])
HISTORY_WIDTHS = (23, 3, 23, 23, 3)


def rotations(quat):
    quat = np.asarray(quat)
    return Rotation.from_quat(quat.reshape(-1, 4)[:, [1, 2, 3, 0]]).as_matrix().reshape(*quat.shape[:-1], 3, 3)


def backward_velocity(values):
    values = np.asarray(values)
    return np.diff(values, axis=0, prepend=values[:1]) / .02


def backward_omega(quat):
    quat = np.asarray(quat)
    flat = quat.reshape(len(quat), -1, 4)
    out = np.zeros((*flat.shape[:2], 3))
    for k in range(flat.shape[1]):
        r = Rotation.from_quat(flat[:, k, [1, 2, 3, 0]])
        out[1:, k] = (r[1:] * r[:-1].inv()).as_rotvec() / .02
    return out.reshape(*quat.shape[:-1], 3)


def prepare_reference(motion, original, contract):
    feet = [contract['body_names'].index(s + '_ankle_roll_link') for s in ('left', 'right')]
    # All quantities at frame t depend only on frames <= t.
    return dict(
        joint=motion['joint_pos'].copy(), joint_velocity=backward_velocity(motion['joint_pos']),
        root=motion['body_pos_w'][:, 0].copy(), root_rotation=rotations(motion['body_quat_w'][:, 0]),
        root_velocity=backward_velocity(motion['body_pos_w'][:, 0]),
        root_omega=backward_omega(motion['body_quat_w'][:, 0]),
        feet=motion['body_pos_w'][:, feet].copy(), feet_velocity=backward_velocity(motion['body_pos_w'][:, feet]),
        tasks=original['source_task_position_w'].copy(),
        task_rotation=rotations(original['source_task_quaternion_wxyz']),
        task_velocity=backward_velocity(original['source_task_position_w']),
        task_omega=backward_omega(original['source_task_quaternion_wxyz']))


def features_numpy(qpos, qvel, reference, frame, default, prior, history):
    frame = int(frame)
    if frame < 0 or frame >= len(reference['joint']):
        raise ValueError('received reference frame out of range')
    slots = np.maximum(frame + OFFSETS, 0)
    rot = rotations(np.asarray(qpos[3:7]))
    yaw = np.arctan2(rot[1, 0], rot[0, 0])
    c, s = np.cos(yaw), np.sin(yaw)
    heading = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    root = qpos[:3]
    proprio = np.r_[qpos[7:] - default, qvel[6:], qvel[3:6], rot.T @ [0., 0., -1.], qvel[:3] @ heading, qpos[2]]
    r = {k:v[slots] for k,v in reference.items()}
    goal = np.concatenate((r['joint'], r['joint_velocity'], (r['root'] - root) @ heading,
        np.einsum('ij,njk->nik', heading.T, r['root_rotation'])[:, :, :2].reshape(8, 6),
        r['root_velocity'] @ heading, r['root_omega'] @ heading,
        ((r['feet'] - root) @ heading).reshape(8, 6), (r['feet_velocity'] @ heading).reshape(8, 6),
        ((r['tasks'] - root) @ heading).reshape(8, 9),
        np.einsum('ij,ntjk->ntik', heading.T, r['task_rotation'])[:, :, :, :2].reshape(8, 18),
        (r['task_velocity'] @ heading).reshape(8, 9), (r['task_omega'] @ heading).reshape(8, 9)), axis=-1)
    result = np.concatenate((proprio, goal.ravel(), prior, history)).astype(np.float32)
    if result.shape != (1323,) or not np.isfinite(result).all():
        raise ValueError('invalid received native23 features')
    return result


class MeasuredHistory:
    def __init__(self, contract):
        self.contract = contract
        self.prior = np.zeros(23, np.float32)
        self.terms = [np.zeros((4, width), np.float32) for width in HISTORY_WIDTHS]

    def vector(self):
        return np.concatenate([x.ravel() for x in self.terms])

    def commit(self, qpos, qvel, target):
        c = self.contract
        values = (self.prior.copy(), qvel[3:6] * .25, qpos[7:] - c['default_q'], qvel[6:], rotations(qpos[3:7]).T @ [0., 0., -1.])
        for bank, value in zip(self.terms, values):
            bank[1:] = bank[:-1].copy()
            bank[0] = value
        self.prior = ((target - c['default_q']) * c['kp'] / (.25 * c['training_effort'])).astype(np.float32)
