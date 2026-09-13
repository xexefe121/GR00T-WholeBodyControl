"""Exact finite-grid collision queries through the declared pose interpolator.

Every control knot and every original source timestamp is queried. Joint-space
interpolation derivatives map each query back to its two adjacent control knots.
Rigid global root motion does not change robot-robot distances. This checks a
finite grid, not continuous swept collision or a dynamics trajectory.
"""

import numpy as np
from scipy import sparse
from scipy.spatial.transform import Rotation, Slerp


def interpolation_weights(control_times, requested_times):
    times, requested = np.asarray(control_times, dtype=float), np.asarray(requested_times, dtype=float)
    if (
        times.ndim != 1
        or requested.ndim != 1
        or len(times) < 3
        or not len(requested)
        or not np.isfinite(times).all()
        or not np.isfinite(requested).all()
        or np.any(np.diff(times) <= 0)
        or np.any(np.diff(requested) <= 0)
    ):
        raise ValueError("collision interpolation requires finite increasing timestamps")
    if requested[0] != times[0] or requested[-1] != times[-1]:
        raise ValueError("original collision grid must retain both source endpoints")
    left = np.clip(np.searchsorted(times, requested, side="right") - 1, 0, len(times) - 2)
    fraction = (requested - times[left]) / (times[left + 1] - times[left])
    rows = np.repeat(np.arange(len(requested)), 2)
    columns = np.column_stack((left, left + 1)).ravel()
    weights = np.column_stack((1 - fraction, fraction)).ravel()
    result = sparse.csc_matrix((weights, (rows, columns)), shape=(len(requested), len(times)))
    result.eliminate_zeros()
    return result


def interpolate_original_poses(poses, control_source_times, original_times):
    values = np.asarray(poses, dtype=float)
    weights = interpolation_weights(control_source_times, original_times)
    if values.shape != (weights.shape[1], 30) or not np.isfinite(values).all():
        raise ValueError("original collision grid requires finite matching native23 poses and times")
    if np.any(np.abs(np.linalg.norm(values[:, 3:7], axis=1) - 1) > 1e-5):
        raise ValueError("collision interpolation requires unit root quaternions")
    root, joints = weights @ values[:, :3], weights @ values[:, 7:]
    quat = Slerp(control_source_times, Rotation.from_quat(values[:, [4, 5, 6, 3]]))(original_times).as_quat()[
        :, [3, 0, 1, 2]
    ]
    return np.column_stack((root, quat, joints))


class CollisionPathSampler:
    def __init__(self, control_times, original_times):
        # Validate the original grid separately; a union must never hide an
        # omitted endpoint or accept an out-of-order source timeline.
        interpolation_weights(control_times, original_times)
        self.control_times = np.array(control_times, dtype=float, copy=True)
        self.original_times = np.array(original_times, dtype=float, copy=True)
        self.query_times = np.unique(np.r_[self.control_times, self.original_times])
        weights = interpolation_weights(self.control_times, self.query_times)
        # OriginalTaskPath has six root-reference variables then 23 joints.
        self.variable_selector = sparse.kron(
            weights, sparse.hstack((sparse.csc_matrix((23, 6)), sparse.eye(23))), format="csc"
        )

    def path_rows(self, query, poses, joint_dof_addresses):
        interpolated = interpolate_original_poses(poses, self.control_times, self.query_times)
        jacobian, distances, identities = query.path_rows(interpolated, joint_dof_addresses)
        return jacobian @ self.variable_selector, distances, identities
