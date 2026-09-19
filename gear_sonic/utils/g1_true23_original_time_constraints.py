"""Original-source task norms inside the full control-path numerical solve.

Joint/translation interpolation is linear; root attitudes use the same SLERP as
the independent original-time audit. The analytic Jacobian propagates through
both root rotations and both adjacent joint knots. No acceptance gate changes.
"""

from types import SimpleNamespace

import mujoco
import numpy as np
from scipy import sparse
from scipy.spatial.transform import Rotation

from gear_sonic.utils import g1_23dof_task_space_retarget as ik
from gear_sonic.utils.g1_23dof_safe_target_transform import SAFE_TARGET_DEFAULT_Q_HARDWARE
from gear_sonic.utils.g1_true23_collision_sampling import interpolate_original_poses, interpolation_weights
from gear_sonic.utils.g1_true23_generalist_protected_root import residual_groups
from gear_sonic.utils.g1_true23_original_task_trajectory import (
    OriginalTaskPath,
    so3_left_jacobian,
    so3_left_jacobian_inverse,
)


def sampled_task_variables(
    control_problem, original_problem, variables, control_times, original_times, *, derivatives=True
):
    """Map control task coordinates to original-time task coordinates exactly."""
    weights = interpolation_weights(control_times, original_times)
    control_poses = control_problem.qpos(variables)
    poses = interpolate_original_poses(control_poses, control_times, original_times)
    sampled = original_problem.serialized_variables(
        {
            "joint_pos": poses[:, 7:],
            "body_pos_w": poses[:, None, :3],
            "body_quat_w": poses[:, None, 3:7],
        }
    )
    if not derivatives:
        return sampled, None
    mapping = sparse.kron(weights, sparse.diags(np.r_[np.ones(3), np.zeros(3), np.ones(23)]), format="csc")
    left = np.clip(np.searchsorted(control_times, original_times, side="right") - 1, 0, len(control_times) - 2)
    fraction = (np.asarray(original_times) - np.asarray(control_times)[left]) / (
        np.asarray(control_times)[left + 1] - np.asarray(control_times)[left]
    )
    rotations = Rotation.from_quat(control_poses[:, [4, 5, 6, 3]])
    rows, columns, values = [], [], []
    for frame, (lo, amount) in enumerate(zip(left, fraction, strict=True)):
        relative = rotations[lo + 1] * rotations[lo].inv()
        vector = relative.as_rotvec()
        advance = Rotation.from_rotvec(amount * vector).as_matrix()
        transfer = amount * so3_left_jacobian(amount * vector) @ so3_left_jacobian_inverse(vector)
        output_coordinates = so3_left_jacobian_inverse(sampled[frame, 3:6])
        for knot, angular_map in ((lo, advance - transfer @ relative.as_matrix()), (lo + 1, transfer)):
            block = output_coordinates @ angular_map @ so3_left_jacobian(variables[knot, 3:6])
            for i in range(3):
                for j in range(3):
                    rows.append(29 * frame + 3 + i)
                    columns.append(29 * knot + 3 + j)
                    values.append(block[i, j])
    rotation_map = sparse.csc_matrix((values, (rows, columns)), shape=mapping.shape)
    return sampled, mapping + rotation_map


def original_direct_baseline(source_model, target_model, source, config):
    """Recompute the SAME naive native23 baseline used by original-time audit."""
    source_layout, target_layout = ik._model_layout(source_model), ik._model_layout(target_model)
    safe = ik._safe_target_layout(target_layout, config.safe_limit_guard_rad, config.native_action_clip)
    mapping = [source_layout.joint_names.index(name) for name in target_layout.joint_names]
    direct = np.clip(source["joint_pos"][:, mapping], safe.lower, safe.upper)
    source_data, target_data = mujoco.MjData(source_model), mujoco.MjData(target_model)
    count = len(direct)
    diagnostics = {"weighted_task_error_before": np.empty(count)}
    for task in ik.DEFAULT_TASKS:
        diagnostics[f"task_{task.name}_position_error_before_m"] = np.empty(count)
        diagnostics[f"task_{task.name}_orientation_error_before_rad"] = np.empty(count)
    for frame in range(count):
        root, quat = source["root_pos_w"][frame], source["root_quat_wxyz"][frame]
        ik._set_configuration(source_model, source_data, source_layout, root, quat, source["joint_pos"][frame])
        targets = ik._task_targets(source_model, source_data, ik.DEFAULT_TASKS)
        ik._set_configuration(target_model, target_data, target_layout, root, quat, direct[frame])
        jac, residual, position, orientation, _, _ = ik._task_linearization(
            target_model,
            target_data,
            target_layout,
            ik.DEFAULT_TASKS,
            targets,
            tuple(bool(x) for x in source["contact_flags"][frame]),
            config.contact_weight_multiplier,
            np.arange(23),
        )
        diagnostics["weighted_task_error_before"][frame] = ik._weighted_task_error(jac, residual)
        for index, task in enumerate(ik.DEFAULT_TASKS):
            diagnostics[f"task_{task.name}_position_error_before_m"][frame] = position[index]
            diagnostics[f"task_{task.name}_orientation_error_before_rad"][frame] = orientation[index]
    return SimpleNamespace(config=config, diagnostics=diagnostics, contact_flags=source["contact_flags"])


class OriginalTimeProtectedTasks:
    def __init__(self, control_problem, source_model, target_model, source, control_times, config):
        self.control_problem = control_problem
        self.control_times = np.asarray(control_times).copy()
        self.original_times = source["timestamps_s"].copy()
        interpolation_weights(self.control_times, self.original_times)
        requested = np.column_stack((source["root_pos_w"], source["root_quat_wxyz"], source["joint_pos"]))
        # Constant seed only initializes the existing FK/Jacobian interface.
        # It is never a motion, prior, objective, or original-time rate audit.
        seed = np.repeat(np.asarray(SAFE_TARGET_DEFAULT_Q_HARDWARE)[None], len(requested), axis=0)
        self.problem = OriginalTaskPath(source_model, target_model, requested, seed, config=control_problem.config)
        baseline = original_direct_baseline(source_model, target_model, source, config)
        groups, self.width = residual_groups(self.problem, baseline)
        self.groups = [{**group, "name": "original_time_" + group["name"]} for group in groups]

    def protect_upper_landmarks(self, accepted_reference_variables, maximum_position_error_m):
        from gear_sonic.utils.g1_true23_upper_landmark_constraints import upper_landmark_groups

        sampled, _ = sampled_task_variables(
            self.control_problem,
            self.problem,
            accepted_reference_variables,
            self.control_times,
            self.original_times,
            derivatives=False,
        )
        groups, report = upper_landmark_groups(self.problem, sampled, maximum_position_error_m)
        self.groups.extend({**group, "name": "original_time_" + group["name"]} for group in groups)
        return report

    def evaluate(self, variables, *, derivatives=True):
        sampled, mapping = sampled_task_variables(
            self.control_problem,
            self.problem,
            variables,
            self.control_times,
            self.original_times,
            derivatives=derivatives,
        )
        residual, jacobian, _ = self.problem.evaluate(sampled, derivatives=derivatives)
        return residual, jacobian @ mapping if derivatives else None
