"""Offline native124 comparator, NOT a SONIC implementation or live controller.

The selected native23 actor is evaluated on received-only references with the
same native plant, target envelope and range preview used by the SONIC trials.
Only copied state reaches this adapter. Its explicit 994-value diagnostic stub
exists to reuse the old physics referee; it is never an actual SONIC activation.
"""

from pathlib import Path

import mujoco
import numpy as np

from gear_sonic.utils.g1_23dof_contract import MUJOCO_TO_ISAACLAB_DOF
from gear_sonic.utils.g1_23dof_native124_21204_adapter import (
    build_checkpoint21204_observation,
    checkpoint21204_raw_action_to_hardware_targets,
    hardware_targets_to_checkpoint21204_raw_action,
)
from gear_sonic.utils.g1_23dof_safe_target_transform import (
    SAFE_TARGET_DEFAULT_Q_HARDWARE,
    safe_target_transform_numpy,
)
from gear_sonic.utils.g1_true23_generalist_benchmark import MODEL, PHYSICS
from gear_sonic.utils.g1_true23_range_preview import Native23RangePreview
from gear_sonic.utils.g1_true23_source_action_codec import (
    SOURCE_SCALE_NATIVE_IL23,
    source_scaled_precompensation,
)
from gear_sonic.utils.g1_true23_step1b_mujoco import _quaternion_matrix
from gear_sonic.utils.g1_true23_sonic_library_replay import validate_library_motion

PHASES = ("causal_q9", "current_q10")


def target_to_existing_envelope(target):
    """Preserve target units; reuse, do not widen, SONIC's reachable projection."""
    target = np.asarray(target)
    if target.shape != (23,) or target.dtype != np.float32 or not np.isfinite(target).all():
        raise ValueError("native124 target must be finite float32 hardware23")
    delta = target.astype(np.float64) - np.asarray(SAFE_TARGET_DEFAULT_Q_HARDWARE)
    equivalent = (delta[list(MUJOCO_TO_ISAACLAB_DOF)] / np.asarray(SOURCE_SCALE_NATIVE_IL23)).astype(np.float32)
    inverse, projection = source_scaled_precompensation(equivalent)
    return inverse, projection


def relative_rotation6(robot, reference):
    """Stock row-major first two columns of R_robot.T @ R_reference."""
    for q in (robot, reference):
        if np.shape(q) != (4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q) - 1) > 2e-5:
            raise ValueError("torso anchor must be a unit WXYZ quaternion")
    return (_quaternion_matrix(robot).T @ _quaternion_matrix(reference))[:, :2].reshape(6).astype(np.float32)


class Native124PicoComparator:
    """Native124-only diagnostic; no action labels admitted or transport imports."""

    def __init__(self, motion, *, phase, root, assets):
        if phase not in PHASES:
            raise ValueError("unknown native124 reference phase")
        validate_library_motion(motion)
        self.motion = {key: value.copy() for key, value in motion.items()}
        self.phase = phase
        self.preview = Native23RangePreview(model_path=Path(assets) / MODEL, physics_path=Path(root) / PHYSICS)
        self.geometry = mujoco.MjModel.from_xml_path(str(Path(assets) / MODEL))
        self.probe = mujoco.MjData(self.geometry)
        self.torso = self.geometry.body("torso_link").id
        if self.torso != 14 or (self.geometry.nq, self.geometry.nv, self.geometry.nu) != (30, 29, 23):
            raise ValueError("native124 comparator requires exact native23 topology")
        self.previous_qpos = np.r_[motion["body_pos_w"][9, 0], motion["body_quat_w"][9, 0], motion["joint_pos"][9]]
        # The referee initializes its previous-safe action to zero: this is the
        # associated requested target, NOT selected checkpoint's home posture.
        self.previous_target = np.asarray(SAFE_TARGET_DEFAULT_Q_HARDWARE, dtype=np.float32)
        self.next_control = 0
        self.failed = False
        self.attempts = []

    def infer(self, policy, encoder, history, *, control_index, measured_qpos, measured_qvel, **unused):
        del encoder, unused
        if self.failed or control_index != self.next_control:
            raise ValueError("native124 comparator rejects restart, skipped or repeated control")
        if history.shape != (930,) or history.dtype != np.float32 or not np.isfinite(history).all():
            raise ValueError("referee history must be finite float32 history930")
        if measured_qpos.shape != (30,) or measured_qvel.shape != (29,):
            raise ValueError("native124 comparator requires copied native23 state")
        if not np.isfinite(measured_qpos).all() or not np.isfinite(measured_qvel).all():
            raise ValueError("nonfinite measured state")
        q9, q10 = control_index + 9, control_index + 10
        if q10 >= len(self.motion["joint_pos"]):
            raise ValueError("source exhausted; comparator cannot pad or loop")
        anchor = q9 if self.phase == "causal_q9" else q10
        torso_pose = self.previous_qpos if self.phase == "causal_q9" else measured_qpos
        self.probe.qpos[:] = torso_pose
        mujoco.mj_kinematics(self.geometry, self.probe)
        robot_torso = self.probe.xquat[self.torso].copy()
        reference_torso = self.motion["body_quat_w"][anchor, self.torso - 1].copy()
        qd = (
            self.motion["joint_pos"][q10].astype(np.float32) - self.motion["joint_pos"][q9].astype(np.float32)
        ) / np.float32(0.02)
        observation = build_checkpoint21204_observation(
            q_ref_hardware=self.motion["joint_pos"][anchor],
            qd_ref_hardware=qd,
            torso_motion_anchor_ori_b=relative_rotation6(robot_torso, reference_torso),
            base_angular_velocity=history[27:30],
            q_measured_hardware=measured_qpos[7:],
            qd_measured_hardware=measured_qvel[6:],
            previous_applied_raw_action_hardware=hardware_targets_to_checkpoint21204_raw_action(
                self.previous_target
            ),
        )
        row = dict(
            observation124=observation[0].copy(),
            measured_qpos=measured_qpos.copy(),
            measured_qvel=measured_qvel.copy(),
            robot_torso_wxyz=robot_torso,
            reference_torso_wxyz=reference_torso,
            previous_target23=self.previous_target.copy(),
            source_indices=np.array([q9, q10, anchor]),
            source_times_s=np.array([q9, q10, anchor]) * 0.02,
        )
        self.attempts.append(row)
        try:
            action = policy.run(observation)
            row["selected_raw_hw23"] = action.copy()
            if action.shape != (23,) or action.dtype != np.float32 or not np.isfinite(action).all():
                raise ValueError("selected native124 actor ABI mismatch")
            if np.max(np.abs(action)) >= 10:
                raise ValueError("selected native124 action exceeds existing raw bound")
            proposed = checkpoint21204_raw_action_to_hardware_targets(action)
            row["proposed_target23"] = proposed.copy()
            inverse, projection = target_to_existing_envelope(proposed)
            row.update(inverse23=inverse.copy(), projection_native23=projection.copy())
            accepted = self.preview.filter(inverse, measured_qpos, measured_qvel)
            _, target = safe_target_transform_numpy(accepted)
            row.update(accepted23=accepted.copy(), applied_target23=target.copy())
            self.previous_target = target.copy()
            self.previous_qpos = measured_qpos.copy()
            self.next_control += 1
            # This is a named referee compatibility stub, never SONIC inference.
            stub = np.r_[np.zeros(64, np.float32), history].astype(np.float32)
            return accepted, stub
        except Exception:
            self.failed = True
            raise

    def external_force_world(self, step):
        return np.zeros(3)

    def arrays(self):
        keys = set().union(*(row.keys() for row in self.attempts))
        return {key: np.asarray([row[key] for row in self.attempts if key in row]) for key in sorted(keys)}

    def contract(self):
        return dict(
            kind="native124_selected_existing_pico_comparator_v1",
            policy_is_sonic=False,
            observation="hardware124_checkpoint21204",
            reference_phase=self.phase,
            current_q10_is_distinct_from_pinned_causal_manifest=self.phase == "current_q10",
            source_velocity="float32(q10-q9)/0.02_received_only",
            source_clock_latency_s=0.02 if self.phase == "causal_q9" else 0.0,
            oracle_future_samples_used=False,
            measured_pose_velocity_phase="current_q10",
            measured_encoder_bias_rad=0.0,
            previous_action="actual_projected_target_in_checkpoint21204_home_scale_coordinates",
            decoder994_is_non_neural_referee_stub=True,
            source29_missing_hand_torso_axes_reconstructed=False,
            joint_range_preview=self.preview.contract(),
            completed_proposals=self.next_control,
            failed=self.failed,
            teacher_label_admitted=False,
            simulator_qualified=False,
            deployment_ready=False,
            hardware_authorized=False,
        )
