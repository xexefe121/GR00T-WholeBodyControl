"""Declared offline native23 pose-tracking MPC using batched finite differences.

The default 15-knot preview is 300 ms. This is an offline expert experiment,
not the received-only 140 ms live-stream controller and not a robot interface.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from gear_sonic.utils.g1_true23_mjbatch_ilqr_core import EPS, Planner, quat_log, quat_mul


TRACKED = (
    "pelvis",
    "torso_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_roll_rubber_hand",
    "right_wrist_roll_rubber_hand",
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_native_bundle(directory, clip):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    assert sha256(directory / "native_prepared.xml") == manifest["portable_xml_sha256"]
    assert sha256(directory / "prepared_model_arrays.npz") == manifest["prepared_arrays_sha256"]
    for name, digest in manifest["meshes"].items():
        assert sha256(directory / "meshes" / name) == digest, name
    motion_path = directory / clip / "native_original.npz"
    assert sha256(motion_path) == manifest["cases"][clip]["native_original.npz"]
    model = mujoco.MjModel.from_xml_path(str(directory / "native_prepared.xml"))
    with np.load(directory / "prepared_model_arrays.npz", allow_pickle=False) as archive:
        for name in archive.files:
            getattr(model, name)[:] = archive[name]
        mujoco.mj_setConst(model, mujoco.MjData(model))
        for name in archive.files:
            np.testing.assert_array_equal(getattr(model, name), archive[name], err_msg=name)
    contract = json.loads((directory / "contract.json").read_text())
    assert (model.nq, model.nv, model.nu, model.nbody) == (30, 29, 23, 25)
    assert [model.joint(index).name for index in range(1, model.njnt)] == contract["joint_names"]
    assert [model.body(index).name for index in range(1, model.nbody)] == contract["body_names"]
    assert model.opt.timestep == contract["timestep"] == 0.002
    assert model.opt.integrator == mujoco.mjtIntegrator.mjINT_EULER
    assert contract["decimation"] == 10
    with np.load(motion_path, allow_pickle=False) as archive:
        motion = {name: archive[name].copy() for name in archive.files}
    assert motion["joint_pos"].shape[1] == 23 and motion["body_pos_w"].shape[1:] == (24, 3)
    assert float(np.asarray(motion["fps"]).reshape(-1)[0]) == 50.0
    timeline = json.loads((directory / clip / "timeline.json").read_text())
    return model, contract, motion, timeline, manifest


def motion_states(motion):
    qpos = np.concatenate((motion["body_pos_w"][:, 0], motion["body_quat_w"][:, 0], motion["joint_pos"]), axis=1)
    # World root angular velocity rotated to MuJoCo's root-local tangent axes.
    q = motion["body_quat_w"][:, 0]
    w = np.column_stack((np.zeros(len(q)), motion["body_ang_vel_w"][:, 0]))
    spin = quat_mul(quat_mul(q * [1, -1, -1, -1], w), q)[:, 1:]
    qvel = np.concatenate((motion["body_lin_vel_w"][:, 0], spin, motion["joint_vel"]), axis=1)
    return np.concatenate((qpos, qvel), axis=1)


def load_motion_override(path, bundle, clip, native, contract, base_motion, timeline, manifest):
    """Verify a portable retarget receipt and every timed native body pose.

    Derivative/FK checks follow the referee's load_case_motion contract without
    importing its Torch policy dependencies into the isolated planner runtime.
    """
    from scipy.spatial.transform import Rotation

    path = Path(path).resolve()
    receipt_path = path.parent / "portable_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    phase = next(item for item in timeline["phases"] if item["name"] == "source_motion")
    expected = dict(
        schema_version=1,
        clip=clip,
        reference_sha256=sha256(path),
        fps=50,
        original_native_reference_sha256=manifest["cases"][clip]["native_original.npz"],
        original29_reference_sha256=manifest["cases"][clip]["original29.npz"],
        native_model_sha256=manifest["prepared_source_model_sha256"],
        physics_contract_sha256=manifest["physics_sha256"],
        frame_count=len(base_motion["joint_pos"]),
        total_requested_controls=timeline["total_requested_controls"],
        source_requested_controls=phase["requested_controls"],
        source_phase=phase,
        timeline_sha256=sha256(Path(bundle) / clip / "timeline.json"),
    )
    for name, value in expected.items():
        if receipt.get(name) != value:
            raise ValueError("retarget portable receipt mismatch: " + name)
    flags = (
        "all_frames_checked",
        "finite_fields",
        "same_fields_shapes_as_original",
        "fps_50",
        "full_original_timeline",
        "source_phase_counts_unchanged",
        "unit_body_quaternions",
        "native_joint_bounds",
        "adjacent_joint_speed_limits",
        "joint_vel_matches_timed_positions",
        "body_lin_vel_matches_timed_positions",
        "body_ang_vel_matches_world_rotation_intervals",
        "native_body_fk_positions",
        "native_body_fk_rotations",
        "copied_bytes_match_source",
        "source_provenance_matches",
    )
    if any(receipt.get("validation", {}).get(name) is not True for name in flags):
        raise ValueError("retarget portable receipt is not fully validated")
    if sha256(path.parent / "report.json") != receipt["source_artifact_report_sha256"]:
        raise ValueError("retarget original report differs from portable receipt")
    with np.load(path, allow_pickle=False) as archive:
        candidate = {name: archive[name].copy() for name in archive.files}
    if set(candidate) != set(base_motion):
        raise ValueError("retarget fields differ from original")
    for name, original in base_motion.items():
        if candidate[name].shape != original.shape or not np.isfinite(candidate[name]).all():
            raise ValueError("retarget finite shape mismatch: " + name)
    np.testing.assert_array_equal(candidate["fps"], base_motion["fps"])
    if np.max(np.abs(np.linalg.norm(candidate["body_quat_w"], axis=-1) - 1)) > 1e-5:
        raise ValueError("retarget quaternions are not unit length")
    joint, limits, dt = candidate["joint_pos"], native.jnt_range[1:], 0.02
    if np.max(np.maximum(limits[:, 0] - joint, joint - limits[:, 1])) > 1e-8:
        raise ValueError("retarget violates native joint ranges")
    if np.max(np.abs(np.diff(joint, axis=0)) / dt / np.asarray(contract["native_velocity"])) > 1 + 1e-8:
        raise ValueError("retarget adjacent samples exceed native joint speed")
    for name, positions in (("joint_vel", joint), ("body_lin_vel_w", candidate["body_pos_w"])):
        np.testing.assert_allclose(candidate[name], np.gradient(positions, dt, axis=0), atol=1e-6, rtol=1e-5)
    count = len(joint)
    earlier, later = np.maximum(np.arange(count) - 1, 0), np.minimum(np.arange(count) + 1, count - 1)
    for body in range(24):
        rotations = Rotation.from_quat(candidate["body_quat_w"][:, body, [1, 2, 3, 0]])
        velocity = (rotations[later] * rotations[earlier].inv()).as_rotvec() / ((later - earlier) * dt)[:, None]
        np.testing.assert_allclose(candidate["body_ang_vel_w"][:, body], velocity, atol=1e-6, rtol=1e-5)
    data = mujoco.MjData(native)
    for frame in range(count):
        data.qpos[:] = np.r_[candidate["body_pos_w"][frame, 0], candidate["body_quat_w"][frame, 0], joint[frame]]
        mujoco.mj_kinematics(native, data)
        if np.max(np.abs(data.xpos[1:] - candidate["body_pos_w"][frame])) > 1e-7:
            raise ValueError(f"retarget FK position mismatch at frame {frame}")
        dot = np.sum(data.xquat[1:] * candidate["body_quat_w"][frame], axis=-1)
        if np.max(np.abs(np.abs(dot) - 1)) > 1e-7:
            raise ValueError(f"retarget FK rotation mismatch at frame {frame}")
    return candidate, dict(
        path=str(path),
        reference_sha256=expected["reference_sha256"],
        portable_receipt_sha256=sha256(receipt_path),
        base_native_reference_sha256=expected["original_native_reference_sha256"],
        original29_sha256=expected["original29_reference_sha256"],
    )


class Native23Tracker(Planner):
    """Joint-target iLQR with native body poses and explicit native joint penalties."""

    def __init__(self, servo_model, contract, motion, *, horizon=15, threads=4):
        super().__init__(servo_model, horizon, 10, num_threads=threads)
        self.contract, self.motion = contract, motion
        self.start = 10
        self.ids = [servo_model.body(name).id for name in TRACKED]
        source_ids = [contract["body_names"].index(name) for name in TRACKED]
        self.body_count = len(self.ids)
        self.states = motion_states(motion)
        self.reference = np.concatenate(
            (
                motion["body_pos_w"][:, source_ids].reshape(len(self.states), -1),
                motion["body_quat_w"][:, source_ids].reshape(len(self.states), -1),
                self.states,
            ),
            axis=1,
        )
        self.positions = self.batch.bind("xpos")
        self.rotations = self.batch.bind("xquat")
        self.line_positions = self.line.bind("xpos")
        self.line_rotations = self.line.bind("xquat")
        self.position_weights = np.array(
            [[200, 200, 1000], [100, 100, 100], [400, 400, 400], [400, 400, 400], [60, 60, 60], [60, 60, 60]],
            float,
        )
        self.rotation_weights = np.array([100, 50, 50, 50, 20, 20], float)
        self.velocity_weights = np.r_[[2, 2, 5], [1, 1, 1], np.full(23, 0.05)]
        self.joint_weight, self.limit_weight, self.control_weight = 2.0, 200.0, 0.1
        self.limit_margin = 0.01
        self.speed_weight = 5.0

    def window(self, reference_state_frame):
        self.start = reference_state_frame

    def target_reference(self, t):
        index = np.minimum(self.start + t + 1, len(self.states) - 1)
        value = (
            self.motion["joint_pos"][index]
            + np.asarray(self.contract["kd"]) / np.asarray(self.contract["kp"]) * self.motion["joint_vel"][index]
        )
        return np.clip(value, self.lo, self.hi)

    def _features(self, positions, rotations, states):
        return np.concatenate(
            (
                positions[:, self.ids].reshape(len(states), -1),
                rotations[:, self.ids].reshape(len(states), -1),
                states,
            ),
            axis=1,
        )

    def features(self, states):
        n = len(states)
        self.qpos[:n], self.qvel[:n] = states[:, : self.nq], states[:, self.nq :]
        self.batch.forward(np.arange(n))
        return self._features(self.positions[:n], self.rotations[:n], states)

    def step(self, states, targets):
        self.input_states = states.copy()
        return super().step(states, targets)

    def probe(self, n):
        # With forward=False, xpos/xquat still describe the input state after
        # the first mj_step; qpos/qvel have already advanced. Preserve the input
        # generalized state explicitly when assembling the cost derivative.
        columns = 1 + self.nx + self.nu
        self.feat = (
            self._features(self.positions[:n], self.rotations[:n], self.input_states)
            .reshape(-1, columns, self.reference.shape[1])[:, : 1 + self.nx]
            .copy()
        )

    def residual(self, t, features):
        ref = self.reference[np.minimum(self.start + t, len(self.states) - 1)]
        shape = features.shape[:-1]
        positions = features[..., :18].reshape(*shape, 6, 3)
        rotations = features[..., 18:42].reshape(*shape, 6, 4)
        ref_positions = ref[..., :18].reshape(*ref.shape[:-1], 6, 3)
        ref_rotations = ref[..., 18:42].reshape(*ref.shape[:-1], 6, 4)
        state, ref_state = features[..., 42:], ref[..., 42:]
        error_position = (positions - ref_positions) * np.sqrt(self.position_weights)
        error_rotation = quat_log(quat_mul(rotations, ref_rotations * [1, -1, -1, -1])) * np.sqrt(
            self.rotation_weights[:, None]
        )
        joint = state[..., 7:30]
        error_joint = np.sqrt(self.joint_weight) * (joint - ref_state[..., 7:30])
        error_velocity = (state[..., 30:] - ref_state[..., 30:]) * np.sqrt(self.velocity_weights)
        lower = np.sqrt(self.limit_weight) * np.maximum(self.lo + self.limit_margin - joint, 0)
        upper = np.sqrt(self.limit_weight) * np.maximum(joint - self.hi + self.limit_margin, 0)
        speed = np.sqrt(self.speed_weight) * np.maximum(
            np.abs(state[..., 36:]) - 0.8 * np.asarray(self.contract["native_velocity"]), 0
        )
        return np.concatenate(
            (
                error_position.reshape(*shape, -1),
                error_rotation.reshape(*shape, -1),
                error_joint,
                error_velocity,
                lower,
                upper,
                speed,
            ),
            axis=-1,
        )

    def cost(self, t, states, targets):
        features = (
            self.features(states) if t == 0 else self._features(self.line_positions, self.line_rotations, states)
        )
        residual = self.residual(t, features)
        cost = np.sum(residual**2, axis=-1)
        if t < self.T:
            cost += self.control_weight * np.sum((targets - self.target_reference(t)) ** 2, axis=-1)
        return cost

    def expand(self, states, targets):
        last = np.repeat(states[-1:], 1 + self.nx, axis=0)
        delta = EPS * np.eye(1 + self.nx)[:, 1:]
        last[:, : self.nq] = self.integrate(last[:, : self.nq], delta[:, : self.nv])
        last[:, self.nq :] += delta[:, self.nv :]
        features = np.concatenate((self.feat, self.features(last)[None]))
        residual = self.residual(np.arange(self.T + 1)[:, None], features)
        jacobian = (residual[:, 1:] - residual[:, :1]) / EPS
        lx = 2 * np.einsum("tkr,tr->tk", jacobian, residual[:, 0])
        lxx = 2 * np.einsum("tkr,tlr->tkl", jacobian, jacobian, optimize=True)
        lu = 2 * self.control_weight * (targets - self.target_reference(np.arange(self.T)))
        luu = 2 * self.control_weight * np.tile(np.eye(self.nu), (self.T, 1, 1))
        return lx, lxx, lu, luu
