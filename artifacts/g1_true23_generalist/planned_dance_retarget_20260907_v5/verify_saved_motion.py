"""Cold-read, independent full-frame FK audit of the saved V5 artifact.

No solver, policy, simulation step, state write, or robot transport is invoked.
Uses declared task frames but recomputes the gates directly from native/source
model FK, rather than trusting report metrics or the optimizer's qpos buffer.
"""
from pathlib import Path
import json

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from gear_sonic.utils import g1_23dof_task_space_retarget as ik
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file
from gear_sonic.utils.g1_true23_reference_floor import compiled_model_sha256


directory = Path(__file__).resolve().parent
report_path = directory / "report.json"
report = json.loads(report_path.read_text())
assert report["accepted"] is True
for path, expected in report["input_bindings"].items():
    assert sha256_file(Path(path)) == expected, f"source binding changed: {path}"
motion_path = directory / "adapted.true23.npz"
assert sha256_file(motion_path) == report["adapted_motion_sha256"]
diagnostic_path = directory / report["diagnostic_artifact"]["path"]
assert sha256_file(diagnostic_path) == report["diagnostic_artifact"]["sha256"]
with np.load(motion_path, allow_pickle=False) as archive:
    motion = {key: archive[key].copy() for key in archive.files}
with np.load(directory / "planned.named29.npz", allow_pickle=False) as archive:
    named_source = {key: archive[key].copy() for key in archive.files}
with np.load(diagnostic_path, allow_pickle=False) as archive:
    diagnostic = {key: archive[key].copy() for key in archive.files}
source_path = next(Path(path) for path in report["input_bindings"] if path.endswith("original29.mjb"))
target_path = next(Path(path) for path in report["input_bindings"] if path.endswith("g1_23dof_rev_1_0.xml"))
source_model = mujoco.MjModel.from_binary_path(str(source_path))
target_model = mujoco.MjModel.from_xml_path(str(target_path))
assert report["compiled_models"] == {"source": compiled_model_sha256(source_model),
                                      "target": compiled_model_sha256(target_model)}
assert (source_model.nq, target_model.nq, target_model.nu) == (36, 30, 23)
assert motion["joint_pos"].shape == (1091, 23)
assert named_source["joint_pos"].shape == (546, 29)
assert all(np.isfinite(value).all() for value in motion.values() if value.dtype.kind in "fiu")
np.testing.assert_array_equal(motion["source_joint_pos_resampled"][::2], named_source["joint_pos"])
np.testing.assert_array_equal(motion["source_root_pos_w_resampled"][::2], named_source["root_pos_w"])
np.testing.assert_array_equal(motion["source_time_map_s"], np.linspace(0., 10.9, 1091))
np.testing.assert_array_equal(motion["contact_flags"], diagnostic["diagnostic_contact_flags"])
np.testing.assert_array_equal(motion["joint_names"], ik._model_layout(target_model).joint_names)
np.testing.assert_array_equal(motion["source_joint_names"], ik._model_layout(source_model).joint_names)
qpos = np.column_stack((motion["body_pos_w"][:, 0], motion["body_quat_w"][:, 0], motion["joint_pos"]))
requested = diagnostic["diagnostic_requested_qpos29"]
original = np.column_stack((motion["source_root_pos_w_resampled"], motion["source_root_quat_wxyz_resampled"],
                           motion["source_joint_pos_resampled"]))
np.testing.assert_allclose(requested[:, :3], original[0, :3] + .9 * (original[:, :3] - original[0, :3]), atol=0, rtol=0)
np.testing.assert_allclose(requested[:, 7:], original[0, 7:] + .9 * (original[:, 7:] - original[0, 7:]), atol=0, rtol=0)
source_rotations = Rotation.from_quat(original[:, [4, 5, 6, 3]])
scaled = source_rotations[0] * Rotation.from_rotvec((source_rotations[0].inv() * source_rotations).as_rotvec() * .9)
assert np.max((scaled * Rotation.from_quat(requested[:, [4, 5, 6, 3]]).inv()).magnitude()) < 1e-12
root_and_joint_delta = np.abs(qpos - diagnostic["diagnostic_qpos_native23"])
assert np.max(root_and_joint_delta[:, np.r_[0:3, 7:30]]) == 0
assert np.max(root_and_joint_delta[:, 3:7]) <= np.finfo(np.float32).eps


def task_fk(model, pose, source):
    data = mujoco.MjData(model)
    data.qpos[:] = pose
    mujoco.mj_forward(model, data)
    positions, rotations = [], []
    for task in ik.DEFAULT_TASKS:
        body = model.body(task.source_body if source else task.target_body).id
        if task.kind == "subtree_com":
            positions.append(data.subtree_com[body].copy())
            rotations.append(np.eye(3))
        else:
            rotation = data.xmat[body].reshape(3, 3).copy()
            point = task.source_point if source else task.target_point
            positions.append(data.xpos[body] + rotation @ np.asarray(point))
            rotations.append(rotation)
    return np.asarray(positions), np.asarray(rotations), data


layout = ik._safe_target_layout(ik._model_layout(target_model), 1e-5, 10.)
source_names = ik._model_layout(source_model).joint_names
direct = np.clip(requested[:, [7 + source_names.index(name) for name in layout.joint_names]], layout.lower, layout.upper)
np.testing.assert_array_equal(direct, diagnostic["diagnostic_direct_joints_native23"])
before_position, after_position, before_angle, after_angle, distortion = [], [], [], [], []
original_positions, body_position_error, body_orientation_error = [], [], []
for frame in range(len(qpos)):
    wanted_pos, wanted_rot, _ = task_fk(source_model, requested[frame], True)
    orig_pos, _, _ = task_fk(source_model, original[frame], True)
    base_pos, base_rot, _ = task_fk(target_model, np.r_[requested[frame, :7], direct[frame]], False)
    actual_pos, actual_rot, data = task_fk(target_model, qpos[frame], False)
    np.testing.assert_allclose(wanted_pos, motion["adapted_task_pos_w"][frame], atol=1e-12, rtol=0)
    before_position.append(np.linalg.norm(base_pos - wanted_pos, axis=1))
    after_position.append(np.linalg.norm(actual_pos - wanted_pos, axis=1))
    before_angle.append(Rotation.from_matrix(base_rot @ np.transpose(wanted_rot, (0, 2, 1))).magnitude())
    after_angle.append(Rotation.from_matrix(actual_rot @ np.transpose(wanted_rot, (0, 2, 1))).magnitude())
    original_positions.append(orig_pos)
    distortion.append(np.linalg.norm(wanted_pos - orig_pos, axis=1))
    body_position_error.append(np.max(np.abs(data.xpos[1:] - motion["body_pos_w"][frame])))
    exported_rot = Rotation.from_quat(motion["body_quat_w"][frame][:, [1, 2, 3, 0]])
    actual_body_rot = Rotation.from_matrix(data.xmat[1:].reshape(-1, 3, 3))
    body_orientation_error.append(np.max((actual_body_rot * exported_rot.inv()).magnitude()))
before_position, after_position, before_angle, after_angle = map(np.asarray,
    (before_position, after_position, before_angle, after_angle))
names = [task.name for task in ik.DEFAULT_TASKS]
cost_before, cost_after = np.zeros(len(qpos)), np.zeros(len(qpos))
for index, task in enumerate(ik.DEFAULT_TASKS):
    multiplier = np.ones(len(qpos))
    if task.contact_side:
        multiplier[motion["contact_flags"][:, int(task.contact_side == "right")]] = 2.5
    cost_before += multiplier * (task.position_weight * before_position[:, index]**2
                                  + task.orientation_weight * before_angle[:, index]**2)
    cost_after += multiplier * (task.position_weight * after_position[:, index]**2
                                 + task.orientation_weight * after_angle[:, index]**2)
failed = {"weighted_cost": cost_after > cost_before + 1e-7 * np.maximum(1., cost_before)}
for foot in ("left_foot", "right_foot"):
    index = names.index(foot)
    failed[f"{foot}_position"] = after_position[:, index] > .005
    failed[f"{foot}_orientation"] = after_angle[:, index] > before_angle[:, index] + .005
com = names.index("whole_robot_com")
failed["com_regression"] = after_position[:, com] > before_position[:, com] + .001
native = ik._hardware_targets_to_raw_native(qpos[:, 7:])
failed["native_clip"] = np.any(np.abs(native) > 10., axis=1)
low, high = ik.safe_target_joint_bounds(target_model, native_action_clip=9.5, safe_limit_guard_rad=.05)
failed["guarded_rom"] = np.any((qpos[:, 7:] < low - 1e-7) | (qpos[:, 7:] > high + 1e-7), axis=1)
velocity = np.diff(qpos[:, 7:], axis=0) / .02
acceleration = np.diff(velocity, axis=0) / .02
assert np.max(np.abs(velocity)) <= 5.0 and np.max(np.abs(acceleration)) <= 80.
offset = qpos[:, :3] - requested[:, :3]
rotation_offset = (Rotation.from_quat(qpos[:, [4, 5, 6, 3]]) * Rotation.from_quat(requested[:, [4, 5, 6, 3]]).inv()).as_rotvec()
assert np.max(np.abs(offset)) <= .08 + 2e-7
assert np.max(np.abs(rotation_offset).sum(axis=1)) <= .45 + 2e-7
root_derivatives = {}
for key, value, vmax, amax in (("offset", offset, .75, 6.), ("rotation_coordinate", rotation_offset, 1.5, 12.)):
    vel = np.diff(value, axis=0) / .02
    acc = np.diff(vel, axis=0) / .02
    root_derivatives[key] = {"velocity_max": float(np.max(np.abs(vel))), "acceleration_max": float(np.max(np.abs(acc)))}
    assert np.max(np.abs(vel)) <= vmax + 1e-7 and np.max(np.abs(acc)) <= amax + 1e-6
distortion = np.asarray(distortion)
source_excursion = np.linalg.norm(np.asarray(original_positions) - original_positions[0], axis=2).max(axis=0)
fractions = distortion.max(axis=0) / np.maximum(source_excursion, 1e-8)
assert np.max(fractions) <= .2 + 1e-6
p95 = {name: float(np.percentile(after_position[:, names.index(name)], 95))
       for name in ("left_foot", "right_foot", "head_proxy", "left_hand", "right_hand")}
assert all(error <= (.05 if "foot" in name else .10) for name, error in p95.items())
counts = {name: int(mask.sum()) for name, mask in failed.items()}
assert sum(counts.values()) == 0, counts
assert max(body_position_error) <= 2e-5 and max(body_orientation_error) <= 2e-5
verification = {
    "kind": "g1_true23_v5_cold_saved_motion_independent_fk_verification_v1", "passed": True,
    "source_frames": 546, "saved_frames": 1091, "source_time_endpoints_s": [0., 10.9],
    "saved_duration_s": 21.8, "all_original_joint_and_root_position_samples_exact": True,
    "source_contacts_preserved_not_physical_contact_evidence": True,
    "protected_failure_counts": counts, "actual_saved_motion_fk_p95_m": p95,
    "body_position_channel_fk_max_error_m": float(max(body_position_error)),
    "body_orientation_channel_fk_max_error_rad": float(max(body_orientation_error)),
    "second_serialization_quaternion_component_delta_max": float(root_and_joint_delta[:, 3:7].max()),
    "second_serialization_root_position_or_joint_delta_max": 0.,
    "joint_velocity_abs_max_rad_s": float(np.max(np.abs(velocity))),
    "joint_acceleration_abs_max_rad_s2": float(np.max(np.abs(acceleration))),
    "root_reference_derivatives": root_derivatives,
    "max_task_space_adaptation_distortion_fraction": float(np.max(fractions)),
    "input_bindings_verified": report["input_bindings"],
    "report_sha256": sha256_file(report_path), "motion_sha256": sha256_file(motion_path),
    "diagnostic_sha256": sha256_file(diagnostic_path), "verification_script_sha256": sha256_file(Path(__file__)),
    "dynamic_feasibility_verified": False, "controller_qualified": False,
    "hardware_authorized": False, "deployment_ready": False,
}
for path, expected in report["input_bindings"].items():
    assert sha256_file(Path(path)) == expected
with (directory / "saved_motion_verification.json").open("x") as stream:
    json.dump(verification, stream, indent=2, sort_keys=True, allow_nan=False)
print(json.dumps({key: verification[key] for key in ("passed", "protected_failure_counts", "actual_saved_motion_fk_p95_m")}))
