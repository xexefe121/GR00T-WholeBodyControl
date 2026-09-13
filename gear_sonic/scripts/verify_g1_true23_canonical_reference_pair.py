"""Independent quaternion-algebra derivative and original-pose verification."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from gear_sonic.utils.g1_true23_mpc_student import load_inputs, sha256
from gear_sonic.scripts.verify_g1_true23_floor_filter import filtered


def archive(path):
    with np.load(path, allow_pickle=False) as a:
        return {k: a[k].copy() for k in a.files}


def central(x, dt):
    v = np.empty_like(x)
    v[1:-1] = (x[2:] - x[:-2]) / (2 * dt)
    v[0] = (x[1] - x[0]) / dt
    v[-1] = (x[-1] - x[-2]) / dt
    return v


def angular(q, dt):
    n = len(q)
    lo = np.maximum(np.arange(n) - 1, 0)
    hi = np.minimum(np.arange(n) + 1, n - 1)
    later = q[hi].copy(); earlier = q[lo].copy()
    later /= np.linalg.norm(later, axis=-1, keepdims=True)
    earlier /= np.linalg.norm(earlier, axis=-1, keepdims=True)
    earlier[..., 1:] *= -1
    w = later[..., :1] * earlier[..., :1] - np.sum(later[..., 1:] * earlier[..., 1:], axis=-1, keepdims=True)
    v = later[..., :1] * earlier[..., 1:] + earlier[..., :1] * later[..., 1:] + np.cross(later[..., 1:], earlier[..., 1:])
    negative = w[..., 0] < 0
    w[negative] *= -1; v[negative] *= -1
    length = np.linalg.norm(v, axis=-1, keepdims=True)
    scale = np.full_like(length, 2.)
    np.divide(2 * np.arctan2(length, w), length, out=scale, where=length > 1e-14)
    return v * scale / ((hi - lo) * dt)[:, None, None]


def run(args):
    records = []
    for clip in ("pico", "walk002"):
        model, contract, native, _, timeline = load_inputs(args.bundle, clip)
        a_path = args.a / clip / "reference.npz"
        b_path = args.b / clip / "reference.npz"
        a, b = archive(a_path), archive(b_path)
        original_path = args.a / clip / "original_native_reference.npz"
        original = archive(original_path)
        receipt = json.loads((args.a / clip / "portable_receipt.json").read_text())
        checks = dict(original_archive_hash_matches_receipt=sha256(original_path) == receipt["original_native_reference_sha256"],
                      b_input_byte_identical_a=sha256(args.b / clip / "before_floor_reference.npz") == sha256(a_path),
                      all_shapes_original=all(a[k].shape == b[k].shape == original[k].shape for k in original),
                      all_finite=all(np.isfinite(x[k]).all() for x in (a, b) for k in x),
                      full_original_frame_count=len(a["joint_pos"]) == len(native["joint_pos"]),
                      same_original_timeline=json.loads((args.a / clip / "original_timeline.json").read_text()) == timeline)
        for key in ("fps", "joint_pos", "body_pos_w", "body_quat_w"):
            checks["a_original_exact_" + key] = np.array_equal(a[key], original[key])
            checks["a_bundle_exact_" + key] = np.array_equal(a[key], native[key])
        for key in ("fps", "joint_pos", "joint_vel", "body_quat_w", "body_ang_vel_w"):
            checks["b_a_exact_" + key] = np.array_equal(b[key], a[key])
        dt = 1 / float(a["fps"][0]); errors = {}
        for variant, motion in (("a", a), ("b", b)):
            errors[variant + "_joint_velocity"] = float(np.max(np.abs(central(motion["joint_pos"], dt) - motion["joint_vel"])))
            errors[variant + "_linear_velocity"] = float(np.max(np.abs(central(motion["body_pos_w"], dt) - motion["body_lin_vel_w"])))
            errors[variant + "_angular_velocity_quaternion_algebra"] = float(np.max(np.abs(angular(motion["body_quat_w"], dt) - motion["body_ang_vel_w"])))
        data = mujoco.MjData(model); position_error = rotation_error = 0.
        for f in range(len(a["joint_pos"])):
            data.qpos[:] = np.r_[a["body_pos_w"][f, 0], a["body_quat_w"][f, 0], a["joint_pos"][f]]
            mujoco.mj_kinematics(model, data)
            position_error = max(position_error, float(np.max(np.abs(data.xpos[1:] - a["body_pos_w"][f]))))
            expected = a["body_quat_w"][f]
            quaternion_delta = np.minimum(np.linalg.norm(data.xquat[1:] - expected, axis=1), np.linalg.norm(data.xquat[1:] + expected, axis=1))
            rotation_error = max(rotation_error, float(quaternion_delta.max()))
        errors["all_frame_native_fk_position"] = position_error
        errors["all_frame_native_fk_quaternion_sign_invariant"] = rotation_error
        delta = b["body_pos_w"] - a["body_pos_w"]
        lift_data = archive(args.b / clip / "frame_lift.npz")
        lift = lift_data["frame_lift_m"]
        expected = np.zeros_like(delta); expected[:, :, 2] = lift[:, None]
        errors["b_common_saved_z_translation"] = float(np.max(np.abs(delta - expected)))
        raw = archive(args.audit_a / f"{clip}_floor_arrays.npz")["minimum_lift"]
        smooth = filtered(raw, 15., dt)
        reconstructed = np.maximum(raw + .000001, smooth + .020)
        errors["b_raw_independent_floor_lift"] = float(np.max(np.abs(raw - lift_data["raw_required_lift_m"])))
        errors["b_independent_expm_floor_filter"] = float(np.max(np.abs(reconstructed - lift)))
        speed_ratio = float(np.max(np.abs(np.diff(a["joint_pos"], axis=0)) / dt / np.asarray(contract["native_velocity"])))
        joint_ids = model.actuator_trnid[:, 0]
        bounds = model.jnt_range[joint_ids]
        excess = float(np.maximum(np.maximum(bounds[:, 0] - a["joint_pos"], a["joint_pos"] - bounds[:, 1]), 0.).max())
        checks["native_joint_bounds"] = excess <= 1e-12
        checks["native_adjacent_speed"] = speed_ratio <= 1 + 1e-12
        record = dict(clip=clip, frames=len(a["joint_pos"]), a_sha256=sha256(a_path), b_sha256=sha256(b_path),
                      checks=checks, errors=errors, native_speed_ratio=speed_ratio, native_joint_limit_excess=excess,
                      b_guard_activations=int(np.sum(raw + .000001 > smooth + .020)),
                      passed=all(checks.values()) and max(errors.values()) < 1e-10)
        records.append(record)
        print(json.dumps(record), flush=True)
    report = dict(all_passed=all(r["passed"] for r in records), results=records,
                  future_pose_support_seconds=.02, physical_dynamics_qualified=False,
                  method="direct central differences plus independently implemented quaternion multiplication/log; exact prepared MuJoCo3.2.3 FK and floor audit",
                  code_sha256=sha256(Path(__file__)), mujoco=mujoco.__version__)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in ("bundle", "a", "b", "audit-a", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    run(p.parse_args())
