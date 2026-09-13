"""Independently verify stitched accepted reference poses, without dynamics."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from gear_sonic.utils.g1_true23_mpc_student import load_inputs, sha256


def run(args):
    model, contract, _, original, _ = load_inputs(args.bundle, "pico")
    with np.load(args.references / "pico/reference.npz", allow_pickle=False) as a:
        motion = {k: a[k].copy() for k in a.files}
    accepted = {}
    bindings = {}
    for directory in (args.first, args.resume):
        report = json.loads((directory / "report.json").read_text())
        valid = {r["frame"] for r in report["records"] if r["passed"]}
        with np.load(directory / "episode.npz", allow_pickle=False) as a:
            accepted.update({int(f): q.copy() for f, q in zip(a["frames"], a["proposed_current"]) if f in valid})
        bindings[str(directory / "episode.npz")] = sha256(directory / "episode.npz")
    refinement = json.loads((args.refinement / "report.json").read_text())
    assert refinement["passed"]
    with np.load(args.refinement / "poses.npz", allow_pickle=False) as a:
        accepted[refinement["frame"]] = a["proposed"][0].copy()
    bindings[str(args.refinement / "poses.npz")] = sha256(args.refinement / "poses.npz")
    frames = np.array(sorted(accepted)); corrected = np.array([accepted[f] for f in frames])
    assert np.array_equal(frames, np.arange(frames[0], frames[-1] + 1))
    original_poses = np.c_[motion["body_pos_w"][:, 0], motion["body_quat_w"][:, 0], motion["joint_pos"]]
    incoming = np.vstack((original_poses[frames[0] - 1], corrected))
    with_source_return = np.vstack((incoming, original_poses[frames[-1] + 1]))
    speed = np.asarray(contract["native_velocity"])
    actual_speed = np.diff(incoming[:, 7:], axis=0) / .02
    arm_ratio = float(np.max(np.abs(actual_speed[:, 13:]) / (.8 * speed[13:])))
    source_return_ratio = float(np.max(np.abs(with_source_return[-1, 20:] - with_source_return[-2, 20:]) / (.02 * .8 * speed[13:])))
    acceleration = np.diff(with_source_return[:, 20:], n=2, axis=0) / .02**2
    accepted_acceleration = np.diff(incoming[:, 20:], n=2, axis=0) / .02**2
    data = mujoco.MjData(model)
    arm_roots = {model.body(f"{side}_shoulder_pitch_link").id for side in ("left", "right")}
    arm_bodies = set()
    for body in range(1, model.nbody):
        ancestor = body
        while ancestor:
            if ancestor in arm_roots:
                arm_bodies.add(body); break
            ancestor = int(model.body_parentid[ancestor])
    wrists = [int(model.jnt_bodyid[model.joint(f"{side}_wrist_roll_joint").id]) for side in ("left", "right")]
    torso = model.body("torso_link").id
    local = np.array([[.264, -.025, 0.], [.264, .025, 0.]])
    errors = []; contacts = []; bounds = model.jnt_range[model.actuator_trnid[:, 0]]
    for frame, pose in zip(frames, corrected):
        data.qpos[:] = pose; data.qvel[:] = 0.; data.qacc_warmstart[:] = 0.
        mujoco.mj_forward(model, data)
        hands = data.xpos[wrists] + np.einsum("nij,nj->ni", data.xmat[wrists].reshape(2, 3, 3), local)
        head = data.xpos[torso] + data.xmat[torso].reshape(3, 3) @ [0., 0., .35]
        actual = np.vstack((hands, head)) - pose[:3]
        goal = original["source_task_position_w"][frame] - original["source_qpos29"][frame, :3]
        errors.append(np.linalg.norm(actual - goal, axis=1))
        for c in data.contact:
            ga, gb = int(c.geom1), int(c.geom2)
            ba, bb = int(model.geom_bodyid[ga]), int(model.geom_bodyid[gb])
            if not ba or not bb or c.dist >= 0:
                continue
            contacts.append(dict(frame=int(frame), geom_ids=[ga, gb], depth_m=-float(c.dist),
                                 bodies=[model.body(ba).name, model.body(bb).name],
                                 arm_involving=ba in arm_bodies or bb in arm_bodies,
                                 exclude=int(c.exclude), active_constraint=int(c.efc_address)))
    errors = np.array(errors)
    arm_contacts = [c for c in contacts if c["arm_involving"]]
    fixed_contacts = [c for c in contacts if not c["arm_involving"]]
    excess = float(np.maximum(np.maximum(bounds[:, 0] - corrected[:, 7:], corrected[:, 7:] - bounds[:, 1]), 0.).max())
    root_fixed = np.array_equal(corrected[:, :20], original_poses[frames, :20])
    passed = not arm_contacts and errors.max() <= .15 and arm_ratio <= 1 + 1e-10 and excess <= 1e-12 and root_fixed
    report = dict(accepted_frames=len(frames), first_frame=int(frames[0]), last_frame=int(frames[-1]),
                  original_reference_sha256=sha256(args.references / "pico/reference.npz"), input_bindings=bindings,
                  no_frame_gaps=True, root_waist_legs_bit_exact=root_fixed, native_joint_limit_excess_rad=excess,
                  original_relative_hand_head_max_m=errors.max(axis=0).tolist(),
                  original_relative_hand_head_p95_m=np.percentile(errors, 95, axis=0).tolist(),
                  actual_adjacent_arm_80pct_native_speed_ratio=arm_ratio,
                  return_to_original_next_frame_80pct_native_speed_ratio=source_return_ratio,
                  arm_joint_speed_max_radps=float(np.max(np.abs(actual_speed[:, 13:]))),
                  accepted_fragment_arm_acceleration_max_radps2=float(np.max(np.abs(accepted_acceleration))),
                  arm_acceleration_with_source_neighbors_max_radps2=float(np.max(np.abs(acceleration))),
                  arm_acceleration_per_joint_max_radps2=np.max(np.abs(acceleration), axis=0).tolist(),
                  native_acceleration_limit_not_declared=True, arm_contact_count=len(arm_contacts),
                  unresolved_fixed_contact_count=len(fixed_contacts),
                  unresolved_fixed_depth_max_m=max([c["depth_m"] for c in fixed_contacts], default=0.),
                  all_self_collision_free=not contacts, contacts=contacts,
                  accepted_fragment_geometry_passed=bool(passed), completed_requested_episode=int(frames[-1]) == args.expected_stop - 1,
                  return_to_original_next_frame_speed_passed=source_return_ratio <= 1 + 1e-10,
                  physical_dynamics_qualified=False, full_reference_promoted=False,
                  code_sha256=sha256(Path(__file__)), mujoco=mujoco.__version__)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "contacts"}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in ("bundle", "references", "first", "resume", "refinement", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--expected-stop", type=int, default=3866)
    run(p.parse_args())
