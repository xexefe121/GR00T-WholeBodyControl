"""Read-only failure localization; no controller, training or hardware authority.

All root statistics use the existing q2 referee. Episode-local displacement is
reported alongside absolute error, never substituted for it. Command ablations
hold recorded observations fixed and therefore are not counterfactual rollouts.
"""

import numpy as np


def finite_array(value, shape, name):
    value = np.asarray(value)
    if value.shape != shape or value.dtype.kind not in "fi" or not np.isfinite(value).all():
        raise ValueError(f"{name} requires finite numeric shape {shape}")
    return value


def contiguous_runs(mask):
    mask = np.asarray(mask)
    if mask.ndim != 1 or mask.dtype != np.bool_:
        raise ValueError("run mask must be one-dimensional boolean")
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist(), strict=True))


def phase_path_metrics(qpos, reference_qpos, phases, physics_post_qvel, dt=0.002):
    """Verify the Euler translation integral before decomposing fixed-world drift."""
    if dt != 0.002:
        raise ValueError("failure localization requires unchanged 500 Hz physics")
    qpos = np.asarray(qpos)
    if qpos.ndim != 2 or qpos.shape[1] != 30 or len(qpos) < 2:
        raise ValueError("native23 qpos must have shape [controls+1,30]")
    n = len(qpos) - 1
    finite_array(qpos, (n + 1, 30), "qpos")
    reference_qpos = finite_array(reference_qpos, (n + 11, 36), "original29 reference")
    velocity = finite_array(physics_post_qvel, (n * 10, 29), "post-substep velocity")
    actual_step = np.diff(qpos[:, :3], axis=0)
    integrated = dt * velocity[:, :3].reshape(n, 10, 3).sum(axis=1)
    integral_error = float(np.max(np.abs(actual_step - integrated)))
    if integral_error > 2e-10:
        raise ValueError("saved translation does not match actual Euler post-velocity integral")
    # qpos[0] is reference frame10; first measured output is scored against11.
    desired_step = np.diff(reference_qpos[10:, :3], axis=0)
    error = qpos[:, :3] - reference_qpos[10:, :3]
    error_step = actual_step - desired_step
    np.testing.assert_allclose(np.diff(error, axis=0), error_step, atol=2e-12, rtol=0)
    rows, cursor = [], 0
    for phase in phases:
        s, e = phase["control_start"], phase["control_stop"]
        if type(s) is not int or type(e) is not int or s != cursor or not s < e <= n:
            raise ValueError("phases must cover full contiguous control sequence")
        cursor = e
        section = slice(s, e)
        wanted = desired_step[section].sum(0)
        actual = actual_step[section].sum(0)
        direction = wanted[:2] / max(float(np.linalg.norm(wanted[:2])), 1e-12)
        err = error[s + 1 : e + 1]
        speed_wanted = desired_step[section, :2] / 0.02
        speed_actual = actual_step[section, :2] / 0.02
        denominator = float(np.sum(speed_wanted**2))
        rows.append(
            dict(
                name=phase["name"],
                controls=e - s,
                start_error_xyz_m=error[s].tolist(),
                end_error_xyz_m=error[e].tolist(),
                accumulated_velocity_mismatch_xyz_m=error_step[section].sum(0).tolist(),
                root_q2_p95_m=float(np.percentile(np.linalg.norm(err, axis=-1), 95)),
                root_q2_max_m=float(np.linalg.norm(err, axis=-1).max()),
                original_net_displacement_xyz_m=wanted.tolist(),
                actual_net_displacement_xyz_m=actual.tolist(),
                original_xy_path_length_m=float(np.linalg.norm(desired_step[section, :2], axis=-1).sum()),
                actual_xy_path_length_m=float(np.linalg.norm(actual_step[section, :2], axis=-1).sum()),
                net_along_direction_error_m=float((actual[:2] - wanted[:2]) @ direction),
                net_cross_direction_error_m=float(
                    (actual[0] - wanted[0]) * -direction[1] + (actual[1] - wanted[1]) * direction[0]
                ),
                xy_velocity_least_squares_gain=float(np.sum(speed_actual * speed_wanted) / denominator)
                if denominator > 1e-8
                else None,
            )
        )
    if cursor != n:
        raise ValueError("phase coverage is incomplete")
    return dict(translation_integral_max_abs_error_m=integral_error, phases=rows)


def limit_events(trace, ranges, joint_names):
    """Every actual range excursion, including incoming motion and command sign."""
    ranges = finite_array(ranges, (23, 2), "joint ranges")
    if np.any(ranges[:, 0] >= ranges[:, 1]) or len(joint_names) != 23 or len(set(joint_names)) != 23:
        raise ValueError("invalid joint ranges or names")
    q = np.asarray(trace["physics_post_qpos"])
    count = len(q)
    q = finite_array(q, (count, 30), "post qpos")[:, 7:]
    preq = finite_array(trace["physics_pre_qpos"], (count, 30), "pre qpos")[:, 7:]
    prevel = finite_array(trace["physics_pre_qvel"], (count, 29), "pre qvel")[:, 6:]
    postvel = finite_array(trace["physics_post_qvel"], (count, 29), "post qvel")[:, 6:]
    times = finite_array(trace["physics_time"], (count, 2), "physics time")
    torque = finite_array(trace["engine_actuator_force23"], (count, 23), "actual motor torque")
    requested = finite_array(trace["requested_torque23"], (count, 23), "requested motor torque")
    target = finite_array(trace["target23"], (count // 10, 23), "hardware targets")
    if count % 10 or not np.allclose(times[:, 1] - times[:, 0], 0.002, atol=1e-10, rtol=0):
        raise ValueError("substeps must form complete unchanged 50 Hz controls")
    if np.any(target < ranges[:, 0]) or np.any(target > ranges[:, 1]):
        raise ValueError("recorded target lies outside physical joint range")
    rows = []
    for j, name in enumerate(joint_names):
        for side, outward in ((0, -1), (1, 1)):
            excess = np.maximum(outward * (q[:, j] - ranges[j, side]), 0)
            for start, stop in contiguous_runs(excess > 0):
                peak = start + int(np.argmax(excess[start:stop]))

                def point(i):
                    return dict(
                        substep=i,
                        control=i // 10,
                        time_post_s=float(times[i, 1]),
                        pre_q_rad=float(preq[i, j]),
                        post_q_rad=float(q[i, j]),
                        pre_dq_rad_s=float(prevel[i, j]),
                        post_dq_rad_s=float(postvel[i, j]),
                        held_target_rad=float(target[i // 10, j]),
                        requested_torque_nm=float(requested[i, j]),
                        actual_torque_nm=float(torque[i, j]),
                        outward_motor_torque_nm=float(outward * torque[i, j]),
                    )

                rows.append(
                    dict(
                        joint=name,
                        joint_index=j,
                        side="lower" if side == 0 else "upper",
                        boundary_rad=float(ranges[j, side]),
                        start_substep=start,
                        stop_substep_exclusive=stop,
                        duration_s=(stop - start) * 0.002,
                        maximum_excess_rad=float(excess[peak]),
                        first=point(start),
                        peak=point(peak),
                        motor_outward_fraction=float(np.mean(outward * torque[start:stop, j] > 0)),
                        motor_saturated_fraction=float(
                            np.mean(np.abs(requested[start:stop, j] - torque[start:stop, j]) > 1e-10)
                        ),
                    )
                )
    return sorted(rows, key=lambda x: (x["start_substep"], x["joint_index"]))


def fixed_observation_branch_outputs(actor, decoder_input, feedback):
    """Ablate copied inputs/branches without mutating any trained parameter."""
    import torch
    from torch.nn import functional as F

    if decoder_input.ndim != 2 or decoder_input.shape[1] != 994 or feedback.shape != (len(decoder_input), 9):
        raise ValueError("branch diagnostic requires [N,994] decoder and [N,9] root inputs")
    if not torch.isfinite(decoder_input).all() or not torch.isfinite(feedback).all():
        raise ValueError("branch diagnostic requires finite inputs")
    first = actor.core.decoder.module[0](decoder_input)
    pose = F.linear(F.linear(decoder_input, actor.pose_lora_a), actor.pose_lora_b)
    no_position = feedback.clone()
    no_position[:, :3] = 0
    variants = {
        "full": first + actor.root_conditioner(feedback) + pose,
        "no_root_position": first + actor.root_conditioner(no_position) + pose,
        "no_root": first + pose,
        "no_pose": first + actor.root_conditioner(feedback),
    }
    for name, hidden in variants.items():
        for layer in actor.core.decoder.module[1:]:
            hidden = layer(hidden)
        variants[name] = hidden
    return variants
