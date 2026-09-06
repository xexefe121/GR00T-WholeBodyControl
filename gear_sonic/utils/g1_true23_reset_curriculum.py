"""Floor-corrected synthetic reset curriculum; no robot or runtime action changes.

This does not certify self-collision freedom, dynamic contact equilibrium or
physical reachability. Full-source evaluation remains separate and unchanged.
"""

import copy
import hashlib

import numpy as np

from gear_sonic.utils.g1_true23_contact_geometry import lift_reset_floor_overlap


def reset_curriculum_contract(warmup_controls, ramp_end_controls):
    if (
        type(warmup_controls) is not int
        or type(ramp_end_controls) is not int
        or not 0 <= warmup_controls < ramp_end_controls <= 100000000
    ):
        raise ValueError("reset curriculum requires integer 0 <= warmup < ramp end <= 100000000")
    return dict(
        kind="g1_true23_floor_corrected_reset_curriculum_v1",
        warmup_controls=warmup_controls,
        ramp_end_controls=ramp_end_controls,
        initial_scale=0.0,
        final_scale=1.0,
        schedule="linear_in_actual_environment_common_step_counter",
        changed_reset_terms=["pose_range", "velocity_range", "joint_position_range"],
        floor_clearance_m=1e-5,
        maximum_reset_lift_m=0.2,
        floor_correction_all_stages=True,
        preserve_reference_clips_and_sampling=True,
        preserve_action_and_sensor_noise=True,
        preserve_motor_limits_and_gains=True,
        physics_steps_added_during_reset=0,
        self_collision_freedom_proven=False,
        dynamic_equilibrium_proven=False,
        physical_reachability_proven=False,
        hardware_authorized=False,
        deployment_ready=False,
    )


def reset_scale(control, contract):
    if type(control) is not int or control < 0:
        raise ValueError("reset curriculum requires a nonnegative actual control counter")
    expected = reset_curriculum_contract(contract["warmup_controls"], contract["ramp_end_controls"])
    if contract != expected:
        raise ValueError("reset curriculum contract changed")
    start, stop = contract["warmup_controls"], contract["ramp_end_controls"]
    return min(1.0, max(0.0, (control - start) / (stop - start)))


def scaled_reset_ranges(cfg, scale):
    return dict(
        pose_range={key: tuple(scale * x for x in value) for key, value in cfg.pose_range.items()},
        velocity_range={key: tuple(scale * x for x in value) for key, value in cfg.velocity_range.items()},
        joint_position_range=tuple(scale * x for x in cfg.joint_position_range),
    )


def array_digest(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(str(value.dtype).encode() + str(value.shape).encode() + value.tobytes()).hexdigest()


def resample_with_curriculum(command, env_ids, original_resample, contract):
    import torch

    step = int(command._env.common_step_counter)
    scale = reset_scale(step, contract)
    original_ranges = {name: copy.deepcopy(getattr(command.cfg, name)) for name in contract["changed_reset_terms"]}
    try:
        for name, value in scaled_reset_ranges(command.cfg, scale).items():
            setattr(command.cfg, name, value)
        original_resample(command, env_ids)
    finally:
        for name, value in original_ranges.items():
            setattr(command.cfg, name, value)
    if len(env_ids) == 0:
        return
    env = command._env
    qpos_before = env.sim.data.qpos[env_ids].detach().cpu().numpy().copy()
    qvel_before = env.sim.data.qvel[env_ids].clone()
    lifted, evidence = lift_reset_floor_overlap(
        env.sim.mj_model,
        qpos_before,
        clearance_m=contract["floor_clearance_m"],
        maximum_lift_m=contract["maximum_reset_lift_m"],
    )
    root_start = evidence["root_z_qpos_index"] - 2
    root_pose = torch.as_tensor(lifted[:, root_start : root_start + 7], device=env.device)
    command.robot.write_root_link_pose_to_sim(root_pose, env_ids=env_ids)
    command.robot.clear_state(env_ids=env_ids)
    if not torch.equal(env.sim.data.qvel[env_ids], qvel_before):
        raise ValueError("reset curriculum changed generalized velocity during floor correction")
    if not torch.equal(env.sim.data.qpos[env_ids], torch.as_tensor(lifted, device=env.device)):
        raise ValueError("reset curriculum changed more than expected root height")
    events = getattr(command, "_reset_curriculum_events", None)
    if events is None:
        events = command._reset_curriculum_events = []
    distances = lambda rows: [row["minimum_floor_contact_distance_m"] for row in rows]
    events.append(
        dict(
            control=step,
            scale=scale,
            env_indices=env_ids.detach().cpu().tolist(),
            source_phases=command.time_steps[env_ids].detach().cpu().tolist(),
            root_lifts_m=evidence["actual_lifts_m"],
            minimum_floor_distances_before_m=distances(evidence["before"]["rows"]),
            minimum_floor_distances_after_m=distances(evidence["after"]["rows"]),
            qpos_before_sha256=array_digest(qpos_before),
            qpos_after_sha256=array_digest(lifted),
            qvel_before_and_after_sha256=array_digest(qvel_before.detach().cpu().numpy()),
            expected_positions_and_unchanged_velocities_verified=True,
        )
    )


def summarize_reset_events(events, contract):
    if not events:
        raise ValueError("reset curriculum has no actual reset evidence")
    controls = [row["control"] for row in events]
    if controls != sorted(controls):
        raise ValueError("reset curriculum event counter moved backwards")
    for row in events:
        if row["scale"] != reset_scale(row["control"], contract):
            raise ValueError("actual reset amplitude differs from curriculum counter")
        fields = (
            "source_phases",
            "root_lifts_m",
            "minimum_floor_distances_before_m",
            "minimum_floor_distances_after_m",
        )
        if any(len(row[key]) != len(row["env_indices"]) for key in fields):
            raise ValueError("reset curriculum event row counts differ")
        if row["expected_positions_and_unchanged_velocities_verified"] is not True:
            raise ValueError("reset curriculum lacks state-preservation evidence")
        if any(value is not None and value < 0 for value in row["minimum_floor_distances_after_m"]):
            raise ValueError("reset curriculum left detected floor penetration")
        if any(not 0 <= value <= contract["maximum_reset_lift_m"] + 1e-7 for value in row["root_lifts_m"]):
            raise ValueError("reset curriculum lift exceeds its bound")
    lifts = [value for row in events for value in row["root_lifts_m"]]
    return dict(
        events=len(events),
        reset_rows=len(lifts),
        lifted_rows=sum(value > 0 for value in lifts),
        maximum_lift_m=max(lifts, default=0),
        minimum_scale=min(row["scale"] for row in events),
        maximum_scale=max(row["scale"] for row in events),
        final_event_control=controls[-1],
        full_disturbance_reset_rows=sum(len(row["env_indices"]) for row in events if row["scale"] == 1),
        remaining_detected_floor_penetrations=0,
        physics_steps_added_during_reset=0,
        hardware_authorized=False,
        deployment_ready=False,
    )
