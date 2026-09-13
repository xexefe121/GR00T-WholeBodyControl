"""Measured swing-foot loading cost for SIM training, never contact qualification."""

from gear_sonic.utils.g1_true23_sole_tracking import sole_points_world

SWING_LOAD_PROFILE = "root_and_upper_feet_swing_load_v6"
SWING_LOAD_WEIGHT = -2.0
SWING_FORCE_NORMALIZATION_N = 100.0
AIRBORNE_ONSET_M = 0.01
AIRBORNE_FULL_M = 0.03
PHYSICS_SUBSTEPS = 10
FOOT_BODY_NAMES = ("left_ankle_roll_link", "right_ankle_roll_link")
FOOT_SENSOR_NAMES = ("true23_left_foot_load", "true23_right_foot_load")


def swing_load_contract():
    return dict(
        weight=SWING_LOAD_WEIGHT,
        force_normalization_n=SWING_FORCE_NORMALIZATION_N,
        reference_airborne_onset_m=AIRBORNE_ONSET_M,
        reference_airborne_full_m=AIRBORNE_FULL_M,
        reference="same_held_received_sample_as_existing_root_world_rewards",
        reference_clearance="minimum_of_four_native_sole_sphere_bottoms_above_fixed_flat_floor",
        measured_source="separate_left_right_foot_ground_net_world_Z_force_at_all_ten_physics_substeps",
        formula="mean_feet(clip((reference_clearance-0p01)/0p02,0,1)*mean_substeps((Fz/100N)^2))",
        force_cost_clipped=False,
        measured_height_or_requested_pd_target_used_as_contact=False,
        source_timing_actor_inputs_architecture_and_physics_changed=False,
        existing_reward_terms_changed=False,
        stance_load_penalized=False,
        sensor_history_substeps=PHYSICS_SUBSTEPS,
        contact_slip_balance_or_tracking_improvement_proven=False,
        deployment_ready=False,
        hardware_authorized=False,
    )


def swing_load_l2(desired_position, desired_quaternion, force_history_w, floor_height):
    """Penalize actual loading only when the received reference requests flight.

    Squaring each physical substep before averaging retains brief impacts. The
    reference mask never depends on actual foot height, actions or future frames.
    """
    import torch

    points = sole_points_world(desired_position, desired_quaternion)
    count = len(desired_position)
    if (
        not isinstance(force_history_w, torch.Tensor)
        or force_history_w.shape != (count, 2, PHYSICS_SUBSTEPS, 3)
        or force_history_w.dtype != desired_position.dtype
        or force_history_w.device != desired_position.device
        or not isinstance(floor_height, torch.Tensor)
        or floor_height.shape != (count,)
        or floor_height.dtype != desired_position.dtype
        or floor_height.device != desired_position.device
    ):
        raise ValueError("swing loading needs matching [env,2,10,3] forces and [env] fixed floor heights")
    if not torch.isfinite(force_history_w).all() or not torch.isfinite(floor_height).all():
        raise ValueError("swing loading requires finite actual forces and floor heights")
    clearance = points[..., 2].amin(dim=-1) - floor_height[:, None]
    requested_air = ((clearance - AIRBORNE_ONSET_M) / (AIRBORNE_FULL_M - AIRBORNE_ONSET_M)).clamp(0, 1)
    load = (force_history_w[..., 2] / SWING_FORCE_NORMALIZATION_N).square().mean(dim=-1)
    return (requested_air * load).mean(dim=-1)


def received_swing_load_l2(env, command_name="motion"):
    import torch

    from gear_sonic.envs.mjlab.sonic_true23_root_feedback import _q10_body_position, _q10_body_quaternion

    if int(env.cfg.decimation) != PHYSICS_SUBSTEPS or abs(float(env.cfg.sim.mujoco.timestep) - 0.002) > 1e-12:
        raise ValueError("swing loading requires the unchanged ten 2ms physics substeps")
    command = env.command_manager.get_term(command_name)
    names = tuple(command.cfg.body_names)
    if any(names.count(name) != 1 for name in FOOT_BODY_NAMES):
        raise ValueError("swing loading requires exactly one named left and right ankle")
    indices = [names.index(name) for name in FOOT_BODY_NAMES]
    histories = [env.scene[name].data.force_history for name in FOOT_SENSOR_NAMES]
    if any(value is None or value.shape != (env.num_envs, 1, PHYSICS_SUBSTEPS, 3) for value in histories):
        raise ValueError("swing loading requires a complete ten-substep history from each named foot sensor")
    return swing_load_l2(
        _q10_body_position(command)[:, indices],
        _q10_body_quaternion(command)[:, indices],
        torch.cat(histories, dim=1),
        command._env.scene.env_origins[:, 2],
    )


def configure_swing_load(cfg):
    from mjlab.managers.reward_manager import RewardTermCfg
    from mjlab.sensor import ContactMatch, ContactSensorCfg

    # The trainer installs rewards before finalizing the pinned nominal scene.
    # Validate the finalized clock when the reward is executed, not here.
    sensors = tuple(cfg.scene.sensors or ())
    if any(sensor.name in FOOT_SENSOR_NAMES for sensor in sensors):
        raise ValueError("swing loading sensors already configured")
    cfg.scene.sensors = sensors + tuple(
        ContactSensorCfg(
            name=name,
            primary=ContactMatch(mode="body", pattern=body, entity="robot"),
            secondary=ContactMatch(mode="geom", pattern="floor"),
            fields=("force",),
            reduce="netforce",
            num_slots=1,
            history_length=PHYSICS_SUBSTEPS,
        )
        for name, body in zip(FOOT_SENSOR_NAMES, FOOT_BODY_NAMES, strict=True)
    )
    cfg.rewards["measured_swing_load_l2"] = RewardTermCfg(func=received_swing_load_l2, weight=SWING_LOAD_WEIGHT)
    return cfg
