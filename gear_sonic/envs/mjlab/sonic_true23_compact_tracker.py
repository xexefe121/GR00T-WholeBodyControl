"""Add explicit read-only goals to the unchanged native23 training environment."""

import torch

from gear_sonic.envs.mjlab import sonic_true23_original_intent as intent
from gear_sonic.envs.mjlab import sonic_true23_root_feedback as root
from gear_sonic.utils.g1_true23_compact_features import make_goal


def native_goal(env, spec):
    command = env.command_manager.get_term("motion")
    q0, q1 = intent._indices(command)
    desired, _ = root.current_root_reference(command)
    feet = root._body_indices(command, ("left_ankle_roll_link", "right_ankle_roll_link"))
    goal_feet = command.motion.body_pos_w[q1][:, feet] + env.scene.env_origins[:, None, :]
    return make_goal(
        command.motion.joint_pos[q1],
        (command.motion.joint_pos[q1] - command.motion.joint_pos[q0]) / 0.02,
        root.root_feedback_observation(env),
        command.motion.body_quat_w[q1, command.motion_anchor_body_index],
        command.robot_anchor_quat_w,
        intent.original_critic_vr(env, spec),
        goal_feet - command.robot_body_pos_w[:, feet],
        torch.stack(
            (
                desired[:, 2] - env.scene.env_origins[:, 2],
                command.robot_anchor_pos_w[:, 2] - env.scene.env_origins[:, 2],
            ),
            -1,
        ),
    )


def configure(cfg, spec):
    from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg

    if "native_goal" in cfg.observations:
        raise ValueError("compact goal already installed")
    cfg.observations["native_goal"] = ObservationGroupCfg(
        terms={"goal90": ObservationTermCfg(func=native_goal, params=dict(spec=spec))},
        concatenate_terms=True,
        enable_corruption=False,
        nan_policy="error",
    )
    return cfg
