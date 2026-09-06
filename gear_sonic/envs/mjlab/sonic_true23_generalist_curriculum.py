"""Standing-start command with reset-only sampling and no update-time writes."""

from __future__ import annotations

import torch

from gear_sonic.envs.mjlab.sonic_true23_causal_history import CausalHistoryMotionCommand


def advance_lifecycle_command(command):
    """Target-buffer updates only. At the boundary clamp, never resample."""
    current_pos = command.robot_anchor_pos_w.clone()
    current_quat = command.robot_anchor_quat_w.clone()
    resampled = command._causal_resampled.clone()
    advancing = ~resampled
    command.time_steps[advancing] = torch.minimum(
        command.time_steps[advancing] + 1, command._lifecycle_last_anchor[advancing]
    )
    if torch.any(advancing):
        command._causal_robot_anchor_pos_w[advancing] = command._causal_last_current_anchor_pos_w[advancing]
        command._causal_robot_anchor_quat_w[advancing] = command._causal_last_current_anchor_quat_w[advancing]
    indices = resampled.nonzero(as_tuple=False).flatten()
    if len(indices):
        pos, quat = command._virtual_anchor_at_q9(indices)
        command._causal_robot_anchor_pos_w[indices] = pos
        command._causal_robot_anchor_quat_w[indices] = quat
    command._refresh_targets_from_causal_anchor()
    command._causal_last_current_anchor_pos_w.copy_(current_pos)
    command._causal_last_current_anchor_quat_w.copy_(current_quat)
    command._causal_resampled.zero_()


class GeneralistCurriculumMotionCommand(CausalHistoryMotionCommand):
    """State writes inherited only under CommandTerm.reset()."""

    def __init__(self, cfg, env, *, spans):
        self._inside_environment_reset = False
        super().__init__(cfg, env)
        self._curriculum_spans = spans
        self._lifecycle_starts = torch.tensor(
            [r["start"] + r["timeline"]["prehistory_frames"] - 1 for r in spans], device=self.device
        )
        self._lifecycle_ends = torch.tensor([r["start"] + r["length"] - 2 for r in spans], device=self.device)
        if int(self._lifecycle_ends.max()) + 2 != self.motion.time_step_total:
            raise ValueError("lifecycle spans do not cover loaded motion")
        self._lifecycle_choice = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._lifecycle_last_anchor = self._lifecycle_ends[self._lifecycle_choice].clone()
        self._env_clip_stop = self._lifecycle_last_anchor + 1
        self.time_steps[:] = self._lifecycle_starts[0]
        self.reset_sample_count = 0
        self.completed_reference_timelines = 0
        self.command_update_count = 0

    def reset(self, env_ids):
        if self._inside_environment_reset:
            raise RuntimeError("nested lifecycle reset")
        self._inside_environment_reset = True
        try:
            return super().reset(env_ids)
        finally:
            self._inside_environment_reset = False

    def _resample_command(self, env_ids):
        if not self._inside_environment_reset:
            raise RuntimeError("lifecycle state resampling permitted only during environment reset")
        self.completed_reference_timelines += int(
            (self.time_steps[env_ids] >= self._lifecycle_last_anchor[env_ids]).sum().item()
        )
        super()._resample_command(env_ids)
        self.reset_sample_count += len(env_ids)

    def _uniform_sampling(self, env_ids):
        choice = torch.randint(len(self._curriculum_spans), (len(env_ids),), device=self.device)
        self._lifecycle_choice[env_ids] = choice
        self.time_steps[env_ids] = self._lifecycle_starts[choice]
        self._lifecycle_last_anchor[env_ids] = self._lifecycle_ends[choice]
        self._env_clip_stop[env_ids] = self._lifecycle_last_anchor[env_ids] + 1

    def _adaptive_sampling(self, env_ids):
        self._uniform_sampling(env_ids)

    def compute(self, dt):
        # Bypass CommandTerm's timed _resample(): that writes robot state even
        # when no episode terminated. Reset itself still samples normally.
        self._update_metrics()
        self._update_command()
        self.command_update_count += 1

    def _update_command(self):
        advance_lifecycle_command(self)

    def curriculum_receipt(self):
        return {
            "kind": "g1_native23_curriculum_command_mechanics_v1",
            "reset_sample_count": self.reset_sample_count,
            "command_update_count": self.command_update_count,
            "completed_reference_timelines": self.completed_reference_timelines,
            "active_reference_control_indices": (self.time_steps - self._lifecycle_starts[self._lifecycle_choice])
            .cpu()
            .tolist(),
            "full_lifecycle_tracking_verified": False,
            "physical_state_write_instrumentation_complete": False,
            "deployment_ready": False,
            "hardware_authorized": False,
        }


def install_curriculum_command(spans):
    from gear_sonic.envs.mjlab import sonic_true23_causal_history as task

    def build(cfg, env):
        return GeneralistCurriculumMotionCommand(cfg, env, spans=spans)

    task.CausalHistoryMotionCommandCfg.build = build


def configure_curriculum_environment(cfg, spans):
    longest = max(r["timeline"]["total_requested_controls"] for r in spans)
    cfg.episode_length_s = (longest + 2) * 0.02
    command = cfg.commands["motion"]
    command.sampling_mode = "uniform"
    command.pose_range = {}
    command.velocity_range = {}
    command.joint_position_range = (0.0, 0.0)
    # Not used by our compute(), but make the nominal intent explicit.
    command.resampling_time_range = (cfg.episode_length_s + 1,) * 2
    return cfg
