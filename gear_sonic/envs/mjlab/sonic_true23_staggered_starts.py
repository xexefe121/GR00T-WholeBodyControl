"""Phase-diverse training batches without initializing robots inside a dance."""

from __future__ import annotations

import torch

from gear_sonic.envs.mjlab.sonic_true23_generalist_curriculum import advance_lifecycle_command
from gear_sonic.envs.mjlab.sonic_true23_root_feedback import RootFeedbackCurriculumCommand
from gear_sonic.utils.g1_true23_start_schedule import start_schedule_contract


def standing_start_delays(env_ids, lifecycle_controls, num_envs):
    if num_envs < 1 or torch.any(env_ids < 0) or torch.any(env_ids >= num_envs):
        raise ValueError("standing-start schedule requires valid environment indices")
    if torch.any(lifecycle_controls < 1):
        raise ValueError("standing-start schedule requires a complete lifecycle")
    return torch.div(env_ids * lifecycle_controls, num_envs, rounding_mode="floor")


def advance_staggered_command(command):
    # Command priming after reset is not a physical control interval.
    held = (command._start_hold_remaining > 0) & ~command._causal_resampled
    advance_lifecycle_command(command, hold_reference=held)
    command._start_hold_remaining[held] -= 1
    command._start_hold_executed[held] += 1


def assign_standing_start_delays(command, env_ids):
    controls = command._lifecycle_ends[command._lifecycle_choice[env_ids]] - command.time_steps[env_ids] + 1
    first_reset = ~command._start_schedule_initialized[env_ids]
    delay = standing_start_delays(env_ids, controls, command.num_envs)
    command._start_hold_remaining[env_ids] = torch.where(first_reset, delay, 0)
    first_ids = env_ids[first_reset]
    command._start_hold_assigned[first_ids] = delay[first_reset]
    command._start_schedule_initialized[env_ids] = True


class StaggeredRootFeedbackCurriculumCommand(RootFeedbackCurriculumCommand):
    def __init__(self, cfg, env, *, spans):
        super().__init__(cfg, env, spans=spans)
        self._start_hold_remaining = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._start_hold_assigned = torch.zeros_like(self._start_hold_remaining)
        self._start_hold_executed = torch.zeros_like(self._start_hold_remaining)
        self._start_schedule_initialized = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        # Holding a moving reference would change its intended velocity. Fail
        # before learning unless the whole initial history/proof is stationary.
        for row in spans:
            start = row["start"]
            stop = start + row["timeline"]["prehistory_frames"] + 1
            for name in ("joint_pos", "body_pos_w", "body_quat_w"):
                prefix = getattr(self.motion, name)[start:stop]
                if not torch.allclose(prefix, prefix[:1].expand_as(prefix), atol=1e-6, rtol=0):
                    raise ValueError("staggered starts require stationary standing prehistory and proof")

    def _uniform_sampling(self, env_ids):
        super()._uniform_sampling(env_ids)
        assign_standing_start_delays(self, env_ids)

    def _update_command(self):
        advance_staggered_command(self)

    def curriculum_receipt(self):
        return {
            **super().curriculum_receipt(),
            "start_schedule": start_schedule_contract("staggered_standing_start_v1"),
            "assigned_standing_hold_controls": self._start_hold_assigned.cpu().tolist(),
            "remaining_standing_hold_controls": self._start_hold_remaining.cpu().tolist(),
            "executed_standing_hold_controls": self._start_hold_executed.cpu().tolist(),
            "distinct_current_reference_control_indices": int(
                torch.unique(self.time_steps - self._lifecycle_starts[self._lifecycle_choice]).numel()
            ),
        }


def configure_staggered_start_timeout(cfg, spans):
    longest = max(row["timeline"]["total_requested_controls"] for row in spans)
    # Every delay is strictly shorter than one lifecycle; keep the entire
    # selected timeline available even to the last-starting environment.
    cfg.episode_length_s = (2 * longest + 2) * 0.02
    cfg.commands["motion"].resampling_time_range = (cfg.episode_length_s + 1,) * 2
    return cfg
