"""Teach source-motion tracking with reset-time reference-state initialization.

One quarter of environments retain complete standing starts. Other environments
start at a sampled source frame, then execute every remaining frame through the
standing return. Training suffixes never qualify the full-lifecycle CPU test.
"""

from __future__ import annotations

import torch

from gear_sonic.envs.mjlab.sonic_true23_root_feedback import RootFeedbackCurriculumCommand
from gear_sonic.utils.g1_true23_start_schedule import start_schedule_contract


def mixed_reference_anchors(env_ids, standing_anchors, source_first_anchors, source_lengths, fractions):
    if any(
        value.shape != env_ids.shape
        for value in (standing_anchors, source_first_anchors, source_lengths, fractions)
    ):
        raise ValueError("mixed reference reset arrays must share the environment shape")
    if (
        torch.any(env_ids < 0)
        or torch.any(source_lengths < 1)
        or torch.any(source_first_anchors < standing_anchors)
        or not torch.isfinite(fractions).all()
        or torch.any(fractions < 0)
        or torch.any(fractions >= 1)
    ):
        raise ValueError("invalid mixed reference reset indices or fractions")
    standing = env_ids.remainder(4) == 0
    sampled = source_first_anchors + torch.floor(fractions * source_lengths).long()
    return torch.where(standing, standing_anchors, sampled), standing


class MixedReferenceRootFeedbackCommand(RootFeedbackCurriculumCommand):
    def __init__(self, cfg, env, *, spans):
        super().__init__(cfg, env, spans=spans)
        self._source_first_anchors = torch.tensor(
            [r["start"] + r["timeline"]["source_start_frame"] - 1 for r in spans], device=self.device
        )
        self._source_lengths = torch.tensor([r["timeline"]["source_frames"] for r in spans], device=self.device)
        self._episode_began_standing = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._episode_initial_anchor = self.time_steps.clone()
        self.standing_reset_samples = 0
        self.source_reset_samples = 0
        self.completed_reference_suffixes = 0
        self._source_reset_bin_counts = torch.zeros(10, dtype=torch.long, device=self.device)

    def _uniform_sampling(self, env_ids):
        super()._uniform_sampling(env_ids)
        choice = self._lifecycle_choice[env_ids]
        fractions = torch.rand(len(env_ids), device=self.device)
        anchors, standing = mixed_reference_anchors(
            env_ids,
            self.time_steps[env_ids],
            self._source_first_anchors[choice],
            self._source_lengths[choice],
            fractions,
        )
        self.time_steps[env_ids] = anchors
        self._episode_initial_anchor[env_ids] = anchors
        self._episode_began_standing[env_ids] = standing
        self.standing_reset_samples += int(standing.sum().item())
        self.source_reset_samples += int((~standing).sum().item())
        self._source_reset_bin_counts += torch.bincount((fractions[~standing] * 10).long(), minlength=10)

    def _resample_command(self, env_ids):
        if not self._inside_environment_reset:
            raise RuntimeError("mixed reference initialization permitted only during environment reset")
        suffixes = int(
            (
                (self.time_steps[env_ids] >= self._lifecycle_last_anchor[env_ids])
                & ~self._episode_began_standing[env_ids]
            )
            .sum()
            .item()
        )
        super()._resample_command(env_ids)
        # The base counter includes every end-of-reference reset. Keep actual
        # standing-start completions distinct from sampled training suffixes.
        self.completed_reference_timelines -= suffixes
        self.completed_reference_suffixes += suffixes

    def curriculum_receipt(self):
        return {
            **super().curriculum_receipt(),
            "start_schedule": start_schedule_contract("mixed_reference_reset_v1"),
            "standing_reset_samples": self.standing_reset_samples,
            "sampled_source_reset_samples": self.source_reset_samples,
            "completed_reference_suffixes_not_full_lifecycles": self.completed_reference_suffixes,
            "source_reset_decile_counts": self._source_reset_bin_counts.cpu().tolist(),
            "current_episode_began_standing": self._episode_began_standing.cpu().tolist(),
            "current_episode_initial_reference_control_indices": (
                self._episode_initial_anchor - self._lifecycle_starts[self._lifecycle_choice]
            )
            .cpu()
            .tolist(),
        }
