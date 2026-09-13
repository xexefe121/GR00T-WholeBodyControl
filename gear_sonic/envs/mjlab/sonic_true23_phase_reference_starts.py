"""Explicit phase-balanced reference-state resets for SIM training only.

Keep one quarter of environments starting from standing; allocate the other
quarters to complete-reference entry, source and return phases. All sampled
episodes execute the remaining unchanged timeline. They never qualify the full
standing-start benchmark or dynamically validate a sampled initial state.
"""

import torch

from gear_sonic.envs.mjlab.sonic_true23_mixed_reference_starts import MixedReferenceRootFeedbackCommand
from gear_sonic.envs.mjlab.sonic_true23_root_feedback import RootFeedbackCurriculumCommand
from gear_sonic.utils.g1_true23_start_schedule import start_schedule_contract

PROFILE = "phase_balanced_reference_reset_v1"
PHASES = ("initial_standing", "acquisition_ramp", "source_motion", "return_ramp")


def phase_reference_anchors(env_ids, standing_anchors, first_anchors, phase_lengths, fractions):
    count = len(env_ids)
    if (
        env_ids.ndim != 1
        or standing_anchors.shape != (count,)
        or first_anchors.shape != (count, 3)
        or phase_lengths.shape != (count, 3)
        or fractions.shape != (count,)
        or torch.any(env_ids < 0)
        or torch.any(phase_lengths < 1)
        or torch.any(first_anchors < standing_anchors[:, None])
        or not torch.isfinite(fractions).all()
        or torch.any(fractions < 0)
        or torch.any(fractions >= 1)
    ):
        raise ValueError("phase reset requires finite fractions and complete ordered reference ranges")
    if torch.any(first_anchors[:, 1:] < first_anchors[:, :-1] + phase_lengths[:, :-1]):
        raise ValueError("phase reset ranges overlap or change source ordering")
    roles = env_ids.remainder(4)
    column = torch.clamp(roles - 1, min=0)
    selected_first = first_anchors.gather(1, column[:, None]).flatten()
    selected_length = phase_lengths.gather(1, column[:, None]).flatten()
    sampled = selected_first + torch.floor(fractions * selected_length).long()
    return torch.where(roles == 0, standing_anchors, sampled), roles


class PhaseReferenceRootFeedbackCommand(MixedReferenceRootFeedbackCommand):
    def __init__(self, cfg, env, *, spans):
        super().__init__(cfg, env, spans=spans)
        first, lengths = [], []
        for span in spans:
            phases = {phase["name"]: phase for phase in span["timeline"]["phases"]}
            first.append([span["start"] + phases[name]["frame_start"] - 1 for name in PHASES[1:]])
            lengths.append([phases[name]["requested_controls"] for name in PHASES[1:]])
        self._phase_first_anchors = torch.tensor(first, device=self.device)
        self._phase_lengths = torch.tensor(lengths, device=self.device)
        self._phase_reset_counts = torch.zeros(4, dtype=torch.long, device=self.device)
        self._phase_reset_bins = torch.zeros((4, 10), dtype=torch.long, device=self.device)
        self._initial_phase = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def _uniform_sampling(self, env_ids):
        # Bypass source-only sampling; all physical writes remain in the inherited
        # environment-reset routine with its mid-episode write guard.
        RootFeedbackCurriculumCommand._uniform_sampling(self, env_ids)
        choices = self._lifecycle_choice[env_ids]
        fractions = torch.rand(len(env_ids), device=self.device)
        anchors, roles = phase_reference_anchors(
            env_ids,
            self.time_steps[env_ids],
            self._phase_first_anchors[choices],
            self._phase_lengths[choices],
            fractions,
        )
        self.time_steps[env_ids] = anchors
        self._episode_initial_anchor[env_ids] = anchors
        self._episode_began_standing[env_ids] = roles == 0
        self._initial_phase[env_ids] = roles
        self.standing_reset_samples += int((roles == 0).sum().item())
        self.source_reset_samples += int((roles == 2).sum().item())
        self._source_reset_bin_counts += torch.bincount((fractions[roles == 2] * 10).long(), minlength=10)
        self._phase_reset_counts += torch.bincount(roles, minlength=4)
        bins = torch.where(roles == 0, 0, torch.floor(fractions * 10).long())
        self._phase_reset_bins += torch.bincount(roles * 10 + bins, minlength=40).reshape(4, 10)

    def curriculum_receipt(self):
        receipt = super().curriculum_receipt()
        suffixes = receipt.pop("completed_reference_suffixes_not_full_lifecycles")
        return {
            **receipt,
            "start_schedule": start_schedule_contract(PROFILE),
            "phase_reset_counts": dict(zip(PHASES, self._phase_reset_counts.cpu().tolist(), strict=True)),
            "phase_reset_decile_counts": dict(zip(PHASES, self._phase_reset_bins.cpu().tolist(), strict=True)),
            "current_episode_initial_phase": [PHASES[i] for i in self._initial_phase.cpu().tolist()],
            "completed_phase_suffixes_not_full_lifecycles": suffixes,
            "sampled_phase_initial_contact_or_dynamics_qualified": False,
        }
