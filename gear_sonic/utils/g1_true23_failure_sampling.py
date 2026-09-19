"""Clip-contained failure-weighted source reset sampling, never motion editing."""

import torch

BIN_CONTROLS = 50
ALPHA = 0.01
UNIFORM_MIX = 0.20
BACKWARD_WEIGHTS = (1.0, 0.5, 0.25)


def sampling_contract():
    return dict(
        name="failure_weighted_source_reset_v1",
        standing_environments="env_id_modulo_4_equals_0_unchanged_full_lifecycle_start",
        clip_selection="unchanged_uniform_selected_training_clip",
        other_environments="inverse_CDF_failure_weighted_q1_within_complete_source",
        failure_source="existing_true_termination_at_pre_reset_received_q0_anchor",
        timeout_is_failure=False,
        failures_outside_original_source_ignored=True,
        bin_controls=BIN_CONTROLS,
        ema_alpha_per_actual_control=ALPHA,
        backward_failure_kernel=list(BACKWARD_WEIGHTS),
        uniform_per_frame_mixture=UNIFORM_MIX,
        no_failures_fallback="bit_identical_original_uniform_fraction_mapping",
        sampler_extra_random_draws=0,
        uniform_support_on_every_original_source_frame=True,
        physical_state_initialization="unchanged_reference_q1_at_environment_reset_only",
        reference_after_reset="every_remaining_original_frame_and_return_without_skips",
        sampled_suffix_completion_is_full_lifecycle_completion=False,
        mid_episode_pose_writes=False,
        reference_reward_actor_physics_and_limits_changed=False,
        evaluation_requires_full_standing_start_lifecycle=True,
        deployment_ready=False,
        hardware_authorized=False,
    )


class FailureBinSampler:
    def __init__(self, source_first, source_lengths, *, device):
        self.first = torch.as_tensor(source_first, device=device, dtype=torch.long).clone()
        self.lengths = torch.as_tensor(source_lengths, device=device, dtype=torch.long).clone()
        if self.first.ndim != 1 or self.first.shape != self.lengths.shape or len(self.first) == 0:
            raise ValueError("sampler requires nonempty matching source ranges")
        if (self.first < 0).any() or (self.lengths <= 0).any():
            raise ValueError("invalid source range")
        if (self.first[1:] < self.first[:-1] + self.lengths[:-1]).any():
            raise ValueError("source ranges overlap or are unordered")
        self.bin_count = (self.lengths + BIN_CONTROLS - 1) // BIN_CONTROLS
        self.width = int(self.bin_count.max())
        offsets = torch.arange(self.width, device=device) * BIN_CONTROLS
        self.bin_lengths = torch.clamp(self.lengths[:, None] - offsets, min=0, max=BIN_CONTROLS)
        self.ema = torch.zeros(self.bin_lengths.shape, dtype=torch.float64, device=device)
        self.total_failures = torch.zeros_like(self.bin_lengths)
        self.controls = 0

    def observe(self, anchors, terminated):
        if anchors.ndim != 1 or anchors.dtype != torch.long or terminated.shape != anchors.shape:
            raise ValueError("failure observation requires matching anchor and done vectors")
        if (
            terminated.dtype != torch.bool
            or anchors.device != self.first.device
            or terminated.device != anchors.device
        ):
            raise ValueError("failure observation device or boolean type differs")
        offsets = anchors[:, None] - self.first
        inside = (offsets >= 0) & (offsets < self.lengths)
        eligible = inside.any(-1) & terminated
        clips = inside.long().argmax(-1)
        bins = torch.div(offsets.gather(1, clips[:, None]).flatten(), BIN_CONTROLS, rounding_mode="floor")
        selected = clips[eligible] * self.width + bins[eligible]
        counts = torch.bincount(selected, minlength=self.ema.numel()).reshape(self.ema.shape)
        self.ema.mul_(1 - ALPHA).add_(counts, alpha=ALPHA)
        self.total_failures += counts
        self.controls += 1
        return counts

    def probabilities(self):
        # A failure also gives weight to earlier source bins, never across a clip
        # seam. No prediction or future reference is added to policy observations.
        score = torch.zeros_like(self.ema)
        for shift, weight in enumerate(BACKWARD_WEIGHTS):
            if shift < self.width:
                score[:, : self.width - shift] += weight * self.ema[:, shift:]
        score = torch.where(self.bin_lengths > 0, score, 0)
        uniform = self.bin_lengths.double() / self.lengths[:, None]
        sums = score.sum(-1, keepdim=True)
        adaptive = score / sums.clamp_min(torch.finfo(torch.float64).tiny)
        result = torch.where(sums > 0, UNIFORM_MIX * uniform + (1 - UNIFORM_MIX) * adaptive, uniform)
        if not torch.isfinite(result).all() or (result < UNIFORM_MIX * uniform).any():
            raise ValueError("sampler lost finite uniform source support")
        return result

    def sample(self, choices, fractions):
        if choices.ndim != 1 or fractions.shape != choices.shape or choices.dtype != torch.long:
            raise ValueError("sampler requires matching clip choices and uniform fractions")
        if choices.device != self.first.device or fractions.device != choices.device:
            raise ValueError("sampler inputs must stay on simulator device")
        if (choices < 0).any() or (choices >= len(self.first)).any():
            raise ValueError("sampler clip choice outside source set")
        if (
            not fractions.is_floating_point()
            or not torch.isfinite(fractions).all()
            or (fractions < 0).any()
            or (fractions >= 1).any()
        ):
            raise ValueError("sampler fraction must be finite in[0,1)")
        probabilities = self.probabilities()[choices]
        cdf = probabilities.cumsum(-1)
        final_or_padding = (
            torch.arange(self.width, device=choices.device)[None] >= self.bin_count[choices, None] - 1
        )
        cdf = torch.where(final_or_padding, 1.0, cdf)
        indices = torch.searchsorted(cdf.contiguous(), fractions.double()[:, None], right=True).flatten()
        probability = probabilities.gather(1, indices[:, None]).flatten()
        previous = torch.where(indices > 0, cdf.gather(1, (indices - 1).clamp_min(0)[:, None]).flatten(), 0)
        width = self.bin_lengths[choices, indices]
        within = torch.floor((fractions.double() - previous) / probability * width).long()
        within = torch.minimum(within.clamp_min(0), width - 1)
        adaptive = self.first[choices] + indices * BIN_CONTROLS + within
        # Preserve the exact old float32 multiplication and random draw count
        # during initialization or for a clip with no observed source failures.
        uniform = self.first[choices] + torch.floor(fractions * self.lengths[choices]).long()
        result = torch.where(self.ema[choices].sum(-1) > 0, adaptive, uniform)
        if (result < self.first[choices]).any() or (result >= self.first[choices] + self.lengths[choices]).any():
            raise ValueError("sample escaped its original source")
        return result

    def state(self):
        return dict(
            contract=sampling_contract(),
            controls=self.controls,
            first=self.first.detach().cpu().clone(),
            lengths=self.lengths.detach().cpu().clone(),
            ema=self.ema.detach().cpu().clone(),
            total_failures=self.total_failures.detach().cpu().clone(),
        )
