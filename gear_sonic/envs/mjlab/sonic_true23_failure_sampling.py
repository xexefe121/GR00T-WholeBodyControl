"""New reset distribution; existing simulation/termination/reset mechanics stay."""

import numpy as np
import torch

from gear_sonic.envs.mjlab.sonic_true23_mixed_reference_starts import MixedReferenceRootFeedbackCommand
from gear_sonic.envs.mjlab.sonic_true23_root_feedback import RootFeedbackCurriculumCommand
from gear_sonic.utils.g1_true23_failure_sampling import FailureBinSampler, sampling_contract


def cpu(value):
    return value.detach().cpu().clone()


class FailureAdaptiveRootFeedbackCommand(MixedReferenceRootFeedbackCommand):
    def __init__(self, cfg, env, *, spans):
        super().__init__(cfg, env, spans=spans)
        self.failure_sampler = FailureBinSampler(
            self._source_first_anchors, self._source_lengths, device=self.device
        )
        self.failure_observations, self.failure_reset_samples = [], []
        self._failure_monitor_installed = False

    def _uniform_sampling(self, env_ids):
        # Keep the original uniform clip draw and exactly one fraction per env.
        # Only the fraction -> source frame mapping changes after actual failures.
        RootFeedbackCurriculumCommand._uniform_sampling(self, env_ids)
        choices = self._lifecycle_choice[env_ids]
        fractions = torch.rand(len(env_ids), device=self.device)
        standing = env_ids.remainder(4) == 0
        source_anchors = self.failure_sampler.sample(choices, fractions)
        anchors = torch.where(standing, self.time_steps[env_ids], source_anchors)
        self.time_steps[env_ids] = anchors
        self._episode_initial_anchor[env_ids] = anchors
        self._episode_began_standing[env_ids] = standing
        self.standing_reset_samples += int(standing.sum())
        self.source_reset_samples += int((~standing).sum())
        actual_fraction = (anchors - self._source_first_anchors[choices]).double() / self._source_lengths[choices]
        self._source_reset_bin_counts += torch.bincount((actual_fraction[~standing] * 10).long(), minlength=10)
        self.failure_reset_samples.append(
            dict(
                control=self.failure_sampler.controls,
                **{
                    name: cpu(value)
                    for name, value in dict(
                        env_ids=env_ids,
                        choices=choices,
                        fractions=fractions,
                        anchors=anchors,
                        standing=standing,
                        probabilities=self.failure_sampler.probabilities(),
                    ).items()
                },
            )
        )

    def install_failure_monitor(self):
        if self._failure_monitor_installed:
            raise ValueError("failure sampler monitor already installed")
        manager = self._env.termination_manager
        original_compute = manager.compute

        def compute(*args, **kwargs):
            result = original_compute(*args, **kwargs)
            if self.failure_sampler.controls + 1 != self._env.common_step_counter:
                raise ValueError("failure sampler requires one observation per real control")
            anchors, terminated = self.time_steps.clone(), manager.terminated.clone()
            counts = self.failure_sampler.observe(anchors, terminated)
            self.failure_observations.append(
                dict(
                    control=self.failure_sampler.controls,
                    anchors=cpu(anchors),
                    terminated=cpu(terminated),
                    counts=cpu(counts),
                    ema=cpu(self.failure_sampler.ema),
                )
            )
            return result

        manager.compute = compute
        self._failure_compute = compute
        self._failure_monitor_installed = True

    def require_sampling_runtime(self):
        if (
            not self._failure_monitor_installed
            or self._env.termination_manager.compute is not self._failure_compute
        ):
            raise ValueError("actual failure sampling monitor is missing or changed")
        if self.cfg.sampling_mode != "uniform":
            raise ValueError("unexpected inherited sampler dispatch")
        return self.failure_sampler.state()

    def curriculum_receipt(self):
        return {
            **super().curriculum_receipt(),
            "start_schedule": sampling_contract(),
            "failure_sampling_controls": self.failure_sampler.controls,
            "observed_source_failures": int(self.failure_sampler.total_failures.sum()),
            "failure_sampling_probabilities": self.failure_sampler.probabilities().cpu().tolist(),
        }

    def capture_failure_sampling(self):
        rows, samples = self.failure_observations, self.failure_reset_samples
        arrays = {
            "observation_" + key: torch.stack([row[key] for row in rows]).numpy() if rows else np.empty((0,))
            for key in ("anchors", "terminated", "counts", "ema")
        }
        arrays["observation_control"] = np.asarray([row["control"] for row in rows], dtype=np.int64)
        arrays["sample_offsets"] = np.cumsum([0] + [len(row["env_ids"]) for row in samples], dtype=np.int64)
        arrays["sample_control"] = np.asarray([row["control"] for row in samples], dtype=np.int64)
        arrays["sample_probabilities"] = torch.stack([row["probabilities"] for row in samples]).numpy()
        for key in ("env_ids", "choices", "fractions", "anchors", "standing"):
            arrays["sample_" + key] = torch.cat([row[key] for row in samples]).numpy()
        return arrays
