"""Scale existing training sensor noise while preserving its random draws.

Diagnostic/curriculum component only. No physical sensor or motor behavior.
Keep the full-amplitude path bit-identical and still sample at zero amplitude
so initial motion/reset sampling does not change merely from fewer RNG calls.
"""

from dataclasses import dataclass
import math

from mjlab.utils.noise import UniformNoiseCfg


def checked_scale(value):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise ValueError("sensor noise scale must be finite within 0..1")
    return float(value)


@dataclass
class ScaledUniformNoiseCfg(UniformNoiseCfg):
    amplitude_scale: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        self.amplitude_scale = checked_scale(self.amplitude_scale)
        if self.operation != "add":
            raise ValueError("sensor amplitude probe supports additive noise only")

    def apply(self, data):
        noisy = super().apply(data)
        if self.amplitude_scale == 1:
            return noisy
        if self.amplitude_scale == 0:
            return data
        return data + self.amplitude_scale * (noisy - data)


EXPECTED_TERMS = {
    ("tokenizer", "motion_anchor_ori_b"): (-0.05, 0.05),
    ("policy", "base_ang_vel"): (-0.2, 0.2),
    ("policy", "projected_gravity"): (-0.05, 0.05),
}


def scale_sensor_noise(cfg, scale):
    scale = checked_scale(scale)
    found, replacements = {}, []
    for group_name in ("tokenizer", "policy"):
        group = cfg.observations[group_name]
        if group.enable_corruption is not True:
            raise ValueError("sensor probe must preserve enabled corruption and its random draws")
        for name, term in group.terms.items():
            if term.noise is None:
                continue
            noise = term.noise
            key = (group_name, name)
            if (
                key not in EXPECTED_TERMS
                or type(noise) is not UniformNoiseCfg
                or noise.operation != "add"
                or (noise.n_min, noise.n_max) != EXPECTED_TERMS[key]
                or noise._tensor_cache
            ):
                raise ValueError("sensor probe requires the unchanged uncached SONIC noise terms")
            found[key] = (noise.n_min, noise.n_max)
            replacements.append(
                (
                    term,
                    ScaledUniformNoiseCfg(
                        n_min=noise.n_min, n_max=noise.n_max, operation=noise.operation, amplitude_scale=scale
                    ),
                )
            )
    if found != EXPECTED_TERMS:
        raise ValueError("sensor probe lacks an expected SONIC noise term")
    for term, replacement in replacements:
        term.noise = replacement
    return dict(
        amplitude_scale=scale,
        terms=[
            dict(group=group, term=name, original_range=list(bounds)) for (group, name), bounds in found.items()
        ],
        full_scale_uses_exact_original_output=True,
        zero_scale_still_consumes_original_random_draws=True,
        critic_noise_changed=False,
        command_reset_distribution_changed=False,
        action_noise_changed=False,
        hardware_authorized=False,
        deployment_ready=False,
    )
