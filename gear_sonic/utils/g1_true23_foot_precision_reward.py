"""SIM-only foot-placement reward overlay; physical gates remain unchanged."""

import torch

from gear_sonic.utils.g1_true23_bounded_progress import finite_vector

RATE = 100.0
STEP_DT = 0.02
FOOT_ERROR_SCALE_M = 0.15
EXISTING_FOOT_NORMALIZATION_M = 0.05


def foot_precision_contract():
    return dict(
        kind="native23_independent_foot_precision_bonus_v1",
        measured_quantity="mean_two_ankle_world_position_squared_error",
        source="existing_synchronized_post_step_world_cost_parts_before_usable_reset_substitution",
        formula="2/(1+existing_normalized_foot_cost/9) on non_done; zero on done",
        rate=RATE,
        step_dt=STEP_DT,
        foot_error_scale_m=FOOT_ERROR_SCALE_M,
        existing_foot_normalization_m=EXISTING_FOOT_NORMALIZATION_M,
        reward_bounds_per_control=[0.0, 2.0],
        reward_scale_is_not_acceptance_threshold=True,
        independent_of_root_and_upper_cost_denominator=True,
        original_rewards_termination_physics_and_reference_unchanged=True,
        changes_optimization_objective=True,
        hardware_authorized=False,
        deployment_ready=False,
    )


def foot_precision_bonus(normalized_foot_cost, terminated, timeouts):
    finite_vector(normalized_foot_cost, "normalized foot cost")
    if (normalized_foot_cost < 0).any():
        raise ValueError("foot precision cost must be nonnegative")
    for flags in (terminated, timeouts):
        if (
            flags.shape != normalized_foot_cost.shape
            or flags.dtype != torch.bool
            or flags.device != normalized_foot_cost.device
        ):
            raise ValueError("foot precision requires matching boolean done flags")
    value = RATE * STEP_DT / (1 + normalized_foot_cost / 9.0)
    return torch.where(terminated | timeouts, torch.zeros_like(value), value)


class FootPrecisionStep:
    """Append one reward term; no extra environment/command/observation calls."""

    def __init__(self, original):
        self.original = original

    @property
    def rows(self):
        return self.original.rows

    def capture(self):
        return self.original.capture()

    def __call__(self, actions):
        before = len(self.rows)
        obs, reward, terminated, timeouts, extras = self.original(actions)
        if len(self.rows) != before + 1:
            raise ValueError("foot precision requires exactly one recorded reward transition")
        row = self.rows[-1]
        if "foot_precision_bonus" in row:
            raise ValueError("foot precision cannot be applied twice")
        cost = row["world_cost_parts_after_including_reset_states"][:, 1].to(reward)
        bonus = foot_precision_bonus(cost, terminated, timeouts)
        result = reward + bonus
        finite_vector(result, "foot precision returned reward")
        row["pre_foot_precision_returned_reward"] = row["returned_reward"].clone()
        row["foot_precision_bonus"] = bonus.detach().cpu().clone()
        row["returned_reward"] = result.detach().cpu().clone()
        return obs, result, terminated, timeouts, extras
