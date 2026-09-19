"""SIM-only joint world-task quality reward, separate from bounded-base shaping."""

import torch

from gear_sonic.utils.g1_true23_bounded_progress import finite_vector

BONUS_RATE = 100.0
STEP_DT = 0.02


def quality_contract():
    return dict(
        kind="native23_joint_world_task_quality_bonus_v1",
        rate=BONUS_RATE,
        step_dt=STEP_DT,
        error="10*root_position_error_squared+normalized_feet_world_cost+original29_hand_head_world_cost",
        bonus="100*0.02/(1+error_after) on non-done transitions; zero on termination or timeout",
        source="existing synchronized post-step world potential before any usable reset substitution",
        non_done_bonus_bounds=[0.0, 2.0],
        done_bonus=0.0,
        post_reset_potential_rewarded=False,
        all_existing_base_rewards_and_potential_shaping_unchanged=True,
        actor_inputs_weights_initialization_physics_and_failures_unchanged=True,
        changes_optimization_objective=True,
        reward_backpropagation_through_physics_claimed=False,
        training_resume_supported=False,
        hardware_authorized=False,
        deployment_ready=False,
    )


def world_quality_bonus(phi_after, terminated, timeouts):
    finite_vector(phi_after, "post-step potential")
    if (phi_after > 1e-6).any():
        raise ValueError("world potential must be nonpositive")
    for flags in (terminated, timeouts):
        if flags.shape != phi_after.shape or flags.dtype != torch.bool or flags.device != phi_after.device:
            raise ValueError("quality bonus needs matching boolean done flags")
    # Reset potentials may be present in the capture, but never earn a bonus.
    result = torch.where(
        terminated | timeouts, torch.zeros_like(phi_after), BONUS_RATE * STEP_DT * torch.exp(phi_after)
    )
    if (result < 0).any() or (result > 2.0 + 3e-6).any() or not torch.isfinite(result).all():
        raise ValueError("quality bonus exceeds declared finite bounds")
    return result


class WorldQualityStep:
    """Wrap the original reward step without another observation/physics call."""

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
            raise ValueError("quality step needs exactly one original reward transition")
        row = self.rows[-1]
        if "world_quality_bonus" in row:
            raise ValueError("quality reward may not be applied twice")
        # All original captured rows are CPU copies; arithmetic stays on the
        # original reward device and dtype, with explicit captured result bytes.
        phi = row["phi_after_including_reset_states"].to(device=reward.device, dtype=reward.dtype)
        bonus = world_quality_bonus(phi, terminated, timeouts)
        result = reward + bonus
        row["pre_quality_returned_reward"] = row["returned_reward"].clone()
        row["world_quality_bonus"] = bonus.detach().cpu().clone()
        row["returned_reward"] = result.detach().cpu().clone()
        return obs, result, terminated, timeouts, extras
