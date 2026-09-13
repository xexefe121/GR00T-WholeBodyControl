"""Append owned pre-update context without changing the BFM history clock."""
import numpy as np

WIDTHS = {'actions':23, 'base_ang_vel':3, 'dof_pos':23, 'dof_vel':23, 'projected_gravity':3}


def incoming_context(seed):
    prior = seed.previous_action
    if prior.shape != (23,) or prior.dtype != np.float32 or not np.isfinite(prior).all():
        raise ValueError('Finite incoming raw prior float32[23] required')
    named = seed.history.data
    if set(named) != set(WIDTHS):
        raise ValueError('Complete incoming named history required')
    for name, width in WIDTHS.items():
        value = named[name]
        if value.shape != (4, width) or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError('Finite incoming four-sample history required: '+name)
    return np.concatenate([prior] + [named[name].reshape(-1) for name in sorted(WIDTHS)])


class CausalFeatures:
    def __init__(self, current, seed, condition, context_mean):
        if condition not in ('causal', 'blinded'):
            raise ValueError('Explicit selected context-study condition required')
        if context_mean.shape != (323,) or context_mean.dtype != np.float32 or not np.isfinite(context_mean).all():
            raise ValueError('Frozen context mean float32[323] required')
        self.current, self.seed, self.condition = current, seed, condition
        self.context_mean = context_mean.copy()

    def __call__(self, qpos, qvel, frame):
        current = self.current(qpos, qvel, frame)
        if current.shape != (1000,) or current.dtype != np.float32 or not np.isfinite(current).all():
            raise ValueError('Original finite current feature float32[1000] required')
        # propose() only stages before_update on a deepcopy. Seed history remains
        # the incoming lagged history here and changes only after commit().
        context = incoming_context(self.seed)
        if self.condition == 'blinded':
            context = self.context_mean
        return np.concatenate((current, context))
