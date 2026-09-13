"""Same native dynamics and recovery reset pools, one learned actor throughout."""
from gear_sonic.envs.mjlab.g1_true23_causal_dynamics import CausalNative23Env
from gear_sonic.envs.mjlab.g1_true23_balanced_causal_dynamics import BalancedCausalNative23Env


class SinglePolicyNative23Env(BalancedCausalNative23Env):
    def step(self,targets):
        # Reuse actual expert/recovery-state resets, but do not call the
        # supervisor's target blend or frozen balance inference during steps.
        return CausalNative23Env.step(self,targets)
