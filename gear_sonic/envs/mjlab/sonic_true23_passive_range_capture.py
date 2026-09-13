"""Passive SIM diagnostic of real physics substeps and existing failures.

No new termination is installed. In particular, this observer does not call a
termination predicate twice or substitute post-reset states for terminal states.
The caller explicitly brackets the unchanged vector-environment step with run().
"""

import numpy as np
import torch

from gear_sonic.envs.mjlab.sonic_true23_measured_range_failure import TERM_NAME, MeasuredRangeLatch
from gear_sonic.utils.g1_23dof_contract import HARDWARE_23_JOINT_NAMES
from gear_sonic.utils.g1_23dof_safe_target_transform import (
    SAFE_TARGET_HARD_LOWER_HARDWARE,
    SAFE_TARGET_HARD_UPPER_HARDWARE,
)


def snapshot(value):
    return value.detach().cpu().clone()


class PassiveRangeCapture:
    def __init__(self, env):
        if hasattr(env, "_native_passive_range_capture"):
            raise ValueError("passive range capture already installed")
        if hasattr(env, "_native_measured_range_latch") or TERM_NAME in env.termination_manager.active_terms:
            raise ValueError("passive diagnostic requires unchanged training failures")
        if abs(env.physics_dt - 0.002) > 1e-12 or abs(env.step_dt - 0.02) > 1e-12:
            raise ValueError("passive capture requires unchanged500/50Hz physics")
        model = env.sim.mj_model
        names = tuple(model.joint(i).name for i in range(1, model.njnt))
        # MJLab attaches this single robot under its explicit scene namespace.
        # Bare names are accepted only for the standalone native-model fixture.
        expected = tuple(HARDWARE_23_JOINT_NAMES)
        allowed = (expected, tuple("robot/" + name for name in expected))
        if model.nq != 30 or model.nv != 29 or names not in allowed:
            raise ValueError("passive capture requires native23 physical joint order")
        np.testing.assert_array_equal(model.jnt_range[1:, 0], SAFE_TARGET_HARD_LOWER_HARDWARE)
        np.testing.assert_array_equal(model.jnt_range[1:, 1], SAFE_TARGET_HARD_UPPER_HARDWARE)
        self.env = env
        self.latch = MeasuredRangeLatch(env.num_envs, env.device)
        self.term_names = tuple(env.termination_manager.active_terms)
        self.term_timeouts = tuple(env.termination_manager.get_term_cfg(n).time_out for n in self.term_names)
        self.rows, self.pending = [], None
        physics_step, compute = env.sim.step, env.termination_manager.compute

        def physics(*args, **kwargs):
            result = physics_step(*args, **kwargs)
            if self.pending is not None:
                if self.pending["termination_calls"]:
                    raise ValueError("physics unexpectedly ran after termination computation")
                self.latch.observe(env.sim.data.qpos)
                for key in ("qpos", "qvel"):
                    self.pending["physics_" + key].append(snapshot(getattr(env.sim.data, key)))
            return result

        def termination(*args, **kwargs):
            result = compute(*args, **kwargs)
            if self.pending is not None:
                if self.latch.calls != 10 or self.pending["termination_calls"]:
                    raise ValueError("termination capture requires exactly ten actual substeps")
                manager = env.termination_manager
                if tuple(manager.active_terms) != self.term_names:
                    raise ValueError("existing failure inventory changed")
                self.pending["termination_calls"] += 1
                self.pending["term_flags"] = torch.stack(
                    [snapshot(manager.get_term(n)) for n in self.term_names], -1
                )
                self.pending["terminated"] = snapshot(manager.terminated)
                self.pending["timeouts"] = snapshot(manager.time_outs)
                self.pending["done"] = snapshot(result)
                self.pending["range_failure"] = snapshot(self.latch.failure)
                self.pending["max_excess"] = snapshot(self.latch.max_excess)
                # These are direct qpos/qvel BEFORE the environment reset block.
                for key in ("qpos", "qvel"):
                    self.pending["terminal_" + key] = snapshot(getattr(env.sim.data, key))
            return result

        self.original_physics, self.original_compute = physics_step, compute
        self.physics_wrapper, self.compute_wrapper = physics, termination
        env.sim.step, env.termination_manager.compute = physics, termination
        env._native_passive_range_capture = self

    def run(self, control_step, *args, **kwargs):
        if self.pending is not None:
            raise ValueError("passive control capture nested")
        self.latch.begin()
        self.pending = dict(physics_qpos=[], physics_qvel=[], termination_calls=0)
        for key in ("qpos", "qvel"):
            self.pending["pre_" + key] = snapshot(getattr(self.env.sim.data, key))
        try:
            result = control_step(*args, **kwargs)
            self.latch.end()
            if self.pending["termination_calls"] != 1:
                raise ValueError("passive capture missed existing termination computation")
            for key in ("qpos", "qvel"):
                self.pending["physics_" + key] = torch.stack(self.pending["physics_" + key])
                self.pending["post_reset_" + key] = snapshot(getattr(self.env.sim.data, key))
                if not torch.equal(self.pending["physics_" + key][-1], self.pending["terminal_" + key]):
                    raise ValueError("captured terminal state differs from final physics state")
            self.rows.append(self.pending)
            return result
        finally:
            self.pending = None
            self.latch.active = False

    def capture(self):
        if self.pending is not None or not self.rows:
            raise ValueError("passive capture requires completed controls")
        return {
            key: torch.stack([row[key] for row in self.rows]).numpy()
            for key in self.rows[0]
            if key != "termination_calls"
        }
