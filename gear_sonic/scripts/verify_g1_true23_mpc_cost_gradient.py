"""Independent directional check of native MPC cost and knot derivatives."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from gear_sonic.utils.g1_true23_mjbatch_model import position_servo_copy
from gear_sonic.utils.g1_true23_mjbatch_mpc import Native23Tracker, load_native_bundle, sha256


def run(args):
    native, contract, motion, _, _ = load_native_bundle(args.bundle, "walk002")
    servo = position_servo_copy(native, *[np.asarray(contract[k]) for k in ("kp", "kd", "native_effort")])
    tracker = Native23Tracker(servo, contract, motion, horizon=4, threads=1)
    rng = np.random.default_rng(913)

    def perturb(state, delta):
        out = state.copy()
        out[:30] = tracker.integrate(out[None, :30], delta[None, :29])[0]
        out[30:] += delta[29:]
        return out

    def scalar_cost(t, state, target):
        if t == 0:
            return float(tracker.cost(t, state[None], target[None])[0])
        tracker.line_fields[0][:] = state[:30]
        tracker.line_fields[1][:] = state[30:]
        tracker.line.forward()
        # Cost line buffers intentionally have all nine line-search rows.
        return float(tracker.cost(t, np.tile(state, (9, 1)), np.tile(target, (9, 1)))[0])

    checks = []
    alignments = []
    for start in (10, 400, 1000):
        tracker.window(start)
        states = tracker.states[start : start + 5].copy()
        for i in range(5):
            delta = rng.normal(size=58) * np.r_[np.full(29, .015), np.full(29, .08)]
            states[i] = perturb(states[i], delta)
        targets = np.clip(tracker.target_reference(np.arange(4)) + rng.normal(0, .03, (4, 23)), tracker.lo, tracker.hi)
        tracker.linearize(states, targets)
        lx, lxx, lu, luu = tracker.expand(states, targets)
        for t in (0, 2, 4):
            target = targets[min(t, 3)]
            for direction_index in range(8):
                direction = rng.normal(size=58)
                direction /= np.linalg.norm(direction)
                eps = 2e-6
                fd = (scalar_cost(t, perturb(states[t], eps * direction), target)
                      - scalar_cost(t, perturb(states[t], -eps * direction), target)) / (2 * eps)
                predicted = float(lx[t] @ direction)
                error = abs(fd - predicted)
                checks.append(dict(start=start, t=t, direction=direction_index, kind="state",
                                   finite_difference=fd, expanded=predicted, absolute_error=error,
                                   scaled_error=error / max(1, abs(fd))))
                if t < 4:
                    control_direction = rng.normal(size=23)
                    control_direction /= np.linalg.norm(control_direction)
                    fd = (scalar_cost(t, states[t], target + eps * control_direction)
                          - scalar_cost(t, states[t], target - eps * control_direction)) / (2 * eps)
                    predicted = float(lu[t] @ control_direction)
                    checks.append(dict(start=start, t=t, direction=direction_index, kind="control",
                                       finite_difference=fd, expanded=predicted, absolute_error=abs(fd-predicted),
                                       scaled_error=abs(fd-predicted)/max(1, abs(fd))))
            feature = tracker.features(tracker.states[start+t:start+t+1])
            residual = tracker.residual(t, feature)[0]
            alignments.append(dict(start=start, t=t, state_reference_frame=start+t,
                                   target_reference_frame=min(start+t+1, len(tracker.states)-1),
                                   pose_velocity_residual_max=float(np.max(np.abs(residual[:88]))),
                                   hessian_symmetry_max=float(np.max(np.abs(lxx[t]-lxx[t].T))),
                                   hessian_min_eigenvalue=float(np.linalg.eigvalsh(lxx[t]).min())))
    passed = max(c["scaled_error"] for c in checks) < 1e-4
    assert max(c["pose_velocity_residual_max"] for c in alignments) < 1e-10
    result = dict(passed=passed, mujoco=mujoco.__version__, checks=checks, alignments=alignments,
                  max_scaled_error=max(c["scaled_error"] for c in checks),
                  max_absolute_error=max(c["absolute_error"] for c in checks),
                  cost_semantics="state at start+t; nonterminal target references start+t+1; terminal has no control cost",
                  source_sha256=sha256(Path(__file__).resolve().parents[1]/"utils/g1_true23_mjbatch_mpc.py"))
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ("checks","alignments")}))
    assert passed


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
