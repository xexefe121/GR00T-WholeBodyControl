"""Fixed five-versus-ten-iteration comparison from one recorded actual planning state."""

import argparse
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from gear_sonic.utils.g1_true23_mjbatch_ilqr_core import ilqr
from gear_sonic.utils.g1_true23_mjbatch_model import position_servo_copy
from gear_sonic.utils.g1_true23_mjbatch_mpc import (
    Native23Tracker, load_motion_override, load_native_bundle, sha256,
)


def run(args):
    args.output.mkdir(exist_ok=False)
    native, contract, original, timeline, manifest = load_native_bundle(args.bundle, "pico")
    motion, override = load_motion_override(
        args.motion_override, args.bundle, "pico", native, contract, original, timeline, manifest
    )
    kp, kd, effort, speed = [
        np.asarray(contract[key]) for key in ("kp", "kd", "native_effort", "native_velocity")
    ]
    with np.load(args.trace, allow_pickle=False) as source:
        actual = np.r_[source["qpos"][args.control], source["qvel"][args.control]]
    with np.load(args.seed_archive, allow_pickle=False) as seed:
        targets = seed["fresh_targets"].copy()
        np.testing.assert_array_equal(seed["fresh_actual_states"][0], actual)
    assert targets.shape == (30, 23)
    planner = Native23Tracker(
        position_servo_copy(native, kp, kd, effort), contract, motion,
        horizon=30, threads=args.threads, relative_foot_weight=400,
    )
    planner.window(args.control + 10)
    initial_xs, _, initial_costs = planner.rollout(actual, targets)
    initial_rollout = (initial_xs[:, 0].copy(), float(initial_costs[0]))
    results, arrays = [], {}
    for iterations in (5, 10):
        started = time.perf_counter()
        xs, us, gains, cost = ilqr(
            planner, actual, targets.copy(), iters=iterations, initial_rollout=initial_rollout
        )
        elapsed = time.perf_counter() - started
        data = mujoco.MjData(native)
        data.qpos[:], data.qvel[:] = actual[:30], actual[30:]
        mujoco.mj_forward(native, data)
        physics = [np.r_[data.qpos, data.qvel]]
        excess = velocity = 0.0
        min_height, max_tilt = float(data.qpos[2]), 0.0
        for target in us:
            for _ in range(10):
                data.ctrl[:] = np.clip(kp * (target - data.qpos[7:]) - kd * data.qvel[6:], -effort, effort)
                mujoco.mj_step(native, data)
                physics.append(np.r_[data.qpos, data.qvel])
                excess = max(excess, float(np.maximum(planner.lo - data.qpos[7:], data.qpos[7:] - planner.hi).max()))
                velocity = max(velocity, float(np.max(np.abs(data.qvel[6:]) / speed)))
                min_height = min(min_height, float(data.qpos[2]))
                max_tilt = max(max_tilt, float(np.arccos(np.clip(1 - 2 * np.sum(data.qpos[4:6]**2), -1, 1))))
        physics = np.asarray(physics)
        results.append(dict(
            iterations=iterations, planning_seconds=elapsed, initial_seed_cost=initial_rollout[1],
            accepted_cost=float(cost), actual_substep_range_excess=excess,
            actual_substep_velocity_ratio=velocity, actual_root_height_min=min_height,
            actual_tilt_max=max_tilt, affine_vs_manual_state_max=float(np.max(np.abs(physics[::10] - xs))),
            native_substep_gate=bool(excess <= 1e-6 and velocity <= 1 and min_height >= 0.25 and max_tilt <= 1.2),
        ))
        arrays.update({f"iterations{iterations}_" + name: value for name, value in
                       dict(states=xs, targets=us, gains=gains, physical_substeps=physics).items()})
    ratio = results[0]["accepted_cost"] / results[1]["accepted_cost"]
    report = dict(
        kind="fixed_actual_state_fresh_seed_five_vs_ten_iterations",
        actual_control_zero_based=args.control, horizon=30, threads=args.threads,
        initial_actual_state=actual.tolist(), results=results, five_to_ten_cost_ratio=ratio,
        five_iteration_gate=bool(ratio <= 1.1 and all(item["native_substep_gate"] for item in results)),
        goal=override, relative_foot_weight=400, finite_difference_epsilon=1e-6,
        input_hashes={str(path): sha256(path) for path in (args.trace, args.seed_archive, Path(__file__))},
        sample_selection="one previously recorded actual PICO state; no source trial completed by this probe",
        seed_selection="fresh original-goal BFM candidate only; no retrospective stitched future MPC plan",
        timing_scope="offline fixed-state probe on shared host with concurrent full walking trials",
        full_trial_qualified=False,
    )
    np.savez_compressed(args.output / "arrays.npz", **arrays)
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    (args.output / "source.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("bundle", "trace", "seed-archive", "motion-override", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--control", type=int, default=800)
    parser.add_argument("--threads", type=int, default=8)
    run(parser.parse_args())
