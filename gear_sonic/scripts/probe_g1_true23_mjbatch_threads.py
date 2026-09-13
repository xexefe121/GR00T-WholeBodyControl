"""Same-input planner thread-count parity and shared-host timing diagnostic."""

import argparse
import gc
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from gear_sonic.utils.g1_true23_mjbatch_ilqr_core import ilqr
from gear_sonic.utils.g1_true23_mjbatch_model import position_servo_copy
from gear_sonic.utils.g1_true23_mjbatch_mpc import Native23Tracker, load_native_bundle, sha256


def run(args):
    native, contract, motion, _, _ = load_native_bundle(args.bundle, "walk002")
    with np.load(args.trace, allow_pickle=False) as trace:
        state = np.r_[trace["qpos"][args.control], trace["qvel"][args.control]]
    with np.load(args.seed, allow_pickle=False) as seed:
        targets = seed["target"][args.control : args.control + 30].copy()
    results, records = [], []
    for threads in (4, 8, 8, 4):
        servo = position_servo_copy(native, contract["kp"], contract["kd"], contract["native_effort"])
        planner = Native23Tracker(servo, contract, motion, horizon=30, threads=threads)
        planner.window(args.control + 10)
        started = time.perf_counter()
        xs, us, gains, cost = ilqr(planner, state, targets.copy(), iters=10)
        elapsed = time.perf_counter() - started
        records.append(dict(threads=threads, solve_seconds=elapsed, cost=float(cost)))
        results.append((xs, us, gains))
        print(json.dumps(records[-1]), flush=True)
        del planner, servo
        gc.collect()
    parity = {
        name: max(float(np.max(np.abs(result[index] - results[0][index]))) for result in results[1:])
        for index, name in enumerate(("state", "target", "feedback_gain"))
    }
    assert all(value == 0.0 for value in parity.values()), parity
    report = dict(
        kind="same_input_native23_planner_thread_parity",
        mujoco=mujoco.__version__,
        trace_sha256=sha256(args.trace),
        seed_sha256=sha256(args.seed),
        code_sha256=sha256(Path(__file__)),
        control=args.control,
        horizon=30,
        iterations=10,
        records=records,
        bit_exact=True,
        parity_max_abs=parity,
        timing_environment="shared host; root replays and separate MuJoCo3.2.3 build may run",
        qualification="Relative timing diagnostic only, not a real-time claim",
    )
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--seed", type=Path, required=True)
    parser.add_argument("--control", type=int, default=400)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
