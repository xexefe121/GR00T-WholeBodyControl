"""Exact returned-controller parity against a frozen pre-optimization solver."""

import argparse
import gc
import importlib.util
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from gear_sonic.utils.g1_true23_mjbatch_ilqr_core import ilqr
from gear_sonic.utils.g1_true23_mjbatch_model import position_servo_copy
from gear_sonic.utils.g1_true23_mjbatch_mpc import Native23Tracker, load_native_bundle, sha256


def run(args):
    spec = importlib.util.spec_from_file_location("frozen_ilqr", args.frozen_core)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    native, contract, motion, _, _ = load_native_bundle(args.bundle, "walk002")
    with np.load(args.trace, allow_pickle=False) as archive:
        states = np.column_stack((archive["qpos"], archive["qvel"]))
    with np.load(args.seed, allow_pickle=False) as archive:
        targets = archive["target"].copy()
    cases = []
    for control, iterations in ((0, 1), (300, 1), (400, 10), (600, 10)):
        results, timing = [], []
        for variant in ("frozen", "optimized", "optimized_cached"):
            servo = position_servo_copy(native, contract["kp"], contract["kd"], contract["native_effort"])
            planner = Native23Tracker(servo, contract, motion, horizon=30, threads=8)
            planner.window(control + 10)
            state, us = states[control].copy(), targets[control : control + 30].copy()
            # Include the rollout in each timing: it is needed to compare the
            # seed and shifted previous plan in the real evaluator.
            started = time.perf_counter()
            cached_xs, _, totals = planner.rollout(state, us)
            cached = (cached_xs[:, 0].copy(), float(totals[0]))
            if variant == "frozen":
                result = old.ilqr(planner, state, us, iters=iterations)
            else:
                result = ilqr(
                    planner,
                    state,
                    us,
                    iters=iterations,
                    initial_rollout=cached if variant.endswith("cached") else None,
                )
            timing.append(time.perf_counter() - started)
            results.append(result)
            del planner, servo
            gc.collect()
        errors = {
            name: max(
                float(np.max(np.abs(np.asarray(result[i]) - np.asarray(results[0][i])))) for result in results[1:]
            )
            for i, name in enumerate(("state", "target", "feedback_gain", "cost"))
        }
        bit_exact = all(np.array_equal(a, b) for result in results[1:] for a, b in zip(results[0], result))
        case = dict(
            control=control,
            iterations=iterations,
            max_abs_errors=errors,
            bit_exact=bit_exact,
            seconds=dict(zip(("frozen", "optimized", "optimized_cached"), timing)),
        )
        cases.append(case)
        print(json.dumps(case), flush=True)
        if not bit_exact:
            break
    report = dict(
        kind="exact_solver_reuse_native23_parity",
        mujoco=mujoco.__version__,
        frozen_solver_sha256=sha256(args.frozen_core),
        trace_sha256=sha256(args.trace),
        seed_sha256=sha256(args.seed),
        verifier_sha256=sha256(__file__),
        current_solver_sha256=sha256(Path(__file__).resolve().parents[1] / "utils/g1_true23_mjbatch_ilqr_core.py"),
        cases=cases,
        bit_exact=all(case["bit_exact"] for case in cases),
        timing_environment="shared host; diagnostic only, no real-time claim",
    )
    args.output.write_text(json.dumps(report, indent=2))
    assert report["bit_exact"], report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("bundle", "trace", "seed", "frozen-core", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    run(parser.parse_args())
