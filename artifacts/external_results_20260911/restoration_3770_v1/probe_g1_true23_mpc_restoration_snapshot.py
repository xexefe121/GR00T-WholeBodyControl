"""One fixed H30 physical-margin restoration solve at actual control3770."""

import argparse
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from gear_sonic.scripts.evaluate_g1_true23_mjbatch_mpc import finite_json
from gear_sonic.utils.g1_true23_feasibility_referee import inspect_native_segment
from gear_sonic.utils.g1_true23_mjbatch_ilqr_core import ilqr
from gear_sonic.utils.g1_true23_mjbatch_model import position_servo_copy
from gear_sonic.utils.g1_true23_mjbatch_mpc import (
    Native23Tracker,
    load_motion_override,
    load_native_bundle,
    sha256,
)
from gear_sonic.utils.g1_true23_mjbatch_restoration import Native23RestorationTracker


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    native, contract, original, timeline, manifest = load_native_bundle(args.bundle, "pico")
    motion, override = load_motion_override(
        args.motion_override, args.bundle, "pico", native, contract, original, timeline, manifest
    )
    with np.load(args.fixture, allow_pickle=False) as archive:
        initial = archive["final_integration"].copy()
        spec = int(archive["integration_state_spec"])
        seed = archive["final_warm_targets"].copy()
    data = mujoco.MjData(native)
    mujoco.mj_setState(native, data, initial, spec)
    mujoco.mj_forward(native, data)
    mujoco.mj_setState(native, data, initial, spec)
    x0 = np.r_[data.qpos, data.qvel]
    servo = position_servo_copy(native, contract["kp"], contract["kd"], contract["native_effort"])
    restorer = Native23RestorationTracker(servo, contract, motion, seed, threads=8)
    restorer.window(3780)
    certify = Native23Tracker(servo, contract, motion, horizon=30, threads=8,
                              all_joint_limit_margin=.05, all_joint_limit_weight=2000,
                              relative_foot_weight=400, hard_feasibility=True)
    certify.window(3780)
    directory = Path(__file__).resolve().parents[1]
    sources = [Path(__file__), directory / "scripts/evaluate_g1_true23_mjbatch_mpc.py"]
    sources += [directory / "utils" / (name + ".py") for name in (
        "g1_true23_mjbatch_model", "g1_true23_mjbatch_mpc", "g1_true23_mjbatch_ilqr_core",
        "g1_true23_mjbatch_restoration", "g1_true23_relative_foot_cost", "g1_true23_feasibility_referee",
    )]
    request = dict(kind="one_bounded_private_seed_restoration", control=3770, horizon=30,
                   maximum_iterations=10, threads=8, actual_controls_executed=0,
                   merit=restorer.merit_contract(), acceptance=certify.feasibility.contract(),
                   original_tracking_goal=override, hardware_authorized=False,
                   input_hashes={str(p): sha256(p) for p in [*sources, args.fixture, args.motion_override]})
    (args.output / "request.json").write_text(json.dumps(request, indent=2))
    for path in sources:
        (args.output / (path.stem + "_snapshot.py")).write_bytes(path.read_bytes())
    seed_states, _, seed_costs = restorer.rollout(x0, seed)
    if not np.isfinite(seed_costs[0]):
        raise ValueError("restoration seed has nonfinite/engine-invalid merit; no derivative search")
    # Independent scalar/gradient witness for precisely the declared merit.
    feature = restorer.reference[0:1].copy()
    feature[0, 49:72] = (restorer.lo + restorer.hi) / 2
    feature[0, 72:] = 0
    feature[0, 45:49] = [np.cos(1.15 / 2), np.sin(1.15 / 2), 0, 0]
    feature[0, 44] = .275
    feature[0, 49] = restorer.lo[0] + .01
    feature[0, 78] = .95 * contract["native_velocity"][0]
    witness_cost = float(np.sum(restorer.residual(0, feature) ** 2))
    np.testing.assert_allclose(witness_cost, 1, rtol=0, atol=1e-12)
    started = time.perf_counter()
    states, targets, gains, cost = ilqr(
        restorer, x0, seed.copy(), iters=10, initial_rollout=(seed_states[:, 0], seed_costs[0])
    )
    elapsed = time.perf_counter() - started
    certified_states, _, certified_cost = certify.rollout(x0, targets)
    oracle, actual_trace = inspect_native_segment(native, data, targets, contract)
    feasible = bool(np.isfinite(certified_cost[0]) and oracle["feasible"])
    np.savez_compressed(args.output / "counterfactual.npz", targets=targets, merit_states=states,
                        nominal_certification_states=certified_states[:, 0], generating_feedback_K=gains,
                        original_seed=seed, initial_integration=initial, integration_state_spec=spec,
                        **{"oracle_" + k: v for k, v in actual_trace.items()})
    if feasible:
        np.savez_compressed(args.output / "feasible_seed.npz", targets=targets,
                            initial_integration=initial, integration_state_spec=spec)
    report = finite_json(dict(
        request=request, initial_merit=float(seed_costs[0]), final_merit=float(cost), solve_seconds=elapsed,
        physical_margin_scalar_witness=witness_cost, hard_nominal=certify.last_rollout_feasibility,
        independent_actual_state_oracle=oracle, feasible_seed_exported=feasible,
        main_tracking_cost_if_feasible=float(certified_cost[0]) if np.isfinite(certified_cost[0]) else None,
        actual_controls_executed=0, recovery_qualified=False, full_source_replay=False,
        source_files_unchanged=all(sha256(p) == request["input_hashes"][str(p)] for p in sources),
    ))
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({key: report[key] for key in (
        "initial_merit", "final_merit", "solve_seconds", "feasible_seed_exported", "actual_controls_executed"
    )}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("bundle", "fixture", "motion-override", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    run(parser.parse_args())
