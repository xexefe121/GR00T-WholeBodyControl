"""Instrument one identical control3805 restoration solve without execution."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from gear_sonic.scripts.evaluate_g1_true23_mjbatch_mpc import finite_json
from gear_sonic.utils import g1_true23_mjbatch_ilqr_core as core
from gear_sonic.utils.g1_true23_feasibility_referee import inspect_native_segment
from gear_sonic.utils.g1_true23_mjbatch_model import position_servo_copy
from gear_sonic.utils.g1_true23_mjbatch_mpc import (
    Native23Tracker,
    load_motion_override,
    load_native_bundle,
    sha256,
)
from gear_sonic.utils.g1_true23_mjbatch_restoration import Native23RestorationTracker


def backward_failure_location(arguments, mu):
    """Reproduce the original backward algebra solely to identify its None exit."""
    A, B, lx, lxx, lu, luu, lo, hi = arguments
    T, nu, nx = len(lu), lu.shape[1], lx.shape[1]
    vx, vxx = lx[-1], lxx[-1]
    k = np.zeros((T + 1, nu))
    for t in reversed(range(T)):
        qx, qu = lx[t] + A[t].T @ vx, lu[t] + B[t].T @ vx
        qxx = lxx[t] + A[t].T @ vxx @ A[t]
        quu, qux = luu[t] + B[t].T @ vxx @ B[t], B[t].T @ vxx @ A[t]
        reg = vxx + mu * np.eye(nx)
        quu_reg, qux_reg = luu[t] + B[t].T @ reg @ B[t], B[t].T @ reg @ A[t]
        finite = np.isfinite(quu_reg).all()
        eig = np.linalg.eigvalsh(quu_reg) if finite else np.array([np.nan])
        if not finite or eig.min() <= 0:
            return dict(knot=t, quu_finite=bool(finite), minimum_eigenvalue=float(eig.min()),
                        maximum_eigenvalue=float(eig.max()), maximum_absolute_quu=float(np.max(np.abs(quu_reg))),
                        maximum_absolute_B=float(np.max(np.abs(B[t]))))
        k[t], free = core.boxqp(quu_reg, qu, lo[t], hi[t], k[t + 1])
        K = np.zeros((nu, nx))
        K[free] = -np.linalg.solve(quu_reg[np.ix_(free, free)], qux_reg[free])
        vx = qx + K.T @ quu @ k[t] + K.T @ qu + qux.T @ k[t]
        vxx = qxx + K.T @ quu @ K + K.T @ qux + qux.T @ K
        vxx = .5 * (vxx + vxx.T)
    return None


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    native, contract, original, timeline, manifest = load_native_bundle(args.bundle, "pico")
    motion, _ = load_motion_override(args.motion_override, args.bundle, "pico", native,
                                     contract, original, timeline, manifest)
    with np.load(args.fixture, allow_pickle=False) as archive:
        integration = archive["final_integration"].copy()
        spec = int(archive["integration_state_spec"])
        seed = archive["final_warm_targets"].copy()
    data = mujoco.MjData(native)
    mujoco.mj_setState(native, data, integration, spec)
    mujoco.mj_forward(native, data)
    mujoco.mj_setState(native, data, integration, spec)
    initial = np.r_[data.qpos, data.qvel]
    full_initial_oracle, full_initial_trace = inspect_native_segment(
        native, data, seed, contract, stop_on_failure=False, retain_trace=True
    )
    lo, hi = native.jnt_range[1:].T
    full_q = full_initial_trace["physics_qpos"][:, 7:]
    excess = np.maximum(np.maximum(lo - full_q, full_q - hi), 0)
    full_initial_oracle["joint_excess_max_rad"] = excess.max(axis=0).tolist()
    full_initial_oracle["joint_excess_final_rad"] = excess[-1].tolist()
    full_initial_oracle["joint_names"] = contract["joint_names"]
    np.savez_compressed(args.output / "initial_full_horizon.npz", **full_initial_trace,
                        targets=seed, initial_integration=integration, integration_state_spec=spec)
    servo = position_servo_copy(native, contract["kp"], contract["kd"], contract["native_effort"])
    restorer = Native23RestorationTracker(servo, contract, motion, seed, threads=8)
    restorer.window(3815)
    certifier = Native23Tracker(servo, contract, motion, horizon=30, threads=8,
                                all_joint_limit_margin=.05, all_joint_limit_weight=2000,
                                relative_foot_weight=400, hard_feasibility=True)
    certifier.window(3815)
    calls, searches, sequences, directions = [], [], [], []
    backward = core.backward
    rollout = restorer.rollout

    def observed_backward(*values):
        *arguments, mu = values
        result = backward(*values)
        names = ("A", "B", "lx", "lxx", "lu", "luu", "lower", "upper")
        entry = dict(mu=float(mu), returned_sweep=result is not None,
                     derivatives={name: dict(all_finite=bool(np.isfinite(a).all()),
                                             max_abs=float(np.max(np.abs(a))))
                                  for name, a in zip(names, arguments)})
        if result is None:
            entry["failure_location"] = backward_failure_location(arguments, mu)
        else:
            directions.append((result[0].copy(), result[1].copy()))
        calls.append(entry)
        return result

    def observed_rollout(*values, **kwargs):
        if args.zero_rollout_feedback and len(values) >= 3:
            x0, us, (xs, k, K) = values
            values = (x0, us, (xs, k, np.zeros_like(K)))
        states, targets, costs = rollout(*values, **kwargs)
        searches.append(dict(kind="alpha_search" if len(values) >= 3 else "initial",
                             costs=costs.tolist(), finite_count=int(np.isfinite(costs).sum())))
        sequences.append(targets.copy())
        return states, targets, costs

    restorer.rollout = observed_rollout
    seed_states, _, initial_costs = restorer.rollout(initial, seed)
    residual = restorer.residual(0, restorer.features(seed_states[:, 0]))
    try:
        core.backward = observed_backward
        states, targets, gains, cost = core.ilqr(
            restorer, initial, seed.copy(), iters=10, initial_rollout=(seed_states[:, 0], initial_costs[0])
        )
    finally:
        core.backward = backward
    # Inspect every generated alpha, including candidates the merit criterion rejected.
    certificates, exports = [], []
    for call, sequences_at_call in enumerate(sequences):
        for lane in range(9 if call else 1):
            candidate = sequences_at_call[:, lane]
            checked_states, _, costs = certifier.rollout(initial, candidate)
            witness = certifier.last_rollout_feasibility["first_violation"][0]
            failure = None
            if witness is not None:
                q = checked_states[-1, 0, 7:30]
                gap = np.maximum(certifier.lo - q, q - certifier.hi)
                joint = int(np.argmax(gap))
                failure = dict(**witness)
                if "joint_range" in witness["reasons"]:
                    failure.update(joint_index=joint, joint_name=contract["joint_names"][joint],
                                   q=float(q[joint]), lower=float(certifier.lo[joint]),
                                   upper=float(certifier.hi[joint]),
                                   direction="below_lower" if q[joint] < certifier.lo[joint] else "above_upper")
            entry = dict(rollout_call=call, lane=lane, merit=searches[call]["costs"][lane],
                         hard_nominal_pass=bool(np.isfinite(costs[0])), first_failure=failure)
            if np.isfinite(costs[0]):
                oracle, _ = inspect_native_segment(native, data, candidate, contract, retain_trace=False)
                entry["independent_actual_oracle"] = oracle
                if oracle["feasible"]:
                    filename = "feasible_%02d_%d.npz" % (call, lane)
                    np.savez_compressed(args.output / filename, targets=candidate, initial_integration=integration,
                                        integration_state_spec=spec)
                    exports.append(dict(file=filename, merit=entry["merit"], tracking_cost=float(costs[0])))
            certificates.append(entry)
    source_paths = [Path(__file__), Path(core.__file__)]
    for name in ("g1_true23_mjbatch_mpc", "g1_true23_mjbatch_model", "g1_true23_mjbatch_restoration",
                 "g1_true23_feasibility_referee"):
        source_paths.append(Path(core.__file__).with_name(name + ".py"))
    source_hashes = {str(path): sha256(path) for path in source_paths}
    for path in source_paths:
        (args.output / path.name).write_bytes(path.read_bytes())
    report = finite_json(dict(
        kind="one_identical_instrumented_restoration_solve", control=3805, horizon=30, maximum_iterations=10,
        zero_rollout_feedback=args.zero_rollout_feedback,
        zero_feedback_scope="private restoration line search only; backward feedforward algebra unchanged",
        initial_full_horizon=full_initial_oracle,
        initial_merit=float(initial_costs[0]), final_merit=float(cost),
        final_targets_equal_initial=np.array_equal(targets, seed),
        initial_fixed_state_merit=float(np.sum(residual[0] ** 2)),
        state_merit_by_knot=np.sum(residual**2, axis=1).tolist(),
        solver_tolerance=core.TOL, maximum_regularization=1e6, source_hashes=source_hashes,
        backward_calls=calls, alpha_searches=searches, all_candidate_certificates=certificates,
        feasible_generated_candidates=exports, actual_controls_executed=0,
        input_hashes={str(path): sha256(path) for path in (args.fixture, args.motion_override, Path(__file__))},
    ))
    np.savez_compressed(args.output / "all_generated.npz", targets_by_search=sequences,
                        final_targets=targets, final_states=states, final_K=gains,
                        backward_feedforward=[x[0] for x in directions],
                        backward_feedback=[x[1] for x in directions],
                        original_seed=seed, initial_integration=integration, integration_state_spec=spec)
    (args.output / "source.py").write_bytes(Path(__file__).read_bytes())
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(dict(initial=report["initial_merit"], final=report["final_merit"],
                         backward_calls=len(calls), alpha_searches=len(searches) - 1,
                         feasible_candidates=len(exports))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("bundle", "fixture", "motion-override", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--zero-rollout-feedback", action="store_true")
    run(parser.parse_args())
