"""Read-only local contact derivative sensitivity; never a tracking pass."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from gear_sonic.utils import g1_true23_mjbatch_ilqr_core as core
from gear_sonic.utils import g1_true23_mjbatch_mpc as mpc
from gear_sonic.utils.g1_true23_mjbatch_model import position_servo_copy


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    native, contract, motion, _, _ = mpc.load_native_bundle(args.bundle, "walk002")
    kp, kd, effort = [np.asarray(contract[key]) for key in ("kp", "kd", "native_effort")]
    servo = position_servo_copy(native, kp, kd, effort)
    planner = mpc.Native23Tracker(servo, contract, motion, horizon=30, threads=4)
    with np.load(args.trace, allow_pickle=False) as archive:
        xs = np.c_[archive["qpos"], archive["qvel"]][args.control:args.control + 31].copy()
        us = archive["target"][args.control:args.control + 30].copy()
    if xs.shape != (31, 59) or us.shape != (30, 23):
        raise ValueError("need 30 full physical controls")
    planner.window(args.control + 10)
    rng = np.random.default_rng(731)
    directions = rng.normal(size=(16, 58))
    directions[:6] = 0
    directions[:6, :6] = np.eye(6)
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    probe_data = mujoco.MjData(native)

    def physical_step(x, target):
        mujoco.mj_resetData(native, probe_data)
        probe_data.qpos[:], probe_data.qvel[:] = x[:30], x[30:]
        mujoco.mj_forward(native, probe_data)
        probe_data.qacc_warmstart[:] = 0
        for _ in range(10):
            probe_data.ctrl[:] = np.clip(kp * (target - probe_data.qpos[7:]) - kd * probe_data.qvel[6:], -effort, effort)
            mujoco.mj_step(native, probe_data)
        return np.r_[probe_data.qpos, probe_data.qvel]

    truth = {}
    for knot in (0, 5, 10, 15, 20, 25):
        nominal = physical_step(xs[knot], us[knot])
        for amplitude in (1e-5, 1e-4, 1e-3):
            tangent = amplitude * directions
            perturbed = np.tile(xs[knot], (len(tangent), 1))
            perturbed[:, :30] = planner.integrate(perturbed[:, :30], tangent[:, :29])
            perturbed[:, 30:] += tangent[:, 29:]
            outputs = np.array([physical_step(state, us[knot]) for state in perturbed])
            difference = planner.difference(np.tile(nominal, (len(tangent), 1)), outputs)
            truth[knot, amplitude] = difference
    results = []
    for epsilon in (1e-6, 1e-5, 1e-4, 1e-3):
        # This is a separate process. Both dynamics and cost expansion need the
        # same perturbation spacing because the cost reuses perturbed features.
        core.EPS = mpc.EPS = epsilon
        tick = time.perf_counter()
        A, B = planner.linearize(xs, us)
        lx, lxx, lu, luu = planner.expand(xs, us)
        sweep = core.backward(A, B, lx, lxx, lu, luu, planner.lo - us, planner.hi - us, 1.)
        K = None if sweep is None else sweep[1]
        errors = []
        for (knot, amplitude), actual in truth.items():
            predicted = amplitude * directions @ A[knot].T
            err = np.linalg.norm(predicted - actual, axis=1)
            actual_norm = np.linalg.norm(actual, axis=1)
            errors.append(dict(knot=knot, global_control=args.control + knot, amplitude=amplitude,
                               absolute_error_p50_p95_max=np.percentile(err, [50, 95, 100]).tolist(),
                               relative_error_p50_p95_max=np.percentile(err / np.maximum(actual_norm, 1e-9), [50, 95, 100]).tolist()))
        noise = np.r_[[.001] * 3, [.001] * 3, [.0001] * 23, [.01] * 3, [.001] * 3, [.001] * 23]
        results.append(dict(epsilon=epsilon, A_max=float(np.max(np.abs(A))), B_max=float(np.max(np.abs(B))),
                            K_max=None if K is None else float(np.max(np.abs(K))),
                            assumed_sensor_sigma=noise.tolist(),
                            linearized_target_noise_sigma_max=None if K is None else float(np.max(np.linalg.norm(K * noise, axis=-1))),
                            elapsed_seconds=time.perf_counter() - tick, directional_errors=errors))
        print(json.dumps({key: value for key, value in results[-1].items() if key not in ("directional_errors", "assumed_sensor_sigma")}), flush=True)
    report = dict(kind="read_only_mpc_finite_difference_contact_sensitivity", mujoco=mujoco.__version__,
                  start_control=args.control, horizon=30, results=results,
                  input_sha256=sha(args.trace), script_sha256=sha(__file__),
                  physical_model_unchanged=True, new_controller_qualified=False,
                  caveat="Short directional derivative witnesses, not closed-loop replanning or full-source tests")
    (args.output / "report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--control", type=int, default=300)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
