"""Bounded hard-feasibility experiment at an archived actual PICO state."""

import argparse
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from gear_sonic.utils.g1_true23_mjbatch_ilqr_core import NoFeasiblePlan, ilqr
from gear_sonic.utils.g1_true23_mjbatch_model import position_servo_copy, preview_native_control
from gear_sonic.utils.g1_true23_mjbatch_mpc import (
    Native23Tracker,
    load_motion_override,
    load_native_bundle,
    sha256,
)


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    native, contract, original, timeline, manifest = load_native_bundle(args.bundle, "pico")
    motion, override = load_motion_override(
        args.motion_override, args.bundle, "pico", native, contract, original, timeline, manifest
    )
    kp, kd, caps = [np.asarray(contract[name]) for name in ("kp", "kd", "native_effort")]
    tracker = Native23Tracker(
        position_servo_copy(native, kp, kd, caps), contract, motion, horizon=30, threads=8,
        all_joint_limit_margin=.05, all_joint_limit_weight=2000, relative_foot_weight=400,
        fd_epsilon=1e-6, hard_feasibility=True,
    )
    with np.load(args.fixture, allow_pickle=False) as archive:
        fixture = {name: archive[name].copy() for name in archive.files}
    data = mujoco.MjData(native)
    spec = int(fixture["state_spec"])
    mujoco.mj_setState(native, data, fixture["state_vector"], spec)
    mujoco.mj_forward(native, data)
    mujoco.mj_setState(native, data, fixture["state_vector"], spec)
    state = np.r_[data.qpos, data.qvel]
    seed = fixture["targets"]
    tracker.window(3750)
    started = time.perf_counter()
    failed = None
    try:
        states, targets, gains, cost = ilqr(tracker, state, seed.copy(), iters=5)
    except NoFeasiblePlan as error:
        failed = dict(kind="no_feasible_seed", diagnostics=error.diagnostics)
        states, targets, gains, cost = None, None, None, None
    solve_seconds = time.perf_counter() - started
    actual_states, actual_targets, physics_states, physics_torques = [state], [], [state], []
    checks = []
    if failed is None:
        for local in range(5):
            current = np.r_[data.qpos, data.qvel]
            raw = gains[local] @ tracker.difference(states[local:local + 1], current[None])[0]
            target = np.clip(targets[local] + np.clip(raw, -.1, .1), tracker.lo, tracker.hi)
            witness, predicted, predicted_force = preview_native_control(
                native, data, target, contract, tracker.feasibility
            )
            checks.append(witness)
            if not witness["feasible"]:
                failed = dict(kind="imminent_control_infeasible", control=3740 + local, witness=witness)
                break
            actual_targets.append(target)
            for substep in range(10):
                data.ctrl[:] = np.clip(kp * (target - data.qpos[7:]) - kd * data.qvel[6:], -caps, caps)
                mujoco.mj_step(native, data)
                actual = np.r_[data.qpos, data.qvel]
                np.testing.assert_array_equal(actual, predicted[substep + 1])
                np.testing.assert_array_equal(data.qfrc_actuator[6:], predicted_force[substep])
                physics_states.append(actual)
                physics_torques.append(data.qfrc_actuator[6:].copy())
            actual_states.append(np.r_[data.qpos, data.qvel])
    report = dict(
        kind="bounded_actual_state_hard_feasibility_probe", actual_control=3740,
        original_source_seconds=67.8, horizon=30, commit=5, iterations=5, threads=8,
        source_frame_initial=3750, planning_seconds=solve_seconds,
        feasible_incumbent_found=cost is not None, cost=cost,
        solver=tracker.last_solve_feasibility, imminent_control_checks=checks,
        executed_controls=len(actual_targets), failure=failed,
        future_preview_packet_seconds=.74, future_raw_pose_support_seconds=.76,
        native_warning_counts=data.warning.number.tolist(),
        full_lifecycle_run=False, full_source_completed=False, recovery_qualified=False,
        deployment_ready=False, hardware_authorized=False,
        input_hashes={str(path): sha256(path) for path in (args.fixture, args.motion_override, Path(__file__))},
        motion_override=override,
    )
    arrays = dict(actual_states=actual_states, actual_targets=actual_targets,
                  physics_states=physics_states, physics_torques=physics_torques, initial_seed=seed)
    if targets is not None:
        arrays.update(planned_states=states, planned_targets=targets, feedback_K=gains)
    np.savez_compressed(args.output / "trace.npz", **arrays)
    report["trace_sha256"] = sha256(args.output / "trace.npz")
    utilities = Path(__file__).resolve().parents[1] / "utils"
    for name in ("g1_true23_mjbatch_ilqr_core", "g1_true23_mjbatch_mpc", "g1_true23_mjbatch_model"):
        source = utilities / (name + ".py")
        (args.output / (name + "_snapshot.py")).write_bytes(source.read_bytes())
        report["input_hashes"][str(source)] = sha256(source)
    (args.output / "source.py").write_bytes(Path(__file__).read_bytes())
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({key: report[key] for key in (
        "feasible_incumbent_found", "cost", "planning_seconds", "executed_controls", "failure"
    )}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("bundle", "fixture", "motion-override", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    run(parser.parse_args())
