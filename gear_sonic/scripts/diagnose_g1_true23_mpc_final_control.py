"""Privately finish one failed control and inspect its declared nominal endpoint."""

import argparse
import json
from pathlib import Path

from mjbatch import Batch
import mujoco
import numpy as np

from gear_sonic.utils.g1_true23_mjbatch_model import position_servo_copy
from gear_sonic.utils.g1_true23_mjbatch_mpc import load_native_bundle, sha256


def run(args):
    args.output.mkdir(exist_ok=False)
    native, contract, _, _, _ = load_native_bundle(args.bundle, "pico")
    kp, kd, effort = [np.asarray(contract[k]) for k in ("kp", "kd", "native_effort")]
    with np.load(args.run / "trace.npz", allow_pickle=False) as source:
        saved = {k: source[k].copy() for k in source.files}
    last = len(saved["target"]) - 1
    assert int(saved["physics_substeps"][-1]) < 10
    assert np.all(saved["physics_substeps"][:-1] == 10)
    data = mujoco.MjData(native)
    data.qpos[:], data.qvel[:] = saved["qpos"][0], saved["qvel"][0]
    mujoco.mj_forward(native, data)
    maximum_q = maximum_dq = maximum_ctrl = 0.0
    actual_last = []
    actual_torque = []
    last_warm = None
    for control, target in enumerate(saved["target"]):
        if control == last:
            last_warm = data.qacc_warmstart.copy()
            actual_last.append(np.r_[data.qpos, data.qvel])
        for substep in range(10):
            data.ctrl[:] = np.clip(kp * (target - data.qpos[7:]) - kd * data.qvel[6:], -effort, effort)
            mujoco.mj_step(native, data)
            sample = control * 10 + substep + 1
            if sample < len(saved["physics_qpos"]):
                maximum_q = max(maximum_q, float(np.max(np.abs(data.qpos - saved["physics_qpos"][sample]))))
                maximum_dq = max(maximum_dq, float(np.max(np.abs(data.qvel - saved["physics_qvel"][sample]))))
                maximum_ctrl = max(
                    maximum_ctrl, float(np.max(np.abs(data.ctrl - saved["physics_torque"][sample - 1])))
                )
            if control == last:
                actual_last.append(np.r_[data.qpos, data.qvel])
                actual_torque.append(data.ctrl.copy())
    np.testing.assert_allclose(maximum_q, 0, atol=1e-12, rtol=0)
    np.testing.assert_allclose(maximum_dq, 0, atol=1e-10, rtol=0)
    servo = position_servo_copy(native, kp, kd, effort)
    batch = Batch(servo, 1, num_threads=1, forward=True)
    fields = {k: batch.bind(k) for k in ("qpos", "qvel", "ctrl", "qacc_warmstart")}

    def nominal_control(control):
        state = saved["planned_state"][control]
        fields["qpos"][:] = state[:30]
        fields["qvel"][:] = state[30:]
        fields["ctrl"][:] = saved["planned_target"][control]
        fields["qacc_warmstart"][:] = 0.0
        states = [state.copy()]
        for _ in range(10):
            batch.step(nstep=1)
            states.append(np.r_[fields["qpos"][0], fields["qvel"][0]])
        return np.asarray(states)

    plans = json.loads((args.run / "plans.json").read_text())
    plan_start = plans[-1]["control"]
    nominal_prefix_errors = []
    for control in range(plan_start, last):
        predicted = nominal_control(control)
        nominal_prefix_errors.append(float(np.max(np.abs(predicted[-1] - saved["planned_state"][control + 1]))))
    nominal = nominal_control(last)
    actual_last = np.asarray(actual_last)
    lo, hi = native.jnt_range[1:].T

    def limits(states):
        excess = np.maximum(0, np.maximum(lo - states[:, 7:30], states[:, 7:30] - hi))
        return dict(
            maximum_rad=float(excess.max()), endpoint_maximum_rad=float(excess[-1].max()),
            left_ankle_pitch_q=states[:, 11].tolist(),
            left_ankle_pitch_excess_rad=excess[:, 4].tolist(),
            endpoint_has_native_violation=bool(excess[-1].max() > 1e-6),
        )
    report = dict(
        kind="private_failed_control_endpoint_diagnosis", control_zero_based=last,
        last_plan_start=plan_start, original_failed_substeps=int(saved["physics_substeps"][-1]),
        reproduced_original_q_max=maximum_q, reproduced_original_dq_max=maximum_dq,
        reproduced_original_ctrl_max=maximum_ctrl,
        original_first_five_substeps_reproduced=True,
        actual_private_completion=limits(actual_last), predicted_nominal=limits(nominal),
        earlier_same_plan_endpoint_max_deltas=nominal_prefix_errors,
        nominal_warmstart_contract="zero each20ms advance, matching Planner.advance",
        precontrol_actual_vs_nominal_max=float(np.max(np.abs(actual_last[0] - nominal[0]))),
        applied_vs_planned_target_max=float(np.max(np.abs(saved["target"][last] - saved["planned_target"][last]))),
        actual_private_warning_counts=data.warning.number.tolist(),
        inter_knot_only_excursion=bool(
            limits(actual_last)["maximum_rad"] > 1e-6
            and not limits(actual_last)["endpoint_has_native_violation"]
            and not limits(nominal)["endpoint_has_native_violation"]
        ),
        source_trace_sha256=sha256(args.run / "trace.npz"), source_sha256=sha256(__file__),
        original_trial_modified=False, new_controller_trial=False,
    )
    np.savez_compressed(
        args.output / "arrays.npz", actual_last=actual_last, actual_torque=actual_torque,
        nominal_last=nominal, actual_last_warmstart=last_warm,
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    (args.output / "source.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for key in ("run", "bundle", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    run(parser.parse_args())
