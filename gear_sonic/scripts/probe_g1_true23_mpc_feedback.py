"""Saved-controller sensitivity under tiny physical initial-state perturbations.

This reuses saved local feedback. It is not receding-horizon replanning.
"""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from gear_sonic.utils.g1_true23_mjbatch_mpc import load_native_bundle, sha256


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    model, contract, _, _, _ = load_native_bundle(args.bundle, "walk002")
    with np.load(args.trace, allow_pickle=False) as archive:
        source = {
            name: archive[name].copy()
            for name in ("qpos", "qvel", "planned_target", "planned_state", "feedback_gain")
        }
    kp, kd, effort, speeds = [
        np.asarray(contract[key]) for key in ("kp", "kd", "native_effort", "native_velocity")
    ]
    limits = np.asarray(contract["joint_limits"])
    direction = np.random.default_rng(641).normal(size=58)
    direction /= np.max(np.abs(direction))
    reports = []
    for scale, clip in ((0.0, 0.0), (1e-7, 0.0), (1e-5, 0.0), (1e-3, 0.0), (1e-5, 0.1), (1e-3, 0.1)):
        data = mujoco.MjData(model)
        data.qpos[:], data.qvel[:] = source["qpos"][0], source["qvel"][0]
        initial_delta = direction * scale
        mujoco.mj_integratePos(model, data.qpos, initial_delta[:29], 1.0)
        data.qvel[:] += initial_delta[29:]
        mujoco.mj_forward(model, data)
        state_error = np.empty(58)
        trace = {key: [] for key in ("physics_qpos", "physics_qvel", "torque", "raw_correction", "target")}
        trace["physics_qpos"].append(data.qpos.copy())
        trace["physics_qvel"].append(data.qvel.copy())
        failure, completed, range_max, raw_max = None, 0, 0.0, 0.0
        for control, nominal in enumerate(source["planned_state"]):
            mujoco.mj_differentiatePos(model, state_error[:29], 1.0, nominal[:30], data.qpos)
            state_error[29:] = data.qvel - nominal[30:]
            raw = source["feedback_gain"][control] @ state_error
            correction = np.clip(raw, -clip, clip) if clip else raw
            target = np.clip(source["planned_target"][control] + correction, limits[:, 0], limits[:, 1])
            trace["raw_correction"].append(raw.copy())
            trace["target"].append(target.copy())
            raw_max = max(raw_max, float(np.max(np.abs(raw))))
            for substep in range(10):
                data.ctrl[:] = np.clip(kp * (target - data.qpos[7:]) - kd * data.qvel[6:], -effort, effort)
                mujoco.mj_step(model, data)
                trace["physics_qpos"].append(data.qpos.copy())
                trace["physics_qvel"].append(data.qvel.copy())
                trace["torque"].append(data.ctrl.copy())
                range_max = max(
                    range_max,
                    float(np.max(np.maximum(limits[:, 0] - data.qpos[7:], data.qpos[7:] - limits[:, 1]))),
                )
                tilt = float(np.arccos(np.clip(1 - 2 * np.sum(data.qpos[4:6] ** 2), -1, 1)))
                if (
                    range_max > 0.01
                    or np.max(np.abs(data.qvel[6:]) / speeds) > 1
                    or data.qpos[2] < 0.25
                    or tilt > 1.2
                ):
                    failure = dict(
                        control=control + 1,
                        substep=substep + 1,
                        time=float(data.time),
                        range_excess=range_max,
                        height=float(data.qpos[2]),
                        tilt=tilt,
                    )
                    break
            completed += 1
            if failure:
                break
        name = f"scale_{scale:g}_clip_{clip:g}"
        np.savez_compressed(args.output / (name + ".npz"), **trace, initial_delta=initial_delta)
        reports.append(
            dict(
                name=name,
                perturbation_scale=scale,
                correction_clip=clip,
                completed=completed,
                requested=len(source["planned_target"]),
                failure=failure,
                range_excess_max=range_max,
                raw_correction_max=raw_max,
                final_root=data.qpos[:3].tolist(),
            )
        )
        print(json.dumps(reports[-1]), flush=True)
    gains = source["feedback_gain"]
    location = np.unravel_index(np.abs(gains).argmax(), gains.shape)
    report = dict(
        kind="saved_local_feedback_initial_state_sensitivity",
        mujoco=mujoco.__version__,
        replanning=False,
        trace_sha256=sha256(args.trace),
        code_sha256=sha256(Path(__file__)),
        tangent_difference="native mj_differentiatePos plus generalized velocity difference",
        gain_abs_max=float(np.max(np.abs(gains))),
        gain_abs_max_control_joint_channel=list(map(int, location)),
        gain_row_norm_p50_p95_max=np.percentile(np.linalg.norm(gains, axis=-1), [50, 95, 100]).tolist(),
        cases=reports,
        teacher_or_hardware_qualified=False,
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
