"""Reproduce failed next-control preview and inspect earlier stopping room.

Saved states initialize independent native23 probes only, never a physical
robot or new closed-loop policy trial. Preserve each rejected candidate.
"""

import json
from pathlib import Path

import numpy as np

from gear_sonic.utils.g1_23dof_contract import HARDWARE_23_JOINT_NAMES
from gear_sonic.utils.g1_23dof_safe_target_transform import (
    SAFE_TARGET_HARD_LOWER_HARDWARE,
    SAFE_TARGET_HARD_UPPER_HARDWARE,
    safe_target_transform_numpy,
)
from gear_sonic.utils.g1_sonic_cpp_parameters import file_sha256
from gear_sonic.utils.g1_true23_generalist_benchmark import MODEL, PHYSICS
from gear_sonic.utils.g1_true23_range_preview import Native23RangePreview, RANGE_RESERVE_RAD, choose_range_target

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ASSETS = Path("/mnt/z/codex/GR00T-WholeBodyControl")


def main():
    output = HERE / "stop_diagnostic_v1"
    if output.exists():
        raise FileExistsError("stop diagnosis refuses overwrite")
    output.mkdir()
    inputs = {}

    def bind(path, expected=None):
        path = Path(path).resolve(strict=True)
        digest = file_sha256(path)
        if expected is not None and digest != expected:
            raise ValueError("stop diagnostic input changed: " + str(path))
        inputs[str(path)] = digest
        return path

    bind(__file__)
    report = json.loads(
        bind(
            HERE / "native_actual_v1/report.json",
            "2294d1381ca34ae9470731d27ecdfd7e74c60c19e37f8256ff97e3836df78f6b",
        ).read_text()
    )
    with np.load(
        bind(
            HERE / "native_actual_v1/attempts.npz", report["inputs"][str(HERE / "native_actual_v1/attempts.npz")]
        ),
        allow_pickle=False,
    ) as z:
        attempted = {key: z[key].copy() for key in ("measured_qpos", "measured_qvel", "inverse23")}
    preview = Native23RangePreview(model_path=bind(ASSETS / MODEL), physics_path=bind(ROOT / PHYSICS))
    c = preview.probe
    assert preview.model_sha256 == report["result"]["compiled_model_sha256"]
    low = np.asarray(SAFE_TARGET_HARD_LOWER_HARDWARE) + RANGE_RESERVE_RAD
    high = np.asarray(SAFE_TARGET_HARD_UPPER_HARDWARE) - RANGE_RESERVE_RAD
    integrations = 0

    def forecast(q, v, target, steps):
        nonlocal integrations
        c.module.mj_resetData(c.model, c.data)
        c.reset(
            base_position=q[:3],
            base_quaternion_wxyz=q[3:7],
            joint_position_hardware=q[7:],
            root_velocity=v[:6],
            joint_velocity_hardware=v[6:],
        )
        poses, velocities, forces = [], [], []
        for _ in range(steps):
            request = c.physics.kp * (target.astype(np.float64) - c.data.qpos[7:]) - c.physics.kd * c.data.qvel[6:]
            c.data.ctrl[:] = np.clip(request, -c.physics.effort, c.physics.effort)
            c.module.mj_step(c.model, c.data)
            integrations += 1
            poses.append(c.data.qpos.copy())
            velocities.append(c.data.qvel.copy())
            forces.append(c.data.ctrl.copy())
        return dict(qpos=np.asarray(poses), qvel=np.asarray(velocities), torque23=np.asarray(forces))

    def violations(poses):
        lower, upper = np.maximum(low - poses[:, 7:], 0).max(0), np.maximum(poses[:, 7:] - high, 0).max(0)
        return [
            dict(
                joint=HARDWARE_23_JOINT_NAMES[j],
                hardware_index=int(j),
                lower_reserve_violation_rad=float(lower[j]),
                upper_reserve_violation_rad=float(upper[j]),
            )
            for j in np.flatnonzero(np.maximum(lower, upper) > 0)
        ]

    last = len(attempted["inverse23"]) - 1
    q, v, raw = (attempted[k][last] for k in ("measured_qpos", "measured_qvel", "inverse23"))
    branches = []

    def predict(target):
        trace = forecast(q, v, target, 10)
        branches.append(dict(target=target.copy(), trace=trace, violations=violations(trace["qpos"])))
        return trace["qpos"]

    failure = None
    try:
        choose_range_target(raw, predict)
    except ValueError as error:
        failure = str(error)
    assert failure == report["result"]["failure"]["message"]
    branch_rows = []
    for i, branch in enumerate(branches):
        with (output / f"final_candidate_{i}.npz").open("xb") as stream:
            np.savez_compressed(stream, target=branch["target"], **branch["trace"])
        bind(output / f"final_candidate_{i}.npz")
        branch_rows.append(dict(candidate=i, violations=branch["violations"]))
    earlier = []
    for control in range(max(0, last - 20), last + 1):
        pq, pv = attempted["measured_qpos"][control], attempted["measured_qvel"][control]
        _, target = safe_target_transform_numpy(attempted["inverse23"][control])
        trace = forecast(pq, pv, target, 50)
        earlier.append(
            dict(
                control=control,
                source_time_s=(control - 350) * 0.02,
                first20ms_violations=violations(trace["qpos"][:10]),
                constant_target_100ms_violations=violations(trace["qpos"]),
            )
        )
        with (output / f"prior_{control}.npz").open("xb") as stream:
            np.savez_compressed(stream, initial_qpos=pq, initial_qvel=pv, target=target, **trace)
        bind(output / f"prior_{control}.npz")
    involved = sorted(set(j["hardware_index"] for row in branch_rows for j in row["violations"]))
    initial = [
        dict(
            joint=HARDWARE_23_JOINT_NAMES[j],
            position_rad=float(q[7 + j]),
            velocity_rad_s=float(v[6 + j]),
            hard_lower_rad=float(SAFE_TARGET_HARD_LOWER_HARDWARE[j]),
            hard_upper_rad=float(SAFE_TARGET_HARD_UPPER_HARDWARE[j]),
            lower_distance_rad=float(q[7 + j] - SAFE_TARGET_HARD_LOWER_HARDWARE[j]),
            upper_distance_rad=float(SAFE_TARGET_HARD_UPPER_HARDWARE[j] - q[7 + j]),
        )
        for j in involved
    ]
    result = dict(
        kind="native23_saved_stop_and_prior_stopping_room_diagnostic_v1",
        inputs=inputs,
        failed_control=last,
        reproduced_failure=failure,
        final_initial_joint_states=initial,
        final_candidate_checks=branch_rows,
        earlier_fixed_target_checks=earlier,
        independent_probe_physics_steps=integrations,
        longer_constant_target_prediction_is_not_actual_future_policy=True,
        new_closed_loop_policy_trials=0,
        source_lookahead_added=False,
        hardware_authorized=False,
        deployment_ready=False,
    )
    with (output / "report.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(
        json.dumps(
            dict(
                reproduced_failure=failure,
                initial=initial,
                candidates=branch_rows,
                prior_warning_controls=[
                    r["control"]
                    for r in earlier
                    if not r["first20ms_violations"] and r["constant_target_100ms_violations"]
                ],
                independent_probe_physics_steps=integrations,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
