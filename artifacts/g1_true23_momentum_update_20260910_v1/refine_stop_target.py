"""Refine an already checked offline target; never infer live readiness."""

import json
from pathlib import Path

import numpy as np

from gear_sonic.utils.g1_23dof_safe_target_transform import (
    SAFE_TARGET_HARD_LOWER_HARDWARE,
    SAFE_TARGET_HARD_UPPER_HARDWARE,
    safe_target_transform_numpy,
)
from gear_sonic.utils.g1_sonic_cpp_parameters import file_sha256
from gear_sonic.utils.g1_true23_generalist_benchmark import MODEL, PHYSICS
from gear_sonic.utils.g1_true23_range_preview import Native23RangePreview, RANGE_RESERVE_RAD, target_to_raw


def main():
    here = Path(__file__).resolve().parent
    output = here / "coordinated_refinement_v1"
    if output.exists():
        raise FileExistsError("refinement refuses overwrite")
    output.mkdir()
    inputs = {}

    def bind(path, expected=None):
        path = Path(path).resolve(strict=True)
        digest = file_sha256(path)
        if expected is not None and digest != expected:
            raise ValueError("refinement input changed")
        inputs[str(path)] = digest
        return path

    bind(__file__)
    original = json.loads(bind(here / "native_actual_v1/report.json").read_text())
    found = json.loads(bind(here / "coordinated_stop_v1/report.json").read_text())
    assert found["found_verified_next_control_target"]
    with np.load(
        bind(
            here / "native_actual_v1/attempts.npz", original["inputs"][str(here / "native_actual_v1/attempts.npz")]
        ),
        allow_pickle=False,
    ) as z:
        q, v, raw = (z[k][-1].copy() for k in ("measured_qpos", "measured_qvel", "inverse23"))
    _, nominal = safe_target_transform_numpy(raw)
    _, endpoint = safe_target_transform_numpy(np.asarray(found["raw23"], np.float32))
    c = Native23RangePreview(
        model_path=bind(Path("/mnt/z/codex/GR00T-WholeBodyControl") / MODEL),
        physics_path=bind(here.parents[1] / PHYSICS),
    ).probe
    low = np.asarray(SAFE_TARGET_HARD_LOWER_HARDWARE) + RANGE_RESERVE_RAD
    high = np.asarray(SAFE_TARGET_HARD_UPPER_HARDWARE) - RANGE_RESERVE_RAD
    rows = []

    def check(alpha):
        action, target = target_to_raw(nominal + alpha * (endpoint.astype(float) - nominal))
        c.module.mj_resetData(c.model, c.data)
        c.reset(
            base_position=q[:3],
            base_quaternion_wxyz=q[3:7],
            joint_position_hardware=q[7:],
            root_velocity=v[:6],
            joint_velocity_hardware=v[6:],
        )
        poses, torques = [], []
        for _ in range(10):
            c.data.ctrl[:] = np.clip(
                c.physics.kp * (target.astype(float) - c.data.qpos[7:]) - c.physics.kd * c.data.qvel[6:],
                -c.physics.effort,
                c.physics.effort,
            )
            c.module.mj_step(c.model, c.data)
            poses.append(c.data.qpos.copy())
            torques.append(c.data.ctrl.copy())
        poses = np.asarray(poses)
        good = bool(np.all(poses[:, 7:] >= low) and np.all(poses[:, 7:] <= high))
        row = dict(alpha=alpha, verified=good, raw=action, target=target, qpos=poses, torque23=np.asarray(torques))
        rows.append(row)
        return row

    assert not check(0.0)["verified"]
    accepted = check(1.0)
    assert accepted["verified"]
    lower, upper = 0.0, 1.0
    for _ in range(12):
        alpha = (lower + upper) / 2
        candidate = check(alpha)
        if candidate["verified"]:
            upper, accepted = alpha, candidate
        else:
            lower = alpha
    # The accepted endpoint is always actually checked; no monotonicity claim.
    with (output / "candidates.npz").open("xb") as stream:
        np.savez_compressed(stream, **{key: np.asarray([r[key] for r in rows]) for key in rows[0]})
    bind(output / "candidates.npz")
    result = dict(
        kind="offline_coordinated_target_segment_refinement_v1",
        inputs=inputs,
        nominal_to_endpoint_alpha=accepted["alpha"],
        original_target_change_max_rad=float(np.abs(endpoint - nominal).max()),
        refined_target_change_max_rad=float(np.abs(accepted["target"] - nominal).max()),
        verified_all23_joints_all10_substeps=True,
        raw23=accepted["raw"].tolist(),
        independent_probe_physics_steps=len(rows) * 10,
        includes_cost_of_original_500ms_search=False,
        online_solver_or_realtime_qualified=False,
        whole_source_or_motion_slew_qualified=False,
        new_closed_loop_policy_trials=0,
        hardware_authorized=False,
        deployment_ready=False,
    )
    with (output / "report.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
