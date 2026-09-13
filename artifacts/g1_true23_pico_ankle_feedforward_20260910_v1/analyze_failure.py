"""Saved-state balance/target contrast, no new control or dynamics execution."""

import json
from pathlib import Path

import numpy as np

from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file

HERE = Path(__file__).resolve().parent
DIRECTORY = HERE / "eval1000_v1/pico"
OUT = HERE / "eval1000_v1/pico_balance_failure.json"


def read(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def tilt(q):
    assert np.max(np.abs(np.linalg.norm(q, axis=-1) - 1)) < 1e-6
    return np.arccos(np.clip(1 - 2 * (q[..., 1] ** 2 + q[..., 2] ** 2), -1, 1))


def main():
    if OUT.exists():
        raise FileExistsError("saved balance analysis refuses overwrite")
    report = json.loads((DIRECTORY / "report.json").read_text())
    assert sha256_file(DIRECTORY / "trace.npz") == report["trace_sha256"]
    trace = read(DIRECTORY / "trace.npz")
    motion_path = Path(report["timeline"]["timeline_path"])
    assert sha256_file(motion_path) == report["timeline"]["timeline_sha256"]
    motion = read(motion_path)
    n = report["result"]["completed_controls"]
    indices = np.flatnonzero(np.any(trace["feedforward_torque23"], axis=1))
    assert len(indices) == 1 and indices[0] == 1880
    actual_tilt = tilt(trace["qpos"][:, 3:7])
    source = next(phase for phase in report["timeline"]["phases"] if phase["name"] == "source_motion")
    start = source["control_start"]
    rows = []
    for k in (1880, 1881, 1900, 1920, 1940, 1960, 1980, n):
        ref = k + 10
        target_root = motion["body_pos_w"][ref, 0]
        target_q = motion["body_quat_w"][ref, 0]
        rows.append(
            dict(
                completed_controls=k,
                source_seconds=(k - start) * 0.02,
                actual_root_height_m=float(trace["qpos"][k, 2]),
                target_root_height_m=float(target_root[2]),
                actual_root_tilt_rad=float(actual_tilt[k]),
                target_root_tilt_rad=float(tilt(target_q)),
                root_position_error_m=float(np.linalg.norm(trace["qpos"][k, :3] - target_root)),
                physical_right_ankle_roll_rad=float(trace["qpos"][k, 18]),
                source_target_right_ankle_roll_rad=float(motion["joint_pos"][ref, 11]),
                leg_joint_rmse_rad=float(
                    np.sqrt(np.mean((trace["qpos"][k, 7:19] - motion["joint_pos"][ref, :12]) ** 2))
                ),
                total_feedforward_previous_control_nm=trace["feedforward_torque23"][k - 1].tolist(),
            )
        )
    root_above = np.flatnonzero(trace["pelvis_error_m"][1880:] > 0.30)
    first = int(root_above[0] + 1880) if len(root_above) else None
    result = dict(
        report_sha256=sha256_file(DIRECTORY / "report.json"),
        trace_sha256=report["trace_sha256"],
        analyzer_sha256=sha256_file(Path(__file__)),
        samples=rows,
        only_feedforward_control=1880,
        feedforward_nm=trace["feedforward_torque23"][1880].tolist(),
        first_post_intervention_control_with_root_error_over30cm=first,
        existing_30cm_training_termination_is_not_a_new_or_runtime_acceptance_gate=True,
        integrated_control_count_after_single20ms_correction=n - 1881,
        final_existing_height_stop=bool(trace["qpos"][-1, 2] < 0.12),
        final_existing_tilt_stop=bool(actual_tilt[-1] > 2.2),
        new_policy_physics_or_hardware_execution=False,
        deployment_ready=False,
        hardware_authorized=False,
    )
    with OUT.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
