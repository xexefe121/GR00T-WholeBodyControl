"""Independent transport arithmetic and saved-torque native-physics replay."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from gear_sonic.utils.g1_true23_clean_mujoco_teleop import CleanTrue23MujocoController
from gear_sonic.utils.g1_true23_generalist_benchmark import MODEL, PHYSICS
from gear_sonic.utils.g1_true23_reference_floor import compiled_model_sha256

ROOT = Path("/mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof")
OLD = Path("/mnt/c/Users/camer/sonic23_sim_artifacts/public_vr_legs_20260910_v1/walk002_paired/measured_trace.npz")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(directory):
    report = json.loads((directory / "report.json").read_text())
    request = json.loads((directory / "request.json").read_text())
    for path, expected in request["source_hashes"].items():
        assert sha(path) == expected, path
    assert sha(directory / "trace.npz") == report["trace_sha256"]
    assert sha(directory / "received.json") == report["received_sha256"]
    with np.load(directory / "trace.npz", allow_pickle=False) as archive:
        z = {k: archive[k].copy() for k in archive.files}
    sent = json.loads((directory / "publisher.json").read_text())["sent"]
    received = json.loads((directory / "received.json").read_text())
    delivery = [x["sha256"] for x in sent] == [x["sha256"] for x in received]
    assert delivery == report["all_sent_messages_received_in_order"]
    assert all(a["scheduled_ns"] <= b["received_ns"] for a, b in zip(sent, received, strict=True))
    n = report["completed_controls"]
    assert n == 906 and len(z["qpos"]) == n + 1 and len(z["physics_joint_pos"]) == n * 10
    assert all(np.isfinite(v).all() for v in z.values())
    np.testing.assert_allclose(z["simulation_time"], np.arange(n + 1) * 0.02, rtol=0, atol=1e-8)
    np.testing.assert_allclose(z["physics_time"], (np.arange(n * 10) + 1) * 0.002, rtol=0, atol=1e-8)
    np.testing.assert_array_equal(z["physics_joint_pos"][9::10], z["qpos"][1:, 7:])
    np.testing.assert_array_equal(z["physics_joint_vel"][9::10], z["qvel"][1:, 6:])
    np.testing.assert_array_equal(z["execution_ns"], z["finished_ns"] - z["started_ns"])
    np.testing.assert_array_equal(z["wake_lateness_ns"], z["started_ns"] - z["deadline_ns"])
    missed = z["finished_ns"] > z["deadline_ns"] + 20_000_000
    np.testing.assert_array_equal(missed, z["missed_compute_deadline"])
    np.testing.assert_array_equal(
        z["deadline_ns"][1:],
        np.where(missed[:-1], z["finished_ns"][:-1] + 20_000_000, z["deadline_ns"][:-1] + 20_000_000),
    )
    assert int(missed.sum()) == report["missed_compute_deadlines"]
    assert (z["started_ns"] >= z["deadline_ns"]).all()
    count = int((~z["fallback_mode"]).sum())
    assert count == report["completed_sonic_controls"]
    assert z["fallback_mode"][count:].all() and not z["fallback_mode"][:count].any()
    ages = (z["started_ns"] - z["source_timestamp_ns"])[:count]
    assert (ages >= 0).all() and (ages <= 100_000_000).all()
    if report["runtime_fault"] == "control_finished_after_next_deadline":
        assert report["runtime_fault_tick"] == int(np.flatnonzero(missed)[0])
        assert count == report["runtime_fault_tick"] + 1
    elif report["runtime_fault"] is None:
        assert not missed.any() and np.max(z["wake_lateness_ns"]) < 20_000_000
    else:
        assert report["runtime_fault"] == "wake_missed_entire_control_period"
        assert np.max(z["wake_lateness_ns"]) >= 20_000_000
    with np.load(OLD, allow_pickle=False) as baseline:
        exact = {
            key: bool(np.array_equal(z[key][: count + 1], baseline[key][: count + 1]))
            for key in ("qpos", "qvel", "simulation_time")
        }
    assert all(exact.values())
    root = ROOT.parent / "GR00T-WholeBodyControl"
    c = CleanTrue23MujocoController(model_path=root / MODEL, physics_path=root / PHYSICS, policy=None)
    assert compiled_model_sha256(c.model) == report["model_compiled_sha256"]
    c.reset(
        base_position=z["qpos"][0, :3],
        base_quaternion_wxyz=z["qpos"][0, 3:7],
        joint_position_hardware=z["qpos"][0, 7:],
        root_velocity=z["qvel"][0, :6],
        joint_velocity_hardware=z["qvel"][0, 6:],
    )
    maximum = dict(joint_position=0.0, joint_velocity=0.0, control_qpos=0.0, control_qvel=0.0)
    for index, torque in enumerate(z["physics_command_torque"]):
        c.data.ctrl[:] = torque
        c.module.mj_step(c.model, c.data)
        maximum["joint_position"] = max(
            maximum["joint_position"], float(np.abs(c.data.qpos[7:] - z["physics_joint_pos"][index]).max())
        )
        maximum["joint_velocity"] = max(
            maximum["joint_velocity"], float(np.abs(c.data.qvel[6:] - z["physics_joint_vel"][index]).max())
        )
        if index % 10 == 9:
            control = index // 10 + 1
            maximum["control_qpos"] = max(
                maximum["control_qpos"], float(np.abs(c.data.qpos - z["qpos"][control]).max())
            )
            maximum["control_qvel"] = max(
                maximum["control_qvel"], float(np.abs(c.data.qvel - z["qvel"][control]).max())
            )
    limits = c.model.jnt_range[1:]
    q = z["physics_joint_pos"]
    range_excess = float(np.maximum(np.maximum(limits[:, 0] - q, q - limits[:, 1]), 0).max())
    config = json.loads((root / PHYSICS).read_text())["physics"]
    velocity_ratio = float(
        np.max(np.abs(z["physics_joint_vel"]) / np.asarray(config["velocity_limit_hardware_radps"]))
    )
    torque_ratio = float(np.max(np.abs(z["physics_command_torque"]) / c.physics.effort))
    assert range_excess == report["maximum_joint_range_excess_rad"] == 0
    assert velocity_ratio <= 1 and torque_ratio <= 1
    assert (
        not report["tracking_qualified"] and not report["deployment_ready"] and not report["hardware_authorized"]
    )
    return dict(
        case=directory.name,
        source_pins_verified=len(request["source_hashes"]),
        controls=n,
        substeps=n * 10,
        sonic_controls=count,
        scenario_passed=report["scenario_passed"],
        deadline_and_delivery_arithmetic_verified=True,
        sonic_baseline_exact=exact,
        saved_torque_reintegration_max_abs_error=maximum,
        physics_bit_exact=not any(maximum.values()),
        physical_range_excess_rad=range_excess,
        maximum_velocity_ratio=velocity_ratio,
        maximum_command_effort_ratio=torque_ratio,
        report_sha256=sha(directory / "report.json"),
        publisher_sha256=sha(directory / "publisher.json"),
        hardware_authorized=False,
        deployment_ready=False,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads((args.directory / "summary.json").read_text())
    rows = []
    for case in summary["cases"]:
        row = audit(args.directory / case["case"])
        rows.append(row)
        print(json.dumps(row), flush=True)
    result = dict(
        cases=rows,
        verified_physics_substeps=sum(r["substeps"] for r in rows),
        all_physics_bit_exact=all(r["physics_bit_exact"] for r in rows),
        actual_scenarios_all_passed=summary["all_scenarios_passed"],
        auditor_sha256=sha(Path(__file__)),
        summary_sha256=sha(args.directory / "summary.json"),
        tracking_qualified=False,
        deployment_ready=False,
        robot_commands_sent=False,
    )
    with (args.directory / "independent_audit.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
