"""Independent saved-array/transport audit, not new dynamics or new timing."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(directory):
    report = json.loads((directory / "report.json").read_text())
    request = json.loads((directory / "request.json").read_text())
    assert sha(directory / "trace.npz") == report["trace_sha256"]
    assert sha(directory / "sent.json") == report["sent_sha256"]
    assert sha(directory / "received.json") == report["received_sha256"]
    resolved_sources = {}
    for logical, expected in request["source_hashes"].items():
        path = Path(logical)
        if sha(path) != expected:
            path = directory / "executed_sources" / path.name
        assert sha(path) == expected, logical
        resolved_sources[logical] = str(path)
    physics = (
        Path(request["arguments"]["repository_root"])
        / "gear_sonic/config/sim_validation/g1_23dof_mujoco_sim2sim.json"
    )
    config = json.loads(physics.read_text())["physics"]
    with np.load(directory / "trace.npz", allow_pickle=False) as values:
        z = {key: values[key] for key in values.files}
    sent = json.loads((directory / "sent.json").read_text())
    received = json.loads((directory / "received.json").read_text())
    delivery = [r["sha256"] for r in sent] == [r["sha256"] for r in received]
    assert delivery == report["all_sent_messages_received_in_order"]
    if delivery:
        # sent_ns is sampled after send() returns; another thread may already
        # have received the message. Do not invent a negative-latency failure.
        assert all(a["scheduled_ns"] <= b["received_ns"] for a, b in zip(sent, received, strict=True))
    n = report["completed_controls"]
    assert len(z["qpos"]) == len(z["qvel"]) == n + 1
    assert len(z["fallback_mode"]) == len(z["started_ns"]) == n
    assert len(z["physics_joint_pos"]) == len(z["physics_command_torque"]) == n * 10
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
    assert (z["started_ns"] >= z["deadline_ns"]).all()
    assert report["missed_compute_deadlines"] == int(missed.sum())
    assert report["fallback_controls"] == int(z["fallback_mode"].sum())
    assert report["completed_sonic_controls"] == int((~z["fallback_mode"]).sum())
    fallback = np.flatnonzero(z["fallback_mode"])
    if len(fallback):
        assert z["fallback_mode"][fallback[0] :].all()
    if report["runtime_fault"] is None:
        assert not missed.any()
        assert np.max(z["wake_lateness_ns"]) < 20_000_000
    elif report["runtime_fault"] == "control_finished_after_next_deadline":
        assert report["runtime_fault_tick"] == int(np.flatnonzero(missed)[0])
        assert len(fallback) and fallback[0] == report["runtime_fault_tick"] + 1
    admitted = ~z["fallback_mode"]
    age = z["started_ns"] - z["source_timestamp_ns"]
    assert np.all((age[admitted] >= 0) & (age[admitted] <= 100_000_000))
    velocity_ratio = float(
        np.max(np.abs(z["physics_joint_vel"]) / np.asarray(config["velocity_limit_hardware_radps"]))
    )
    effort_ratio = float(
        np.max(np.abs(z["physics_command_torque"]) / np.asarray(config["effort_limit_hardware_nm"]))
    )
    assert velocity_ratio <= 1 and effort_ratio <= 1 + 1e-12
    # Parse source MJCF range indirectly through the independently saved
    # previous fault campaign's model, not report booleans. No integration.
    import mujoco

    model = mujoco.MjModel.from_xml_path(
        str(Path(request["arguments"]["repository_root"]) / "gear_sonic/data/robots/g1/g1_23dof_rev_1_0.xml")
    )
    ranges = model.jnt_range[1:]
    q = z["physics_joint_pos"]
    excess = float(np.maximum(np.maximum(ranges[:, 0] - q, q - ranges[:, 1]), 0).max())
    assert excess == report["maximum_joint_range_excess_rad"] and excess <= 1e-6
    assert (
        not report["tracking_qualified"] and not report["deployment_ready"] and not report["hardware_authorized"]
    )
    baseline = directory.parent.parent / "g1_true23_vr_clocked_20260910_v1" / "end-of-stream" / "trace.npz"
    with np.load(baseline, allow_pickle=False) as old:
        common = report["completed_sonic_controls"] + 1
        prefix_exact = {
            key: bool(np.array_equal(z[key][:common], old[key][:common]))
            for key in ("qpos", "qvel", "simulation_time")
        }
    assert all(prefix_exact.values())
    return dict(
        case=directory.name,
        audit_passed=True,
        source_hashes_verified=len(resolved_sources),
        resolved_sources=resolved_sources,
        new_dynamics_performed=False,
        scenario_screen_passed=report["scenario_screen_passed"],
        runtime_fault=report["runtime_fault"],
        input_fault=report["input_fault"],
        controls=n,
        physics_substeps=n * 10,
        sonic_controls=report["completed_sonic_controls"],
        maximum_motor_velocity_ratio=velocity_ratio,
        maximum_command_effort_ratio=effort_ratio,
        maximum_joint_range_excess_rad=excess,
        baseline_sonic_prefix_exact=prefix_exact,
        report_sha256=sha(directory / "report.json"),
        tracking_qualified=False,
        deployment_ready=False,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).parent
    rows = [audit(root / case) for case in args.cases]
    result = dict(
        cases=rows,
        verified_physics_substeps=sum(r["physics_substeps"] for r in rows),
        auditor_sha256=sha(Path(__file__)),
        no_physical_commands=True,
        deployment_ready=False,
    )
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
    print(json.dumps({**result, "cases": [{k: v for k, v in r.items() if k != "resolved_sources"} for r in rows]}))


if __name__ == "__main__":
    main()
