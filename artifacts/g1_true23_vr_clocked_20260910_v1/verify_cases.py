"""Read-only recount of saved clocked simulations, including motor velocity."""

import json
from pathlib import Path

import mujoco
import numpy as np

from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file


def main():
    root = Path(__file__).parent
    output = root / "verification.json"
    if output.exists():
        raise FileExistsError("verification refuses overwrite")
    rows = []
    for name in ("nominal", "pause", "gap", "payload", "stale-start", "end-of-stream"):
        path = root / name / "report.json"
        report = json.loads(path.read_text())
        for source, expected in report["source_files"].items():
            assert sha256_file(Path(source)) == expected, source
        assert sha256_file(Path(report["packets"])) == report["packets_sha256"]
        assert sha256_file(Path(report["trace_path"])) == report["trace_sha256"]
        model_path = next(Path(p) for p in report["source_files"] if p.endswith("/g1_23dof_rev_1_0.xml"))
        config_path = next(Path(p) for p in report["source_files"] if p.endswith("/g1_23dof_mujoco_sim2sim.json"))
        model = mujoco.MjModel.from_xml_path(str(model_path))
        physics = json.loads(config_path.read_text())["physics"]
        with np.load(report["trace_path"], allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        n = report["requested_ticks"]
        assert arrays["qpos"].shape == (n + 1, 30)
        assert arrays["physics_joint_pos"].shape == (n * 10, 23)
        np.testing.assert_allclose(arrays["simulation_time"], np.arange(n + 1) * .02, atol=1e-8, rtol=0)
        np.testing.assert_allclose(arrays["physics_time"], np.arange(1, n * 10 + 1) * .002, atol=1e-8, rtol=0)
        np.testing.assert_array_equal(arrays["physics_joint_pos"][9::10], arrays["qpos"][1:, 7:])
        np.testing.assert_array_equal(arrays["physics_joint_vel"][9::10], arrays["qvel"][1:, 6:])
        assert all(np.isfinite(array).all() for array in arrays.values())
        q = arrays["physics_joint_pos"]
        range_excess = float(np.maximum(np.maximum(model.jnt_range[1:, 0] - q, q - model.jnt_range[1:, 1]), 0).max())
        effort_ratio = float((np.abs(arrays["physics_command_torque"]) / physics["effort_limit_hardware_nm"]).max())
        velocity_ratio = float((np.abs(arrays["physics_joint_vel"]) / physics["velocity_limit_hardware_radps"]).max())
        latch = report["expected_fault_tick"]
        expected_mode = np.zeros(n, dtype=bool)
        if latch is not None:
            expected_mode[latch:] = True
        np.testing.assert_array_equal(arrays["fallback_mode"], expected_mode)
        if name == "nominal":
            assert report["nominal_baseline_bit_exact"] is True
        if name == "end-of-stream":
            assert n == report["source_controls"] + 250
            assert float(arrays["simulation_time"][-1]) == np.float64(n * .02) or abs(float(arrays["simulation_time"][-1]) - n * .02) < 1e-8
        assert report["scenario_screen_passed"] and not report["tracking_qualified"] and not report["hardware_authorized"]
        rows.append(dict(scenario=name, controls=n, substeps=n * 10, fault_tick=latch,
                         maximum_joint_range_excess_rad=range_excess,
                         maximum_command_effort_ratio=effort_ratio, maximum_joint_velocity_ratio=velocity_ratio,
                         position_effort_velocity_passed=range_excess <= 1e-6 and effort_ratio <= 1 + 1e-8 and velocity_ratio <= 1,
                         report_sha256=sha256_file(path), trace_sha256=report["trace_sha256"]))
    result = dict(cases=rows, total_new_physics_substeps=sum(r["substeps"] for r in rows),
                  all_position_effort_velocity_screens_passed=all(r["position_effort_velocity_passed"] for r in rows),
                  verification_source_sha256=sha256_file(Path(__file__)),
                  new_simulation_run=False, source_tracking_qualified=False, hardware_authorized=False,
                  real_time_receiver_or_physical_handback_proven=False, deployment_ready=False)
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
