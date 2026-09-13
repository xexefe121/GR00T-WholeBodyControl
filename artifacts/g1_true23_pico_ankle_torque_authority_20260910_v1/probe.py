"""Fixed ankle feedforward counterfactuals; no robot transport or full rollout."""

import json
from pathlib import Path

import mujoco
import numpy as np

from gear_sonic.utils.g1_23dof_safe_target_transform import safe_target_transform_numpy
from gear_sonic.utils.g1_true23_clean_mujoco_teleop import CleanTrue23MujocoController
from gear_sonic.utils.g1_true23_generalist_benchmark import MODEL, PHYSICS
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file
from gear_sonic.utils.g1_true23_reference_floor import compiled_model_sha256

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ASSETS = ROOT.parent / "GR00T-WholeBodyControl"
OUT = HERE / "probe_v1"
FLAGS = dict(deployment_ready=False, hardware_authorized=False, simulator_qualified=False)


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def main():
    if OUT.exists():
        raise FileExistsError("torque authority probes refuse overwrite")
    inputs = {}

    def bind(path, expected=None):
        path = Path(path).resolve(strict=True)
        actual = sha256_file(path)
        if expected is not None:
            assert actual == expected, path
        inputs[str(path)] = actual
        return path

    def read(path, expected):
        with np.load(bind(path, expected), allow_pickle=False) as data:
            return {key: data[key].copy() for key in data.files}

    bind(__file__)
    bind(HERE / "EXPERIMENT.md")
    bind(ROOT / "gear_sonic/utils/g1_true23_clean_mujoco_teleop.py")
    bind(ASSETS / MODEL)
    bind(ROOT / PHYSICS)
    cases = {}
    for name, family, run, expected in (
        (
            "final1000",
            "g1_true23_pico_foot_precision_20260910_v1",
            "eval1000_v1",
            "7da5b318d6d30aca2c1469a5abdf1b587623932d6ac7c7e432b7046734420884",
        ),
        (
            "parent500",
            "g1_true23_pico_training_20260910_v1",
            "eval500_v1",
            "65ba159c632aaddaddae2037bf2ed7e8371a49b914ab409d0d47e837ddc6ba48",
        ),
    ):
        directory = ROOT / "artifacts" / family / run / "pico"
        report = json.loads(bind(directory / "report.json", expected).read_text())
        for path, digest in report["inputs"].items():
            bind(path, digest)
        attempts = read(directory / "attempts.npz", report["attempts_sha256"])
        index = report["result"]["completed_controls"]
        assert len(attempts["inverse23"]) == index + 1
        assert len(attempts["accepted23"]) == index
        assert (
            report["result"]["failure"]["message"]
            == "no verified next-control joint-range target found by bounded inward search"
        )
        cases[name] = dict(
            report=report,
            index=index,
            q=attempts["measured_qpos"][index],
            v=attempts["measured_qvel"][index],
            raw=attempts["inverse23"][index],
        )
    previous = ROOT / "artifacts/g1_true23_pico_ankle_dynamics_20260910_v1/capture_v2"
    prior_report = json.loads(bind(previous / "report.json").read_text())
    prior_zero = read(
        previous / "rejected_preview_0/forces.npz", prior_report["preview_candidates"][0]["arrays_sha256"]
    )
    OUT.mkdir()
    write(
        OUT / "request.json",
        dict(
            inputs=inputs,
            feedforward_offsets_nm=[0, -2.5, -5, -10],
            copied_states=list(cases),
            actual_rollout=False,
            **FLAGS,
        ),
    )
    rows, arrays = [], {}
    for name, case in cases.items():
        c = CleanTrue23MujocoController(model_path=ASSETS / MODEL, physics_path=ROOT / PHYSICS, policy=None)
        assert compiled_model_sha256(c.model) == case["report"]["result"]["compiled_model_sha256"]
        assert c.physics.effort[11] == 35
        _, target = safe_target_transform_numpy(case["raw"])
        for number, correction in enumerate((0.0, -2.5, -5.0, -10.0)):
            mujoco.mj_resetData(c.model, c.data)
            q, v = case["q"], case["v"]
            c.reset(
                base_position=q[:3],
                base_quaternion_wxyz=q[3:7],
                joint_position_hardware=q[7:],
                root_velocity=v[:6],
                joint_velocity_hardware=v[6:],
            )
            positions, velocities, pd_torques, total_requests, applied = ([] for _ in range(5))
            for step in range(50):
                pd = c.physics.kp * (target.astype(float) - c.data.qpos[7:]) - c.physics.kd * c.data.qvel[6:]
                total = pd.copy()
                total[11] += correction
                c.data.ctrl[:] = np.clip(total, -c.physics.effort, c.physics.effort)
                mujoco.mj_step(c.model, c.data)
                np.testing.assert_array_equal(c.data.qfrc_actuator[6:], c.data.ctrl)
                for collection, value in (
                    (positions, c.data.qpos),
                    (velocities, c.data.qvel),
                    (pd_torques, pd),
                    (total_requests, total),
                    (applied, c.data.ctrl),
                ):
                    collection.append(value.copy())
            positions, velocities, pd_torques, total_requests, applied = (
                np.asarray(value) for value in (positions, velocities, pd_torques, total_requests, applied)
            )
            assert all(np.isfinite(value).all() for value in (positions, velocities, applied))
            if name == "final1000" and correction == 0:
                np.testing.assert_array_equal(positions[:10], prior_zero["post_qpos"])
                np.testing.assert_array_equal(velocities[:10], prior_zero["post_qvel"])
                np.testing.assert_array_equal(applied[:10], prior_zero["applied_torque23"])
            low, high = c.model.jnt_range[1:, 0], c.model.jnt_range[1:, 1]
            excess = np.maximum(np.maximum(low + 0.0019 - positions[:, 7:], positions[:, 7:] - high + 0.0019), 0)
            key = f"{name}_{number}"
            for field, value in (
                ("qpos", positions),
                ("qvel", velocities),
                ("target23", target),
                ("pd_requested23", pd_torques),
                ("total_requested23", total_requests),
                ("applied23", applied),
            ):
                arrays[f"{key}_{field}"] = value
            row = dict(
                case=name,
                control_index=case["index"],
                feedforward_right_ankle_roll_nm=correction,
                unchanged_target_right_ankle_roll_rad=float(target[11]),
                next20ms_reserve_excess_rad=float(excess[:10].max()),
                next20ms_violating_hardware_joints=np.flatnonzero(excess[:10].max(0) > 0).tolist(),
                hold100ms_reserve_excess_rad=float(excess.max()),
                hold100ms_violating_hardware_joints=np.flatnonzero(excess.max(0) > 0).tolist(),
                right_ankle_q20_q100=positions[[9, 49], 18].tolist(),
                right_ankle_dq20_dq100=velocities[[9, 49], 17].tolist(),
                right_ankle_next20ms_actuator_torque_min_max=[
                    float(applied[:10, 11].min()),
                    float(applied[:10, 11].max()),
                ],
                maximum_total_effort_ratio=float(np.max(np.abs(applied) / c.physics.effort)),
                next20ms_checked_range_passed=bool(excess[:10].max() == 0),
                counterfactual_only=True,
                **FLAGS,
            )
            rows.append(row)
            print(json.dumps(row), flush=True)
    with (OUT / "probes.npz").open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    for path, expected in inputs.items():
        assert sha256_file(Path(path)) == expected, path
    write(
        OUT / "report.json",
        dict(
            rows=rows,
            inputs=inputs,
            probes_sha256=sha256_file(OUT / "probes.npz"),
            prior_final1000_zero_offset10substeps_bit_exact=True,
            controller_or_training_integrated=False,
            applied_to_actual_rollout=False,
            **FLAGS,
        ),
    )


if __name__ == "__main__":
    main()
