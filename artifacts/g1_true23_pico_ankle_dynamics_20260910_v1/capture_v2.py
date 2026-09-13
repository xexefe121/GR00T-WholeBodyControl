"""Fixed saved-torque and rejected-preview force diagnosis, simulation only."""

import json
from pathlib import Path

import mujoco
import numpy as np

from gear_sonic.utils.g1_23dof_contract import HARDWARE_23_JOINT_NAMES
from gear_sonic.utils.g1_23dof_safe_target_transform import (
    SAFE_TARGET_HARD_LOWER_HARDWARE,
    SAFE_TARGET_HARD_UPPER_HARDWARE,
)
from gear_sonic.utils.g1_true23_clean_mujoco_teleop import CleanTrue23MujocoController
from gear_sonic.utils.g1_true23_force_balance import COMPONENTS, SolvedForceObserver
from gear_sonic.utils.g1_true23_generalist_benchmark import MODEL, PHYSICS
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file
from gear_sonic.utils.g1_true23_range_preview import RANGE_RESERVE_RAD, choose_range_target
from gear_sonic.utils.g1_true23_reference_floor import compiled_model_sha256
from gear_sonic.utils.g1_true23_source_action_codec import source_scaled_precompensation

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ASSETS = ROOT.parent / "GR00T-WholeBodyControl"
OUTPUT = HERE / "capture_v2"
STOP, BEGIN = 1880, 1780
FLAGS = dict(deployment_ready=False, hardware_authorized=False, simulator_qualified=False)
FIELDS = (
    "mass",
    "force_components",
    "forward_acceleration",
    "acceleration_components",
    "inverse_mass_diagonal",
    "self_actuator_acceleration",
    "foot_normal_load_n",
    "qfrc_smooth",
    "qfrc_constraint",
    "constraint_partition_error",
    "smooth_partition_error",
    "force_closure_error",
    "acceleration_closure_error",
)


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def reset(controller, q, v):
    controller.module.mj_resetData(controller.model, controller.data)
    controller.reset(
        base_position=q[:3],
        base_quaternion_wxyz=q[3:7],
        joint_position_hardware=q[7:],
        root_velocity=v[:6],
        joint_velocity_hardware=v[6:],
    )


def summarize(rows, pre, post, pre_v, post_v, controls):
    maximums = {key: float(max(row[key] for row in rows)) for key in FIELDS if key.endswith("error")}
    if max(maximums.values()) > 1e-7:
        raise ValueError(
            "solved force partition does not close within declared1e-7 numerical check: " + str(maximums)
        )
    states = np.asarray(post)
    velocities = np.asarray(post_v)
    forces = np.asarray([row["force_components"] for row in rows])
    parts = np.asarray([row["acceleration_components"] for row in rows])
    own = np.asarray([row["self_actuator_acceleration"] for row in rows])
    items = []
    for joint in (4, 5, 10, 11):
        dof = 6 + joint
        item = dict(
            joint=HARDWARE_23_JOINT_NAMES[joint],
            position_min_max_rad=[float(states[:, 7 + joint].min()), float(states[:, 7 + joint].max())],
            velocity_min_max_rad_s=[float(velocities[:, dof].min()), float(velocities[:, dof].max())],
            mean_force_components={name: float(forces[:, i, dof].mean()) for i, name in enumerate(COMPONENTS)},
            mean_acceleration_components={
                name: float(parts[:, i, dof].mean()) for i, name in enumerate(COMPONENTS)
            },
            mean_self_actuator_acceleration=float(own[:, dof].mean()),
            mean_other_actuator_acceleration=float((parts[:, 0, dof] - own[:, dof]).mean()),
            measured_velocity_increment_rad_s=float(velocities[-1, dof] - np.asarray(pre_v)[0, dof]),
        )
        items.append(item)
    return dict(
        physical_steps=len(rows),
        control_start=int(min(controls)),
        control_stop_exclusive=int(max(controls) + 1),
        maximum_numerical_partition_errors=maximums,
        ankles=items,
        average_left_right_foot_normal_load_n=np.mean(
            [row["foot_normal_load_n"] for row in rows], axis=0
        ).tolist(),
        contact_records=sum(len(row["contacts"]) for row in rows),
        integration_state_preserved_by_observer=True,
        preintegration_force_solution_not_post_pose_resolve=True,
        continuous_acceleration_not_assumed_equal_to_discrete_velocity_increment=True,
        full_motion_or_policy_improvement_claimed=False,
        **FLAGS,
    )


def save_capture(directory, rows, pre, post, pre_v, post_v, controls, targets, requested, applied):
    directory.mkdir()
    arrays = {key: np.asarray([row[key] for row in rows]) for key in FIELDS}
    arrays.update(
        pre_qpos=np.asarray(pre),
        post_qpos=np.asarray(post),
        pre_qvel=np.asarray(pre_v),
        post_qvel=np.asarray(post_v),
        control_index=np.asarray(controls),
        target23=np.asarray(targets),
        requested_torque23=np.asarray(requested),
        applied_torque23=np.asarray(applied),
    )
    assert all(np.isfinite(value).all() for value in arrays.values())
    with (directory / "forces.npz").open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    write(directory / "contacts.json", [row["contacts"] for row in rows])
    summary = summarize(rows, pre, post, pre_v, post_v, controls)
    summary.update(
        component_names=list(COMPONENTS),
        arrays_sha256=sha256_file(directory / "forces.npz"),
        contacts_sha256=sha256_file(directory / "contacts.json"),
    )
    write(directory / "summary.json", summary)
    return summary


def main():
    if OUTPUT.exists():
        raise FileExistsError("saved ankle-force diagnostic refuses overwrite")
    inputs = {}

    def bind(path, expected=None):
        path = Path(path).resolve(strict=True)
        digest = sha256_file(path)
        if expected is not None:
            assert digest == expected, path
        inputs[str(path)] = digest
        return path

    def read(path, expected=None):
        with np.load(bind(path, expected), allow_pickle=False) as archive:
            return {key: archive[key].copy() for key in archive.files}

    bind(__file__)
    bind(HERE / "EXPERIMENT.md")
    for name in ("g1_true23_force_balance.py", "g1_true23_range_preview.py", "g1_true23_clean_mujoco_teleop.py"):
        bind(ROOT / "gear_sonic/utils" / name)
    paths = {
        "parent500": ROOT / "artifacts/g1_true23_pico_training_20260910_v1/eval500_v1/pico",
        "candidate1000": ROOT / "artifacts/g1_true23_pico_foot_precision_20260910_v1/eval1000_v1/pico",
    }
    cases = {}
    for name, directory in paths.items():
        report = json.loads(bind(directory / "report.json").read_text())
        bind(directory.parent / "independent_audit.json")
        for path, digest in report["inputs"].items():
            bind(path, digest)
        trace = read(directory / "trace.npz", report["trace_sha256"])
        attempts = read(directory / "attempts.npz", report["attempts_sha256"])
        assert len(trace["qpos"]) - 1 == report["result"]["completed_controls"] >= STOP
        assert np.count_nonzero(trace["physics_external_force_world_n"]) == 0
        cases[name] = (report, trace, attempts)
    assert cases["parent500"][0]["timeline"] == cases["candidate1000"][0]["timeline"]
    OUTPUT.mkdir()
    write(
        OUTPUT / "request.json",
        dict(inputs=inputs, full_prefix_controls_verified=STOP, force_capture_controls=[BEGIN, STOP], **FLAGS),
    )
    summaries = {}
    for name, (report, trace, attempts) in cases.items():
        c = CleanTrue23MujocoController(model_path=ASSETS / MODEL, physics_path=ROOT / PHYSICS, policy=None)
        assert compiled_model_sha256(c.model) == report["result"]["compiled_model_sha256"]
        reset(c, trace["qpos"][0], trace["qvel"][0])
        observer = SolvedForceObserver(c.model)
        rows, pre, post, pre_v, post_v, controls, targets, requested, applied = ([] for _ in range(9))
        for step in range(STOP * 10):
            np.testing.assert_array_equal(c.data.qpos, trace["physics_pre_qpos"][step])
            np.testing.assert_array_equal(c.data.qvel, trace["physics_pre_qvel"][step])
            c.data.ctrl[:] = trace["applied_torque23"][step]
            mujoco.mj_step(c.model, c.data)
            np.testing.assert_array_equal(c.data.qpos, trace["physics_post_qpos"][step])
            np.testing.assert_array_equal(c.data.qvel, trace["physics_post_qvel"][step])
            np.testing.assert_array_equal(c.data.qfrc_actuator[6:], trace["engine_actuator_force23"][step])
            if step >= BEGIN * 10:
                rows.append(observer.capture(c.data))
                for target, value in (
                    (pre, trace["physics_pre_qpos"][step]),
                    (post, c.data.qpos),
                    (pre_v, trace["physics_pre_qvel"][step]),
                    (post_v, c.data.qvel),
                    (targets, trace["target23"][step // 10]),
                    (requested, trace["requested_torque23"][step]),
                    (applied, c.data.ctrl),
                ):
                    target.append(value.copy())
                controls.append(step // 10)
        summary = save_capture(
            OUTPUT / name, rows, pre, post, pre_v, post_v, controls, targets, requested, applied
        )
        summary["prefix_substeps_verified_exact"] = STOP * 10
        summaries[name] = summary
        print(json.dumps(dict(name=name, summary=summary)), flush=True)
    report, trace, attempts = cases["candidate1000"]
    assert report["result"]["completed_controls"] == STOP
    np.testing.assert_array_equal(attempts["measured_qpos"][-1], trace["qpos"][-1])
    np.testing.assert_array_equal(attempts["measured_qvel"][-1], trace["qvel"][-1])
    raw, _ = source_scaled_precompensation(attempts["released_raw23"][-1])
    np.testing.assert_array_equal(raw, attempts["inverse23"][-1])
    probe = CleanTrue23MujocoController(model_path=ASSETS / MODEL, physics_path=ROOT / PHYSICS, policy=None)
    observer = SolvedForceObserver(probe.model)
    candidate_summaries = []

    def predict(target):
        reset(probe, trace["qpos"][-1], trace["qvel"][-1])
        rows, pre, post, pre_v, post_v, controls, targets, requested, applied = ([] for _ in range(9))
        for _ in range(10):
            pre.append(probe.data.qpos.copy())
            pre_v.append(probe.data.qvel.copy())
            torque = (
                probe.physics.kp * (target.astype(np.float64) - probe.data.qpos[7:])
                - probe.physics.kd * probe.data.qvel[6:]
            )
            probe.data.ctrl[:] = np.clip(torque, -probe.physics.effort, probe.physics.effort)
            mujoco.mj_step(probe.model, probe.data)
            rows.append(observer.capture(probe.data))
            post.append(probe.data.qpos.copy())
            post_v.append(probe.data.qvel.copy())
            controls.append(STOP)
            targets.append(target.copy())
            requested.append(torque.copy())
            applied.append(probe.data.ctrl.copy())
        index = len(candidate_summaries)
        summary = save_capture(
            OUTPUT / f"rejected_preview_{index}",
            rows,
            pre,
            post,
            pre_v,
            post_v,
            controls,
            targets,
            requested,
            applied,
        )
        q = np.asarray(post)[:, 7:]
        lower = np.maximum(np.asarray(SAFE_TARGET_HARD_LOWER_HARDWARE) + RANGE_RESERVE_RAD - q, 0).max(0)
        upper = np.maximum(q - np.asarray(SAFE_TARGET_HARD_UPPER_HARDWARE) + RANGE_RESERVE_RAD, 0).max(0)
        summary.update(
            index=index,
            target23=target.tolist(),
            lower_reserve_excess23=lower.tolist(),
            upper_reserve_excess23=upper.tolist(),
        )
        candidate_summaries.append(summary)
        return np.asarray(post)

    try:
        choose_range_target(raw, predict)
    except ValueError as error:
        assert str(error) == report["result"]["failure"]["message"]
        failure = str(error)
    else:
        raise AssertionError("original rejected range search unexpectedly accepted")
    assert len(candidate_summaries) == 8
    for path, expected in inputs.items():
        assert sha256_file(Path(path)) == expected, path
    result = dict(
        kind="pico1000_saved_force_partition_and_exact_guard_rejection_v1",
        inputs=inputs,
        source_reports={name: sha256_file(path / "report.json") for name, path in paths.items()},
        cases=summaries,
        preview_candidates=candidate_summaries,
        reproduced_failure=failure,
        changed_physics_policy_reference_or_guard=False,
        **FLAGS,
    )
    write(OUTPUT / "report.json", result)
    print(
        json.dumps(dict(report=str(OUTPUT / "report.json"), candidates=len(candidate_summaries), failure=failure)),
        flush=True,
    )


if __name__ == "__main__":
    main()
