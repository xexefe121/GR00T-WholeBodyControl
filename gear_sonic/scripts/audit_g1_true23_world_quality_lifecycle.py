"""Check saved GPU lifecycle measurements without running physics or a robot."""

import argparse
import json
from pathlib import Path

import numpy as np

from gear_sonic.envs.mjlab.sonic_true23_world_tracking_termination import THRESHOLD_M, termination_contract
from gear_sonic.utils.g1_23dof_safe_target_transform import (
    SAFE_TARGET_HARD_LOWER_HARDWARE,
    SAFE_TARGET_HARD_UPPER_HARDWARE,
)
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file
from gear_sonic.utils.g1_true23_native_model_actuation import NativeModelActuationProfile
from gear_sonic.utils.g1_true23_world_quality import quality_contract


def audit_trace(trace, completed, profile):
    steps = len(trace["physics_post_qpos"])
    if not completed * 10 <= steps <= completed * 10 + 10:
        raise ValueError("completed controls do not match ten-substep physics")
    for key, shape in {
        "qpos": (completed + 1, 30),
        "qvel": (completed + 1, 29),
        "physics_pre_qpos": (steps, 30),
        "physics_post_qpos": (steps, 30),
        "physics_pre_qvel": (steps, 29),
        "physics_post_qvel": (steps, 29),
        "requested_torque23": (steps, 23),
        "applied_torque23": (steps, 23),
        "physics_time": (steps, 2),
    }.items():
        if trace[key].shape != shape or not np.isfinite(trace[key]).all():
            raise ValueError("bad measured shape or nonfinite: " + key)
    if steps == 0:
        raise ValueError("no integrated measurements to audit")
    for suffix in ("qpos", "qvel"):
        pre, post = trace["physics_pre_" + suffix], trace["physics_post_" + suffix]
        np.testing.assert_array_equal(pre[0], trace[suffix][0])
        np.testing.assert_array_equal(pre[1:], post[:-1])
        np.testing.assert_array_equal(trace[suffix][1:], post[9 : completed * 10 : 10])
    times = trace["physics_time"]
    np.testing.assert_array_equal(times[1:, 0], times[:-1, 1])
    np.testing.assert_allclose(times[:, 1] - times[:, 0], 0.002, rtol=0, atol=2e-6)
    # Recompute the actual eager float32 PD operations, not the double CPU plant.
    targets = np.repeat(trace["target23"], 10, axis=0)[:steps].astype(np.float32)
    q = trace["physics_pre_qpos"][:, 7:].astype(np.float32)
    v = trace["physics_pre_qvel"][:, 6:].astype(np.float32)
    requested = np.asarray(profile.kp, np.float32) * (targets - q) - np.asarray(profile.kd, np.float32) * v
    effort = np.asarray(profile.effort, np.float32)
    np.testing.assert_array_equal(requested, trace["requested_torque23"])
    np.testing.assert_array_equal(np.clip(requested, -effort, effort), trace["applied_torque23"])
    for key, saved in (("batch32_raw_repeats", "released_model_raw23"), ("batch32_token_repeats", "decoder994")):
        rows = trace[key]
        width = 23 if key == "batch32_raw_repeats" else 64
        if rows.shape != (len(trace[saved]), 8, width) or not np.isfinite(rows).all():
            raise ValueError("bad repeated-inference evidence")
        np.testing.assert_array_equal(rows[:, 0], trace[saved][:, :width])
    post_joints = trace["physics_post_qpos"][:, 7:]
    excess = np.maximum(
        np.maximum(
            np.asarray(SAFE_TARGET_HARD_LOWER_HARDWARE) - post_joints,
            post_joints - np.asarray(SAFE_TARGET_HARD_UPPER_HARDWARE),
        ),
        0,
    )
    return dict(
        completed_controls=completed,
        actual_substeps=steps,
        state_continuity_and_control_boundaries_exact=True,
        float32_native_PD_and_effort_saturation_exact=True,
        actual_range_excess_max_rad=float(excess.max()),
        actual_effort_excess_max_nm=float(np.maximum(np.abs(trace["applied_torque23"]) - effort, 0).max()),
        target_step_max_rad=float(np.max(np.abs(np.diff(trace["target23"], axis=0))))
        if len(trace["target23"]) > 1
        else 0.0,
        raw_repeat_max_abs_difference=float(
            np.max(np.abs(trace["batch32_raw_repeats"] - trace["batch32_raw_repeats"][:, :1]))
        ),
        token_repeats_exact=bool(
            np.array_equal(
                trace["batch32_token_repeats"], np.repeat(trace["batch32_token_repeats"][:, :1], 8, axis=1)
            )
        ),
        nominal_integrated_duration_s=steps * 0.002,
        actual_GPU_clock_duration_s=float(times[-1, 1] - times[0, 0]),
    )


def compare_prefix(gpu, cpu):
    count = min(len(gpu["qpos"]), len(cpu["qpos"]))
    root = np.linalg.norm(gpu["qpos"][:count, :3] - cpu["qpos"][:count, :3], axis=-1)
    joints = np.abs(gpu["qpos"][:count, 7:] - cpu["qpos"][:count, 7:]).max(axis=-1)
    tokens = min(len(gpu["decoder994"]), len(cpu["decoder994"]))
    mismatches = np.flatnonzero(np.any(gpu["decoder994"][:tokens, :64] != cpu["decoder994"][:tokens, :64], axis=1))

    def first(values, threshold):
        hits = np.flatnonzero(values > threshold)
        return int(hits[0]) if len(hits) else None

    return dict(
        shared_control_boundary_states=count,
        root_difference_max_m=float(root.max()),
        joint_difference_max_rad=float(joints.max()),
        first_boundary_root_difference_over_1cm=first(root, 0.01),
        first_boundary_joint_difference_over_1mrad=first(joints, 0.001),
        first_different_token_control=int(mismatches[0]) if len(mismatches) else None,
        different_token_control_count=len(mismatches),
        comparison_is_of_diverging_closed_loops_not_a_same_state_input_probe=True,
    )


def audit_world_measurements(trace, completed, span_start, source_qpos29):
    """Independently check the actual pre-final-forward failure measurements."""
    shapes = {
        "desired_position_w": (completed, 3),
        "measured_position_w": (completed, 3),
        "error_m": (completed,),
        "failure": (completed,),
        "reference_q0": (completed,),
        "common_step_counter": (completed,),
    }
    for key, shape in shapes.items():
        value = trace.get("world_tracking_" + key)
        if value is None or value.shape != shape or not np.isfinite(value).all():
            raise ValueError("bad actual world failure measurement: " + key)
    if completed == 0:
        raise ValueError("no completed world tracking calls")
    desired = trace["world_tracking_desired_position_w"]
    measured = trace["world_tracking_measured_position_w"]
    error = trace["world_tracking_error_m"]
    failure = trace["world_tracking_failure"]
    if any(value.dtype != np.float32 for value in (desired, measured, error)) or failure.dtype != np.bool_:
        raise ValueError("world measurement must preserve float32 and boolean execution types")
    expected_error = np.linalg.norm(desired.astype(np.float64) - measured.astype(np.float64), axis=-1)
    maximum_error = float(np.max(np.abs(expected_error - error)))
    if maximum_error > 5e-7:
        raise ValueError("saved world error differs from measured positions")
    np.testing.assert_array_equal(failure, error > np.float32(THRESHOLD_M))
    if np.any(failure & ~trace["training_episode_terminated"]):
        raise ValueError("world failure omitted from actual training termination")
    if any(trace["world_tracking_" + key].dtype != np.int64 for key in ("reference_q0", "common_step_counter")):
        raise ValueError("world reference and control counters must be int64")
    np.testing.assert_array_equal(trace["world_tracking_reference_q0"], span_start + 9 + np.arange(completed))
    np.testing.assert_array_equal(trace["world_tracking_common_step_counter"], 1 + np.arange(completed))
    # The original manager phase uses the state BEFORE the last 2ms step.
    np.testing.assert_array_equal(measured, trace["physics_pre_qpos"][9 : completed * 10 : 10, :3])
    expected_desired = source_qpos29[span_start + 10 : span_start + 10 + completed, :3].astype(np.float32)
    np.testing.assert_array_equal(desired, expected_desired)
    return dict(
        actual_world_failure_calls=completed,
        desired_q1_matches_unmodified_original_source=True,
        measured_pelvis_matches_pre_last_substep_exact=True,
        strict_threshold_and_true_failure_union_exact=True,
        maximum_norm_reconstruction_error_m=maximum_error,
        world_failure_controls=np.flatnonzero(failure).tolist(),
        error_max_m=float(error.max()),
        training_failures_recorded_without_reference_resets=True,
        training_failure_count_is_not_motion_success=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--cpu-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("audit refuses overwrite")
    inputs = {}

    def bind(path, expected=None):
        path = Path(path).resolve(strict=True)
        digest = sha256_file(path)
        if expected is not None and digest != expected:
            raise ValueError("changed audit input: " + str(path))
        inputs[str(path)] = digest
        return path

    def read(path, expected=None):
        return json.loads(bind(path, expected).read_text())

    def arrays(path, expected=None):
        with np.load(bind(path, expected), allow_pickle=False) as archive:
            return {key: archive[key].copy() for key in archive.files}

    bind(__file__)
    report = read(args.run / "report.json")
    if (
        report["kind"] != "native23_world_quality_full_lifecycle_diagnostic_v1"
        or report.get("additional_training_failure") != termination_contract()
        or report.get("world_quality_bonus") != quality_contract()
    ):
        raise ValueError("audit requires the declared four-world tracking candidate")
    physics = next(path for path in report["inputs"] if path.endswith("/g1_23dof_mujoco_sim2sim.json"))
    profile = NativeModelActuationProfile.from_sim_config(bind(physics, report["inputs"][physics]))
    spec = read(args.run / "evaluation_source_spec.json")
    merged_reference = arrays(args.run / "original_reference.npz")
    merged_motion = arrays(args.run / "motion.npz")
    cases = []
    for record in report["cases"]:
        name = record["name"]
        saved = read(args.run / name / "report.json")
        if saved != record:
            raise ValueError("case report disagrees with aggregate")
        trace = arrays(args.run / name / "trace.npz", record["trace_sha256"])
        completed = record["result"]["completed_controls"]
        row = audit_trace(trace, completed, profile)
        if row["actual_substeps"] != record["actual_integrated_substeps"]:
            raise ValueError("substep report disagrees with measured arrays")
        if row["actual_range_excess_max_rad"] > 0 and record["result"]["failure"] is None:
            raise ValueError("actual range violation omitted from outcome")
        directory = args.cpu_run / ("cpu100_" + name)
        cpu = arrays(directory / ("nominal.npz" if (directory / "nominal.npz").exists() else "failed_physics.npz"))
        if "decoder994" not in cpu:
            cpu.update(arrays(directory / "failed_adapter_attempts.npz"))
        span = next(item for item in spec["spans"] if item["name"] == name)
        world = audit_world_measurements(trace, completed, span["start"], merged_reference["source_qpos29"])
        if world["world_failure_controls"] != record["world_tracking_failure_controls"]:
            raise ValueError("world failure report differs from actual captured calls")
        phase = next(item for item in span["timeline"]["phases"] if item["name"] == "source_motion")
        start, stop = phase["control_start"], min(completed, phase["control_stop"])
        metric = dict(full_source_completed=stop == phase["control_stop"], source_controls=max(0, stop - start))
        if stop > start:
            source_rows = slice(span["start"] + 11 + start, span["start"] + 11 + stop)
            states = trace["qpos"][1 + start : 1 + stop]
            root_error = np.linalg.norm(
                states[:, :3] - merged_reference["source_qpos29"][source_rows, :3], axis=-1
            )
            joint_error = states[:, 7:19] - merged_motion["joint_pos"][source_rows, :12]
            metric["root_world_position_p95_m"] = float(np.percentile(root_error, 95))
            metric["leg_joint_rmse_rad"] = float(np.sqrt(np.mean(joint_error**2)))
            expected = record["original_task_metrics"]["unchanged_referee_post_control_q2"]
            for key in ("root_world_position_p95_m", "leg_joint_rmse_rad"):
                if metric[key] != expected[key]:
                    raise ValueError("source metric differs from recorded value: " + key)
        cases.append(
            dict(
                name=name,
                trace_audit=row,
                world_tracking=world,
                CPU_prefix_comparison=compare_prefix(trace, cpu),
                source_tracking=metric,
                result=record["result"],
                original_task_metrics=record["original_task_metrics"],
                lifecycle=record["lifecycle"],
            )
        )
    for path, digest in inputs.items():
        if sha256_file(Path(path)) != digest:
            raise ValueError("audit input changed before completion")
    result = dict(
        kind="native23_world_quality_saved_trace_audit_v1",
        inputs=inputs,
        cases=cases,
        recording_mode=report["mode"],
        original_fatal_runtime_error=report["fatal_runtime_error"],
        checked_trace_structure_not_independent_physics_reexecution=True,
        linked_run_report_binds_full_execution_source_closure=True,
        audit_rehashes_only_referenced_measurement_and_metric_inputs=True,
        no_new_integration_or_inference=True,
        hardware_authorized=False,
        deployment_ready=False,
    )
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(
        json.dumps(
            dict(
                output=str(args.output),
                sha256=sha256_file(args.output),
                actual_substeps=sum(c["trace_audit"]["actual_substeps"] for c in cases),
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
