"""Offline full-reference collision repair; never robot control or qualification.

Bind an accepted named29/native23 retarget and its retained fixed-root seed.
Reuse every original task/COM/foot/derivative bound, add self-contact checks on
the exact CPU physics model, and preserve rejected candidates separately.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

import mujoco
import numpy as np

from gear_sonic.scripts.audit_g1_true23_reference_bank_self_contacts import (
    measure_self_contacts,
    validated_reference_qpos,
)
from gear_sonic.utils import g1_23dof_task_space_retarget as ik
from gear_sonic.utils.g1_23dof_trajectory_projection import project_nearest_trajectory
from gear_sonic.utils.g1_true23_bones_seed_source_audit import audit_original_timestamps
from gear_sonic.utils.g1_true23_collision_path import fit_collision_clearance_path
from gear_sonic.utils.g1_true23_collision_sampling import interpolate_original_poses
from gear_sonic.utils.g1_true23_generalist_benchmark import MODEL, PHYSICS
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file
from gear_sonic.utils.g1_true23_generalist_retarget import (
    AdaptationLimits,
    _candidate_assessment,
    _reduce_excursion,
    _refined_result_from_qpos,
    _resample,
    _restore_retained_diagnostic,
    _source_task_poses,
    validate_named_motion,
)
from gear_sonic.utils.g1_true23_generalist_source_closure import collect_local_source_closure
from gear_sonic.utils.g1_true23_original_task_trajectory import OriginalTaskConfig, OriginalTaskPath
from gear_sonic.utils.g1_true23_reference_floor import compiled_model_sha256
from gear_sonic.utils.g1_true23_step1b_mujoco import prepare_true23_model


def validate_warm_start_bindings(
    audit, source_bindings, compiled, physical_hash, *, restore_original_time_tasks=False
):
    """A rejected numerical iterate is never promoted by reusing its poses."""
    if (
        audit.get("kind") != "g1_true23_full_collision_repair_audit_v1"
        or audit.get("acceptance", {}).get("passed") is not False
        or audit.get("compiled_models") != compiled
        or audit.get("compiled_physics_model_sha256") != physical_hash
        or not audit.get("rejected_candidate")
    ):
        raise ValueError("warm start requires a bound rejected collision candidate on identical models")
    for path, expected in source_bindings.items():
        if audit.get("input_bindings", {}).get(path) != expected:
            raise ValueError(f"warm start source identity differs: {path}")
    for gate in (
        "complete_control_grid_fidelity_passed",
        "serialized_declared_path_bounds_passed",
    ):
        if audit["acceptance"].get("checks", {}).get(gate) is not True:
            raise ValueError("warm start must retain all previous full-path fidelity and temporal gates")
    if audit["acceptance"].get("checks", {}).get("all_original_timestamp_fidelity_passed") is not True:
        original = audit.get("after_original_time") or {}
        allowed = {
            "weighted_task_cost_regression",
            "com_position_regression",
            "left_foot_position",
            "right_foot_position",
            "left_foot_orientation_regression",
            "right_foot_orientation_regression",
        }
        if (
            not restore_original_time_tasks
            or original.get("all_original_timestamps_and_endpoints_evaluated") is not True
            or original.get("original_time_sample_indices") != list(range(audit["original_frames"]))
            or not original.get("failures")
            or not set(original["failures"]) <= allowed
        ):
            raise ValueError(
                "warm start needs passed original fidelity or explicit fully covered task restoration"
            )


def collision_repair_acceptance(
    *, control_failures, original_audit, serialized_path, control_contacts, original_contacts
):
    checks = {
        "complete_control_grid_fidelity_passed": control_failures == [],
        "all_original_timestamp_fidelity_passed": original_audit is not None
        and original_audit.get("original_timestamp_fidelity_passed") is True,
        "serialized_declared_path_bounds_passed": serialized_path.get("passed") is True,
        "no_control_grid_robot_robot_penetration": control_contacts["frames_with_robot_robot_penetration"] == 0,
        "no_original_grid_robot_robot_penetration": original_contacts["frames_with_robot_robot_penetration"] == 0,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "dynamic_feasibility_proven": False,
        "continuous_between_sample_clearance_proven": False,
        "hardware_authorized": False,
        "deployment_ready": False,
    }


def load_arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def retained_diagnostic_input(parent, retained_directory):
    descriptor = parent.get("diagnostic_artifact", parent.get("diagnostic_output"))
    if descriptor:
        return Path(retained_directory) / descriptor["path"], descriptor["sha256"]
    # Earlier original-time repairs retained the hash-bound seed in input
    # provenance rather than copying it into each new output directory.
    matches = [
        (Path(path), digest)
        for path, digest in parent.get("input_bindings", {}).items()
        if Path(path).name in {"fit.diagnostic.npz", "solver.diagnostic-only.npz"}
    ]
    if len(matches) != 1:
        raise ValueError("exactly one bound original fixed-root diagnostic seed is required")
    return matches[0]


def joint_search_config(profile, low, high, maximum_iterations):
    """Change only an explicit numerical seed neighborhood, never safe limits."""
    lower, upper = np.asarray(low), np.asarray(high)
    if (
        lower.shape != (23,)
        or upper.shape != (23,)
        or not np.isfinite([lower, upper]).all()
        or np.any(lower >= upper)
    ):
        raise ValueError("joint search requires finite ordered native23 safe bounds")
    if profile == "original_seed_neighborhood_v1":
        return OriginalTaskConfig(maximum_iterations=maximum_iterations)
    if profile == "full_existing_safe_envelope_v2":
        # OriginalTaskPath still intersects this neighborhood with the exact
        # unchanged native safe-target envelope. The radius is large enough
        # to avoid constraining any seed that is inside that envelope.
        return OriginalTaskConfig(
            maximum_iterations=maximum_iterations, maximum_joint_change_rad=float(np.max(upper - lower))
        )
    raise ValueError("unknown joint search profile")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "retained-directory",
        "named-source",
        "source-model",
        "target-model",
        "asset-root",
        "output-directory",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--maximum-iterations", type=int, default=48)
    parser.add_argument("--initial-candidate-audit", type=Path)
    parser.add_argument("--protect-original-time-tasks", action="store_true")
    parser.add_argument("--protect-upper-landmark-p95", action="store_true")
    parser.add_argument("--pose-escape-audit", type=Path)
    parser.add_argument(
        "--line-search-profile", choices=("four_step_v1", "extended_bisection_v2"), default="four_step_v1"
    )
    parser.add_argument(
        "--collision-merit-profile",
        choices=("maximum_clearance_v1", "sum_squared_clearance_v2"),
        default="maximum_clearance_v1",
    )
    parser.add_argument(
        "--objective-profile",
        choices=("task_lsq_v1", "minimum_change_v2", "pose_escape_guided_v3"),
        default="task_lsq_v1",
    )
    parser.add_argument(
        "--collision-grid",
        choices=("control_only_v1", "control_and_all_original_timestamps_v2"),
        default="control_only_v1",
    )
    parser.add_argument(
        "--solver-form",
        choices=("augmented_residual_v1", "eliminated_residual_v2"),
        default="augmented_residual_v1",
    )
    parser.add_argument(
        "--joint-search-profile",
        choices=("original_seed_neighborhood_v1", "full_existing_safe_envelope_v2"),
        default="original_seed_neighborhood_v1",
    )
    parser.add_argument(
        "--restoration-strategy",
        choices=("all_contacts_step_v1", "worst_contact_first_v2", "elastic_squared_clearance_v3"),
        default="all_contacts_step_v1",
    )
    args = parser.parse_args(argv)
    if args.restoration_strategy == "elastic_squared_clearance_v3" and (
        args.solver_form != "eliminated_residual_v2"
        or args.objective_profile == "task_lsq_v1"
        or args.collision_merit_profile != "sum_squared_clearance_v2"
    ):
        parser.error(
            "elastic clearance requires eliminated movement objective and explicit squared collision merit"
        )
    if args.protect_upper_landmark_p95 and not args.protect_original_time_tasks:
        parser.error("upper landmark constraints require both complete control and original timestamp grids")
    if not 1 <= args.maximum_iterations <= 64:
        parser.error("bounded offline numerical repair requires 1..64 iterations")
    if args.objective_profile != "task_lsq_v1" and args.solver_form != "eliminated_residual_v2":
        parser.error("minimum-change objective requires --solver-form eliminated_residual_v2")
    if (args.pose_escape_audit is not None) != (args.objective_profile == "pose_escape_guided_v3") or (
        args.pose_escape_audit is not None and args.initial_candidate_audit is None
    ):
        parser.error(
            "pose-guided objective requires both a bound pose-escape audit and its rejected starting path"
        )
    output = args.output_directory.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError("collision repair refuses overwrite")
    bindings = {}

    def bind(path, expected=None):
        path = Path(path).resolve(strict=True)
        actual = sha256_file(path)
        if expected is not None and actual != expected:
            raise ValueError(f"collision repair input hash mismatch: {path}")
        bindings[str(path)] = actual
        return path

    root = Path(__file__).resolve().parents[2]
    parent_path = bind(args.retained_directory / "report.json")
    parent = json.loads(parent_path.read_text())
    if parent.get("accepted") is not True:
        raise ValueError("collision repair requires an accepted full control-grid source")
    motion_path = bind(
        args.retained_directory / "adapted.true23.npz",
        parent.get("adapted_motion_sha256", parent.get("output", {}).get("sha256")),
    )
    named_path = args.named_source.resolve(strict=True)
    named_expected = parent.get("named_source_sha256", parent["input_bindings"].get(str(named_path)))
    if named_expected is None:
        raise ValueError("named source must already be bound by the retained report")
    bind(named_path, named_expected)
    diagnostic_file, diagnostic_hash = retained_diagnostic_input(parent, args.retained_directory)
    diagnostic_path = bind(diagnostic_file, diagnostic_hash)
    for path in [
        args.source_model,
        args.target_model,
        args.asset_root / MODEL,
        root / PHYSICS,
    ]:
        bind(path)
    source_identity_bindings = dict(bindings)
    for path in collect_local_source_closure(root, [Path(__file__)]).files:
        bind(path)
    source_model = (
        mujoco.MjModel.from_binary_path(str(args.source_model))
        if args.source_model.suffix == ".mjb"
        else mujoco.MjModel.from_xml_path(str(args.source_model))
    )
    target_model = mujoco.MjModel.from_xml_path(str(args.target_model))
    compiled = {"source": compiled_model_sha256(source_model), "target": compiled_model_sha256(target_model)}
    if compiled != parent.get("compiled_models", parent.get("compiled_model_sha256")):
        raise ValueError("collision repair FK model identities differ from retained source")
    _, physical_model, _ = prepare_true23_model(args.asset_root / MODEL, root / PHYSICS)
    physical_hash = compiled_model_sha256(physical_model)
    if ik._model_layout(physical_model).joint_names != ik._model_layout(target_model).joint_names:
        raise ValueError("collision and FK native23 joint orders disagree")
    named, adapted, stored = load_arrays(named_path), load_arrays(motion_path), load_arrays(diagnostic_path)
    before_audit = audit_original_timestamps(source_model, target_model, named, adapted, parent)
    if before_audit["original_timestamp_fidelity_passed"] is not True:
        raise ValueError("collision repair cannot hide an already failed original-time gate")
    poses, fk = validated_reference_qpos(physical_model, adapted)
    before_contacts = measure_self_contacts(physical_model, poses)
    source = validate_named_motion(named, source_model, allow_source_limit_excess=True)
    config = ik.RetargetConfig(**parent["ik_config"])
    if "contact_flags" not in source:
        feet = ik._source_foot_positions(
            source_model,
            ik._model_layout(source_model),
            source["root_pos_w"],
            source["root_quat_wxyz"],
            source["joint_pos"],
        )
        source["contact_flags"] = ik.infer_foot_contacts(
            feet,
            fps=float(source["fps"][0]),
            height_tolerance_m=config.contact_height_tolerance_m,
            speed_tolerance_m_s=config.contact_speed_tolerance_m_s,
        )
    declared = deepcopy(parent["limits"])
    for key in ("duration_scales", "excursion_scales"):
        declared[key] = tuple(declared[key])
    limits = AdaptationLimits(**declared)
    attempt = parent["attempts"][parent["selected_attempt"]]
    if attempt["requested_excursion_scale"] != 1:
        raise ValueError("collision repair must retain full original source excursion")
    original, time_map, _ = _resample(source, attempt["requested_duration_scale"], limits)
    candidate = _reduce_excursion(original, 1.0)
    baseline = _restore_retained_diagnostic(source_model, target_model, candidate, stored, config)
    low, high = ik.safe_target_joint_bounds(target_model, native_action_clip=9.5, safe_limit_guard_rad=0.05)
    cfg = joint_search_config(args.joint_search_profile, low, high, args.maximum_iterations)
    prior = stored["diagnostic_fixed_root_qpos_native23"][:, 7:]
    projection = project_nearest_trajectory(
        prior,
        lower_bounds=low,
        upper_bounds=high,
        dt=0.02,
        max_velocity=cfg.joint_velocity_rad_s * cfg.serialization_margin_fraction,
        max_acceleration=cfg.joint_acceleration_rad_s2 * cfg.serialization_margin_fraction,
        initial_velocity=np.clip(
            (prior[1] - prior[0]) / 0.02, -cfg.joint_velocity_rad_s, cfg.joint_velocity_rad_s
        ),
    )
    problem = OriginalTaskPath(
        source_model, target_model, stored["diagnostic_requested_qpos29"], projection.projected_path, config=cfg
    )
    initial = problem.serialized_variables(adapted)
    accepted_reference_variables = initial.copy()
    warm_start = None
    if args.initial_candidate_audit is not None:
        warm_path = bind(args.initial_candidate_audit)
        warm_audit = json.loads(warm_path.read_text())
        validate_warm_start_bindings(
            warm_audit,
            source_identity_bindings,
            compiled,
            physical_hash,
            restore_original_time_tasks=args.protect_original_time_tasks,
        )
        descriptor = warm_audit["rejected_candidate"]
        candidate_path = (warm_path.parent / descriptor["path"]).resolve(strict=True)
        if candidate_path.parent != warm_path.parent:
            raise ValueError("warm-start candidate must reside beside its audit")
        rejected = load_arrays(bind(candidate_path, descriptor["sha256"]))
        if (
            set(rejected) != {"diagnostic_only_not_accepted_motion", "qpos_native23"}
            or rejected["diagnostic_only_not_accepted_motion"].shape != (1,)
            or rejected["diagnostic_only_not_accepted_motion"].dtype != np.dtype(bool)
            or not rejected["diagnostic_only_not_accepted_motion"][0]
        ):
            raise ValueError("warm start must be explicitly marked diagnostic-only, not an accepted motion")
        warm_poses = rejected["qpos_native23"]
        if (
            warm_poses.shape != poses.shape
            or not np.isfinite(warm_poses).all()
            or np.any(np.abs(np.linalg.norm(warm_poses[:, 3:7], axis=1) - 1) > 1e-5)
        ):
            raise ValueError("warm start cannot change the full native23 pose timeline")
        initial = problem.serialized_variables(
            {
                "joint_pos": warm_poses[:, 7:],
                "body_pos_w": warm_poses[:, None, :3],
                "body_quat_w": warm_poses[:, None, 3:7],
            }
        )
        if not problem.audit(initial)["passed"]:
            raise ValueError("warm start fails current unchanged full-path bounds")
        warm_start = {
            "audit_path": str(warm_path),
            "audit_sha256": bindings[str(warm_path)],
            "candidate_sha256": descriptor["sha256"],
            "previous_candidate_remains_rejected": True,
            "all_final_acceptance_tests_rerun": True,
            "numerical_objective_and_sampling_may_change_as_explicitly_declared": True,
            "original_time_task_restoration_explicitly_requested": args.protect_original_time_tasks,
        }
    original_tasks = None
    if args.protect_original_time_tasks:
        from gear_sonic.utils.g1_true23_original_time_constraints import OriginalTimeProtectedTasks

        original_tasks = OriginalTimeProtectedTasks(problem, source_model, target_model, source, time_map, config)
    upper_groups, upper_proof = None, None
    if args.protect_upper_landmark_p95:
        from gear_sonic.utils.g1_true23_upper_landmark_constraints import upper_landmark_groups

        upper_groups, upper_proof = upper_landmark_groups(
            problem, accepted_reference_variables, limits.hand_head_p95_m
        )
        upper_proof = {
            "control_grid": upper_proof,
            "original_time_grid": original_tasks.protect_upper_landmarks(
                accepted_reference_variables, limits.hand_head_p95_m
            ),
            "anchor_is_accepted_parent_not_rejected_warm_start": True,
            "anchor_motion_sha256": bindings[str(motion_path)],
        }
    objective_target, guidance = None, None
    if args.pose_escape_audit is not None:
        from gear_sonic.utils.g1_true23_generalist_protected_root import audit_norms, residual_groups
        from gear_sonic.utils.g1_true23_pose_escape_corridor import pose_escape_objective_target

        escape_path = bind(args.pose_escape_audit)
        escape = json.loads(escape_path.read_text())
        if (
            escape.get("kind") != "g1_true23_multistart_single_pose_collision_diagnostic_v1"
            or escape.get("compiled_models") != compiled
            or escape.get("compiled_physics_model_sha256") != physical_hash
            or escape.get("input_bindings", {}).get(str(warm_path)) != bindings[str(warm_path)]
            or escape.get("accepted_training_motion") is not False
        ):
            raise ValueError("pose escape does not bind this same rejected path and exact models")
        descriptor = escape["diagnostic_output"]
        escaped_path = (escape_path.parent / descriptor["path"]).resolve(strict=True)
        if escaped_path.parent != escape_path.parent:
            raise ValueError("pose candidates must reside beside their bound diagnostic")
        escaped = load_arrays(bind(escaped_path, descriptor["sha256"]))
        options = [i for i, row in enumerate(escape["results"]) if row.get("pose_only_feasible") is True]
        frame = escape["frame"]
        if (
            not options
            or type(frame) is not int
            or not 0 <= frame < len(initial)
            or escaped["qpos_candidates"].shape != (len(escape["results"]), 30)
            or not np.array_equal(escaped.get("diagnostic_only_not_motion"), [True])
        ):
            raise ValueError("pose guidance requires a declared feasible isolated native23 pose")
        choice = options[0]  # Deterministic first passing seed, not a best-performance claim.
        trial_poses = problem.qpos(initial)
        trial_poses[frame] = escaped["qpos_candidates"][choice]
        trial_variables = problem.serialized_variables(
            {
                "joint_pos": trial_poses[:, 7:],
                "body_pos_w": trial_poses[:, None, :3],
                "body_quat_w": trial_poses[:, None, 3:7],
            }
        )
        residual = problem.evaluate(trial_variables, derivatives=False)[0]
        groups = [group for group in residual_groups(problem, baseline)[0] if group["frame"] == frame]
        if (
            not audit_norms(residual, groups)["passed"]
            or np.any(trial_variables[frame] < problem.lower[frame])
            or np.any(trial_variables[frame] > problem.upper[frame])
            or np.abs(trial_variables[frame, 3:6]).sum() > cfg.maximum_root_rotation_l1_rad
            or measure_self_contacts(physical_model, trial_poses[frame : frame + 1])[
                "frames_with_robot_robot_penetration"
            ]
            != 0
        ):
            raise ValueError("isolated guidance pose failed current independent per-frame physical/task checks")
        objective_target, guidance = pose_escape_objective_target(initial, frame, trial_variables[frame])
        guidance.update(
            audit_path=str(escape_path),
            audit_sha256=bindings[str(escape_path)],
            selected_seed_index=choice,
            guidance_pose_independently_rechecked=True,
            isolated_pose_temporal_feasibility_claimed=False,
        )
    print(
        json.dumps(
            {
                "frames": len(poses),
                "reference_penetrating_frames": before_contacts["frames_with_robot_robot_penetration"],
                "reference_maximum_penetration_m": before_contacts["maximum_penetration_m"],
            }
        ),
        flush=True,
    )
    fitted, solver = fit_collision_clearance_path(
        problem,
        initial,
        baseline,
        physical_model,
        progress=lambda row: print(json.dumps(row), flush=True),
        strategy=args.restoration_strategy,
        solver_form=args.solver_form,
        objective_profile=args.objective_profile,
        control_source_times=time_map if args.collision_grid != "control_only_v1" else None,
        original_times=source["timestamps_s"] if args.collision_grid != "control_only_v1" else None,
        original_time_tasks=original_tasks,
        objective_target=objective_target,
        collision_merit_profile=args.collision_merit_profile,
        upper_landmark_constraints=upper_groups,
        line_search_profile=args.line_search_profile,
    )
    solver["warm_start"] = warm_start
    solver["pose_escape_objective_guidance"] = guidance
    solver["upper_landmark_constraints"] = upper_proof
    solver["seed_projection"] = {"iterations": projection.iterations, "audit": asdict(projection.audit)}
    solver["joint_search"] = {
        "profile": args.joint_search_profile,
        "original_seed_neighborhood_rad": 0.6,
        "numerical_seed_radius_rad": cfg.maximum_joint_change_rad,
        "original_seed_neighborhood_preserved": args.joint_search_profile == "original_seed_neighborhood_v1",
        "existing_safe_joint_envelope_changed": False,
        "velocity_acceleration_root_or_fidelity_gates_changed": False,
        "config": asdict(cfg),
    }
    motion = problem.serialize(fitted)
    serialized = problem.serialized_variables(motion)
    serialized_pose = np.column_stack(
        (motion["body_pos_w"][:, 0], motion["body_quat_w"][:, 0], motion["joint_pos"])
    )
    result = _refined_result_from_qpos(source_model, target_model, candidate, baseline, serialized_pose, solver)
    summary, fidelity, failures = _candidate_assessment(
        result, _source_task_poses(source_model, original)[0], limits
    )
    rebuilt = {**adapted, **motion, "achieved_task_pos_w": result.achieved_task_pos_w}
    report = deepcopy(parent)
    selected = report["attempts"][report["selected_attempt"]]
    selected.update(accepted=not failures, failures=failures, ik_summary=summary, fidelity=fidelity)
    selected.setdefault("solver_attempts", []).append(
        {"strategy": "self_collision_clearance_v1", "fit_report": solver}
    )
    report.update(
        accepted=not failures, input_bindings=bindings, hardware_authorized=False, deployment_ready=False
    )
    after_audit = (
        audit_original_timestamps(source_model, target_model, named, rebuilt, report) if not failures else None
    )
    after_contacts = measure_self_contacts(physical_model, serialized_pose)
    original_contacts = measure_self_contacts(
        physical_model, interpolate_original_poses(serialized_pose, time_map, source["timestamps_s"])
    )
    acceptance = collision_repair_acceptance(
        control_failures=failures,
        original_audit=after_audit,
        serialized_path=problem.audit(serialized),
        control_contacts=after_contacts,
        original_contacts=original_contacts,
    )
    audit = {
        "kind": "g1_true23_full_collision_repair_audit_v1",
        "acceptance": acceptance,
        "before_original_time": before_audit,
        "after_original_time": after_audit,
        "source_fk_audit": fk,
        "before_self_contacts": before_contacts,
        "after_control_self_contacts": after_contacts,
        "after_original_self_contacts": original_contacts,
        "control_failures": failures,
        "original_frames": len(source["joint_pos"]),
        "control_frames": len(poses),
        "solver": solver,
        "input_bindings": bindings,
        "compiled_models": compiled,
        "compiled_physics_model_sha256": physical_hash,
        "hardware_authorized": False,
        "deployment_ready": False,
    }
    if compiled_model_sha256(physical_model) != physical_hash:
        raise ValueError("physical collision model mutated during offline fit")
    for path, expected in bindings.items():
        if sha256_file(Path(path)) != expected:
            raise ValueError(f"collision repair input changed during fit: {path}")
    output.mkdir(parents=True, exist_ok=False)
    for key in ("adapted_motion_sha256", "output", "diagnostic_artifact", "diagnostic_output"):
        report.pop(key, None)
    passed = acceptance["passed"]
    report["accepted"] = passed
    if passed:
        destination = output / "adapted.true23.npz"
        with destination.open("xb") as stream:
            np.savez_compressed(stream, **rebuilt)
        report["output"] = {"path": destination.name, "sha256": sha256_file(destination)}
        audit["accepted_candidate"] = dict(report["output"])
    else:
        report["selected_attempt"] = None
        destination = output / "collision.rejected.npz"
        with destination.open("xb") as stream:
            np.savez_compressed(
                stream, diagnostic_only_not_accepted_motion=np.array([True]), qpos_native23=serialized_pose
            )
        audit["rejected_candidate"] = {"path": destination.name, "sha256": sha256_file(destination)}
    for name, value in (("report.json", report), ("collision_repair.json", audit)):
        with (output / name).open("x") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
    print(
        json.dumps(
            {
                "path": str(output),
                "passed": passed,
                "control_failures": failures,
                "original_time_failures": after_audit["failures"] if after_audit else None,
                "remaining_penetrating_control_frames": after_contacts["frames_with_robot_robot_penetration"],
                "remaining_maximum_penetration_m": after_contacts["maximum_penetration_m"],
            }
        ),
        flush=True,
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
