"""SIM-only endpoint extension of an accepted full native23 reference.

Retain all original control/original-time fidelity and self-contact constraints;
add discrete terminal braking room. Separate entry point leaves implementation
bound by an already running training campaign untouched. No robot operations.
"""

import argparse
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
import shutil

import numpy as np

from gear_sonic.scripts import refine_g1_true23_collision_clearance as base
from gear_sonic.utils.g1_true23_original_time_constraints import OriginalTimeProtectedTasks
from gear_sonic.utils.g1_true23_terminal_braking import TerminalBrakingTaskExtension, audit_terminal_braking
from gear_sonic.utils.g1_true23_upper_landmark_constraints import upper_landmark_groups


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
    parser.add_argument("--planned-trace", type=Path)
    parser.add_argument("--maximum-iterations", type=int, default=32)
    args = parser.parse_args(argv)
    if not 1 <= args.maximum_iterations <= 64:
        parser.error("terminal refinement is bounded to 1..64 iterations")
    output = args.output_directory.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError("terminal refinement refuses overwrite")
    bindings = {}

    def bind(path, expected=None):
        path = Path(path).resolve(strict=True)
        digest = base.sha256_file(path)
        if expected is not None and digest != expected:
            raise ValueError("terminal refinement source or model bytes differ: " + str(path))
        bindings[str(path)] = digest
        return path

    parent_path = bind(args.retained_directory / "report.json")
    history_path = bind(args.retained_directory / "collision_repair.json")
    parent, history = json.loads(parent_path.read_text()), json.loads(history_path.read_text())
    if parent.get("accepted") is not True or history.get("acceptance", {}).get("passed") is not True:
        raise ValueError("terminal refinement requires an accepted full collision-clear parent")
    motion_path = bind(args.retained_directory / "adapted.true23.npz", parent["output"]["sha256"])
    named_path = Path(args.named_source).resolve(strict=True)
    expected = parent.get("named_source_sha256", parent["input_bindings"].get(str(named_path)))
    if expected is None:
        raise ValueError("terminal refinement requires the unchanged bound named29 source")
    bind(named_path, expected)
    diagnostic = bind(*base.retained_diagnostic_input(parent, args.retained_directory))
    root = Path(__file__).resolve().parents[2]
    for path in (args.source_model, args.target_model, args.asset_root / base.MODEL, root / base.PHYSICS):
        path = Path(path).resolve(strict=True)
        if str(path) not in parent["input_bindings"]:
            raise ValueError("terminal refinement may not substitute model paths")
        bind(path, parent["input_bindings"][str(path)])
    source_model = (
        base.mujoco.MjModel.from_binary_path(str(args.source_model))
        if args.source_model.suffix == ".mjb"
        else base.mujoco.MjModel.from_xml_path(str(args.source_model))
    )
    target_model = base.mujoco.MjModel.from_xml_path(str(args.target_model))
    compiled = {
        "source": base.compiled_model_sha256(source_model),
        "target": base.compiled_model_sha256(target_model),
    }
    if compiled != history["compiled_models"]:
        raise ValueError("terminal refinement compiled FK models differ")
    _, physical_model, _ = base.prepare_true23_model(args.asset_root / base.MODEL, root / base.PHYSICS)
    physical_hash = base.compiled_model_sha256(physical_model)
    if physical_hash != history["compiled_physics_model_sha256"]:
        raise ValueError("terminal refinement changes physical collision model")
    named, adapted, stored = [base.load_arrays(p) for p in (named_path, motion_path, diagnostic)]
    if parent.get("source_field") == "planned_qpos50":
        if args.planned_trace is None:
            raise ValueError("planned choreography requires independent original trace reconstruction")
        from gear_sonic.scripts.retarget_g1_true23_generalist_planned_trace import planned_named_source

        trace_path = bind(args.planned_trace)
        with np.load(trace_path, allow_pickle=False) as trace:
            rebuilt_named = planned_named_source(trace, source_model)
        if set(named) != set(rebuilt_named) or any(not np.array_equal(named[k], rebuilt_named[k]) for k in named):
            raise ValueError("named29 source differs from complete original planned trace")
    elif args.planned_trace is not None:
        raise ValueError("planned trace cannot relabel another source type")
    for path in base.collect_local_source_closure(root, [Path(__file__)]).files:
        bind(path)
    poses, fk = base.validated_reference_qpos(physical_model, adapted)
    before_original = base.audit_original_timestamps(source_model, target_model, named, adapted, parent)
    before_contacts = base.measure_self_contacts(physical_model, poses)
    if (
        not before_original["original_timestamp_fidelity_passed"]
        or before_contacts["frames_with_robot_robot_penetration"]
    ):
        raise ValueError("terminal refinement parent fails independent original-time or physical audit")
    source = base.validate_named_motion(named, source_model, allow_source_limit_excess=True)
    config = base.ik.RetargetConfig(**parent["ik_config"])
    if "contact_flags" not in source:
        feet = base.ik._source_foot_positions(
            source_model,
            base.ik._model_layout(source_model),
            source["root_pos_w"],
            source["root_quat_wxyz"],
            source["joint_pos"],
        )
        source["contact_flags"] = base.ik.infer_foot_contacts(
            feet,
            fps=float(source["fps"][0]),
            height_tolerance_m=config.contact_height_tolerance_m,
            speed_tolerance_m_s=config.contact_speed_tolerance_m_s,
        )
    declared = deepcopy(parent["limits"])
    for key in ("duration_scales", "excursion_scales"):
        declared[key] = tuple(declared[key])
    limits = base.AdaptationLimits(**declared)
    attempt = parent["attempts"][parent["selected_attempt"]]
    if attempt["requested_excursion_scale"] != 1:
        raise ValueError("terminal refinement requires full source excursion")
    original, time_map, _ = base._resample(source, attempt["requested_duration_scale"], limits)
    candidate = base._reduce_excursion(original, 1.0)
    baseline = base._restore_retained_diagnostic(source_model, target_model, candidate, stored, config)
    low, high = base.ik.safe_target_joint_bounds(target_model, native_action_clip=9.5, safe_limit_guard_rad=0.05)
    cfg = replace(
        base.OriginalTaskConfig(**history["solver"]["joint_search"]["config"]),
        maximum_iterations=args.maximum_iterations,
    )
    prior = stored["diagnostic_fixed_root_qpos_native23"][:, 7:]
    projection = base.project_nearest_trajectory(
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
    problem = base.OriginalTaskPath(
        source_model, target_model, stored["diagnostic_requested_qpos29"], projection.projected_path, config=cfg
    )
    initial = problem.serialized_variables(adapted)
    original_tasks = OriginalTimeProtectedTasks(problem, source_model, target_model, source, time_map, config)
    upper_groups, upper_control = upper_landmark_groups(problem, initial, limits.hand_head_p95_m)
    upper_original = original_tasks.protect_upper_landmarks(initial, limits.hand_head_p95_m)
    extended = TerminalBrakingTaskExtension(original_tasks, initial, low, high)
    before_braking = audit_terminal_braking(adapted["joint_pos"], low, high)
    print(json.dumps({"stage": "terminal_braking_input", **before_braking}), flush=True)
    fitted, solver = base.fit_collision_clearance_path(
        problem,
        initial,
        baseline,
        physical_model,
        progress=lambda row: print(json.dumps(row), flush=True),
        strategy="elastic_squared_clearance_v3",
        solver_form="eliminated_residual_v2",
        objective_profile="pose_escape_guided_v3",
        objective_target=initial,
        control_source_times=time_map,
        original_times=source["timestamps_s"],
        original_time_tasks=extended,
        collision_merit_profile="sum_squared_clearance_v2",
        upper_landmark_constraints=upper_groups,
        line_search_profile="extended_bisection_v2",
    )
    solver.update(
        terminal_braking_constraint_added=True,
        original_noncontact_and_final_acceptance_constraints_changed=True,
        original_fidelity_physical_and_rate_limits_relaxed=False,
        objective_target_is_complete_accepted_parent=True,
        upper_landmark_constraints={"control_grid": upper_control, "original_time_grid": upper_original},
        joint_search={**history["solver"]["joint_search"], "config": asdict(cfg)},
        seed_projection={"iterations": projection.iterations, "audit": asdict(projection.audit)},
    )
    motion = problem.serialize(fitted)
    serialized = problem.serialized_variables(motion)
    serialized_poses = np.column_stack(
        (motion["body_pos_w"][:, 0], motion["body_quat_w"][:, 0], motion["joint_pos"])
    )
    result = base._refined_result_from_qpos(
        source_model, target_model, candidate, baseline, serialized_poses, solver
    )
    summary, fidelity, failures = base._candidate_assessment(
        result, base._source_task_poses(source_model, original)[0], limits
    )
    rebuilt = {**adapted, **motion, "achieved_task_pos_w": result.achieved_task_pos_w}
    report = deepcopy(parent)
    report["attempts"][report["selected_attempt"]].update(
        accepted=not failures, failures=failures, ik_summary=summary, fidelity=fidelity
    )
    report.update(
        input_bindings=bindings,
        terminal_braking_refinement=solver,
        hardware_authorized=False,
        deployment_ready=False,
    )
    after_original = base.audit_original_timestamps(source_model, target_model, named, rebuilt, report)
    contacts = base.measure_self_contacts(physical_model, serialized_poses)
    original_contacts = base.measure_self_contacts(
        physical_model, base.interpolate_original_poses(serialized_poses, time_map, source["timestamps_s"])
    )
    acceptance = base.collision_repair_acceptance(
        control_failures=failures,
        original_audit=after_original,
        serialized_path=problem.audit(serialized),
        control_contacts=contacts,
        original_contacts=original_contacts,
    )
    braking = audit_terminal_braking(motion["joint_pos"], low, high)
    acceptance["checks"]["terminal_joint_braking_room_passed"] = braking["passed"]
    acceptance["passed"] = all(acceptance["checks"].values())
    audit = dict(
        kind="g1_true23_full_collision_repair_audit_v1",
        acceptance=acceptance,
        before_original_time=before_original,
        after_original_time=after_original,
        source_fk_audit=fk,
        before_self_contacts=before_contacts,
        after_control_self_contacts=contacts,
        after_original_self_contacts=original_contacts,
        control_failures=failures,
        before_terminal_braking=before_braking,
        after_terminal_braking=braking,
        original_frames=len(source["joint_pos"]),
        control_frames=len(poses),
        solver=solver,
        input_bindings=bindings,
        compiled_models=compiled,
        compiled_physics_model_sha256=physical_hash,
        hardware_authorized=False,
        deployment_ready=False,
    )
    if base.compiled_model_sha256(physical_model) != physical_hash or any(
        base.sha256_file(Path(p)) != h for p, h in bindings.items()
    ):
        raise ValueError("terminal refinement input or physical model changed")
    output.mkdir(parents=True, exist_ok=False)
    for key in ("adapted_motion_sha256", "output", "diagnostic_artifact", "diagnostic_output"):
        report.pop(key, None)
    report["accepted"] = acceptance["passed"]
    if acceptance["passed"]:
        destination = output / "adapted.true23.npz"
        with destination.open("xb") as stream:
            np.savez_compressed(stream, **rebuilt)
        report["output"] = {"path": destination.name, "sha256": base.sha256_file(destination)}
        report["adapted_motion_sha256"] = report["output"]["sha256"]
        audit["accepted_candidate"] = report["output"]
        if parent.get("source_field") == "planned_qpos50":
            shutil.copyfile(named_path, output / "planned.named29.npz")
            if base.sha256_file(output / "planned.named29.npz") != bindings[str(named_path)]:
                raise ValueError("named29 source copy changed bytes")
    else:
        report["selected_attempt"] = None
        destination = output / "collision.rejected.npz"
        with destination.open("xb") as stream:
            np.savez_compressed(
                stream, diagnostic_only_not_accepted_motion=np.array([True]), qpos_native23=serialized_poses
            )
        audit["rejected_candidate"] = {"path": destination.name, "sha256": base.sha256_file(destination)}
    for name, value in (("report.json", report), ("collision_repair.json", audit)):
        with (output / name).open("x") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
    print(
        json.dumps(
            {
                "output": str(output),
                "passed": acceptance["passed"],
                "checks": acceptance["checks"],
                "terminal_braking": braking,
            }
        ),
        flush=True,
    )
    return 0 if acceptance["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
