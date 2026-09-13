"""Fresh independent full-grid audit of an unchanged collision-repaired reference.

Historical reports retain their historical implementation hashes. This command
recomputes all current FK, fidelity, temporal and physical collision gates, then
copies the identical source NPZ bytes beside new evidence. No retargeting, policy,
training, physics integration or hardware operations occur here.
"""

import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import shutil

import mujoco
import numpy as np

from gear_sonic.scripts.audit_g1_true23_reference_bank_self_contacts import (
    measure_self_contacts,
    validated_reference_qpos,
)
from gear_sonic.scripts.build_g1_true23_audited_reference_bank import validate_collision_evidence
from gear_sonic.scripts.refine_g1_true23_collision_clearance import (
    collision_repair_acceptance,
    load_arrays,
    retained_diagnostic_input,
)
from gear_sonic.utils import g1_23dof_task_space_retarget as ik
from gear_sonic.utils.g1_23dof_trajectory_projection import project_nearest_trajectory
from gear_sonic.utils.g1_true23_bones_seed_source_audit import audit_original_timestamps
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
    args = parser.parse_args(argv)
    output = args.output_directory.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError("reference reaudit refuses overwrite")
    inputs = {}

    def bind(path, expected=None):
        path = Path(path).resolve(strict=True)
        digest = sha256_file(path)
        if expected is not None and digest != expected:
            raise ValueError(f"reference reaudit input changed: {path}")
        inputs[str(path)] = digest
        return path

    root = Path(__file__).resolve().parents[2]
    parent_path = bind(args.retained_directory / "report.json")
    history_path = bind(args.retained_directory / "collision_repair.json")
    parent, history = json.loads(parent_path.read_text()), json.loads(history_path.read_text())
    motion_path = bind(args.retained_directory / "adapted.true23.npz", parent["output"]["sha256"])
    for path in (args.named_source, args.source_model, args.target_model, args.asset_root / MODEL, root / PHYSICS):
        resolved = Path(path).resolve(strict=True)
        expected = parent["input_bindings"].get(str(resolved))
        if expected is None:
            raise ValueError("reference reaudit cannot substitute original source or model inputs")
        bind(resolved, expected)
    diagnostic_path, diagnostic_hash = retained_diagnostic_input(parent, args.retained_directory)
    stored = load_arrays(bind(diagnostic_path, diagnostic_hash))
    for path in collect_local_source_closure(root, [Path(__file__)]).files:
        bind(path)
    source_model = (
        mujoco.MjModel.from_binary_path(str(args.source_model))
        if args.source_model.suffix == ".mjb"
        else mujoco.MjModel.from_xml_path(str(args.source_model))
    )
    target_model = mujoco.MjModel.from_xml_path(str(args.target_model))
    _, physical_model, _ = prepare_true23_model(args.asset_root / MODEL, root / PHYSICS)
    compiled = {"source": compiled_model_sha256(source_model), "target": compiled_model_sha256(target_model)}
    physics_hash = compiled_model_sha256(physical_model)
    named, adapted = load_arrays(args.named_source), load_arrays(motion_path)
    if (
        compiled != history["compiled_models"]
        or validate_collision_evidence(
            history,
            motion_sha256=inputs[str(motion_path)],
            original_frames=parent["source_frame_count"],
            control_frames=len(adapted["joint_pos"]),
            target_model_sha256=compiled["target"],
        )
        != physics_hash
    ):
        raise ValueError("reference reaudit model or historical motion identity differs")
    poses, fk = validated_reference_qpos(physical_model, adapted)
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
    selected = parent["attempts"][parent["selected_attempt"]]
    if selected["requested_excursion_scale"] != 1.0:
        raise ValueError("reference reaudit requires every full-excursion source frame")
    original, times, _ = _resample(source, selected["requested_duration_scale"], limits)
    candidate = _reduce_excursion(original, 1.0)
    baseline = _restore_retained_diagnostic(source_model, target_model, candidate, stored, config)
    cfg = OriginalTaskConfig(**history["solver"]["joint_search"]["config"])
    low, high = ik.safe_target_joint_bounds(target_model, native_action_clip=9.5, safe_limit_guard_rad=0.05)
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
    proof = {
        "kind": "g1_true23_independent_unchanged_reference_reaudit_v1",
        "historical_collision_audit_path": str(history_path),
        "historical_collision_audit_sha256": inputs[str(history_path)],
        "historical_implementation_hashes_not_claimed_current": True,
        "all_current_source_files_bound": True,
        "reference_bytes_changed": False,
        "retargeting_performed": False,
        "iterations": [],
        "declared_path_config": asdict(cfg),
        "seed_projection_audit": asdict(projection.audit),
        "seed_projection": {"iterations": projection.iterations, "audit": asdict(projection.audit)},
    }
    refreshed = _refined_result_from_qpos(source_model, target_model, candidate, baseline, poses, proof)
    summary, fidelity, failures = _candidate_assessment(
        refreshed, _source_task_poses(source_model, original)[0], limits
    )
    original_audit = audit_original_timestamps(source_model, target_model, named, adapted, parent)
    control_contacts = measure_self_contacts(physical_model, poses)
    original_contacts = measure_self_contacts(
        physical_model, interpolate_original_poses(poses, times, source["timestamps_s"])
    )
    acceptance = collision_repair_acceptance(
        control_failures=failures,
        original_audit=original_audit,
        serialized_path=problem.audit(problem.serialized_variables(adapted)),
        control_contacts=control_contacts,
        original_contacts=original_contacts,
    )
    report = deepcopy(parent)
    report["attempts"][report["selected_attempt"]].update(
        accepted=not failures, failures=failures, ik_summary=summary, fidelity=fidelity
    )
    report.update(
        accepted=acceptance["passed"],
        input_bindings=inputs,
        independent_reference_reaudit=proof,
        hardware_authorized=False,
        deployment_ready=False,
    )
    for key in ("adapted_motion_sha256", "output", "diagnostic_artifact", "diagnostic_output"):
        report.pop(key, None)
    audit = {
        "kind": "g1_true23_full_collision_repair_audit_v1",
        "acceptance": acceptance,
        "after_original_time": original_audit,
        "source_fk_audit": fk,
        "after_control_self_contacts": control_contacts,
        "after_original_self_contacts": original_contacts,
        "control_failures": failures,
        "original_frames": len(source["joint_pos"]),
        "control_frames": len(poses),
        "solver": proof,
        "input_bindings": inputs,
        "compiled_models": compiled,
        "compiled_physics_model_sha256": physics_hash,
        "hardware_authorized": False,
        "deployment_ready": False,
    }
    if compiled_model_sha256(physical_model) != physics_hash or any(
        sha256_file(Path(p)) != h for p, h in inputs.items()
    ):
        raise ValueError("reference reaudit source or physical model changed during verification")
    output.mkdir(parents=True, exist_ok=False)
    if acceptance["passed"]:
        destination = output / "adapted.true23.npz"
        shutil.copyfile(motion_path, destination)
        if sha256_file(destination) != inputs[str(motion_path)]:
            raise ValueError("reference reaudit must preserve exact source file bytes")
        report["output"] = {"path": destination.name, "sha256": inputs[str(motion_path)]}
        audit["accepted_candidate"] = report["output"]
    else:
        report["selected_attempt"] = None
    for name, value in (("report.json", report), ("collision_repair.json", audit)):
        with (output / name).open("x") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
    print(
        json.dumps(
            {
                "output": str(output),
                "passed": acceptance["passed"],
                "control_failures": failures,
                "original_failures": original_audit["failures"],
                "source_file_sha256_unchanged": inputs[str(motion_path)],
            }
        ),
        flush=True,
    )
    return 0 if acceptance["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
