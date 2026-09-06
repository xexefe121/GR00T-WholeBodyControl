"""Compare preserved full-length simulator receipts without promoting a policy."""

import hashlib
import json
from pathlib import Path


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    directory = Path(__file__).resolve().parent
    root = directory.parents[2]
    base = directory.parent
    bindings = {str(Path(__file__).resolve()): digest(Path(__file__))}

    def read(path):
        bindings[str(path)] = digest(path)
        return json.loads(path.read_text())

    runs = {
        "preserved_parent_root0": "planned_v5_endpoint_root0_20260907_v1",
        "rejected_uniform_root100": "planned_v5_endpoint_root100_20260907_v1",
        "experimental_feedback_priority100": "planned_v5_endpoint_priority100_20260907_v1",
    }
    reports = {name: read(base / folder / "report.json") for name, folder in runs.items()}
    shared_keys = (
        "initial_state_and_history_sha256",
        "compiled_model_sha256",
        "physics_config_sha256",
        "motion_sha256",
        "kp_hardware",
        "kd_hardware",
        "effort_limit_hardware_nm",
    )
    reference = next(iter(reports.values()))["records"][0]["result"]
    rows = []
    for name, report in reports.items():
        if [row["case"] for row in report["records"]] != ["nominal", "standing_push_x", "standing_push_y"]:
            raise ValueError("comparison requires all three ordered cases")
        for record in report["records"]:
            result, lifecycle = record["result"], record["lifecycle"]
            if any(result[key] != reference[key] for key in shared_keys):
                raise ValueError("comparison initial state, reference or physics changed")
            if result["state_pose_writes_after_reset"] or result["history_resets_during_motion"]:
                raise ValueError("comparison contains a state/history reset")
            rows.append(
                dict(
                    policy=name,
                    case=record["case"],
                    completed_controls=result["completed_controls"],
                    requested_controls=result["available_controls"],
                    failure=result["failure"],
                    root_position_p95_m=result["root_response"]["current_root_position_error_p95_m"],
                    source_motion_tracking=lifecycle["source_motion_tracking"],
                    final_proof={key: value for key, value in lifecycle.items() if key.startswith("final_proof_")},
                    source_frames_all_evaluated=lifecycle["every_original_source_frame_evaluated"],
                )
            )
    changes = {}
    parent = reports["preserved_parent_root0"]["records"]
    candidate = reports["experimental_feedback_priority100"]["records"]
    for old, new in zip(parent, candidate):
        if any(row["result"]["completed_controls"] != 1841 for row in (old, new)):
            raise ValueError("percentage comparison requires both complete denominators")
        old_errors = old["lifecycle"]["source_motion_tracking"]["landmark_position_p95_m"]
        new_errors = new["lifecycle"]["source_motion_tracking"]["landmark_position_p95_m"]
        changes[old["case"]] = {
            key: 100 * (1 - new_errors[key] / old_errors[key]) for key in old_errors
        }

    lineage = read(directory / "train/lineage.json")
    source_rows = []
    for entry in lineage["materials"]["source_files"]["files"]:
        logical = entry["logical_path"]
        if logical.startswith("gear_sonic/"):
            path = root / logical
        elif logical.startswith("causal_recovery/"):
            name = Path(logical).name
            category = "envs/mjlab" if name.startswith("sonic_") else "scripts"
            if name == "causal_history_runner.py":
                category = "trl/mjlab"
            path = root / "gear_sonic" / category / name
        elif logical.startswith("unitree_rl_mjlab/"):
            path = root.parent / "GR00T-WholeBodyControl/external_dependencies" / logical
        elif logical == "native23_generalist/actuation_config.json":
            path = root / "gear_sonic/config/sim_validation/g1_23dof_mujoco_sim2sim.json"
        elif logical == "native23_root_feedback/curriculum.spans.json":
            path = directory / "inputs/curriculum.spans.json"
        else:
            raise ValueError(f"unresolved training source material: {logical}")
        actual = digest(path)
        if actual != entry["sha256"] or path.stat().st_size != entry["size_bytes"]:
            raise ValueError(f"training material changed: {logical}")
        bindings[str(path)] = actual
        source_rows.append(dict(logical_path=logical, path=str(path), sha256=actual))

    update = read(directory / "update_verification.json")
    exported = read(directory / "export_updated/root_feedback.diagnostic.json")
    runtime = read(directory / "train/curriculum_runtime_100.json")
    optimizer = read(directory / "train/root_feedback_runtime.json")["optimizer_groups"]
    if not update["update_verification_passed"] or not exported["decoder"]["parity"]["parity_passed"]:
        raise ValueError("training/export verification did not pass")
    for component in ("encoder", "decoder"):
        path = directory / "export_updated" / exported[component]["filename"]
        bindings[str(path)] = digest(path)
        if bindings[str(path)] != exported[component]["sha256"]:
            raise ValueError("export component bytes changed")
    result = dict(
        kind="g1_root_feedback_endpoint_regression_comparison_v1",
        inputs=bindings,
        shared_simulation_contract={key: reference[key] for key in shared_keys},
        records=rows,
        complete_source_landmark_error_reduction_percent=changes,
        positive_percent_means_lower_error_negative_means_worse=True,
        stopped_uniform_candidate_not_used_for_percentage_comparison=True,
        source_materials_verified=source_rows,
        source_material_count=len(source_rows),
        training_updates=update["completed_update_count"],
        environment_transitions=51200,
        completed_training_reference_timelines=runtime["completed_reference_timelines"],
        completed_training_timelines_are_not_tracking_qualification=True,
        effective_optimizer_groups=optimizer,
        export_parity={name: exported[name]["parity"] for name in ("encoder", "decoder")},
        candidate_selection="no_promotion_tracking_and_standing_still_fail",
        overall_policy_improvement_proven=False,
        dynamics_or_contact_qualified=False,
        physical_estimator_qualified=False,
        deployment_ready=False,
        hardware_authorized=False,
    )
    for path, expected in bindings.items():
        if digest(Path(path)) != expected:
            raise ValueError(f"comparison input changed: {path}")
    with (directory / "comparison.json").open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
    print(json.dumps(dict(source_material_count=len(source_rows), landmark_error_reduction_percent=changes)))


if __name__ == "__main__":
    main()
