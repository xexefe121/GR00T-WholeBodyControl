"""Offline geometric audit of complete evaluated lifecycle references and states.

Checks acquisition/return ramps as well as source motions on the exact bound
physical model. Reuses recorded states without integration or policy inference.
Signed geometric overlap is not a force-history cost or dynamic qualification.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np

from gear_sonic.scripts.audit_g1_true23_reference_bank_self_contacts import measure_self_contacts
from gear_sonic.utils.g1_true23_generalist_benchmark import MODEL, PHYSICS
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file
from gear_sonic.utils.g1_true23_generalist_source_closure import collect_local_source_closure
from gear_sonic.utils.g1_true23_reference_floor import compiled_model_sha256, motion_qpos
from gear_sonic.utils.g1_true23_sim_acquisition import audit_reference_kinematics
from gear_sonic.utils.g1_true23_step1b_mujoco import prepare_true23_model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-directory", action="append", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if not 1 <= len(args.evaluation_directory) <= 4:
        parser.error("select 1..4 complete local evaluations")
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("lifecycle geometry audit refuses overwrite")
    inputs = {}

    def bind(path, expected=None):
        path = Path(path).resolve(strict=True)
        digest = sha256_file(path)
        if expected is not None and digest != expected:
            raise ValueError(f"lifecycle geometry input changed: {path}")
        inputs[str(path)] = digest
        return path

    root = Path(__file__).resolve().parents[2]
    for path in (
        args.asset_root / MODEL,
        root / PHYSICS,
        *collect_local_source_closure(root, [Path(__file__)]).files,
    ):
        bind(path)
    _, model, _ = prepare_true23_model(args.asset_root / MODEL, root / PHYSICS)
    model_hash = compiled_model_sha256(model)
    evaluations = []
    for directory in args.evaluation_directory:
        report_path = bind(directory / "report.json")
        report = json.loads(report_path.read_text())
        if report.get("kind") != "g1_true23_root_feedback_single_policy_lifecycle_campaign_v1":
            raise ValueError("requires a complete scheduled native23 lifecycle evaluation")
        if [row["case"] for row in report["records"]] != ["nominal", "standing_push_x", "standing_push_y"]:
            raise ValueError("all three scheduled evaluation cases must be retained")
        timeline = report["timeline"]
        motion_path = bind(timeline["timeline_path"], timeline["timeline_sha256"])
        bind(timeline["source_motion_path"], timeline["source_motion_sha256"])
        with np.load(motion_path, allow_pickle=False) as archive:
            motion = {
                key: archive[key].copy()
                for key in (
                    "fps",
                    "joint_pos",
                    "joint_vel",
                    "body_pos_w",
                    "body_quat_w",
                    "body_lin_vel_w",
                    "body_ang_vel_w",
                )
            }
        poses = motion_qpos(model, motion)
        fk = audit_reference_kinematics(SimpleNamespace(model=model, module=mujoco), motion)
        if not fk["position_fk_consistent"] or not fk["orientation_fk_consistent"]:
            raise ValueError("generated lifecycle channels differ from exact physical-model FK")
        reference_contacts = measure_self_contacts(model, poses)
        penetrating = np.asarray(reference_contacts["penetration_frame_indices"])
        phases = [
            {
                "name": phase["name"],
                "frames": phase["frame_stop"] - phase["frame_start"],
                "penetrating_frames": int(
                    np.sum((penetrating >= phase["frame_start"]) & (penetrating < phase["frame_stop"]))
                ),
            }
            for phase in timeline["phases"]
        ]
        measured = []
        for record in report["records"]:
            result = record["result"]
            if (
                result["compiled_model_sha256"] != model_hash
                or result["physics_config_sha256"] != inputs[str((root / PHYSICS).resolve())]
                or result["completed_controls"] != timeline["total_requested_controls"]
                or result["requested_controls"] != result["completed_controls"]
                or result["state_pose_writes_after_reset"] != 0
                or result["history_resets_during_motion"] != 0
            ):
                raise ValueError("geometry comparison requires unchanged physics and complete no-reset runs")
            trace_path = bind(record["trace_path"], record["trace_sha256"])
            with np.load(trace_path, allow_pickle=False) as archive:
                states = archive["qpos"].copy()
            if states.shape != (result["completed_controls"] + 1, 30):
                raise ValueError("recorded state timeline is incomplete")
            measured.append({"case": record["case"], **measure_self_contacts(model, states)})
        entry = {
            "evaluation_report": str(report_path),
            "reference_frames": len(poses),
            "reference_fk": fk,
            "reference_self_contacts": reference_contacts,
            "reference_phases": phases,
            "measured_states": measured,
            "all_reference_control_poses_collision_clear": reference_contacts[
                "frames_with_robot_robot_penetration"
            ]
            == 0,
        }
        evaluations.append(entry)
        print(
            json.dumps(
                {
                    "evaluation": str(directory),
                    "reference_frames": len(poses),
                    "reference_penetrating_frames": reference_contacts["frames_with_robot_robot_penetration"],
                    "reference_maximum_penetration_m": reference_contacts["maximum_penetration_m"],
                    "phases": phases,
                }
            ),
            flush=True,
        )
    if compiled_model_sha256(model) != model_hash or any(sha256_file(Path(p)) != h for p, h in inputs.items()):
        raise ValueError("lifecycle geometry inputs or physical model changed")
    with args.output.open("x") as stream:
        json.dump(
            {
                "kind": "g1_true23_complete_generated_lifecycle_geometry_audit_v1",
                "evaluations": evaluations,
                "input_bindings": inputs,
                "compiled_physics_model_sha256": model_hash,
                "physics_integration_steps": 0,
                "continuous_between_sample_clearance_proven": False,
                "policy_tracking_qualified": False,
                "hardware_authorized": False,
                "deployment_ready": False,
            },
            stream,
            indent=2,
            allow_nan=False,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
