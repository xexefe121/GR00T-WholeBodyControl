"""Create a separately labelled once-registered source and offline lifecycle.

This is a source-frame calibration diagnostic, not accepted training data or
the old unregistered benchmark. No policy inference or robot control occurs.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from gear_sonic.scripts.audit_g1_true23_reference_bank_self_contacts import (
    measure_self_contacts,
    validated_reference_qpos,
)
from gear_sonic.utils.g1_true23_generalist_benchmark import MODEL, PHYSICS
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file
from gear_sonic.utils.g1_true23_generalist_lifecycle import build_lifecycle_timeline
from gear_sonic.utils.g1_true23_generalist_source_closure import collect_local_source_closure
from gear_sonic.utils.g1_true23_reference_floor import compiled_model_sha256, motion_qpos
from gear_sonic.utils.g1_true23_start_registration import register_motion_start
from gear_sonic.utils.g1_true23_step1b_mujoco import prepare_true23_model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-motion", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output_directory.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError("start registration diagnostic refuses overwrite")
    root = Path(__file__).resolve().parents[2]
    source_path = args.source_motion.resolve(strict=True)
    inputs = {
        str(path.resolve()): sha256_file(path)
        for path in (
            source_path,
            args.asset_root / MODEL,
            root / PHYSICS,
            *collect_local_source_closure(root, [Path(__file__)]).files,
        )
    }
    with np.load(source_path, allow_pickle=False) as archive:
        source = {key: archive[key].copy() for key in archive.files}
    _, model, _ = prepare_true23_model(args.asset_root / MODEL, root / PHYSICS)
    compiled = compiled_model_sha256(model)
    original_poses, original_fk = validated_reference_qpos(model, source)
    registered, registration = register_motion_start(source)
    registered_poses, registered_fk = validated_reference_qpos(model, registered)
    original_contacts = measure_self_contacts(model, original_poses)
    registered_contacts = measure_self_contacts(model, registered_poses)
    if (
        original_contacts["frames_with_robot_robot_penetration"]
        or registered_contacts["frames_with_robot_robot_penetration"]
    ):
        raise ValueError("registration diagnostic requires collision-clear source before and after registration")
    lifecycle, timeline = build_lifecycle_timeline(
        {
            key: registered[key]
            for key in (
                "fps",
                "joint_pos",
                "joint_vel",
                "body_pos_w",
                "body_quat_w",
                "body_lin_vel_w",
                "body_ang_vel_w",
            )
        },
        model=model,
        simulation_config=root / PHYSICS,
        return_target="planned_endpoint",
    )
    contacts = measure_self_contacts(model, motion_qpos(model, lifecycle))
    if compiled_model_sha256(model) != compiled or any(sha256_file(Path(p)) != h for p, h in inputs.items()):
        raise ValueError("registration diagnostic inputs or physical model changed")
    output.mkdir(parents=True)
    source_output = output / "registered.source.diagnostic.npz"
    lifecycle_output = output / "lifecycle_reference.npz"
    for path, arrays in ((source_output, registered), (lifecycle_output, lifecycle)):
        with path.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
    timeline.update(
        source_motion_path=str(source_output),
        source_motion_sha256=sha256_file(source_output),
        timeline_path=str(lifecycle_output),
        timeline_sha256=sha256_file(lifecycle_output),
    )
    report = {
        "kind": "g1_true23_registered_source_lifecycle_geometry_diagnostic_v1",
        "registration": registration,
        "input_bindings": inputs,
        "compiled_physics_model_sha256": compiled,
        "original_source_fk": original_fk,
        "registered_source_fk": registered_fk,
        "original_source_self_contacts": original_contacts,
        "registered_source_self_contacts": registered_contacts,
        "generated_lifecycle_self_contacts": contacts,
        "timeline": timeline,
        "policy_evaluation_performed": False,
        "unchanged_unregistered_source_claimed": False,
        "training_reference_accepted": False,
        "hardware_authorized": False,
        "deployment_ready": False,
    }
    with (output / "report.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(
        json.dumps(
            {
                "output": str(output),
                "source_frames": len(registered_poses),
                "removed_arbitrary_start_yaw_deg": float(np.rad2deg(registration["fixed_world_yaw_rotation_rad"])),
                "source_collisions": registered_contacts["frames_with_robot_robot_penetration"],
                "generated_lifecycle_collisions": contacts["frames_with_robot_robot_penetration"],
                "deployment_ready": False,
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
