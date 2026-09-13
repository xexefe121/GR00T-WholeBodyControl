"""Offline collision geometry at all reference and measured source poses.

Scratch forward kinematics only. Zero velocity is used because signed contact
distance depends on pose; no static force is claimed as a measured contact
force. This does not reproduce the training sensor's 10-N force-history cost.
No dynamics integration, acceptance-rule changes, or robot operations.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np

from gear_sonic.utils.g1_23dof_contract import HARDWARE_23_JOINT_NAMES
from gear_sonic.utils.g1_true23_generalist_benchmark import MODEL, PHYSICS
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file
from gear_sonic.utils.g1_true23_reference_floor import compiled_model_sha256, motion_qpos
from gear_sonic.utils.g1_true23_self_collision import self_contact_rows
from gear_sonic.utils.g1_true23_sim_acquisition import audit_reference_kinematics
from gear_sonic.utils.g1_true23_step1b_mujoco import prepare_true23_model


def measure_self_contacts(model, poses):
    poses = np.asarray(poses)
    if model.nq != 30 or poses.ndim != 2 or poses.shape[1] != 30 or not len(poses) or not np.isfinite(poses).all():
        raise ValueError("self-contact diagnostic requires finite native23 qpos")
    if not np.allclose(np.linalg.norm(poses[:, 3:7], axis=1), 1, rtol=0, atol=1e-6):
        raise ValueError("self-contact qpos requires normalized wxyz root quaternions")
    data = mujoco.MjData(model)
    pairs, any_penetration, maximum, worst_frame = defaultdict(dict), [], 0.0, None
    for frame, pose in enumerate(poses):
        data.qpos[:] = pose
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        hit = False
        for contact in self_contact_rows(model, data):
            geoms = contact["geoms"]
            if contact["distance_m"] >= 0:
                continue
            depth = -contact["distance_m"]
            pairs[geoms][frame] = depth
            if depth > maximum:
                maximum, worst_frame = depth, frame
            hit = True
        if hit:
            any_penetration.append(frame)
    return dict(
        frames=len(poses),
        frames_with_robot_robot_penetration=len(any_penetration),
        penetration_frame_indices=any_penetration,
        maximum_penetration_m=maximum,
        worst_frame=worst_frame,
        pairs=[
            dict(
                geoms=[model.geom(geom).name or f"geom_{geom}" for geom in pair],
                geom_ids=list(pair),
                bodies=[model.body(int(model.geom_bodyid[geom])).name for geom in pair],
                frames=len(values),
                frame_indices=sorted(values),
                maximum_penetration_m=max(values.values()),
                worst_frame=max(values, key=values.get),
            )
            for pair, values in sorted(pairs.items())
        ],
        world_ground_contacts_excluded=True,
        positive_margin_contacts_excluded=True,
        force_history_training_penalty_reproduced=False,
        physics_integration_steps=0,
        hardware_authorized=False,
        deployment_ready=False,
    )


def validated_reference_qpos(model, arrays):
    if tuple(arrays["joint_names"].tolist()) != tuple(HARDWARE_23_JOINT_NAMES):
        raise ValueError("self-contact reference must preserve exact native23 joint order")
    core = {
        key: arrays[key]
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
    poses = motion_qpos(model, core)
    audit = audit_reference_kinematics(SimpleNamespace(model=model, module=mujoco), arrays)
    if not audit["position_fk_consistent"] or not audit["orientation_fk_consistent"]:
        raise ValueError("self-contact reference body channels differ from exact-model FK")
    return poses, audit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--evaluation-index", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("self-contact diagnostic refuses overwrite")
    inputs = {}

    def bind(path, expected=None):
        path = Path(path).resolve(strict=True)
        digest = sha256_file(path)
        if expected is not None and expected != digest:
            raise ValueError(f"self-contact input hash mismatch: {path}")
        inputs[str(path)] = digest
        return path

    bank_path, evaluation_index = bind(args.bank), bind(args.evaluation_index)
    bank, evaluations = json.loads(bank_path.read_text()), json.loads(evaluation_index.read_text())
    if evaluations["bank_report_sha256"] != inputs[str(bank_path)]:
        raise ValueError("self-contact comparison index belongs to another bank")
    root = Path(__file__).resolve().parents[2]
    from gear_sonic.utils.g1_true23_generalist_source_closure import collect_local_source_closure

    for path in collect_local_source_closure(root, [Path(__file__)]).files:
        bind(path)
    model_path, config_path = bind(args.asset_root / MODEL), bind(root / PHYSICS)
    _, model, _ = prepare_true23_model(model_path, config_path)
    compiled_identity = compiled_model_sha256(model)
    rows = []
    for clip, evaluation in zip(bank["clips"], evaluations["evaluations"], strict=True):
        if clip["name"] != evaluation["name"]:
            raise ValueError("self-contact comparison changed bank order")
        source_path = bind(clip["source_path"], clip["source_sha256"])
        with np.load(source_path, allow_pickle=False) as data:
            reference, fk_audit = validated_reference_qpos(model, {key: data[key] for key in data.files})
        report_path = bind(evaluation["report_path"], evaluation["report_sha256"])
        report = json.loads(report_path.read_text())
        nominal = report["records"][0]
        if (
            nominal["case"] != "nominal"
            or nominal["result"]["model_sha256"] != inputs[str(model_path)]
            or nominal["result"]["physics_config_sha256"] != inputs[str(config_path)]
            or nominal["result"]["compiled_model_sha256"] != compiled_identity
        ):
            raise ValueError("self-contact comparison requires the same nominal physical model")
        trace_path = bind(nominal["trace_path"], nominal["trace_sha256"])
        phase = next(row for row in report["timeline"]["phases"] if row["name"] == "source_motion")
        with np.load(trace_path, allow_pickle=False) as data:
            measured = data["qpos"][phase["control_start"] + 1 : phase["control_stop"] + 1].copy()
        if len(reference) != clip["length"] or len(measured) != len(reference):
            raise ValueError("self-contact comparison requires every complete source control")
        rows.append(
            dict(
                name=clip["name"],
                source_fk_audit=fk_audit,
                reference=measure_self_contacts(model, reference),
                measured_nominal=measure_self_contacts(model, measured),
            )
        )
        print(
            json.dumps(
                {
                    "name": clip["name"],
                    **{
                        key: {
                            "frames": rows[-1][key]["frames"],
                            "penetrating_frames": rows[-1][key]["frames_with_robot_robot_penetration"],
                            "maximum_depth_m": rows[-1][key]["maximum_penetration_m"],
                        }
                        for key in ("reference", "measured_nominal")
                    },
                }
            ),
            flush=True,
        )
    for path, expected in inputs.items():
        if sha256_file(Path(path)) != expected:
            raise ValueError(f"self-contact diagnostic input changed: {path}")
    with args.output.open("x") as stream:
        json.dump(
            dict(
                kind="native23_all_reference_and_measured_source_pose_self_contact_geometry_v2",
                compiled_physics_model_sha256=compiled_identity,
                clips=rows,
                input_bindings=inputs,
                training_force_cost_reproduced=False,
                acceptance_gates_changed=False,
                hardware_authorized=False,
                deployment_ready=False,
            ),
            stream,
            indent=2,
            allow_nan=False,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
