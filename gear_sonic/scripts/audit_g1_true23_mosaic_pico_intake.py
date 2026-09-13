"""Read-only public PICO-derived reference intake; no integration or motor channel."""

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation

from gear_sonic.utils.g1_23dof_contract import SOURCE_IL29_JOINT_NAMES
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file

SAMPLE_SHA256 = "521caae77299a9fe9aedaf14285dbea5ab525fdd909a031328cedb5618a2b993"
URDF_SHA256 = "dbc8471004e9b0b85c27a4ebc1f0d90a727c1cf7ca9ef507b805596f2004d64b"


def urdf_links(path):
    tree = ET.parse(path).getroot()
    joints = []
    for element in tree.findall("joint"):
        origin, axis = element.find("origin"), element.find("axis")
        joints.append(
            dict(
                name=element.attrib["name"],
                type=element.attrib["type"],
                parent=element.find("parent").attrib["link"],
                child=element.find("child").attrib["link"],
                xyz=np.fromstring(origin.get("xyz", "0 0 0") if origin is not None else "0 0 0", sep=" "),
                rpy=np.fromstring(origin.get("rpy", "0 0 0") if origin is not None else "0 0 0", sep=" "),
                axis=np.fromstring(axis.get("xyz", "1 0 0") if axis is not None else "1 0 0", sep=" "),
            )
        )
    roots = {link.attrib["name"] for link in tree.findall("link")} - {joint["child"] for joint in joints}
    if len(roots) != 1 or any(joint["type"] not in ("fixed", "revolute", "continuous") for joint in joints):
        raise ValueError("unsupported publisher robot tree")
    return joints, roots.pop()


def poses_for_order(arrays, joints, root, order):
    count = len(arrays["joint_pos"])
    measured_root = arrays["body_quat_w"][:, 0]
    transforms = {
        root: (
            arrays["body_pos_w"][:, 0].astype(np.float64),
            Rotation.from_quat(measured_root[:, [1, 2, 3, 0]]).as_matrix(),
        )
    }
    angles = {name: arrays["joint_pos"][:, i].astype(np.float64) for i, name in enumerate(order)}
    pending = list(joints)
    while pending:
        progress = False
        for joint in pending[:]:
            if joint["parent"] not in transforms:
                continue
            position, rotation = transforms[joint["parent"]]
            child_pos = position + np.einsum("nij,j->ni", rotation, joint["xyz"])
            child_rotation = rotation @ Rotation.from_euler("xyz", joint["rpy"]).as_matrix()
            if joint["type"] != "fixed":
                axis = joint["axis"]
                if abs(np.linalg.norm(axis) - 1) > 1e-8:
                    raise ValueError("publisher axis is not unit length")
                child_rotation = (
                    child_rotation @ Rotation.from_rotvec(angles[joint["name"]][:, None] * axis).as_matrix()
                )
            transforms[joint["child"]] = child_pos, child_rotation
            pending.remove(joint)
            progress = True
        if not progress:
            raise ValueError("publisher tree has unresolved parents")
    names = [root] + [joint["child"] for joint in joints if joint["type"] != "fixed"]
    if len(names) != 30:
        raise ValueError("reference requires pelvis plus29 actuated child links")
    positions = np.stack([transforms[name][0] for name in names], axis=1)
    rotations = np.stack([transforms[name][1] for name in names], axis=1)
    if positions.shape != (count, 30, 3):
        raise ValueError("invalid FK output")
    return names, positions, rotations


def compare_order(arrays, joints, root, order):
    names, positions, rotations = poses_for_order(arrays, joints, root, order)
    measured = arrays["body_pos_w"].astype(np.float64)
    quaternion = arrays["body_quat_w"].reshape(-1, 4)
    measured_rotation = Rotation.from_quat(quaternion[:, [1, 2, 3, 0]]).as_matrix().reshape(-1, 30, 3, 3)
    cost = np.empty((30, 30))
    for i in range(30):
        position_cost = np.mean((positions[:, i, None] - measured) ** 2, axis=(0, 2))
        rotation_cost = np.mean((rotations[:, i, None] - measured_rotation) ** 2, axis=(0, 2, 3))
        cost[i] = position_cost + 0.01 * rotation_cost
    predicted_ids, measured_ids = linear_sum_assignment(cost)
    position_error = np.linalg.norm(positions[:, predicted_ids] - measured[:, measured_ids], axis=-1)
    relative = np.swapaxes(rotations[:, predicted_ids], -1, -2) @ measured_rotation[:, measured_ids]
    angle = Rotation.from_matrix(relative.reshape(-1, 3, 3)).magnitude()
    return dict(
        joint_order=list(order),
        body_link_to_array_index={names[i]: int(j) for i, j in zip(predicted_ids, measured_ids, strict=True)},
        max_body_position_error_m=float(position_error.max()),
        mean_body_position_error_m=float(position_error.mean()),
        max_body_orientation_error_rad=float(angle.max()),
        all250_frames_all30_body_poses_match=bool(position_error.max() <= 5e-4 and angle.max() <= 1e-3),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("PICO intake refuses overwrite")
    if sha256_file(args.sample) != SAMPLE_SHA256 or sha256_file(args.urdf) != URDF_SHA256:
        raise ValueError("public sample or publisher geometry changed")
    with np.load(args.sample, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    expected = dict(
        fps=(1,),
        joint_pos=(250, 29),
        joint_vel=(250, 29),
        body_pos_w=(250, 30, 3),
        body_quat_w=(250, 30, 4),
        body_lin_vel_w=(250, 30, 3),
        body_ang_vel_w=(250, 30, 3),
    )
    if set(arrays) != set(expected) or any(arrays[key].shape != shape for key, shape in expected.items()):
        raise ValueError("unexpected public motion schema")
    if arrays["fps"].tolist() != [50] or any(not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("invalid50Hz public source")
    norms = np.linalg.norm(arrays["body_quat_w"], axis=-1)
    if np.max(np.abs(norms - 1)) > 1e-6:
        raise ValueError("source quaternion is not normalized")
    joints, root = urdf_links(args.urdf)
    urdf_order = tuple(joint["name"] for joint in joints if joint["type"] != "fixed")
    if set(urdf_order) != set(SOURCE_IL29_JOINT_NAMES):
        raise ValueError("publisher geometry does not have the same29 named source joints")
    candidates = {"canonical_isaaclab29": tuple(SOURCE_IL29_JOINT_NAMES), "publisher_urdf_order": urdf_order}
    reports = {name: compare_order(arrays, joints, root, order) for name, order in candidates.items()}
    matches = [name for name, result in reports.items() if result["all250_frames_all30_body_poses_match"]]
    accepted = len(matches) == 1
    result = dict(
        kind="mosaic_pico_public_reference_source_intake_v1",
        inputs={
            str(args.sample.resolve()): SAMPLE_SHA256,
            str(args.urdf.resolve()): URDF_SHA256,
            str(Path(__file__).resolve()): sha256_file(Path(__file__)),
        },
        publisher_dataset_revision="55aa4c537fe1b9d0f5cdeb915f87c01ba46cfded",
        publisher_code_revision="d73e88556ceaeb30c18e4db8b15f66ea951c124c",
        publisher_label="G1/adaptor_data/pico_VR",
        processed_smoothed_reference_not_raw_headset_packets=True,
        frames=250,
        fps=50,
        sample_span_s=249 / 50,
        all_array_values_finite=True,
        root_body_assumption=root,
        candidate_joint_orders=reports,
        unique_FK_verified_order=matches[0] if accepted else None,
        source_semantics_verified=accepted,
        saved_joint_velocity_vs_central_position_derivative_max_rad_s=float(
            np.abs(np.gradient(arrays["joint_pos"], 0.02, axis=0) - arrays["joint_vel"]).max()
        ),
        saved_root_linear_velocity_vs_central_position_derivative_max_m_s=float(
            np.abs(np.gradient(arrays["body_pos_w"][:, 0], 0.02, axis=0) - arrays["body_lin_vel_w"][:, 0]).max()
        ),
        policy_executed=False,
        physics_integrated=False,
        hardware_authorized=False,
        deployment_ready=False,
    )
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result), flush=True)
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
