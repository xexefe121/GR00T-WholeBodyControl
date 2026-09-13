"""Prepare one complete public PICO-FreeDancing OPTICAL-reference SIM experiment."""

import argparse
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys
import time
import types
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from gear_sonic.scripts.prepare_g1_true23_contact_step_lifecycle import motion_from_poses
from gear_sonic.scripts.record_g1_sonic_public29_baseline import lifecycle29
from gear_sonic.utils.g1_23dof_contract import SOURCE_MJ29_KEEP_INDICES
from gear_sonic.utils.g1_true23_generalist_benchmark import MODEL, PHYSICS
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file
from gear_sonic.utils.g1_true23_generalist_lifecycle import build_lifecycle_timeline
from gear_sonic.utils.g1_true23_original29_reference import SOURCE_JOINT_NAMES, build_original29_reference, verify_unmodified_native_pair
from gear_sonic.utils.g1_true23_pico_optical import NAMES, register_initial_se2, resample_optical, validate_pair

PINS = {
    "public_pair/gt_body_parms.pt": "8f6524c192807f4b2b7bdd6984118ef8d34e31b29400b3df50729f461cc7589f",
    "public_pair/hmd_sensor_data.pt": "947d9f5706480215b2c63b87487b39b44b052265c1dd6ceb9071bf62e4e93b6d",
    "gmr_source/motion_retarget.py": "3a4f9fb0d13b204c8dcbb2ab62db59bce9de81b1f5911167aa73f079e0c90663",
    "gmr_source/smplx_to_g1.json": "db35f79822fb3bc712cd473d672e60bb91214528524d363e44d7c9da0c5cfc9b",
    "gmr_source/g1_mocap_29dof.xml": "85a25128d42ad3cc8dd071c089806a2773a59ee0689edcc646aeb0201ae8f4c8",
    "gmr_source/LICENSE": "c5a85c0b0012230739a0ab8f30eafba7f8ee7a1b29e025d01e27fa330298e522",
}


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def save(path, **arrays):
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def load_tensors(path):
    value = torch.load(path, map_location="cpu", weights_only=True)
    if type(value) is not dict:
        raise ValueError("public tensor file must be a dict")
    return {key: item.numpy() if isinstance(item, torch.Tensor) else item for key, item in value.items()}


def ik_xml(source, output):
    tree = ET.parse(source)
    # Retain all inertials, body/joint transforms, ranges, actuators and sensors.
    # This derived model is only for IK; it is never the actual23 physics model.
    removed = []
    for parent in list(tree.iter()):
        for node in list(parent):
            if node.tag == "asset" or (node.tag == "geom" and ("mesh" in node.attrib or "material" in node.attrib)):
                removed.append(dict(tag=node.tag, attributes=node.attrib))
                parent.remove(node)
    with output.open("xb") as stream:
        tree.write(stream, encoding="utf-8", xml_declaration=True)
    return removed


def retargeter(source, model):
    package = types.ModuleType("_pico_gmr_reference")
    package.__path__ = [str(source)]
    params = types.ModuleType(package.__name__ + ".params")
    params.ROBOT_XML_DICT = {"unitree_g1": model}
    params.IK_CONFIG_DICT = {"smplx": {"unitree_g1": source / "smplx_to_g1.json"}}
    sys.modules[package.__name__] = package
    sys.modules[params.__name__] = params
    spec = importlib.util.spec_from_file_location(package.__name__ + ".motion_retarget", source / "motion_retarget.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.GeneralMotionRetargeting("smplx", "unitree_g1", actual_human_height=None, verbose=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    inputs = {}
    for relative, expected in PINS.items():
        path = args.experiment / relative
        if sha256_file(path) != expected:
            raise ValueError(f"public source pin mismatch: {relative}")
        inputs[str(path)] = expected
    for path in (Path(__file__), root / PHYSICS, args.experiment / "PROTOCOL.md", args.asset_root / MODEL,
                 args.asset_root / "gear_sonic/data/robots/g1/g1_29dof.xml"):
        inputs[str(path)] = sha256_file(path)
    gt = load_tensors(args.experiment / "public_pair/gt_body_parms.pt")
    sensors = load_tensors(args.experiment / "public_pair/hmd_sensor_data.pt")
    local, intake = validate_pair(gt, sensors)
    positions, global_r, times, queries = resample_optical(gt["joints"], local)
    quats = Rotation.from_matrix(global_r.reshape(-1, 3, 3)).as_quat()[:, [3, 0, 1, 2]].reshape(-1, 24, 4)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    write(output / "intake.json", dict(**intake, inputs=inputs, source_protocol_hz=60,
          output_hz=50, original_duration_s=float(times[-1]), resampled_frames=len(queries),
          last_resampled_timestamp_s=float(queries[-1]), unsampled_endpoint_fraction_s=float(times[-1] - queries[-1]),
          optical_reference_only=True, source_speed_factor=1.0))
    save(output / "optical_input_50hz.npz", human_joint_position_w=positions, human_global_quaternion_wxyz=quats,
         original_protocol_timestamps_s=times, resampled_protocol_timestamps_s=queries)
    removed = ik_xml(args.experiment / "gmr_source/g1_mocap_29dof.xml", output / "gmr_ik_only.xml")
    engine = retargeter(args.experiment / "gmr_source", output / "gmr_ik_only.xml")
    if (engine.model.nq, engine.model.nv, engine.model.nu) != (36, 35, 29):
        raise ValueError("GMR XML is not all29")
    names = tuple(engine.model.joint(j).name for j in range(1, 30))
    if names != tuple(SOURCE_JOINT_NAMES):
        raise ValueError("GMR joint order differs from source29")
    poses, errors = [], []
    started = time.monotonic()
    for frame in range(len(queries)):
        human = {name: (positions[frame, j].copy(), quats[frame, j].copy()) for j, name in enumerate(NAMES[:22])}
        pose = engine.retarget(human, offset_to_ground=False)
        if not np.isfinite(pose).all():
            raise ValueError(f"nonfinite GMR pose at {frame}")
        poses.append(pose)
        errors.append((engine.error1(), engine.error2()))
        if (frame + 1) % 250 == 0:
            print(json.dumps(dict(retargeted=frame + 1, total=len(queries), elapsed_s=round(time.monotonic()-started, 1))), flush=True)
    unregistered = np.asarray(poses)
    registered, registration = register_initial_se2(unregistered)
    save(output / "gmr29_source.npz", unregistered_qpos29=unregistered, registered_qpos29=registered,
         ik_task_residual_norms=np.asarray(errors), timestamps_s=queries)
    native = mujoco.MjModel.from_xml_path(str(args.asset_root / MODEL))
    geometry = mujoco.MjModel.from_xml_path(str(args.asset_root / "gear_sonic/data/robots/g1/g1_29dof.xml"))
    selected = np.column_stack((registered[:, :7], registered[:, 7:][:, SOURCE_MJ29_KEEP_INDICES]))
    source = motion_from_poses(native, selected)
    motion, timeline = build_lifecycle_timeline(source, model=native, simulation_config=root / PHYSICS,
                                                return_target="planned_endpoint")
    standing = np.asarray(timeline["configured_standing_qpos"])
    standing29 = np.r_[standing[:7], np.zeros(29)]
    standing29[7 + np.asarray(SOURCE_MJ29_KEEP_INDICES)] = standing[7:]
    all29, phases = lifecycle29(registered, standing29)
    reference = build_original29_reference(geometry, all29)
    pair = verify_unmodified_native_pair(reference, motion)
    for name, arrays in (("source23.npz", source), ("lifecycle23.npz", motion), ("original29.npz", reference.arrays())):
        save(output / name, **arrays)
    timeline.update(timeline_path=str(output / "lifecycle23.npz"), timeline_sha256=sha256_file(output / "lifecycle23.npz"))
    write(output / "timeline.json", timeline)
    for name, module in list(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if name.startswith("gear_sonic.") and path and Path(path).suffix == ".py":
            inputs[str(Path(path).resolve())] = sha256_file(Path(path))
    for path, digest in inputs.items():
        if sha256_file(Path(path)) != digest:
            raise ValueError("source changed during preparation")
    write(output / "report.json", dict(kind="pico_freedancing_optical_reference_preparation_v1", inputs=inputs,
          selected_recording="female1/20231025_032712", optical_intake=intake, registration=registration,
          gmr_commit="bb1bbe40774794fceb2a7c579a3464a28e68c844", gmr_code_and_config_unchanged=True,
          mesh_stripped_ik_xml_sha256=sha256_file(output / "gmr_ik_only.xml"), removed_xml_elements=removed,
          versions={name: importlib.metadata.version(name) for name in ("mujoco", "mink", "qpsolvers", "daqp", "numpy", "scipy", "torch")},
          resampled_frames=len(queries), original_frames=len(times), requested_controls=timeline["total_requested_controls"],
          pairing=pair, timeline=timeline, source_ik_task_residual_p95=np.percentile(errors, 95, axis=0).tolist(),
          measured_sensor_only_fullbody_input=False, optical_reference_only=True,
          actual_controller_dynamics_tested=False, trajectory_dynamic_feasibility_qualified=False,
          deployment_ready=False, hardware_authorized=False,
          outputs={p.name: sha256_file(p) for p in output.iterdir() if p.is_file()}))
    print(json.dumps(dict(prepared=True, frames=len(queries), controls=timeline["total_requested_controls"],
                         report=str(output / "report.json"))), flush=True)


if __name__ == "__main__":
    main()
