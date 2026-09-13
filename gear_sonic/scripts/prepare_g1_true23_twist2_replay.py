"""Import three pinned public TWIST2 recordings for simulator-only SONIC replay.

Inputs are already robot-retargeted 29-joint recordings, not raw XR24 packets.
Only native23 joint selection and rate conversion are performed, not training,
trajectory repair, speed changes, or physical robot control.
"""

import argparse
import hashlib
import io
import json
from pathlib import Path
import pickle

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from gear_sonic.scripts.prepare_g1_true23_contact_step_lifecycle import motion_from_poses
from gear_sonic.utils.g1_23dof_contract import HARDWARE_23_JOINT_NAMES, SOURCE_MJ29_KEEP_INDICES
from gear_sonic.utils.g1_true23_generalist_benchmark import MODEL, PHYSICS
from gear_sonic.utils.g1_true23_motion_reference_packet_bundle import (
    build_motion_reference_packet_bundle,
    sha256_file,
    write_exclusive_packet_bundle,
)
from gear_sonic.utils.g1_true23_step1b_mujoco import prepare_true23_model

COMMIT = "d5c7108e9ef82d1b8770e5b692f27a1294f3aa8a"
BLOBS = {
    "0807_yanjie_walk_002.pkl": "71597b5788a2c44b68dccf313b4860cf1020fe9e",
    "0807_yanjie_walk_003.pkl": "3905de55bf48eee593a521d3b561133665b5c3d5",
    "0807_yanjie_walk_008.pkl": "a1dd65ae10dc0f4abfc44eb7cf75fea8ed78e634",
}


class NumpyOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "numpy" and name in ("ndarray", "dtype"):
            return getattr(np, name)
        if module in ("numpy.core.multiarray", "numpy._core.multiarray") and name in ("_reconstruct", "scalar"):
            return getattr(np._core.multiarray, name)
        raise pickle.UnpicklingError(f"unsupported pickle global: {module}.{name}")


def load_pinned_recording(path):
    path = Path(path)
    if path.name not in BLOBS or not 0 < path.stat().st_size < 2_000_000:
        raise ValueError("only the three bounded, pinned TWIST2 recordings are accepted")
    raw = path.read_bytes()
    blob = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
    if blob != BLOBS[path.name]:
        raise ValueError("TWIST2 recording differs from pinned Git blob")
    value = NumpyOnlyUnpickler(io.BytesIO(raw)).load()
    if not isinstance(value, dict) or set(value) != {
        "fps",
        "root_pos",
        "root_rot",
        "dof_pos",
        "local_body_pos",
        "link_body_list",
    }:
        raise ValueError("TWIST2 recording schema mismatch")
    fps = float(value["fps"])
    if not np.isfinite(fps) or not 10 <= fps <= 120:
        raise ValueError("invalid recorded frame rate")
    count = len(value["root_pos"])
    for name, width in (("root_pos", 3), ("root_rot", 4), ("dof_pos", 29)):
        array = np.asarray(value[name])
        if array.shape != (count, width) or array.dtype.kind not in "fiu" or not np.isfinite(array).all():
            raise ValueError(f"invalid numeric source array: {name}")
    if not 12 <= count <= 10_000:
        raise ValueError("recording length outside diagnostic bounds")
    if not np.allclose(np.linalg.norm(value["root_rot"], axis=1), 1.0, rtol=0, atol=1e-4):
        raise ValueError("source root quaternion is not normalized")
    return value


def native23_poses(recording):
    """Preserve elapsed time; interpolate XYZ/joints linearly and XYZW with SLERP."""
    count, fps = len(recording["root_pos"]), float(recording["fps"])
    source_times = np.arange(count, dtype=np.float64) / fps
    target_times = np.arange(int(np.floor(source_times[-1] * 50)) + 1, dtype=np.float64) / 50
    poses = np.empty((len(target_times), 30), dtype=np.float64)
    for column in range(3):
        poses[:, column] = np.interp(target_times, source_times, recording["root_pos"][:, column])
    for native, source in enumerate(SOURCE_MJ29_KEEP_INDICES):
        poses[:, 7 + native] = np.interp(target_times, source_times, recording["dof_pos"][:, source])
    quaternions = Slerp(source_times, Rotation.from_quat(recording["root_rot"]))(target_times).as_quat()
    poses[:, 3:7] = quaternions[:, [3, 0, 1, 2]]
    return poses, source_times, target_times


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    source = load_pinned_recording(args.source)
    poses, source_times, target_times = native23_poses(source)
    _, model, _ = prepare_true23_model(args.asset_root / MODEL, root / PHYSICS)
    if (model.nq, model.nv, model.nu) != (30, 29, 23):
        raise ValueError("expected the actual native23 physics model")
    if tuple(model.joint(i).name for i in range(1, model.njnt)) != tuple(HARDWARE_23_JOINT_NAMES):
        raise ValueError("native23 model joint order differs from retained source order")
    motion = motion_from_poses(model, poses)
    motion_path = args.output_directory / "motion.native23.npz"
    with motion_path.open("xb") as stream:
        np.savez_compressed(stream, **motion)
    bundle = build_motion_reference_packet_bundle(motion, source_motion_sha256=sha256_file(motion_path))
    packet_path = args.output_directory / "causal_packets.json"
    write_exclusive_packet_bundle(packet_path, bundle)
    report = {
        "kind": "g1_true23_public_twist2_replay_import_v1",
        "source_path": str(args.source.resolve()),
        "source_url": f"https://github.com/amazon-far/TWIST2/blob/{COMMIT}/assets/example_motions/{args.source.name}",
        "source_commit": COMMIT,
        "source_git_blob_sha1": BLOBS[args.source.name],
        "source_sha256": sha256_file(args.source),
        "pickle_loader": "pinned_git_blob_and_restricted_numpy_only",
        "source_frames": len(source_times),
        "source_fps": float(source["fps"]),
        "source_duration_seconds": float(source_times[-1]),
        "resampled_frames": len(target_times),
        "resampled_duration_seconds": float(target_times[-1]),
        "unsampled_final_fraction_seconds": float(source_times[-1] - target_times[-1]),
        "output_fps": 50,
        "speed_factor": 1.0,
        "added_hold_frames": 0,
        "source_dof": 29,
        "retained_dof": 23,
        "retained_mj29_indices": list(SOURCE_MJ29_KEEP_INDICES),
        "retained_joint_values_clipped": False,
        "feasibility_or_tracking_fidelity_qualified": False,
        "retargeting": "direct_retained_joints_then_native23_FK_not_task_space_retargeting",
        "packet_count": len(bundle["robot_independent_reference_packets"]),
        "packet_warmup_and_tail_frames_omitted": 11,
        "packet_path": str(packet_path.resolve()),
        "packet_sha256": sha256_file(packet_path),
        "motion_path": str(motion_path.resolve()),
        "motion_sha256": sha256_file(motion_path),
        "physical_model_sha256": sha256_file(args.asset_root / MODEL),
        "source_29dof_physics_used": False,
        "raw_headset_packets": False,
        "live_headset_source_proven": False,
        "authorization": {"simulator_only": True, "hardware_authorized": False, "dds_opened": False},
    }
    with (args.output_directory / "source_report.json").open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
