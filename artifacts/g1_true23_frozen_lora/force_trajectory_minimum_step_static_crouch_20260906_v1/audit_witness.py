"""Independent serialized single-pose audit; never an accepted motion or controller."""

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np

from gear_sonic.scripts import condition_g1_true23_reference_floor as geometry
from gear_sonic.scripts.refine_g1_true23_stance_contacts import load_motion, refinement_problem, root_orientation_error
from gear_sonic.utils import g1_true23_contact_patch, g1_true23_reference_support
from gear_sonic.utils.g1_23dof_task_space_retarget import build_mjlab_motion_arrays
from gear_sonic.utils.g1_true23_actuation_profile import SIM_CONFIG, NativeSupportActuationProfile
from gear_sonic.utils.g1_true23_contact_trajectory import ContactLinearization, ContactTrajectoryConfig
from gear_sonic.utils.g1_true23_reference_floor import compiled_model_sha256, motion_qpos, reference_geometry
from gear_sonic.utils.g1_true23_reference_support import audit_reference_support, pose_path_derivatives
from gear_sonic.utils.g1_true23_sim_acquisition import audit_reference_kinematics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ARTIFACTS = ROOT / "artifacts/g1_true23_frozen_lora"


def main():
    output, archive = HERE / "serialized_witness_audit.json", HERE / "static_witness_16frames.npz"
    if output.exists() or archive.exists():
        raise FileExistsError(output)
    witness = json.loads((HERE / "report.json").read_text())
    assert witness["solver"]["failure"] is None
    assert witness["full_clip_test"] is False and witness["hardware_authorized"] is False
    identities = dict(witness["files"])
    assert all(geometry.sha256(Path(path)) == digest for path, digest in identities.items())
    old_path = ARTIFACTS / "stance_retarget_20260906_v2/pico_crouch_anchor.report.json"
    old = json.loads(old_path.read_text())
    candidate_path = ARTIFACTS / "contact_trajectory_20260906_v2/pico_crouch_anchor.npz"
    source_path = Path(old["source"])
    model_path = ROOT / geometry.DEFAULT_TARGET_MODEL
    model = mujoco.MjModel.from_xml_path(str(model_path))
    training, runtime_sources = geometry.build_training_geometry()
    models = {"retarget_mesh": model, "training_capsules": training}
    compiled = {key: compiled_model_sha256(value) for key, value in models.items()}
    assert compiled == witness["models"] == old["retarget"]["collision_model_sha256"]
    profile = NativeSupportActuationProfile.from_sim_config(ROOT / SIM_CONFIG)
    limits = np.asarray(profile.effort) * 0.2375
    problem = refinement_problem(model, load_motion(source_path), load_motion(candidate_path), old["retarget"])
    frame, repeated = witness["original_source_frame"], 16
    original_pose = problem["source_qpos"][frame]
    path = np.asarray(witness["static_path"])
    arrays = build_mjlab_motion_arrays(model, SimpleNamespace(
        root_pos_w=np.tile(original_pose[:3] + path[:3], (repeated, 1)).astype(np.float32),
        root_quat_wxyz=np.tile(original_pose[3:7], (repeated, 1)),
        joint_pos_hardware=np.tile(path[3:], (repeated, 1)).astype(np.float32), fps=50.0,
    ))
    with archive.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    serialized = load_motion(archive)
    poses = motion_qpos(model, serialized)
    serialized_path = np.column_stack((poses[:, :3] - original_pose[:3], poses[:, 7:]))
    contact = ContactLinearization(
        models, np.tile(original_pose, (repeated, 1)), [problem["supports"][frame]] * repeated,
        ContactTrajectoryConfig(),
    ).audit(serialized_path)
    orientation_error = root_orientation_error(np.tile(original_pose[3:7], (repeated, 1)), poses[:, 3:7])
    velocity, acceleration = pose_path_derivatives(model, poses, 0.02)
    bound_violation = max(
        float(np.maximum(problem["lower"][frame] - serialized_path, 0).max()),
        float(np.maximum(serialized_path - problem["upper"][frame], 0).max()),
    )
    fk = audit_reference_kinematics(SimpleNamespace(module=mujoco, model=model), serialized)
    assert contact["passed"] and bound_violation <= 2e-7 and orientation_error <= 2e-7
    assert np.max(np.abs(velocity)) == 0 and np.max(np.abs(acceleration)) == 0
    assert fk["position_fk_consistent"] and fk["orientation_fk_consistent"]
    support = {
        key: audit_reference_support(value, serialized, limits, gap_tolerance_m=0.002, reference_dynamics=True)
        for key, value in models.items()
    }
    assert all(value["frames_with_conditional_solution_within_effort_limits"] == repeated for value in support.values())
    for path in (
        Path(__file__), HERE / "report.json", archive, old_path, candidate_path, source_path, model_path,
        ROOT / SIM_CONFIG, *runtime_sources, Path(inspect.getfile(g1_true23_reference_support)),
        Path(inspect.getfile(g1_true23_contact_patch)),
    ):
        actual = geometry.sha256(path)
        assert identities.get(str(path.resolve()), actual) == actual
        identities[str(path.resolve())] = actual
    assert all(geometry.sha256(Path(path)) == digest for path, digest in identities.items())
    assert compiled == {key: compiled_model_sha256(value) for key, value in models.items()}
    result = {
        "kind": "g1_true23_independent_serialized_single_pose_witness_v1",
        "source_frame": frame, "original_motion_frames": len(problem["source_qpos"]),
        "repeated_static_witness_frames": repeated,
        "serialized_contact_audit": contact, "original_correction_bound_violation": bound_violation,
        "root_orientation_error_rad": orientation_error,
        "maximum_pose_derived_velocity": float(np.max(np.abs(velocity))),
        "maximum_pose_derived_acceleration": float(np.max(np.abs(acceleration))),
        "fk": fk, "support": support,
        "geometry": {key: reference_geometry(value, serialized) for key, value in models.items()},
        "actuation_profile": profile.contract(), "torque_limit_multiplier": 0.2375,
        "compiled_models": compiled, "files": identities,
        "single_pose_has_conditional_force_solution_in_both_models": True,
        "full_clip_test": False, "robust_effort_headroom_proven": False,
        "teacher_accepted": False, "dynamic_feasibility_proven": False,
        "hardware_authorized": False, "deployment_ready": False,
        "limitations": [
            "One stationary pose is repeated, not the full original crouch transition.",
            "Mesh peak effort ratio is effectively one, providing negligible effort headroom.",
            "Candidate floor contacts and friction assistance are optimistic, not measured forces.",
            "No closed-loop tracking, contact complementarity, no-slip, standing return or live teleop is proven.",
        ],
    }
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({"output": str(output), "sha256": geometry.sha256(output), "contact_passed": contact["passed"],
        "conditional_force_frames": {key: value["frames_with_conditional_solution_within_effort_limits"] for key, value in support.items()},
        "rechecked_files": len(identities)}))


if __name__ == "__main__":
    main()
