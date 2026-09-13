"""Single-pose numerical probe; never qualifies or replaces a complete motion."""

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np

from gear_sonic.scripts import condition_g1_true23_reference_floor as geometry
from gear_sonic.scripts.refine_g1_true23_stance_contacts import load_motion, refinement_problem
from gear_sonic.utils import g1_true23_force_restoration, g1_true23_force_trajectory
from gear_sonic.utils.g1_23dof_task_space_retarget import build_mjlab_motion_arrays
from gear_sonic.utils.g1_true23_actuation_profile import SIM_CONFIG, NativeSupportActuationProfile
from gear_sonic.utils.g1_true23_contact_trajectory import ContactLinearization, ContactTrajectoryConfig
from gear_sonic.utils.g1_true23_force_restoration import ForceRestorationConfig, restore_force_trajectory
from gear_sonic.utils.g1_true23_force_trajectory import ForceLinearization
from gear_sonic.utils.g1_true23_reference_floor import compiled_model_sha256
from gear_sonic.utils.g1_true23_reference_support import audit_reference_support

ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS = ROOT / "artifacts/g1_true23_frozen_lora"
OUTPUT = Path(__file__).resolve().parent / "report.json"


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    old_path = ARTIFACTS / "stance_retarget_20260906_v2/pico_crouch_anchor.report.json"
    old = json.loads(old_path.read_text())
    candidate_path = ARTIFACTS / "contact_trajectory_20260906_v2/pico_crouch_anchor.npz"
    model_path = ROOT / geometry.DEFAULT_TARGET_MODEL
    model = mujoco.MjModel.from_xml_path(str(model_path))
    training, runtime_sources = geometry.build_training_geometry()
    models = {"retarget_mesh": model, "training_capsules": training}
    model_hashes = {key: compiled_model_sha256(value) for key, value in models.items()}
    source = load_motion(Path(old["source"]))
    assert geometry.sha256(Path(old["source"])) == old["source_sha256"]
    assert model_hashes == old["retarget"]["collision_model_sha256"]
    problem = refinement_problem(model, source, load_motion(candidate_path), old["retarget"])
    frame = 512
    paths = [
        Path(__file__), old_path, Path(old["source"]), candidate_path, model_path, ROOT / SIM_CONFIG,
        *runtime_sources,
        Path(inspect.getfile(g1_true23_force_restoration)), Path(inspect.getfile(g1_true23_force_trajectory)),
    ]
    identities = {str(path.resolve()): geometry.sha256(path) for path in paths}
    for key in ("source_qpos", "desired", "lower", "upper"):
        problem[key] = np.repeat(problem[key][frame:frame + 1], 3, axis=0)
    problem["supports"] = [problem["supports"][frame]] * 3
    # Strictly stationary numerical fixture, not a change to corpus bounds.
    problem["velocity"] = np.full(26, 1e-12)
    problem["acceleration"] = np.full(26, 1e-12)
    problem["initial_velocity"] = np.zeros(26)
    limits = np.asarray(NativeSupportActuationProfile.from_sim_config(ROOT / SIM_CONFIG).effort) * 0.2375
    contacts = ContactLinearization(models, problem["source_qpos"], problem["supports"], ContactTrajectoryConfig())
    forces = ForceLinearization(models, problem["source_qpos"], limits)
    candidate, report = restore_force_trajectory(
        *[problem[key] for key in ("desired", "lower", "upper", "velocity", "acceleration", "initial_velocity")],
        contacts, forces, config=ForceRestorationConfig(),
        progress=lambda value: print(json.dumps(value), flush=True),
    )
    static_path = np.repeat(candidate.mean(axis=0, keepdims=True), 3, axis=0)
    arrays = build_mjlab_motion_arrays(
        model,
        SimpleNamespace(
            root_pos_w=np.repeat((problem["source_qpos"][:1, :3] + static_path[:1, :3]), 16, axis=0).astype(np.float32),
            root_quat_wxyz=np.repeat(problem["source_qpos"][:1, 3:7], 16, axis=0),
            joint_pos_hardware=np.repeat(static_path[:1, 3:], 16, axis=0).astype(np.float32),
            fps=50.0,
        ),
    )
    support = {
        key: audit_reference_support(value, arrays, limits, gap_tolerance_m=0.002, reference_dynamics=False)
        for key, value in models.items()
    }
    assert all(geometry.sha256(Path(path)) == digest for path, digest in identities.items())
    assert model_hashes == {key: compiled_model_sha256(value) for key, value in models.items()}
    result = {
        "kind": "g1_true23_single_stationary_crouch_solver_probe_v1",
        "original_clip_frames": len(source["joint_pos"]),
        "original_source_frame": frame,
        "numerical_fixture_repeated_frames": 3,
        "independent_static_lp_repeated_pose_count": 16,
        "maximum_fixture_frame_deviation_from_mean": float(np.max(np.abs(candidate - static_path))),
        "static_path": static_path[0].tolist(),
        "solver": report,
        "mean_pose_contact_audit": contacts.audit(static_path),
        "mean_pose_static_support": support,
        "models": model_hashes,
        "files": identities,
        "full_clip_test": False,
        "full_clip_feasibility_proven": False,
        "teacher_accepted": False,
        "dynamic_feasibility_proven": False,
        "hardware_authorized": False,
        "deployment_ready": False,
    }
    with OUTPUT.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({
        "output": str(OUTPUT), "sha256": geometry.sha256(OUTPUT), "failure": report["failure"],
        "contact_passed": result["mean_pose_contact_audit"]["passed"],
        "static_conditional_frames": {
            key: value["frames_with_conditional_solution_within_effort_limits"] for key, value in support.items()
        },
        "last_qp": report["iterations"][-1]["qp"] if report["iterations"] else None,
    }), flush=True)


if __name__ == "__main__":
    main()
