"""Restore every native23 frame with contact patches and current-centered force steps.

Writes a separate unaccepted corpus, including failed attempts. This is not
training, a real-time controller, live teleop, deployment or hardware authority.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import inspect
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import clarabel
import mujoco
import numpy as np
import scipy

from gear_sonic.scripts import (
    condition_g1_true23_reference_floor as geometry,
    refine_g1_true23_reference_forces as previous_force_cli,
)
from gear_sonic.scripts.refine_g1_true23_reference_forces import bind_identity, dump, validate_manifest
from gear_sonic.scripts.refine_g1_true23_stance_contacts import (
    load_motion,
    refinement_problem,
    root_orientation_error,
)
from gear_sonic.scripts.retarget_g1_true23_stance import audit_rebuilt_causal_terms
from gear_sonic.utils import (
    g1_true23_box_qp,
    g1_true23_contact_force_optimizer,
    g1_true23_contact_patch,
    g1_true23_force_trajectory,
    g1_true23_reference_support,
)
from gear_sonic.utils.g1_23dof_task_space_retarget import build_mjlab_motion_arrays
from gear_sonic.utils.g1_23dof_trajectory_projection import audit_trajectory_constraints
from gear_sonic.utils.g1_true23_actuation_profile import SIM_CONFIG, NativeSupportActuationProfile
from gear_sonic.utils.g1_true23_contact_force_optimizer import (
    ContactForceConfig,
    PatchedForceLinearization,
    restore_contact_force_trajectory,
)
from gear_sonic.utils.g1_true23_contact_trajectory import ContactLinearization, ContactTrajectoryConfig
from gear_sonic.utils.g1_true23_reference_floor import compiled_model_sha256, motion_qpos, reference_geometry
from gear_sonic.utils.g1_true23_reference_support import audit_reference_support
from gear_sonic.utils.g1_true23_sim_acquisition import audit_reference_kinematics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contact-dir", type=Path, required=True)
    parser.add_argument("--stance-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--maximum-iterations", type=int, default=64)
    parser.add_argument("--qp-maximum-iterations", type=int, default=200)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    contact_dir, stance_dir, output = (
        getattr(args, name).resolve() for name in ("contact_dir", "stance_dir", "output_dir")
    )
    if output.exists():
        raise FileExistsError(output)
    config = ContactForceConfig(
        maximum_iterations=args.maximum_iterations, qp_maximum_iterations=args.qp_maximum_iterations
    )
    entries = validate_manifest(json.loads((contact_dir / "motions.json").read_text()))
    stance_entries = validate_manifest(json.loads((stance_dir / "motions.json").read_text()))
    if [(row["name"], row["weight"]) for row in entries] != [
        (row["name"], row["weight"]) for row in stance_entries
    ]:
        raise ValueError("force refinement must retain every original stance clip, order and weight")
    profile = NativeSupportActuationProfile.from_sim_config(root / SIM_CONFIG)
    limits = np.asarray(profile.effort) * 0.95 * 0.25
    model_path = root / geometry.DEFAULT_TARGET_MODEL
    model = mujoco.MjModel.from_xml_path(str(model_path))
    training, runtime_sources = geometry.build_training_geometry()
    models = {"retarget_mesh": model, "training_capsules": training}
    model_hashes = {name: compiled_model_sha256(value) for name, value in models.items()}
    # Reuse the already-bound original geometry/projection dependency set.
    previous_summary = json.loads((contact_dir / "report.json").read_text())
    if previous_summary.get("kind") != "g1_true23_full_corpus_contact_refinement_v1" or (
        [row["name"] for row in previous_summary["records"]] != [row["name"] for row in entries]
        or previous_summary["clips_in"] != len(entries)
        or previous_summary["clips_out"] != len(entries)
    ):
        raise ValueError("force refinement requires a completed matching contact-refinement corpus")
    identities = dict(previous_summary["inputs"])
    if any(geometry.sha256(Path(path)) != digest for path, digest in identities.items()):
        raise ValueError("previous contact-refinement dependencies changed")
    for path in [
        model_path,
        root / SIM_CONFIG,
        contact_dir / "report.json",
        contact_dir / "motions.json",
        stance_dir / "motions.json",
        Path(__file__),
        *runtime_sources,
        *[
            Path(inspect.getfile(module))
            for module in (
                previous_force_cli,
                g1_true23_box_qp,
                g1_true23_contact_force_optimizer,
                g1_true23_contact_patch,
                g1_true23_force_trajectory,
                g1_true23_reference_support,
            )
        ],
        root / "gear_sonic/utils/g1_true23_actuation_profile.py",
    ]:
        bind_identity(identities, path)
    loaded = []
    for entry, stance_entry in zip(entries, stance_entries):
        name = entry["name"]
        path = Path(entry["path"]).resolve(strict=True)
        old_path = Path(stance_entry["path"]).resolve(strict=True)
        if path.parent != contact_dir or old_path.parent != stance_dir:
            raise ValueError("candidate paths must remain beside their corresponding reports")
        prior_report = json.loads((contact_dir / f"{name}.report.json").read_text())
        if prior_report != next(row for row in previous_summary["records"] if row["name"] == name):
            raise ValueError("per-clip contact report differs from completed corpus evidence")
        stance_report = json.loads((stance_dir / f"{name}.report.json").read_text())
        original = Path(stance_report["source"]).resolve(strict=True)
        if (
            geometry.sha256(path) != prior_report["output_sha256"]
            or geometry.sha256(old_path) != stance_report["output_sha256"]
        ):
            raise ValueError("contact or stance candidate identity changed")
        if (
            geometry.sha256(original) != stance_report["source_sha256"]
            or prior_report["source_sha256"] != stance_report["source_sha256"]
        ):
            raise ValueError("original source identity changed")
        if (
            stance_report["retarget"]["collision_model_sha256"] != model_hashes
            or previous_summary["compiled_models"] != model_hashes
        ):
            raise ValueError("force refinement collision model differs from source retargeting")
        for item in (
            path,
            old_path,
            original,
            contact_dir / f"{name}.report.json",
            stance_dir / f"{name}.report.json",
        ):
            bind_identity(identities, item)
        loaded.append((entry, stance_report, load_motion(original), load_motion(path)))
    if any(geometry.sha256(Path(path)) != digest for path, digest in identities.items()):
        raise ValueError("force refinement source/dependency bytes changed before execution")
    output.mkdir(parents=True)
    dump(
        output / "started.json",
        {
            "kind": "g1_true23_contact_force_experiment_started_v2",
            "inputs": identities,
            "config": asdict(config),
            "argv": sys.argv,
            "complete": False,
            "teacher_accepted": False,
            "hardware_authorized": False,
            "deployment_ready": False,
        },
    )
    records = []
    for entry, old, original, candidate in loaded:
        name = entry["name"]
        print(json.dumps({"starting_clip": name, "frames": len(candidate["joint_pos"])}), flush=True)
        problem = refinement_problem(model, original, candidate, old["retarget"])
        contacts = ContactLinearization(
            models, problem["source_qpos"], problem["supports"], ContactTrajectoryConfig()
        )
        forces = PatchedForceLinearization(
            models,
            problem["source_qpos"],
            limits * config.effort_limit_fraction,
            patch_guard_m=config.patch_guard_m,
            progress=lambda status: print(json.dumps({"clip": name, **status}), flush=True),
        )
        refined, report = restore_contact_force_trajectory(
            *[
                problem[key]
                for key in ("desired", "lower", "upper", "velocity", "acceleration", "initial_velocity")
            ],
            contacts,
            forces,
            config=config,
            progress=lambda status: print(json.dumps({"clip": name, **status}), flush=True),
        )
        source = problem["source_qpos"]
        arrays = build_mjlab_motion_arrays(
            model,
            SimpleNamespace(
                root_pos_w=(source[:, :3] + refined[:, :3]).astype(np.float32),
                root_quat_wxyz=source[:, 3:7],
                joint_pos_hardware=refined[:, 3:].astype(np.float32),
                fps=50.0,
            ),
        )
        destination = output / f"{name}.npz"
        with destination.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        arrays = load_motion(destination)
        qpos = motion_qpos(model, arrays)
        orientation_error = root_orientation_error(source[:, 3:7], qpos[:, 3:7])
        if orientation_error > 2e-7:
            raise ValueError("force refinement changed original root orientation")
        path = np.column_stack((qpos[:, :3] - source[:, :3], qpos[:, 7:]))
        temporal = audit_trajectory_constraints(
            path,
            lower_bounds=problem["lower"],
            upper_bounds=problem["upper"],
            dt=0.02,
            max_velocity=problem["velocity"],
            max_acceleration=problem["acceleration"],
            initial_velocity=problem["initial_velocity"],
            tolerance=2e-7,
        )
        fk = audit_reference_kinematics(SimpleNamespace(module=mujoco, model=model), arrays)
        if not temporal.passed or not fk["position_fk_consistent"] or not fk["orientation_fk_consistent"]:
            raise ValueError("serialized force refinement violated immutable derivative/FK contract")
        causal = audit_rebuilt_causal_terms(arrays)
        contact_audit = contacts.audit(path)
        support = {
            key: audit_reference_support(value, arrays, limits, gap_tolerance_m=0.002, reference_dynamics=True)
            for key, value in models.items()
        }
        support_path = output / f"{name}.support.json"
        dump(support_path, {"name": name, "output_sha256": geometry.sha256(destination), "models": support})
        record = {
            "name": name,
            "weight": entry["weight"],
            "output": str(destination),
            "output_sha256": geometry.sha256(destination),
            "original_source_sha256": old["source_sha256"],
            "previous_candidate_sha256": identities[str(Path(entry["path"]).resolve())],
            "frames_in": len(source),
            "frames_out": len(qpos),
            "frames_dropped": 0,
            "controlled_joint_count": 23,
            "time_scale": 1.0,
            "root_orientation_changed": False,
            "root_orientation_serialization_error_rad": orientation_error,
            "refinement": report,
            "serialized_contact_audit": contact_audit,
            "serialized_temporal_audit": asdict(temporal),
            "serialized_fk_audit": fk,
            "causal_terms": causal,
            "geometry_after": {key: reference_geometry(value, arrays) for key, value in models.items()},
            "independent_support_report": {"path": str(support_path), "sha256": geometry.sha256(support_path)},
            "independent_support": {
                key: {k: v for k, v in value.items() if k != "rows"} for key, value in support.items()
            },
            "requested_effort_fraction": config.effort_limit_fraction,
            "frames_with_requested_effort_headroom": {
                key: sum(
                    row["within_supplied_effort_limits"]
                    and row["minimum_peak_effort_ratio"] <= config.effort_limit_fraction + 1e-8
                    for row in value["rows"]
                )
                for key, value in support.items()
            },
            "headroom_is_algebraic_screen_not_robust_control_guarantee": True,
            "all_frames_with_conditional_force_solution": all(
                value["frames_with_conditional_solution_within_effort_limits"] == len(qpos)
                for value in support.values()
            ),
            "dynamic_feasibility_proven": False,
            "teacher_accepted": False,
            "hardware_authorized": False,
            "deployment_ready": False,
        }
        dump(output / f"{name}.report.json", record)
        records.append(record)
        print(
            json.dumps(
                {
                    "clip": name,
                    "contact_passed": contact_audit["passed"],
                    "conditional_force_frames": {
                        key: value["frames_with_conditional_solution_within_effort_limits"]
                        for key, value in support.items()
                    },
                    "failure": report["failure"],
                }
            ),
            flush=True,
        )
    if any(geometry.sha256(Path(path)) != digest for path, digest in identities.items()):
        raise ValueError("force refinement input/source bytes changed during execution")
    if model_hashes != {name: compiled_model_sha256(value) for name, value in models.items()}:
        raise ValueError("force refinement changed supplied models")
    summary = {
        "kind": "g1_true23_complete_contact_force_refinement_v2",
        "argv": sys.argv,
        "python_version": sys.version,
        "numeric_thread_environment": {
            key: os.environ.get(key) for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "records": records,
        "inputs": identities,
        "compiled_models": model_hashes,
        "clips_in": len(entries),
        "clips_out": len(records),
        "versions": {
            "mujoco": mujoco.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "clarabel": clarabel.__version__,
        },
        "actuation_profile": profile.contract(),
        "torque_limit_multiplier": 0.95 * 0.25,
        "optimizer_effort_fraction": config.effort_limit_fraction,
        "optimizer": asdict(config),
        "production_training_changed": False,
        "dynamic_feasibility_proven": False,
        "teacher_accepted": False,
        "hardware_authorized": False,
        "deployment_ready": False,
    }
    dump(output / "report.json", summary)
    dump(
        output / "motions.json",
        {
            "kind": "g1_true23_stance_candidate_manifest_v1",
            "refinement_kind": summary["kind"],
            "motions": [{"name": row["name"], "path": row["output"], "weight": row["weight"]} for row in records],
            "teacher_accepted": False,
            "hardware_authorized": False,
            "deployment_ready": False,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
