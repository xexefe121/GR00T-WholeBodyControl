"""Audit/condition whole native23 clips against mesh and training collision models.

Writes separate offline artifacts, never edits source clips or policies. The
optional root-height conditioning only establishes geometric floor clearance;
it does not establish support contacts, COM balance or dynamic qualification.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path

import mujoco
import numpy as np

from gear_sonic.utils import g1_true23_contact_geometry, g1_true23_reference_floor
from gear_sonic.utils.g1_23dof_task_space_retarget import DEFAULT_TARGET_MODEL
from gear_sonic.utils.g1_23dof_trajectory_projection import TrajectoryProjectionError
from gear_sonic.utils.g1_true23_reference_floor import (
    compiled_model_sha256,
    condition_reference_floor,
    reference_geometry,
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_training_geometry():
    """Use actual training Entity collision configuration, with flat world floor.

    This is a CPU geometric model, not an alternate simulation dynamics profile.
    The scene's robot namespace and terrain wrapper are immaterial to positions.
    """
    from mjlab.entity import Entity
    from src.assets.robots.unitree_g1 import g1_23dof_constants as asset

    entity = Entity(asset.get_g1_23dof_robot_cfg())
    spec = entity.spec
    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05])
    return spec.compile(), [Path(inspect.getfile(asset)), Path(inspect.getfile(Entity)), asset.G1_23DOF_XML]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--asset-root", type=Path, help="Root resolving source motion paths in the manifest")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mesh-model", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--condition", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("floor reference artifacts require a new output directory")
    root = args.repository_root.resolve()
    asset_root = (args.asset_root or root).resolve()
    manifest = json.loads(args.manifest.read_text())
    entries = manifest["motions"]
    names = [entry["name"] for entry in entries]
    if not entries or len(names) != len(set(names)):
        raise ValueError("manifest must contain unique, nonempty clips")
    if any(
        not isinstance(name, str)
        or not name
        or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name)
        for name in names
    ):
        raise ValueError("clip names must be safe output names")
    mesh_path = args.mesh_model or root / DEFAULT_TARGET_MODEL
    mesh = mujoco.MjModel.from_xml_path(str(mesh_path.resolve()))
    training, asset_sources = build_training_geometry()
    models = {"retarget_mesh": mesh, "training_capsules": training}
    input_files = [args.manifest, mesh_path] + [asset_root / entry["path"] for entry in entries]
    source_files = [
        Path(__file__),
        Path(inspect.getfile(g1_true23_contact_geometry)),
        Path(inspect.getfile(g1_true23_reference_floor)),
        root / "gear_sonic/utils/g1_23dof_task_space_retarget.py",
        root / "gear_sonic/utils/g1_23dof_trajectory_projection.py",
        root / "gear_sonic/utils/g1_true23_sim_acquisition.py",
        *asset_sources,
    ]
    input_hashes = {str(path.resolve()): sha256(path) for path in input_files}
    source_hashes = {str(path.resolve()): sha256(path) for path in source_files}
    model_hashes = {name: compiled_model_sha256(model) for name, model in models.items()}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    records = []
    for entry in entries:
        path = asset_root / entry["path"]
        with np.load(path, allow_pickle=False) as archive:
            motion = {key: archive[key].copy() for key in archive.files}
        record = {
            "name": entry["name"],
            "source_path": str(path.resolve()),
            "source_sha256": input_hashes[str(path.resolve())],
            "weight": entry.get("weight", 1.0),
        }
        if args.condition:
            try:
                arrays, evidence = condition_reference_floor(motion, models, output_model=mesh)
            except (ValueError, TrajectoryProjectionError) as error:
                record.update(
                    conditioning_failed=True,
                    failure=str(error),
                    before={name: reference_geometry(model, motion) for name, model in models.items()},
                )
            else:
                output = args.output_dir / f"{entry['name']}.npz"
                with output.open("xb") as stream:
                    np.savez_compressed(stream, **arrays)
                record.update(
                    conditioning_failed=False,
                    output=str(output.resolve()),
                    output_sha256=sha256(output),
                    conditioning=evidence,
                )
        else:
            record["before"] = {name: reference_geometry(model, motion) for name, model in models.items()}
        records.append(record)
        before = record.get("before", record.get("conditioning", {}).get("before"))
        print(
            json.dumps(
                {
                    "clip": entry["name"],
                    "conditioning_failed": record.get("conditioning_failed"),
                    "geometry_before": {
                        name: {
                            key: report[key]
                            for key in ("frames", "frames_with_floor_overlap", "worst_floor_overlap_m")
                        }
                        for name, report in before.items()
                    },
                }
            ),
            flush=True,
        )
    if input_hashes != {str(path.resolve()): sha256(path) for path in input_files}:
        raise RuntimeError("reference inputs changed during conditioning")
    if source_hashes != {str(path.resolve()): sha256(path) for path in source_files}:
        raise RuntimeError("reference conditioning sources changed during execution")
    if model_hashes != {name: compiled_model_sha256(model) for name, model in models.items()}:
        raise RuntimeError("reference conditioning mutated collision models")
    all_conditioned = args.condition and all(not record["conditioning_failed"] for record in records)
    report = {
        "kind": "g1_true23_full_corpus_floor_geometry_v1",
        "records": records,
        "inputs": input_hashes,
        "sources": source_hashes,
        "compiled_models": model_hashes,
        "mujoco_version": mujoco.__version__,
        "all_clips_geometrically_conditioned": all_conditioned,
        "clips_dropped": 0,
        "production_training_changed": False,
        "dynamic_feasibility_proven": False,
        "full_clip_tracking_qualified": False,
        "hardware_authorized": False,
        "deployment_ready": False,
    }
    if all_conditioned:
        generated = {
            "kind": "g1_true23_floor_conditioned_motion_manifest_v1",
            "motions": [{"name": r["name"], "path": r["output"], "weight": r["weight"]} for r in records],
            "source_manifest_sha256": input_hashes[str(args.manifest.resolve())],
            "hardware_authorized": False,
            "deployment_ready": False,
        }
        with (args.output_dir / "motions.json").open("x") as stream:
            json.dump(generated, stream, indent=2, allow_nan=False)
    with (args.output_dir / "report.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    return 0 if not args.condition or all_conditioned else 2


if __name__ == "__main__":
    raise SystemExit(main())
