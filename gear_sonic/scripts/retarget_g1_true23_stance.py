"""Whole-corpus offline stance retargeting; no robot or teacher promotion.

Every manifest clip receives a result, including failures. A candidate manifest
is written only when every full clip was generated. Generation is NOT force,
tracking, live-PICO, or hardware qualification; run the independent audits.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path

import mujoco
import numpy as np
import scipy

from gear_sonic.scripts import condition_g1_true23_reference_floor as geometry_cli
from gear_sonic.utils import g1_true23_clean_mujoco_teleop, g1_true23_stance_retarget
from gear_sonic.utils.g1_23dof_trajectory_projection import TrajectoryProjectionError
from gear_sonic.utils.g1_true23_clean_mujoco_teleop import motion_reference_terms, validate_reference_terms
from gear_sonic.utils.g1_true23_reference_floor import compiled_model_sha256
from gear_sonic.utils.g1_true23_sonic_library_replay import validate_library_motion
from gear_sonic.utils.g1_true23_stance_retarget import FAMILIES, StanceRetargetConfig, retarget_stance_motion


def validate_manifest(manifest, hypotheses):
    entries = manifest["motions"]
    names = [entry["name"] for entry in entries]
    if not entries or len(set(names)) != len(names):
        raise ValueError("stance corpus requires nonempty unique clips")
    if any(
        not isinstance(name, str)
        or not name
        or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name)
        for name in names
    ):
        raise ValueError("clip names must be safe output names")
    if hypotheses.get("observed_or_verified_contacts") is not False:
        raise ValueError("stance inference must not claim observed or verified contacts")
    families = hypotheses["families"]
    if set(families) != set(names) or any(value not in FAMILIES for value in families.values()):
        raise ValueError("explicit stance families must match every manifest clip exactly")
    if any(not np.isfinite(entry.get("weight", 1.0)) or entry.get("weight", 1.0) <= 0 for entry in entries):
        raise ValueError("motion weights must be finite and positive")
    return entries, families


def audit_rebuilt_causal_terms(motion):
    count = validate_library_motion(motion)
    digest = hashlib.sha256()
    for q9 in range(9, count - 1):
        packet = motion_reference_terms(motion, q9)
        validate_reference_terms(packet)
        digest.update(json.dumps(packet, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
        digest.update(b"\n")
    return {
        "packets_rebuilt_and_validated": count - 10,
        "first_q9": 9,
        "last_q9": count - 2,
        "canonical_jsonl_sha256": digest.hexdigest(),
        "stored_or_sent_to_robot": False,
        "live_pico_input_qualified": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--families", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--maximum-frame-evaluations", type=int, default=80)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("stance retarget requires a new output directory")
    root = args.repository_root.resolve()
    asset_root = (args.asset_root or root).resolve()
    entries, families = validate_manifest(
        json.loads(args.manifest.read_text()), json.loads(args.families.read_text())
    )
    config = StanceRetargetConfig(maximum_frame_evaluations=args.maximum_frame_evaluations)
    mesh_path = root / geometry_cli.DEFAULT_TARGET_MODEL
    model = mujoco.MjModel.from_xml_path(str(mesh_path))
    training, training_sources = geometry_cli.build_training_geometry()
    models = {"retarget_mesh": model, "training_capsules": training}
    inputs = [args.manifest, args.families, mesh_path] + [asset_root / entry["path"] for entry in entries]
    sources = [
        Path(__file__),
        Path(inspect.getfile(geometry_cli)),
        Path(inspect.getfile(g1_true23_stance_retarget)),
        Path(inspect.getfile(g1_true23_clean_mujoco_teleop)),
        *[
            root / "gear_sonic/utils" / name
            for name in (
                "g1_23dof_task_space_retarget.py",
                "g1_23dof_trajectory_projection.py",
                "g1_23dof_safe_target_transform.py",
                "g1_true23_contact_geometry.py",
                "g1_true23_reference_floor.py",
                "g1_true23_sim_acquisition.py",
                "g1_true23_sonic_library_replay.py",
            )
        ],
        *training_sources,
    ]
    input_hashes = {str(path.resolve()): geometry_cli.sha256(path) for path in inputs}
    source_hashes = {str(path.resolve()): geometry_cli.sha256(path) for path in sources}
    model_hashes = {name: compiled_model_sha256(value) for name, value in models.items()}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    records = []
    for entry in entries:
        path = asset_root / entry["path"]
        with np.load(path, allow_pickle=False) as archive:
            motion = {key: archive[key].copy() for key in archive.files}
        record = {
            "name": entry["name"],
            "family": families[entry["name"]],
            "source": str(path.resolve()),
            "source_sha256": input_hashes[str(path.resolve())],
            "weight": entry.get("weight", 1.0),
            "source_frame_count": int(len(motion["joint_pos"])),
        }
        print(
            json.dumps({"starting_clip": entry["name"], "source_frames": record["source_frame_count"]}), flush=True
        )
        try:
            arrays, evidence = retarget_stance_motion(
                model,
                models,
                motion,
                record["family"],
                config=config,
                progress=lambda status: print(json.dumps({"clip": entry["name"], **status}), flush=True),
            )
            causal = audit_rebuilt_causal_terms(arrays)
        except (ValueError, TrajectoryProjectionError) as error:
            record.update(candidate_generated=False, failure=str(error))
        else:
            output = args.output_dir / f"{entry['name']}.npz"
            with output.open("xb") as stream:
                np.savez_compressed(stream, **arrays)
            record.update(
                candidate_generated=True,
                output=str(output.resolve()),
                output_sha256=geometry_cli.sha256(output),
                retarget=evidence,
                rebuilt_causal_terms=causal,
            )
        records.append(record)
        with (args.output_dir / f"{entry['name']}.report.json").open("x") as stream:
            json.dump(record, stream, indent=2, allow_nan=False)
        print(
            json.dumps(
                {
                    "clip": entry["name"],
                    "candidate_generated": record["candidate_generated"],
                    "failure": record.get("failure"),
                }
            ),
            flush=True,
        )
    if input_hashes != {str(path.resolve()): geometry_cli.sha256(path) for path in inputs}:
        raise RuntimeError("stance retarget inputs changed during execution")
    if source_hashes != {str(path.resolve()): geometry_cli.sha256(path) for path in sources}:
        raise RuntimeError("stance retarget sources changed during execution")
    if model_hashes != {name: compiled_model_sha256(value) for name, value in models.items()}:
        raise RuntimeError("stance retarget changed a supplied model")
    complete = all(record["candidate_generated"] for record in records)
    report = {
        "kind": "g1_true23_full_corpus_stance_retarget_v1",
        "records": records,
        "inputs": input_hashes,
        "sources": source_hashes,
        "compiled_models": model_hashes,
        "versions": {"mujoco": mujoco.__version__, "numpy": np.__version__, "scipy": scipy.__version__},
        "all_clips_generated": complete,
        "clips_omitted_from_report": 0,
        "source_manifest_clip_count": len(entries),
        "controlled_joint_count": 23,
        "production_training_changed": False,
        "force_feasibility_proven": False,
        "full_clip_tracking_qualified": False,
        "hardware_authorized": False,
        "deployment_ready": False,
    }
    if complete:
        manifest = {
            "kind": "g1_true23_stance_candidate_manifest_v1",
            "motions": [
                {"name": record["name"], "path": record["output"], "weight": record["weight"]}
                for record in records
            ],
            "source_manifest_sha256": input_hashes[str(args.manifest.resolve())],
            "teacher_accepted": False,
            "hardware_authorized": False,
            "deployment_ready": False,
        }
        with (args.output_dir / "motions.json").open("x") as stream:
            json.dump(manifest, stream, indent=2, allow_nan=False)
    with (args.output_dir / "report.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
