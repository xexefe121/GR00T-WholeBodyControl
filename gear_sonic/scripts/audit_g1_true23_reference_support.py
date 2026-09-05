"""Offline full-reference force-balance screen, not a dynamic/hardware gate."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import mujoco
import numpy as np
import scipy

from gear_sonic.scripts import condition_g1_true23_reference_floor as geometry_cli
from gear_sonic.utils import g1_true23_reference_support
from gear_sonic.utils.g1_true23_actuation_profile import SIM_CONFIG, NativeSupportActuationProfile
from gear_sonic.utils.g1_true23_reference_support import audit_reference_support


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-floor-gap-m", type=float, default=0.002)
    parser.add_argument(
        "--reference-dynamics", action="store_true", help="Use independent pose derivatives, not stationary poses"
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("support audit requires a new output directory")
    root = args.repository_root.resolve()
    asset_root = (args.asset_root or root).resolve()
    entries = json.loads(args.manifest.read_text())["motions"]
    if not entries or len({entry["name"] for entry in entries}) != len(entries):
        raise ValueError("support audit needs nonempty unique motion entries")
    profile = NativeSupportActuationProfile.from_sim_config(root / SIM_CONFIG)
    limits = 0.95 * 0.25 * np.array(profile.effort)
    mesh_path = root / geometry_cli.DEFAULT_TARGET_MODEL
    training, training_sources = geometry_cli.build_training_geometry()
    models = {"retarget_mesh": mujoco.MjModel.from_xml_path(str(mesh_path)), "training_capsules": training}
    inputs = [args.manifest, root / SIM_CONFIG, mesh_path] + [asset_root / entry["path"] for entry in entries]
    sources = [
        Path(__file__),
        Path(inspect.getfile(g1_true23_reference_support)),
        Path(inspect.getfile(geometry_cli)),
        root / "gear_sonic/utils/g1_true23_contact_geometry.py",
        root / "gear_sonic/utils/g1_true23_reference_floor.py",
        root / "gear_sonic/utils/g1_true23_sonic_library_replay.py",
        root / "gear_sonic/utils/g1_true23_actuation_profile.py",
        *training_sources,
    ]
    input_hashes = {str(path.resolve()): geometry_cli.sha256(path) for path in inputs}
    source_hashes = {str(path.resolve()): geometry_cli.sha256(path) for path in sources}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    records = []
    for entry in entries:
        path = asset_root / entry["path"]
        with np.load(path, allow_pickle=False) as archive:
            motion = {key: archive[key].copy() for key in archive.files}
        evidence = {
            name: audit_reference_support(
                model,
                motion,
                limits,
                gap_tolerance_m=args.candidate_floor_gap_m,
                reference_dynamics=args.reference_dynamics,
            )
            for name, model in models.items()
        }
        record = {
            "name": entry["name"],
            "source": str(path.resolve()),
            "source_sha256": input_hashes[str(path.resolve())],
            "models": evidence,
        }
        records.append(record)
        print(
            json.dumps(
                {
                    "clip": entry["name"],
                    "models": {
                        name: {key: value for key, value in report.items() if key.startswith("frames_")}
                        for name, report in evidence.items()
                    },
                }
            ),
            flush=True,
        )
    if input_hashes != {str(path.resolve()): geometry_cli.sha256(path) for path in inputs}:
        raise RuntimeError("support audit inputs changed during execution")
    if source_hashes != {str(path.resolve()): geometry_cli.sha256(path) for path in sources}:
        raise RuntimeError("support audit sources changed during execution")
    report = {
        "kind": "g1_true23_full_reference_force_balance_audit_v1",
        "mode": "reference_inverse_dynamics" if args.reference_dynamics else "quasistatic",
        "records": records,
        "inputs": input_hashes,
        "sources": source_hashes,
        "versions": {"mujoco": mujoco.__version__, "numpy": np.__version__, "scipy": scipy.__version__},
        "actuation_profile": profile.contract(),
        "torque_limit_multiplier": 0.95 * 0.25,
        "controlled_joint_count": 23,
        "frames_dropped": 0,
        "production_training_changed": False,
        "dynamic_feasibility_proven": False,
        "hardware_authorized": False,
        "deployment_ready": False,
    }
    with (args.output_dir / "report.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
