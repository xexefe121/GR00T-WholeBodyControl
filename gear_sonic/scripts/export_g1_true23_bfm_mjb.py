"""Export the exact BFM native23 MuJoCo model as a mesh-free compiled MJB."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco

from gear_sonic.scripts.evaluate_g1_true23_bfmzero import MODEL, PHYSICS, ROOT
from gear_sonic.utils.g1_true23_reference_floor import compiled_model_sha256
from gear_sonic.utils.g1_true23_step1b_mujoco import prepare_true23_model


def main(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite model export: {args.output}")
    _, model, _ = prepare_true23_model(ROOT.parent / "GR00T-WholeBodyControl" / MODEL, ROOT / PHYSICS)
    identity = compiled_model_sha256(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mujoco.mj_saveModel(model, str(args.output))
    result = {
        "kind": "g1_true23_bfm_compiled_mjb_v1",
        "mujoco_version": mujoco.__version__,
        "compiled_model_sha256": identity,
        "mjb_file_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "nq": model.nq,
        "nv": model.nv,
        "nu": model.nu,
        "ngeom": model.ngeom,
        "nsite": model.nsite,
        "output": str(args.output),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    main(parser.parse_args())
