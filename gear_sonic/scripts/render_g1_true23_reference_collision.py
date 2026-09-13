"""Render one declared reference pose, not a policy rollout or live robot view."""

import argparse
import json
from pathlib import Path

import imageio.v3 as iio
import mujoco
import numpy as np

from gear_sonic.scripts.audit_g1_true23_reference_bank_self_contacts import validated_reference_qpos
from gear_sonic.utils.g1_true23_generalist_benchmark import MODEL, PHYSICS
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file
from gear_sonic.utils.g1_true23_reference_floor import compiled_model_sha256
from gear_sonic.utils.g1_true23_self_collision import self_contact_rows
from gear_sonic.utils.g1_true23_step1b_mujoco import prepare_true23_model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = args.output.with_suffix(".json")
    if any(path.exists() or path.is_symlink() for path in (args.output, receipt)):
        raise FileExistsError("collision pose render refuses overwrite")
    root = Path(__file__).resolve().parents[2]
    paths = [args.motion, args.asset_root / MODEL, root / PHYSICS, Path(__file__)]
    bindings = {str(path.resolve(strict=True)): sha256_file(path) for path in paths}
    _, model, _ = prepare_true23_model(args.asset_root / MODEL, root / PHYSICS)
    identity = compiled_model_sha256(model)
    with np.load(args.motion, allow_pickle=False) as archive:
        poses, fk = validated_reference_qpos(model, {key: archive[key] for key in archive.files})
    if not 0 <= args.frame < len(poses):
        raise ValueError("render frame is outside the full reference")
    data = mujoco.MjData(model)
    data.qpos[:] = poses[args.frame]
    mujoco.mj_forward(model, data)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = data.qpos[:3] + np.array([0.0, 0.0, 0.2])
    camera.distance, camera.azimuth, camera.elevation = 1.75, 145.0, -10.0
    options = mujoco.MjvOption()
    options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    with mujoco.Renderer(model, height=480, width=640) as renderer:
        renderer.update_scene(data, camera=camera, scene_option=options)
        pixels = renderer.render().copy()
    if compiled_model_sha256(model) != identity or any(
        sha256_file(Path(path)) != digest for path, digest in bindings.items()
    ):
        raise ValueError("collision render input changed")
    iio.imwrite(args.output, pixels)
    contacts = [
        {
            "geoms": list(row["geoms"]),
            "bodies": [model.body(int(model.geom_bodyid[g])).name for g in row["geoms"]],
            "distance_m": row["distance_m"],
        }
        for row in self_contact_rows(model, data)
        if row["distance_m"] < 0
    ]
    with receipt.open("x") as stream:
        json.dump(
            {
                "kind": "native23_reference_pose_not_policy_rollout_v1",
                "frame": args.frame,
                "source_frames": len(poses),
                "fk": fk,
                "contacts": contacts,
                "input_bindings": bindings,
                "compiled_model_sha256": identity,
                "output_sha256": sha256_file(args.output),
                "physics_steps": 0,
                "hardware_authorized": False,
                "deployment_ready": False,
            },
            stream,
            indent=2,
        )
    print(json.dumps({"output": str(args.output), "frame": args.frame, "contacts": contacts}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
