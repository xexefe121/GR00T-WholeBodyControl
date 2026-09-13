"""Independent planner-side all-frame retarget receipt/FK/clock validation."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from gear_sonic.utils.g1_true23_mjbatch_mpc import load_motion_override, load_native_bundle, sha256


def run(args):
    records = {}
    for clip in args.clips:
        native, contract, original, timeline, manifest = load_native_bundle(args.bundle, clip)
        motion, receipt = load_motion_override(
            args.pack / clip / "reference.npz", args.bundle, clip, native, contract, original, timeline, manifest
        )
        records[clip] = dict(
            receipt=receipt,
            all_frames_checked=True,
            frames=len(motion["joint_pos"]),
            source_controls=next(
                p["requested_controls"] for p in timeline["phases"] if p["name"] == "source_motion"
            ),
            root_initial_change_m=float(
                np.linalg.norm(motion["body_pos_w"][10, 0] - original["body_pos_w"][10, 0])
            ),
            joint_initial_change_max_rad=float(
                np.max(np.abs(motion["joint_pos"][10] - original["joint_pos"][10]))
            ),
        )
        print(json.dumps({clip: records[clip]}), flush=True)
    report = dict(
        kind="planner_portable_retarget_numeric_validation",
        mujoco=mujoco.__version__,
        verifier_sha256=sha256(__file__),
        records=records,
        total_frames_checked=sum(record["frames"] for record in records.values()),
        all_passed=True,
        qualification="Reference geometry and time only; no physical tracking proof",
    )
    args.output.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("bundle", "pack", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--clips", nargs="+", default=["pico", "walk003", "walk008", "walk002"])
    run(parser.parse_args())
