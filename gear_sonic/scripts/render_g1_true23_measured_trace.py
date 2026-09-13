"""Render an existing measured simulator trace; never execute a policy or robot."""

import argparse
import json
from pathlib import Path

import numpy as np

from gear_sonic.utils.g1_true23_generalist_benchmark import render_recorded_rollout
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-directory", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--case", choices=("nominal", "standing_push_x", "standing_push_y"), default="nominal")
    args = parser.parse_args(argv)
    directory = args.evaluation_directory.resolve(strict=True)
    report_path = directory / f"{args.case}.json"
    report_hash = sha256_file(report_path)
    report = json.loads(report_path.read_text())
    trace_path = Path(report["trace_path"])
    if sha256_file(trace_path) != report["trace_sha256"]:
        raise ValueError("recorded state trace differs from evaluated bytes")
    output, receipt_path = (
        directory / f"{args.case}.measured-simulation.mp4",
        directory / f"{args.case}.render.json",
    )
    if receipt_path.exists() or receipt_path.is_symlink():
        raise FileExistsError("render receipt already exists")
    with np.load(trace_path, allow_pickle=False) as archive:
        arrays = {"qpos": archive["qpos"].copy()}
    receipt = render_recorded_rollout(report["result"], arrays, asset_root=args.asset_root, output=output)
    if sha256_file(report_path) != report_hash or sha256_file(trace_path) != report["trace_sha256"]:
        raise ValueError("evaluated input changed during rendering")
    receipt.update(
        source_trace_sha256=report["trace_sha256"],
        source_report_sha256=report_hash,
        renderer_script_sha256=sha256_file(Path(__file__)),
        dance_tracking_screen_passed=report["result"]["tracking"]["provisional_reference_landmark_screen_passed"],
    )
    with receipt_path.open("x") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
    print(json.dumps(receipt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
