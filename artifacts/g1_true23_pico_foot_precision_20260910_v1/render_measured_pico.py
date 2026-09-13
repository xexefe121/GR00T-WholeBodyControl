"""Render every saved failed-tracking state; no new controller or robot action."""

import argparse
import json
from pathlib import Path

import numpy as np

from gear_sonic.utils.g1_true23_generalist_benchmark import render_recorded_rollout
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--update", type=int, choices=(600, 1000), required=True)
    args = parser.parse_args()
    directory = HERE / f"eval{args.update}_v1/pico"
    receipt = directory / "failed_simulation_render.json"
    output = directory / f"FAILED_TRACKING_SIM_pico_checkpoint{args.update}.mp4"
    if receipt.exists() or output.exists():
        raise FileExistsError("measured render refuses overwrite")
    report_path = directory / "report.json"
    report_hash = sha256_file(report_path)
    report = json.loads(report_path.read_text())
    assert report["identity"]["completed_training_updates"] == args.update
    assert not report["lifecycle"]["source_motion_tracking"]["provisional_reference_landmark_screen_passed"]
    trace = directory / "trace.npz"
    assert sha256_file(trace) == report["trace_sha256"]
    with np.load(trace, allow_pickle=False) as bundle:
        arrays = dict(qpos=bundle["qpos"].copy())
    value = render_recorded_rollout(
        report["result"], arrays, asset_root=ROOT.parent / "GR00T-WholeBodyControl", output=output
    )
    assert sha256_file(report_path) == report_hash and sha256_file(trace) == report["trace_sha256"]
    value.update(
        source_report_sha256=report_hash,
        source_trace_sha256=report["trace_sha256"],
        renderer_script_sha256=sha256_file(Path(__file__)),
        source_seconds_completed=report["entire_executed_source"]["source_controls"] / 50,
        source_seconds_requested=115.60,
        includes_initial_standing_and_entry=True,
        terminal_failure=report["result"]["failure"],
        full_motion_tracking_succeeded=False,
        physical_robot_operated=False,
        deployment_ready=False,
    )
    with receipt.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
    print(json.dumps(value), flush=True)


if __name__ == "__main__":
    main()
