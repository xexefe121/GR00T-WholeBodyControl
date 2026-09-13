"""Visualize all saved500-checkpoint PICO states; failed SIM, not robot replay."""

import json
from pathlib import Path

import numpy as np

from gear_sonic.utils.g1_true23_generalist_benchmark import render_recorded_rollout
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CASE = HERE / "eval500_v1/pico"


def main():
    receipt = CASE / "failed_simulation_render.json"
    if receipt.exists():
        raise FileExistsError("failed PICO render refuses overwrite")
    report_path = CASE / "report.json"
    report_hash = sha256_file(report_path)
    report = json.loads(report_path.read_text())
    assert report["result"]["failure"] is not None
    trace = CASE / "trace.npz"
    assert sha256_file(trace) == report["trace_sha256"]
    with np.load(trace, allow_pickle=False) as z:
        arrays = dict(qpos=z["qpos"].copy())
    output = CASE / "FAILED_SIM_pico_69p34_of_115p60_seconds.mp4"
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
        full_motion_succeeded=False,
        physical_robot_operated=False,
    )
    with receipt.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
    print(json.dumps(value), flush=True)


if __name__ == "__main__":
    main()
