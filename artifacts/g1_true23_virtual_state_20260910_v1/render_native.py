"""Render every measured state of the REJECTED internal-source experiment."""

import json
from pathlib import Path
import subprocess

import numpy as np

from gear_sonic.utils.g1_true23_generalist_benchmark import render_recorded_rollout
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file


def main():
    directory = Path(__file__).resolve().parent / "native_actual_v1"
    report_path, trace_path = directory / "report.json", directory / "trace.npz"
    report_sha, trace_sha = sha256_file(report_path), sha256_file(trace_path)
    report = json.loads(report_path.read_text())
    assert report["inputs"][str(trace_path)] == trace_sha
    output = directory / "rejected_virtual_model.measured.mp4"
    with np.load(trace_path, allow_pickle=False) as z:
        arrays = {"qpos": z["qpos"].copy()}
    receipt = render_recorded_rollout(
        report["result"], arrays, asset_root=Path("/mnt/z/codex/GR00T-WholeBodyControl"), output=output
    )
    probe = json.loads(
        subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name,width,height,nb_frames,r_frame_rate,duration",
                "-of",
                "json",
                str(output),
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    assert int(probe["streams"][0]["nb_frames"]) == len(arrays["qpos"])
    assert sha256_file(report_path) == report_sha and sha256_file(trace_path) == trace_sha
    receipt.update(
        source_trace_sha256=trace_sha,
        source_report_sha256=report_sha,
        render_driver_sha256=sha256_file(Path(__file__)),
        ffprobe=probe,
        case_outcome="REJECTED_incomplete_source_virtual_waist_roll_range_failure",
        source_motion_completed_s=25.58,
        requested_source_motion_s=115.60,
        camera_follows_root=True,
        world_tracking_comparison_video=False,
        candidate_promoted=False,
    )
    with (directory / "render.json").open("x") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
