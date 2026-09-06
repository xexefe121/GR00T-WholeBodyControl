"""Bind full available SONIC/PICO requests for diagnostic breadth training.

These are *requested reference motions*, not force-qualified action teachers.
Preserve every available source frame and record unavailable requests. Add the
synthetic stationary prerequisite as its own clip, never as a substitute for
an unavailable elbow or as a motion-parity result. No retiming/retargeting here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from gear_sonic.scripts.evaluate_g1_true23_interior_effort import PICO_NAMES
from gear_sonic.scripts.record_g1_sonic_original29_baseline import CLIPS, dump
from gear_sonic.utils.g1_23dof_multi_motion import MOTION_ARRAY_NAMES
from gear_sonic.utils.g1_sonic_cpp_parameters import file_sha256
from gear_sonic.utils.g1_true23_sonic_library_replay import validate_library_motion


def full_request_rows(suite):
    if suite.get("kind") != "g1_true23_interior_effort_full_request_comparison_v1" or any(
        suite.get(flag) is not False
        for flag in (
            "teacher_accepted",
            "hardware_authorized",
            "deployment_ready",
            "full_eight_clip_qualification",
        )
    ):
        raise ValueError("requires explicitly unqualified full-request source comparison")
    rows = [row for row in suite["records"] if row["inner_fraction"] == 1 and not row["historical_start"]]
    if [row["name"] for row in rows] != [*CLIPS, *PICO_NAMES]:
        raise ValueError("all original SONIC and five PICO requests must remain in order")
    if [row["name"] for row in rows if row.get("not_executed")] != ["elbow_crawling"]:
        raise ValueError("only the already unavailable complete elbow source may be absent")
    return rows


def assemble_clips(clips):
    arrays = {key: [] for key in MOTION_ARRAY_NAMES}
    spans, cursor = [], 0
    if len({name for name, _ in clips}) != len(clips):
        raise ValueError("request corpus names must be unique")
    for name, motion in clips:
        count = validate_library_motion(motion)
        if count < 500:
            raise ValueError(
                "available request is too short for the existing 500-frame trainer; do not silently drop it"
            )
        for key in MOTION_ARRAY_NAMES:
            arrays[key].append(np.asarray(motion[key], dtype=np.float32))
        spans.append({"name": name, "start": cursor, "length": count})
        cursor += count
    if not spans:
        raise ValueError("request corpus must contain clips")
    joined = {key: np.concatenate(values) for key, values in arrays.items()}
    joined["fps"] = np.array([50.0], dtype=np.float64)
    return joined, spans


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-request-report", type=Path, required=True)
    parser.add_argument("--stationary-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output, inputs = args.output_dir.resolve(), {}

    def bind(path, expected=None):
        path = Path(path).resolve(strict=True)
        digest = file_sha256(path)
        if digest != inputs.get(str(path), digest) or (expected is not None and digest != expected):
            raise ValueError(f"request corpus input changed: {path}")
        inputs[str(path)] = digest
        return path

    suite = json.loads(bind(args.full_request_report).read_text())
    stationary = json.loads(bind(args.stationary_report).read_text())
    rows = full_request_rows(suite)
    if (
        stationary.get("kind") != "g1_true23_stationary_actor_comparison_diagnostic_v1"
        or stationary.get("stationary_only") is not True
    ):
        raise ValueError("requires explicitly synthetic stationary prerequisite reference")
    for report in (suite, stationary):
        for path, digest in report["inputs"].items():
            bind(path, digest)
    clips, ledger = [], []
    for row in rows:
        if row.get("not_executed"):
            ledger.append({"name": row["name"], "available": False, "reason": row["reason"], "replacement": None})
            continue
        path = bind(row["source_motion_path"], row["source_motion_sha256"])
        with np.load(path, allow_pickle=False) as archive:
            motion = {key: archive[key].copy() for key in archive.files}
        count = validate_library_motion(motion)
        if count - 11 != row["result"]["requested_transitions"]:
            raise ValueError("full source frame count differs from requested comparison")
        clips.append((row["name"], motion))
        ledger.append(
            {
                "name": row["name"],
                "available": True,
                "path": str(path),
                "sha256": inputs[str(path)],
                "full_source_frames": count,
                "source_arrays_cast_to_float32_only": True,
                "requested_task_not_action_teacher": True,
                "physical_reference_qualified": False,
            }
        )
    path = bind(args.stationary_report.parent / "stationary_reference.npz")
    with np.load(path, allow_pickle=False) as archive:
        motion = {key: archive[key].copy() for key in archive.files}
    if not np.array_equal(motion["joint_pos"], np.tile(motion["joint_pos"][0], (len(motion["joint_pos"]), 1))):
        raise ValueError("stationary prerequisite source no longer stationary")
    clips.append(("synthetic_standing_prerequisite", motion))
    joined, spans = assemble_clips(clips)
    bind(Path(__file__))
    for name, module in list(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if name.startswith("gear_sonic.") and path and Path(path).suffix == ".py":
            bind(path)
    output.mkdir(parents=True, exist_ok=False)
    corpus_path = output / "corpus.npz"
    with corpus_path.open("xb") as stream:
        np.savez_compressed(stream, **joined)
    bind(corpus_path)
    # Reopen and independently verify each complete source span after the
    # necessary float32 serialization boundary; no episode concatenation leak.
    with np.load(corpus_path, allow_pickle=False) as archive:
        for (_, original), span in zip(clips, spans, strict=True):
            for key in MOTION_ARRAY_NAMES:
                np.testing.assert_array_equal(
                    archive[key][span["start"] : span["start"] + span["length"]], original[key].astype(np.float32)
                )
    span_path = output / "corpus.spans.json"
    dump(
        span_path,
        {
            "kind": "g1_true23_motion_corpus_spans_v1",
            "corpus": str(corpus_path),
            "episode_frames": 500,
            "fps": 50.0,
            "clip_count": len(spans),
            "total_frames": len(joined["joint_pos"]),
            "dropped_short": 0,
            "dropped_bad": 0,
            "spans": spans,
        },
    )
    bind(span_path)
    metadata = {
        "schema": "g1_true23_low_latency_recovery_motion_v1",
        "kind": "g1_true23_unqualified_full_request_breadth_curriculum_v1",
        "output": {
            "filename": corpus_path.name,
            "sha256": inputs[str(corpus_path)],
            "fps": 50.0,
            "frames": len(joined["joint_pos"]),
            "duration_s": len(joined["joint_pos"]) / 50,
        },
        "corpus": {
            "clip_count": len(spans),
            "span_sidecar_filename": span_path.name,
            "span_sidecar_sha256": inputs[str(span_path)],
            "full_body_controlled_joint_count": 23,
        },
        "segments_inclusive": {
            span["name"]: [span["start"], span["start"] + span["length"] - 1] for span in spans
        },
        "source_request_ledger": ledger,
        "synthetic_standing_is_an_additional_clip_not_a_missing_motion_replacement": True,
        "all_available_requested_frames_preserved": True,
        "frame_cast": "float32_for_existing_MJLab_loader",
        "reference_temporally_resampled": False,
        "reference_retargeted_again": False,
        "action_teacher_or_behavior_bank": False,
        "full_requested_set_qualified": False,
        "physical_reference_qualified": False,
        "supervised_failed_controller_actions_used": False,
        "simulator_only": True,
        "hardware_authorized": False,
        "deployment_ready": False,
        "authorization": {
            "training_input_only": True,
            "robot_network_commands": False,
            "hardware_authorized": False,
        },
    }
    dump(output / "corpus.recovery.json", metadata)
    bind(output / "corpus.recovery.json")
    for path in list(inputs):
        bind(path)
    dump(output / "report.json", {**metadata, "inputs": inputs, "spans": spans})
    print(
        json.dumps(
            {
                "training_clips": len(spans),
                "frames": len(joined["joint_pos"]),
                "unavailable_requests": [row["name"] for row in ledger if not row["available"]],
                "full_request_qualification": False,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
