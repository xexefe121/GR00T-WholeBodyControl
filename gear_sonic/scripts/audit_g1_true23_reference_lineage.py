"""Audit original SONIC planner provenance versus legacy native23 references.

No policy, physics rollout, time warp or robot I/O. The root-path lower bound
shows when force repair around a different reference cannot recover the
original choreography within the unchanged positional screen.
"""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path, PureWindowsPath

import numpy as np

from gear_sonic.scripts import build_g1_true23_sonic_library_motions as library
from gear_sonic.scripts.refine_g1_true23_reference_forces import bind_identity, dump
from gear_sonic.scripts.refine_g1_true23_stance_contacts import load_motion
from gear_sonic.utils import g1_true23_reference_lineage
from gear_sonic.utils.g1_true23_motion_fidelity import MAXIMUM_ERRORS
from gear_sonic.utils.g1_true23_reference_lineage import root_path_repair_bounds
from gear_sonic.utils.g1_true23_sonic_library_replay import validate_library_motion

CLIPS = {
    "original_sonic_hand_crawl": "hand_crawling",
    "original_sonic_elbow_crawl": "elbow_crawling",
    "original_sonic_happy_dance": "happy_dance",
}


def legacy_asset_path(value, asset_root):
    windows = PureWindowsPath(value)
    if windows.drive:
        if tuple(part.lower() for part in windows.parts[:3]) != ("z:\\", "codex", "gr00t-wholebodycontrol"):
            raise ValueError("legacy lineage path is outside the explicit original asset root")
        path = asset_root.joinpath(*windows.parts[3:]).resolve(strict=True)
    else:
        path = (asset_root / value).resolve(strict=True)
    if not path.is_relative_to(asset_root):
        raise ValueError("lineage path escapes original asset root")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--planner-dir", type=Path, required=True)
    parser.add_argument("--stance-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assets = args.asset_root.resolve(strict=True)
    identities = {}

    def bind(path, expected=None):
        path = Path(path).resolve(strict=True)
        bind_identity(identities, path)
        if expected is not None and identities[str(path)] != expected:
            raise ValueError("lineage source differs from its recorded identity")
        return path

    def read(path):
        return json.loads(bind(path).read_text())

    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = read(args.manifest)
    planner = read(args.planner_dir / "report.json")
    assert planner["kind"] == "g1_released_sonic_planner_motion_suite" and planner["fps"] == 30
    raw_records = {row["name"]: row for row in planner["records"]}
    entries = {row["name"]: row for row in manifest["motions"]}
    if len(entries) != len(manifest["motions"]) or not set(CLIPS).issubset(entries):
        raise ValueError("original manifest must include all three SONIC motion families without duplicates")
    records = []
    for name, raw_name in CLIPS.items():
        path = legacy_asset_path(entries[name]["path"], assets)
        native = load_motion(bind(path))
        frames = validate_library_motion(native)
        stance = read(args.stance_dir / f"{name}.report.json")
        bind(path, stance["source_sha256"])
        raw = raw_records[raw_name]
        source_path = bind(args.planner_dir / raw["npz"], raw["npz_sha256"])
        source = load_motion(source_path)
        assert source["qpos"].shape[1] == 36 and float(source["fps"][0]) == 30
        assert int(source["mode"][0]) == raw["mode"] == library.PLANNER_MODES[raw_name]
        xyz, _quat, _joints = library._resample_qpos(source["qpos"])
        assert len(xyz) == frames
        sidecar = path.parent / "manifest.json"
        if sidecar.exists():
            provenance = read(sidecar)
            assert provenance["kind"] == "g1_true23_physical_rollout_motion_reference_v1"
            bind(path, provenance["output_sha256"])
            rollout_path = bind(legacy_asset_path(provenance["source_path"], assets), provenance["source_sha256"])
            rollout_report = read(rollout_path.parent / "report.json")
            rollout_record = next(row for row in rollout_report["records"] if row["name"] == raw_name)
            assert rollout_record["source_sha256"] == raw["npz_sha256"]
            bind(rollout_path, rollout_record["physical_npz_sha256"])
            rollout = load_motion(rollout_path)
            assert rollout["qpos"].shape == (frames, 30)
            # The exported reference contains the initial state, then the
            # preceding post-control states; it is not a same-index copy of
            # the post-control rollout. Verify this explicitly, without
            # shifting the planner/native comparison or hiding its endpoint.
            np.testing.assert_allclose(native["body_pos_w"][1:, 0], rollout["qpos"][:-1, :3], atol=5e-7, rtol=0)
            np.testing.assert_allclose(native["body_pos_w"][0, 0], xyz[0], atol=5e-7, rtol=0)
            reference_kind = "recorded_true23_controller_rollout_not_original_planner_path"
        else:
            provenance = read(path.parent / "report.json")
            record = next(row for row in provenance["records"] if row["name"] == raw_name)
            bind(path, record["output_sha256"])
            assert record["source_sha256"] == raw["npz_sha256"]
            reference_kind = "task_space_retarget_from_original_planner"
        offset = stance["retarget"]["config"]["maximum_root_offset_m"]
        # Include the full saved-path position audit tolerance in the allowed box.
        bounds = root_path_repair_bounds(xyz, native["body_pos_w"][:, 0], maximum_offset_m=offset + 2e-7)
        threshold = MAXIMUM_ERRORS["maximum_pelvis_position_error_m"]
        records.append(
            {
                "name": name,
                "native_reference_kind": reference_kind,
                "original_planner": str(source_path),
                "legacy_reference": str(path),
                "raw_30hz_frames": len(source["qpos"]),
                "resampled_50hz_frames": frames,
                "root_path": bounds,
                "existing_pelvis_error_screen_m": threshold,
                "repair_box_cannot_recover_original_path_within_screen": bounds[
                    "minimum_possible_maximum_horizontal_error_allowing_yaw_m"
                ]
                > threshold,
            }
        )
    for path in (
        Path(__file__),
        Path(inspect.getfile(library)),
        Path(inspect.getfile(g1_true23_reference_lineage)),
        Path(__file__).resolve().parents[1] / "utils/g1_true23_motion_fidelity.py",
    ):
        bind(path)
    for path, digest in list(identities.items()):
        bind(path, digest)
    result = {
        "kind": "g1_true23_original_sonic_reference_lineage_audit_v1",
        "records": records,
        "manifest_clips": list(entries),
        "planner_mapped_clips": list(CLIPS),
        "pico_entries_not_assessed_against_sonic_planner": [name for name in entries if name not in CLIPS],
        "files": identities,
        "frames_dropped_from_compared_clips": 0,
        "thresholds_are_provisional_not_manufacturer_limits": True,
        "comparison_is_reference_lineage_not_policy_or_hardware_evaluation": True,
        "original_choreography_parity_proven": False,
        "hardware_authorized": False,
        "deployment_ready": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dump(args.output, result)
    print(json.dumps({"output": str(args.output), "files": len(identities), "records": records}), flush=True)


if __name__ == "__main__":
    main()
