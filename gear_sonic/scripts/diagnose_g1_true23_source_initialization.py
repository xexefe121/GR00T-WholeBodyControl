"""SIM-only entry ablation: initialize at source frame zero, retain full source/return.

This deliberately omits standing and acquisition. It cannot qualify the normal
lifecycle or become a training-continuation baseline. No state changes occur
after the referee's single initial reset; all physical and policy code is shared.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np
import onnxruntime as ort
import torch

from gear_sonic.scripts.evaluate_g1_true23_generalist_baselines import write_json
from gear_sonic.teleop.buffered_source_simulation import BufferedSourceSimulationAdapter
from gear_sonic.utils.g1_true23_buffered_reference import BUFFERED_TIMING
from gear_sonic.utils.g1_true23_clean_mujoco_teleop import sha256_file
from gear_sonic.utils.g1_true23_generalist_benchmark import FLAGS, MODEL, PHYSICS, run_reference_diagnostic
from gear_sonic.utils.g1_true23_generalist_lifecycle import assess_lifecycle_diagnostic
from gear_sonic.utils.g1_true23_release_compatibility import release_compatibility_contract
from gear_sonic.utils.g1_true23_root_feedback_benchmark import (
    load_root_feedback_pair,
    summarize_root_perturbations,
)
from gear_sonic.utils.g1_true23_sonic_library_replay import validate_library_motion
from gear_sonic.utils.g1_true23_virtual_source_reference import virtual_source_vr_terms

EXPERIMENT_KIND = "g1_true23_source_initialized_full_motion_and_return_diagnostic_v1"
NOMINAL_BASELINE_EXPERIMENT_KIND = "g1_true23_source_initialized_nominal_baseline_diagnostic_v2"
NORMAL_KIND = "g1_true23_root_feedback_single_policy_lifecycle_campaign_v1"


def validate_baseline_cases(baseline, *, nominal_only_diagnostic=False):
    """Nominal-only entry isolation is explicit, never a three-case campaign.

    A failed full-request baseline is valid diagnostic evidence. A prefix,
    duplicate case, source-initialized result or omitted nominal case is not.
    Model, timeline and physics identities are checked separately by the caller.
    """
    records = baseline.get("records")
    expected = {"nominal"} if nominal_only_diagnostic else {"nominal", "standing_push_x", "standing_push_y"}
    if baseline.get("kind") != NORMAL_KIND or not isinstance(records, list):
        raise ValueError("requires a normal full-request lifecycle baseline")
    cases = [row.get("case") for row in records if isinstance(row, dict)]
    if len(cases) != len(records) or len(cases) != len(expected) or set(cases) != expected:
        raise ValueError("baseline cases differ from the explicitly requested diagnostic scope")
    for row in records:
        result = row.get("result", {})
        if (
            result.get("diagnostic_prefix_requested") is not False
            or type(result.get("available_controls")) is not int
            or result["available_controls"] <= 0
            or result.get("requested_controls") != result["available_controls"]
        ):
            raise ValueError("source initialization requires a full-request baseline, including any failure")
    return cases


def source_initialized_tail(motion, timeline):
    """Keep every original source/tail sample, with eleven disclosed warmup copies."""
    frames = validate_library_motion(motion)
    source = next(row for row in timeline["phases"] if row["name"] == "source_motion")
    start, omitted = source["frame_start"], source["control_start"]
    if (
        timeline["prehistory_frames"] != 11
        or start != omitted + 11
        or timeline["total_requested_controls"] != frames - 11
        or source["requested_controls"] != timeline["source_frames"]
        or source["frame_stop"] - start != timeline["source_frames"]
        or timeline["source_frame_indices"] != list(range(timeline["source_frames"]))
        or omitted <= 0
    ):
        raise ValueError("source initialization requires an intact complete lifecycle")
    result = {
        key: value.copy()
        if key == "fps"
        else np.concatenate((np.repeat(value[start : start + 1], 11, axis=0), value[start:]))
        for key, value in motion.items()
    }
    for key in result:
        if key != "fps":
            np.testing.assert_array_equal(result[key][11:], motion[key][start:])
    changed = copy.deepcopy(timeline)
    changed.update(
        kind=EXPERIMENT_KIND,
        total_frames=validate_library_motion(result),
        total_requested_controls=frames - start,
        source_start_frame=11,
        source_stop_frame_exclusive=11 + timeline["source_frames"],
        omitted_standing_and_acquisition_controls=omitted,
        initialization="source_frame_zero_pose_and_velocity_with_eleven_repeated_source_samples",
        synthetic_warmup_is_physical_standing_history=False,
        standing_acquisition_tested=False,
        eligible_parent_for_training_continuation=False,
        **FLAGS,
    )
    changed["phases"] = []
    for phase in timeline["phases"]:
        if phase["control_start"] < omitted:
            continue
        row = dict(phase)
        for key in ("control_start", "control_stop", "frame_start", "frame_stop"):
            row[key] -= omitted
        changed["phases"].append(row)
    if [row["name"] for row in changed["phases"]] != [
        "source_motion",
        "return_ramp",
        "returned_standing",
        "standing_proof_margin",
    ]:
        raise ValueError("source diagnostic must retain the complete source and return tail")
    return result, changed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-campaign", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--release-source-geometry", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument(
        "--nominal-only-baseline-diagnostic",
        action="store_true",
        help="Explicit entry-isolation ablation from one full nominal request; not three-case evidence",
    )
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    baseline_path, assets, manifest_path, geometry = (
        path.resolve(strict=True)
        for path in (
            args.baseline_campaign,
            args.asset_root,
            args.candidate_manifest,
            args.release_source_geometry,
        )
    )
    baseline = json.loads(baseline_path.read_text())
    baseline_cases = validate_baseline_cases(
        baseline, nominal_only_diagnostic=args.nominal_only_baseline_diagnostic
    )
    experiment_kind = (
        NOMINAL_BASELINE_EXPERIMENT_KIND if args.nominal_only_baseline_diagnostic else EXPERIMENT_KIND
    )
    old_timeline = baseline["timeline"]
    old_motion = Path(old_timeline["timeline_path"]).resolve(strict=True)
    if sha256_file(old_motion) != old_timeline["timeline_sha256"]:
        raise ValueError("baseline timeline bytes changed")
    with np.load(old_motion, allow_pickle=False) as archive:
        original = {key: archive[key].copy() for key in archive.files}
    motion, timeline = source_initialized_tail(original, old_timeline)
    timeline["kind"] = experiment_kind
    compatibility = release_compatibility_contract(
        sha256_file(geometry), "released29_scale_bounded_linear_v2", BUFFERED_TIMING
    )
    if baseline["release_compatibility"] != compatibility:
        raise ValueError("source initialization may not change runtime semantics")
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    torch.set_num_threads(1)
    policy, identity = load_root_feedback_pair(
        manifest_path, session_options=options, expected_release_compatibility=compatibility
    )
    if any(row["policy_identity"] != identity for row in baseline["records"]):
        raise ValueError("source initialization must use the identical baseline policy")
    virtual_vr = virtual_source_vr_terms(motion, geometry)
    paths = [baseline_path, old_motion, manifest_path, geometry, assets / MODEL, root / PHYSICS]
    paths.extend(Path(value) for value in identity["component_paths"].values())
    paths.extend(
        Path(module.__file__).resolve()
        for name, module in list(sys.modules.items())
        if name.startswith("gear_sonic.") and (getattr(module, "__file__", None) or "").endswith(".py")
    )
    paths.append(Path(__file__).resolve())
    inputs = {str(path): sha256_file(path) for path in paths}
    output = args.output_directory.resolve()
    if output.exists() or args.output_directory.is_symlink():
        raise ValueError("source initialization requires a new evidence directory")
    output.mkdir(parents=True, exist_ok=False)
    timeline_path = output / "lifecycle_reference.npz"
    with timeline_path.open("xb") as stream:
        np.savez_compressed(stream, **motion)
    timeline.update(timeline_path=str(timeline_path), timeline_sha256=sha256_file(timeline_path))
    receipt = dict(
        kind=experiment_kind,
        baseline_cases=baseline_cases,
        nominal_only_baseline_diagnostic=args.nominal_only_baseline_diagnostic,
        baseline_has_nominal_and_both_push_cases=not args.nominal_only_baseline_diagnostic,
        inputs=inputs,
        policy_identity=identity,
        timeline=timeline,
        standing_acquisition_tested=False,
        initial_state_deliberately_changed=True,
        complete_original_source_and_return_tail_preserved=True,
        synthetic_prehistory_is_observed_robot_history=False,
        no_postinitial_robot_pose_rewrites=True,
        no_fallback_controller=True,
        eligible_parent_for_training_continuation=False,
        hardware_world_state_estimator_qualified=False,
        **FLAGS,
    )
    write_json(output / "started.json", receipt)
    adapter = BufferedSourceSimulationAdapter(motion, virtual_vr)
    result, arrays = run_reference_diagnostic(
        root=root, asset_root=assets, motion_path=timeline_path, policy=policy, runtime_adapter=adapter
    )
    baseline_result = next(row["result"] for row in baseline["records"] if row["case"] == "nominal")
    for key in (
        "compiled_model_sha256",
        "physics_config_sha256",
        "kp_hardware",
        "kd_hardware",
        "actuator_contract",
    ):
        if result[key] != baseline_result[key]:
            raise ValueError(f"source initialization changed physical contract: {key}")
    arrays.update(adapter.root_adapter.arrays())
    arrays["actual_policy_encoder267"] = np.asarray(adapter.actual_encoder_inputs)
    arrays["released_model_raw23"] = np.asarray(adapter.model_outputs)
    arrays["bounded_linear_projection_delta_rad"] = np.asarray(adapter.projection_delta_rad).reshape(-1, 23)
    arrays["source_emission_anchor_setpoint_timestamps_s"] = np.asarray(adapter.timestamps)
    with (output / "nominal.received_source.npz").open("xb") as stream:
        np.savez_compressed(stream, **adapter.source)
    result["root_response"] = summarize_root_perturbations(
        adapter.root_adapter, arrays, completed_controls=result["completed_controls"], failure=result["failure"]
    )
    lifecycle = assess_lifecycle_diagnostic(timeline, result, arrays)
    lifecycle["source_initialized_motion_and_return_integrated"] = lifecycle.pop(
        "single_policy_full_lifecycle_integrated"
    )
    lifecycle["robot_state_resets_after_source_initialization"] = lifecycle.pop(
        "robot_state_resets_after_initial_standing"
    )
    lifecycle.update(
        kind=experiment_kind, standing_acquisition_tested=False, single_policy_full_lifecycle_integrated=False
    )
    trace = output / "nominal.npz"
    with trace.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    row = dict(
        kind=experiment_kind,
        case="nominal_source_initialized",
        result=result,
        lifecycle=lifecycle,
        policy_identity=identity,
        trace_path=str(trace),
        trace_sha256=sha256_file(trace),
        **FLAGS,
    )
    write_json(output / "nominal.json", row)
    for path, digest in inputs.items():
        if sha256_file(Path(path)) != digest:
            raise ValueError(f"source initialization input changed: {path}")
    write_json(output / "report.json", {**receipt, "records": [row]})
    print(
        json.dumps(dict(completed=result["completed_controls"], failure=result["failure"], lifecycle=lifecycle)),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
