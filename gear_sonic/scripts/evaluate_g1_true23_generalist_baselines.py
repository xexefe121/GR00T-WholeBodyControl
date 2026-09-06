"""Compare original native23 and lifecycle LoRA in one nominal CPU simulator.

Reference-reset diagnostics only. Completion does not establish lifecycle,
generalization, teleoperation, hardware safety or deployment readiness.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import onnxruntime as ort
import torch

from gear_sonic.utils.g1_true23_generalist_benchmark import (
    FLAGS,
    MODEL,
    PHYSICS,
    PROFILES,
    compare_original29_source,
    load_generalist_pair,
    load_lifecycle_policy,
    render_recorded_rollout,
    run_reference_diagnostic,
)
from gear_sonic.utils.g1_true23_sonic_library_replay import ExactHashSonicPolicy
from gear_sonic.utils.g1_true23_clean_mujoco_teleop import sha256_file
from gear_sonic.utils.g1_true23_training_precision import ieee_training_precision

ORIGINAL_ENCODER = "artifacts/g1_true23/causal_model_250_20260803/causal_model_250.encoder.onnx"
ORIGINAL_DECODER = "artifacts/g1_true23/pico_internet_fullbody_v14_100_eval/model_100.decoder.onnx"
ORIGINAL_ENCODER_SHA = "733353148bef1eb8dd83a96416b7a89f0b5c3530ceb9e0cec9c25fdb04f56ff2"
ORIGINAL_DECODER_SHA = "f66408ae9a10720a3aff717269d0e2a4e07ab471e449a6fe8f5bae5e8607ef63"
LIFECYCLE = "artifacts/g1_true23_frozen_lora/lifecycle_ppo_20260907_v1/train100/lifecycle_ppo_model_100.pt"
LIFECYCLE_SHA = "987d550a774db21f46284788f90f42e645198245769bea12bed70af063bb10cd"


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--motion", type=Path, required=True, action="append")
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--profiles", choices=PROFILES, nargs="+", default=["native_model"])
    parser.add_argument("--lifecycle-checkpoint", type=Path)
    parser.add_argument("--lifecycle-sha256", default=LIFECYCLE_SHA)
    parser.add_argument(
        "--candidate-manifest",
        type=Path,
        action="append",
        help="Additional matched generalist ONNX diagnostic pairs",
    )
    parser.add_argument("--source29-trace", type=Path, action="append", help="One all-frame trace per --motion")
    parser.add_argument("--source29-lineage", type=Path, action="append", help="One retarget lineage per --motion")
    parser.add_argument("--source29-model", type=Path)
    parser.add_argument(
        "--render", action="store_true", help="Render saved measured trajectories; not new policy execution"
    )
    parser.add_argument(
        "--maximum-controls", type=int, help="Diagnostic prefixes cannot satisfy full-motion completion"
    )
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    assets = args.asset_root.resolve(strict=True)
    output = args.output_directory.resolve()
    if output.exists() or args.output_directory.is_symlink():
        raise ValueError("benchmark requires a new evidence directory")
    if len(set(args.profiles)) != len(args.profiles):
        raise ValueError("duplicate benchmark profiles")
    motions = [path.resolve(strict=True) for path in args.motion]
    if len(set(motions)) != len(motions):
        raise ValueError("duplicate motion paths")
    source_comparison = (
        args.source29_trace is not None or args.source29_lineage is not None or args.source29_model is not None
    )
    if source_comparison and (
        args.source29_trace is None
        or args.source29_lineage is None
        or args.source29_model is None
        or len(args.source29_trace) != len(motions)
        or len(args.source29_lineage) != len(motions)
    ):
        raise ValueError("source comparison needs model and one trace/lineage for each motion")
    checkpoint = (args.lifecycle_checkpoint or root / LIFECYCLE).resolve(strict=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    torch.set_num_threads(1)
    source_inputs = {
        str(path): sha256_file(path)
        for path in [
            *motions,
            assets / ORIGINAL_ENCODER,
            assets / ORIGINAL_DECODER,
            assets / "sonic_release/g1_23dof_rev_1_0_low_latency_init.pt",
            assets / "low_latency/last.pt",
            assets / MODEL,
            root / PHYSICS,
            checkpoint,
        ]
    }
    if source_comparison:
        for path in [args.source29_model, *args.source29_trace, *args.source29_lineage]:
            source_inputs[str(path.resolve(strict=True))] = sha256_file(path)
    output.mkdir(parents=True, exist_ok=False)
    write_json(
        output / "started.json",
        dict(
            kind="g1_true23_shared_cpu_baseline_request_v1",
            source_inputs=source_inputs,
            profiles=args.profiles,
            reference_reset_diagnostic_only=True,
            **FLAGS,
        ),
    )
    rows = []
    with ieee_training_precision() as (precision, guard):
        original = ExactHashSonicPolicy(
            assets / ORIGINAL_ENCODER,
            assets / ORIGINAL_DECODER,
            expected_decoder_sha256=ORIGINAL_DECODER_SHA,
            expected_encoder_sha256=ORIGINAL_ENCODER_SHA,
            session_options=options,
        )
        lifecycle, lifecycle_identity = load_lifecycle_policy(
            checkpoint,
            expected_sha256=args.lifecycle_sha256,
            warm_start=assets / "sonic_release/g1_23dof_rev_1_0_low_latency_init.pt",
            source_checkpoint=assets / "low_latency/last.pt",
        )
        identities = {
            "original_walk_v14_100": dict(
                encoder_sha256=ORIGINAL_ENCODER_SHA, decoder_sha256=ORIGINAL_DECODER_SHA
            ),
            "lifecycle_lora_100": lifecycle_identity,
        }
        policies = [("original_walk_v14_100", original), ("lifecycle_lora_100", lifecycle)]
        for index, path in enumerate(args.candidate_manifest or []):
            candidate, identity = load_generalist_pair(path, session_options=options)
            label = f"generalist_candidate_{index:03d}"
            policies.append((label, candidate))
            identities[label] = identity
            for candidate_path in [path, *(Path(value) for value in identity["component_paths"].values())]:
                source_inputs[str(candidate_path.resolve(strict=True))] = sha256_file(candidate_path)
        for name, module in list(sys.modules.items()):
            path = getattr(module, "__file__", None)
            if name.startswith("gear_sonic.") and path and Path(path).suffix == ".py":
                source_inputs[str(Path(path).resolve())] = sha256_file(Path(path))
        write_json(output / "loaded.json", dict(source_inputs=dict(source_inputs), precision=precision))
        for profile in args.profiles:
            for index, motion in enumerate(motions):
                paired = []
                for label, policy in policies:
                    guard()
                    result, arrays = run_reference_diagnostic(
                        root=root,
                        asset_root=assets,
                        motion_path=motion,
                        policy=policy,
                        profile=profile,
                        maximum_controls=args.maximum_controls,
                    )
                    if source_comparison:
                        result["original29_source_comparison"] = compare_original29_source(
                            result,
                            arrays,
                            source_trace=args.source29_trace[index],
                            source_model=args.source29_model,
                            retarget_lineage=args.source29_lineage[index],
                            asset_root=assets,
                        )
                    base = f"{profile}.{index:03d}.{label}"
                    trace = output / (base + ".npz")
                    with trace.open("xb") as stream:
                        np.savez_compressed(stream, **arrays)
                    row = dict(
                        policy=label,
                        policy_identity=identities[label],
                        result=result,
                        trace_path=str(trace),
                        trace_sha256=sha256_file(trace),
                    )
                    if args.render:
                        row["offline_recorded_video"] = render_recorded_rollout(
                            result, arrays, asset_root=assets, output=output / (base + ".mp4")
                        )
                    write_json(output / (base + ".json"), row)
                    paired.append(row)
                    rows.append(row)
                    print(
                        json.dumps(
                            dict(
                                profile=profile,
                                motion=str(motion),
                                policy=label,
                                completed=result["completed_controls"],
                                available=result["available_controls"],
                                full_source_motion_completed=result["tracking"]["full_source_motion_completed"],
                                failure=result["failure"],
                            )
                        ),
                        flush=True,
                    )
                for field in (
                    "initial_state_and_history_sha256",
                    "compiled_model_sha256",
                    "motion_sha256",
                    "available_controls",
                    "kp_hardware",
                    "kd_hardware",
                ):
                    if any(paired[0]["result"][field] != item["result"][field] for item in paired[1:]):
                        raise ValueError(f"paired comparison differs at {field}")
        newly_loaded_inputs = {}
        for name, module in list(sys.modules.items()):
            path = getattr(module, "__file__", None)
            if name.startswith("gear_sonic.") and path and Path(path).suffix == ".py":
                resolved = str(Path(path).resolve())
                if resolved not in source_inputs:
                    newly_loaded_inputs[resolved] = sha256_file(Path(path))
        for path, digest in source_inputs.items():
            if sha256_file(Path(path)) != digest:
                raise ValueError(f"benchmark input changed: {path}")
        write_json(
            output / "report.json",
            dict(
                kind="g1_true23_shared_cpu_paired_baseline_v1",
                records=rows,
                source_inputs=source_inputs,
                lazily_loaded_postprocessing_inputs=newly_loaded_inputs,
                precision=precision,
                paired_initial_states_and_physics_equal=True,
                standing_acquisition_or_return_tested=False,
                generalization_tested=False,
                reference_reset_diagnostic_only=True,
                gpu_simulation_used=False,
                hardware_or_network_actuation_used=False,
                **FLAGS,
            ),
        )
    return 0  # Evidence collection completed; not a qualification result.


if __name__ == "__main__":
    raise SystemExit(main())
