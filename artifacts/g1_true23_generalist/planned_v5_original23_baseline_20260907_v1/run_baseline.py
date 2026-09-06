"""Reproduce original pinned ONNX policy on accepted planned V5 full lifecycle.

CPU only. No checkpoint/Torch-model loading, robot middleware, force pulses,
post-initial-state resets, fallback controller, or motion-prefix limit.
Use --output-directory NEW_DIRECTORY to reproduce without replacing evidence.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import sys
import time

import numpy as np
import onnxruntime as ort

from gear_sonic.utils.g1_true23_clean_mujoco_teleop import sha256_file
from gear_sonic.utils.g1_true23_generalist_benchmark import FLAGS, MODEL, PHYSICS, run_reference_diagnostic
from gear_sonic.utils.g1_true23_generalist_lifecycle import assess_lifecycle_diagnostic, build_lifecycle_timeline
from gear_sonic.utils.g1_true23_sonic_library_replay import ExactHashSonicPolicy
from gear_sonic.utils.g1_true23_step1b_mujoco import prepare_true23_model

MOTION_SHA = "dd325625b507d4815cae2f2795b1ae38c7f2ec33731f0c7ecd9544f26f005a91"
FIELDS = ("fps", "joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def bind_imported_sources(bindings):
    for name, module in list(sys.modules.items()):
        filename = getattr(module, "__file__", None)
        if name.startswith("gear_sonic.") and filename and Path(filename).suffix == ".py":
            path = Path(filename).resolve(strict=True)
            bindings.setdefault(str(path), sha256_file(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    assets = root.parent / "GR00T-WholeBodyControl"
    output = args.output_directory.resolve()
    if output.exists():
        if output != Path(__file__).resolve().parent or any(
            path.name != "run_baseline.py" for path in output.iterdir()
        ):
            raise ValueError("evidence directory already populated; choose a new output directory")
    else:
        output.mkdir(parents=True, exist_ok=False)
    v5 = root / "artifacts/g1_true23_generalist/planned_dance_retarget_20260907_v5"
    source_path = v5 / "adapted.true23.npz"
    if sha256_file(source_path) != MOTION_SHA:
        raise ValueError("accepted planned V5 motion differs from approved exact payload")
    constants_path = root / "gear_sonic/scripts/evaluate_g1_true23_generalist_baselines.py"
    # Read literal constants without importing the CLI's unrelated Torch/LoRA paths.
    constants = {}
    for node in ast.parse(constants_path.read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            key = node.targets[0].id
            if key in {"ORIGINAL_ENCODER", "ORIGINAL_DECODER", "ORIGINAL_ENCODER_SHA", "ORIGINAL_DECODER_SHA"}:
                constants[key] = ast.literal_eval(node.value)
    if len(constants) != 4:
        raise ValueError("original paired policy constants unavailable")
    encoder, decoder = (assets / constants[key] for key in ("ORIGINAL_ENCODER", "ORIGINAL_DECODER"))
    paths = [
        Path(__file__).resolve(),
        constants_path,
        source_path,
        encoder,
        decoder,
        assets / MODEL,
        root / PHYSICS,
        *(
            v5 / name
            for name in (
                "report.json",
                "planned.named29.npz",
                "saved_motion_verification.json",
                "source_lineage_verification.json",
            )
        ),
    ]
    inputs = {str(path.resolve(strict=True)): sha256_file(path) for path in paths}
    with np.load(source_path, allow_pickle=False) as archive:
        source = {key: archive[key].copy() for key in FIELDS}
    _, model, _ = prepare_true23_model(assets / MODEL, root / PHYSICS)
    motion, timeline = build_lifecycle_timeline(source, model=model, simulation_config=root / PHYSICS)
    timeline_path = output / "lifecycle_reference.npz"
    with timeline_path.open("xb") as stream:
        np.savez_compressed(stream, **motion)
    timeline.update(
        source_motion_path=str(source_path),
        source_motion_sha256=MOTION_SHA,
        timeline_path=str(timeline_path),
        timeline_sha256=sha256_file(timeline_path),
    )
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    policy = ExactHashSonicPolicy(
        encoder,
        decoder,
        expected_encoder_sha256=constants["ORIGINAL_ENCODER_SHA"],
        expected_decoder_sha256=constants["ORIGINAL_DECODER_SHA"],
        session_options=options,
    )
    bind_imported_sources(inputs)
    write_json(
        output / "started.json",
        dict(
            kind="g1_true23_original23_planned_v5_nominal_lifecycle_started_v1",
            source_inputs=inputs,
            timeline=timeline,
            original_policy=constants,
            maximum_controls=None,
            cpu_onnx_only=True,
            **FLAGS,
        ),
    )
    started = time.monotonic()
    result, arrays = run_reference_diagnostic(
        root=root,
        asset_root=assets,
        motion_path=timeline_path,
        policy=policy,
        profile="native_model",
        maximum_controls=None,
    )
    elapsed = time.monotonic() - started
    lifecycle = assess_lifecycle_diagnostic(timeline, result, arrays)
    trace = output / "original_walk_v14_100.native_model.npz"
    with trace.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    bind_imported_sources(inputs)
    for path, digest in inputs.items():
        if sha256_file(Path(path)) != digest:
            raise ValueError(f"baseline input changed during execution: {path}")
    receipt = dict(
        kind="g1_true23_original23_planned_v5_nominal_lifecycle_baseline_v1",
        policy="original_walk_v14_100",
        original_policy=constants,
        result=result,
        lifecycle=lifecycle,
        source_inputs=inputs,
        timeline=timeline,
        trace_path=str(trace),
        trace_sha256=sha256_file(trace),
        elapsed_simulator_wall_seconds=elapsed,
        maximum_controls=None,
        cpu_onnx_only=True,
        heavyweight_checkpoint_models_loaded=False,
        gpu_simulation_used=False,
        no_postinitial_robot_pose_rewrites=True,
        no_fallback_controller=True,
        shared_referee_fresh_observation_phase_not_historical_original_frontend=True,
        source_motion_is_full_planned_v5_retarget=True,
        source_motion_original_29_frames=546,
        source_motion_retimed_native23_frames=timeline["source_frames"],
        kinematic_acceptance_does_not_qualify_dynamics=True,
        hardware_or_network_actuation_used=False,
        **FLAGS,
    )
    write_json(output / "report.json", receipt)
    print(
        json.dumps(
            dict(
                completed=result["completed_controls"],
                requested=result["available_controls"],
                failure=result["failure"],
                phases=lifecycle["phases"],
                source_tracking=lifecycle["source_motion_tracking"],
                report_sha256=sha256_file(output / "report.json"),
            ),
            allow_nan=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
