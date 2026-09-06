"""One-policy native23 standing/full-source-motion/standing CPU diagnostic."""

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from gear_sonic.scripts.evaluate_g1_true23_generalist_baselines import (
    LIFECYCLE,
    LIFECYCLE_SHA,
    ORIGINAL_DECODER,
    ORIGINAL_DECODER_SHA,
    ORIGINAL_ENCODER,
    ORIGINAL_ENCODER_SHA,
    write_json,
)
from gear_sonic.utils.g1_true23_clean_mujoco_teleop import sha256_file
from gear_sonic.utils.g1_true23_generalist_benchmark import (
    FLAGS,
    MODEL,
    PHYSICS,
    load_generalist_pair,
    load_lifecycle_policy,
    run_reference_diagnostic,
)
from gear_sonic.utils.g1_true23_generalist_lifecycle import build_lifecycle_timeline, assess_lifecycle_diagnostic
from gear_sonic.utils.g1_true23_sonic_library_replay import ExactHashSonicPolicy
from gear_sonic.utils.g1_true23_step1b_mujoco import prepare_true23_model
from gear_sonic.utils.g1_true23_training_precision import ieee_training_precision


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, action="append")
    args = parser.parse_args(argv)
    root, assets = Path(__file__).resolve().parents[2], args.asset_root.resolve(strict=True)
    output = args.output_directory.resolve()
    if output.exists() or args.output_directory.is_symlink():
        raise ValueError("lifecycle diagnostic requires new evidence directory")
    motion_path = args.motion.resolve(strict=True)
    inputs = {
        str(path): sha256_file(path)
        for path in (
            motion_path,
            assets / MODEL,
            root / PHYSICS,
            assets / ORIGINAL_ENCODER,
            assets / ORIGINAL_DECODER,
            root / LIFECYCLE,
            assets / "sonic_release/g1_23dof_rev_1_0_low_latency_init.pt",
            assets / "low_latency/last.pt",
        )
    }
    with np.load(motion_path, allow_pickle=False) as archive:
        source = {key: archive[key].copy() for key in archive.files}
    _, model, _ = prepare_true23_model(assets / MODEL, root / PHYSICS)
    motion, timeline = build_lifecycle_timeline(source, model=model, simulation_config=root / PHYSICS)
    output.mkdir(parents=True, exist_ok=False)
    timeline_path = output / "lifecycle_reference.npz"
    with timeline_path.open("xb") as stream:
        np.savez_compressed(stream, **motion)
    timeline.update(
        source_motion_path=str(motion_path),
        source_motion_sha256=sha256_file(motion_path),
        timeline_path=str(timeline_path),
        timeline_sha256=sha256_file(timeline_path),
    )
    write_json(output / "timeline.json", timeline)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    torch.set_num_threads(1)
    rows = []
    with ieee_training_precision() as (precision, guard):
        original = ExactHashSonicPolicy(
            assets / ORIGINAL_ENCODER,
            assets / ORIGINAL_DECODER,
            expected_encoder_sha256=ORIGINAL_ENCODER_SHA,
            expected_decoder_sha256=ORIGINAL_DECODER_SHA,
            session_options=options,
        )
        lora, identity = load_lifecycle_policy(
            root / LIFECYCLE,
            expected_sha256=LIFECYCLE_SHA,
            warm_start=assets / "sonic_release/g1_23dof_rev_1_0_low_latency_init.pt",
            source_checkpoint=assets / "low_latency/last.pt",
        )
        policies = [
            (
                "original_walk_v14_100",
                original,
                dict(encoder_sha256=ORIGINAL_ENCODER_SHA, decoder_sha256=ORIGINAL_DECODER_SHA),
            ),
            ("lifecycle_lora_100", lora, identity),
        ]
        for index, manifest in enumerate(args.candidate_manifest or []):
            policy, identity = load_generalist_pair(manifest, session_options=options)
            policies.append((f"generalist_candidate_{index:03d}", policy, identity))
            for path in [manifest, *(Path(value) for value in identity["component_paths"].values())]:
                inputs[str(path.resolve(strict=True))] = sha256_file(path)
        import sys

        for name, module in list(sys.modules.items()):
            path = getattr(module, "__file__", None)
            if name.startswith("gear_sonic.") and path and Path(path).suffix == ".py":
                inputs[str(Path(path).resolve())] = sha256_file(Path(path))
        write_json(output / "started.json", dict(inputs=dict(inputs), precision=precision, **FLAGS))
        for label, policy, identity in policies:
            guard()
            result, arrays = run_reference_diagnostic(
                root=root, asset_root=assets, motion_path=timeline_path, policy=policy
            )
            assessment = assess_lifecycle_diagnostic(timeline, result, arrays)
            trace_path = output / (label + ".npz")
            with trace_path.open("xb") as stream:
                np.savez_compressed(stream, **arrays)
            row = dict(
                policy=label,
                policy_identity=identity,
                result=result,
                lifecycle=assessment,
                trace_path=str(trace_path),
                trace_sha256=sha256_file(trace_path),
            )
            rows.append(row)
            write_json(output / (label + ".json"), row)
            print(
                json.dumps(
                    dict(
                        policy=label,
                        completed=result["completed_controls"],
                        requested=result["available_controls"],
                        phases=assessment["phases"],
                        failure=result["failure"],
                    )
                ),
                flush=True,
            )
        for path, digest in inputs.items():
            if sha256_file(Path(path)) != digest:
                raise ValueError(f"lifecycle source changed during evaluation: {path}")
        for field in (
            "initial_state_and_history_sha256",
            "compiled_model_sha256",
            "motion_sha256",
            "available_controls",
            "kp_hardware",
            "kd_hardware",
        ):
            if any(rows[0]["result"][field] != row["result"][field] for row in rows[1:]):
                raise ValueError(f"paired lifecycle differs at {field}")
        write_json(
            output / "report.json",
            dict(
                kind="g1_true23_single_policy_paired_lifecycle_diagnostic_v1",
                timeline=timeline,
                records=rows,
                inputs=inputs,
                precision=precision,
                **FLAGS,
            ),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
