"""Compare the actual frozen training encoder with the selected runtime ONNX.

Uses motion-derived causal inputs. No robot transport or deployment mutation.
Exact discrete token agreement is required; a shape/hash check alone is not
accepted as numerical parity. This does not prove live-input distribution fit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from gear_sonic.trl.mjlab.frozen_platform_lora_runner import _state_sha256
from gear_sonic.utils.g1_23dof_artifact import build_true23_policy_pair
from gear_sonic.utils.g1_true23_clean_mujoco_teleop import (
    encoder267_from_reference,
    motion_reference_terms,
    sha256_file,
)
from gear_sonic.utils.g1_true23_frozen_lora_artifact import load_frozen_lora_diagnostic_policy
from gear_sonic.utils.g1_true23_sonic_library_replay import validate_library_motion


def compare_tokens(expected: np.ndarray, actual: np.ndarray) -> tuple[int, float]:
    if expected.shape != (1, 64) or actual.shape != expected.shape:
        raise ValueError("encoder token ABI mismatch")
    if not np.isfinite(expected).all() or not np.isfinite(actual).all():
        raise ValueError("encoder tokens must be finite")
    return int(np.count_nonzero(expected != actual)), float(np.max(np.abs(expected - actual)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic-policy", type=Path, required=True)
    parser.add_argument("--runtime-encoder", type=Path, required=True)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    artifact = load_frozen_lora_diagnostic_policy(args.diagnostic_policy)
    encoder, _, policy_hash = build_true23_policy_pair(artifact)
    encoder.eval()
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    runtime = ort.InferenceSession(str(args.runtime_encoder), options, providers=["CPUExecutionProvider"])
    if len(runtime.get_inputs()) != 1 or runtime.get_inputs()[0].shape != [1, 267]:
        raise ValueError("runtime encoder input ABI mismatch")
    if len(runtime.get_outputs()) != 1 or runtime.get_outputs()[0].shape != [1, 64]:
        raise ValueError("runtime encoder output ABI mismatch")
    with np.load(args.motion, allow_pickle=False) as archive:
        motion = {key: archive[key].copy() for key in archive.files}
    count = validate_library_motion(motion)
    mismatches = []
    maximum_error = 0.0
    with torch.inference_mode():
        for q9 in range(9, count - 2):
            semantic = encoder267_from_reference(motion_reference_terms(motion, q9), motion["body_quat_w"][q9, 0])
            expected = encoder(torch.from_numpy(semantic[None])).numpy()
            actual = runtime.run(None, {runtime.get_inputs()[0].name: semantic[None]})[0]
            different_coordinates, error = compare_tokens(expected, actual)
            maximum_error = max(maximum_error, error)
            if different_coordinates:
                mismatches.append({"q9": q9, "token_coordinates": different_coordinates})
    report = {
        "kind": "g1_true23_training_runtime_encoder_parity_v1",
        "passed": not mismatches,
        "case_count": count - 11,
        "mismatches": mismatches,
        "maximum_abs_error": maximum_error,
        "policy_state_sha256": policy_hash,
        "paired_encoder_state_sha256": _state_sha256(
            {
                name: tensor
                for name, tensor in artifact["policy_state_dict"].items()
                if name.startswith("actor_module.encoders.teleop.module.")
            }
        ),
        "sources": {
            str(path.resolve()): sha256_file(path)
            for path in (args.diagnostic_policy, args.runtime_encoder, args.motion, Path(__file__))
        },
        "boundary": "actual_exported_training_encoder_plus_FSQ_vs_runtime_ONNX",
        "scope": "motion_reference_causal_inputs_not_live_headset_or_off_reference_states",
        "hardware_authorized": False,
        "robot_commands_published": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(
        json.dumps({"passed": report["passed"], "cases": report["case_count"], "mismatch_count": len(mismatches)})
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
