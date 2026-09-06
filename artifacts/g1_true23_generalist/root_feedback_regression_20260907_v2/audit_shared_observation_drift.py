"""Separate full-decoder drift from root-branch effect on identical saved states."""

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file


def main():
    root = Path(__file__).resolve().parents[3]
    base = root / "artifacts/g1_true23_generalist"
    trace_path = base / "planned_v5_endpoint_root0_20260907_v1/nominal.npz"
    paths = [
        base / "root_feedback_smoke_20260907_v1/export_initial/decoder.root_feedback.diagnostic.onnx",
        base / "root_feedback_regression_20260907_v2/export_updated/decoder.root_feedback.diagnostic.onnx",
    ]
    inputs = {str(path): sha256_file(path) for path in [trace_path, *paths, Path(__file__)]}
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    sessions = [
        ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"]) for path in paths
    ]
    with np.load(trace_path, allow_pickle=False) as archive:
        indices = np.arange(0, len(archive["raw23"]), 5)
        observations = archive["decoder994"][indices].copy()
        feedback = archive["root_feedback9"][indices].copy()
        recorded = archive["raw23"][indices].copy()
    zero = np.zeros((1, 9), dtype=np.float32)
    rows = []
    for index, observation, received, previous in zip(indices, observations, feedback, recorded):
        observation = observation.astype(np.float32)[None]
        parent = sessions[0].run(["action"], {"obs_dict": observation, "root_feedback": zero})[0][0]
        changed_core = sessions[1].run(["action"], {"obs_dict": observation, "root_feedback": zero})[0][0]
        full = sessions[1].run(["action"], {"obs_dict": observation, "root_feedback": received[None]})[0][0]
        rows.append(
            dict(
                control_index=int(index),
                saved_parent_reproduction_max_abs=float(np.max(np.abs(parent - previous))),
                decoder_change_rms=float(np.sqrt(np.mean((changed_core - parent) ** 2))),
                feedback_branch_effect_rms=float(np.sqrt(np.mean((full - changed_core) ** 2))),
                total_action_change_rms=float(np.sqrt(np.mean((full - parent) ** 2))),
            )
        )
    if max(row["saved_parent_reproduction_max_abs"] for row in rows) > 1e-5:
        raise ValueError("saved baseline decoder input does not reproduce its own action")
    phases = {
        "standing": (0, 250),
        "acquisition": (250, 350),
        "dance": (350, 1441),
        "return_and_hold": (1441, 1841),
    }
    summaries = {}
    for name, (start, stop) in phases.items():
        selected = [row for row in rows if start <= row["control_index"] < stop]
        summaries[name] = {
            key + "_p95": float(np.percentile([row[key] for row in selected], 95))
            for key in ("decoder_change_rms", "feedback_branch_effect_rms", "total_action_change_rms")
        }
    for path, digest in inputs.items():
        if sha256_file(Path(path)) != digest:
            raise ValueError("drift diagnostic input changed")
    result = dict(
        kind="g1_root_feedback_shared_observation_drift_v1",
        inputs=inputs,
        sampled_controls=len(rows),
        stride=5,
        phase_summaries=summaries,
        rows=rows,
        counterfactual_action_only=True,
        new_dynamics_integrated=False,
        policy_improvement_proven=False,
        deployment_ready=False,
        hardware_authorized=False,
    )
    with (Path(__file__).parent / "shared_observation_drift.json").open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
    print(json.dumps(summaries, sort_keys=True))


if __name__ == "__main__":
    main()
