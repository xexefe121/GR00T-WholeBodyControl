"""Run every original clip with separately recompiled original29 hand collisions.

This explicit source-model ablation preserves the old baseline. It does not
alter native23, disable collisions, increase actuation limits or authorize
hardware. Its output kind is deliberately distinct from accepted references.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
import onnxruntime as ort

from gear_sonic.scripts import simulate_g1_sonic_library_motions as legacy
from gear_sonic.scripts.record_g1_sonic_original29_baseline import (
    CLIPS,
    dump,
    trace_metrics,
    validate_source_report,
)
from gear_sonic.utils.g1_sonic_cpp_parameters import CppParameters, file_sha256
from gear_sonic.utils.g1_sonic_hand_collision_variant import compile_hand_collision_variant, inspect_hand_contacts
from gear_sonic.utils.g1_sonic_original29_trace import trace_original29_motion
from gear_sonic.utils.g1_true23_reference_floor import compiled_model_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    baseline, assets = args.baseline_dir.resolve(strict=True), args.asset_root.resolve(strict=True)
    output = args.output_dir.resolve()
    source = json.loads((baseline / "report.json").read_text())
    flags = dict(teacher_accepted=False, hardware_authorized=False, deployment_ready=False)
    if source.get("kind") != "g1_sonic_original29_recorded_comparison_v1" or any(
        source.get(k) is not v for k, v in flags.items()
    ):
        raise ValueError("requires the explicitly unaccepted recorded legacy comparison")
    inputs = dict(source["inputs"])

    def bind(path, expected=None):
        path = Path(path).resolve(strict=True)
        digest = file_sha256(path)
        if digest != inputs.get(str(path), digest) or (expected is not None and digest != expected):
            raise ValueError(f"source identity changed: {path}")
        inputs[str(path)] = digest
        return path

    for path in list(inputs):
        bind(path)
    bind(baseline / "report.json")
    original, model, model_report = compile_hand_collision_variant(
        bind(root / "gear_sonic_deploy/g1/scene_29dof.xml"),
        expected_baseline_sha256=source["compiled_model_sha256"],
    )
    parameters = CppParameters(json.loads(bind(baseline / "cpp_capture/parameters.json").read_text()))
    encoder_path = bind(assets / "gear_sonic_deploy/policy/release/model_encoder.onnx")
    decoder_path = bind(assets / "gear_sonic_deploy/policy/release/model_decoder.onnx")
    source_dir = assets / "artifacts/g1_true23/sonic_library_motion_suite_v2"
    planner = validate_source_report(json.loads(bind(source_dir / "report.json").read_text()))
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    encoder = ort.InferenceSession(str(encoder_path), sess_options=options, providers=["CPUExecutionProvider"])
    decoder = ort.InferenceSession(str(decoder_path), sess_options=options, providers=["CPUExecutionProvider"])
    legacy._validate_policy_abi(encoder, decoder)
    bind(Path(__file__))
    for name, module in list(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if name.startswith("gear_sonic.") and path and Path(path).suffix == ".py":
            bind(path)
    output.mkdir(parents=True, exist_ok=False)
    model_path = output / "original29_with_hand_collisions.mjb"
    mujoco.mj_saveModel(model, filename=str(model_path))
    bind(model_path, model_report["variant_compiled_sha256"])
    dump(output / "started.json", {"inputs": dict(inputs), "model_variant": model_report, **flags})
    previous = {r["name"]: r for r in source["records"] if r["profile"] == "cpp_parameters_and_float32_targets"}
    if list(previous) != list(CLIPS):
        raise ValueError("the comparison must retain all three original clips")
    records = []
    for name in CLIPS:
        record = {"name": name, "failure": None, "trace_path": None, **flags}
        print(json.dumps({"starting_full_hand_collision_case": name}), flush=True)
        try:
            with np.load(
                bind(source_dir / planner[name]["npz"], planner[name]["npz_sha256"]), allow_pickle=False
            ) as archive:
                if float(archive["fps"][0]) != 30 or int(archive["mode"][0]) != planner[name]["mode"]:
                    raise ValueError("planner metadata drift")
                raw = archive["qpos"].copy()
            arrays, details = trace_original29_motion(model, encoder, decoder, raw, parameters=parameters)
            path = output / f"{name}.npz"
            with path.open("xb") as stream:
                np.savez_compressed(stream, **arrays)
            bind(path)
            old = previous[name]
            with np.load(bind(old["trace_path"], old["trace_sha256"]), allow_pickle=False) as archive:
                old_poses = archive["pre_qpos"].copy()
            record.update(
                trace_path=str(path),
                trace_sha256=file_sha256(path),
                details=details,
                metrics=trace_metrics(model, arrays),
                hand_contacts=inspect_hand_contacts(model, arrays["pre_qpos"]),
                original_motion_under_original_masks=inspect_hand_contacts(original, old_poses),
                original_motion_under_enabled_masks_no_physics=inspect_hand_contacts(model, old_poses),
                original_baseline_completed_frames=old["details"]["frames_completed"],
                counterfactual_contact_check_is_not_a_rollout=True,
            )
        except Exception as exc:
            record["failure"] = f"{type(exc).__name__}: {exc}"
        records.append(record)
        dump(output / f"{name}.report.json", record)
        print(
            json.dumps(
                {
                    "name": name,
                    "error": record["failure"],
                    "completed": record.get("details", {}).get("frames_completed"),
                    "full_clip": record.get("details", {}).get("complete_original_clip"),
                }
            ),
            flush=True,
        )
    for path in list(inputs):
        bind(path)
    if (
        compiled_model_sha256(original) != model_report["baseline_compiled_sha256"]
        or compiled_model_sha256(model) != model_report["variant_compiled_sha256"]
    ):
        raise ValueError("experiment changed a compiled model")
    dump(
        output / "report.json",
        {
            "kind": "g1_sonic_original29_full_hand_collision_recorded_diagnostic_v1",
            "inputs": inputs,
            "model_variant": model_report,
            "records": records,
            "all_requested_cases_recorded": len(records) == 3 and all(row["failure"] is None for row in records),
            "all_original_clips_completed": all(
                row.get("details", {}).get("complete_original_clip") is True for row in records
            ),
            "original29_source_parity_proven": False,
            "native23_model_or_limits_modified": False,
            "runtime": source["runtime"],
            "new_runtime_mujoco": mujoco.__version__,
            "new_runtime_onnxruntime": ort.__version__,
            **flags,
        },
    )


if __name__ == "__main__":
    main()
