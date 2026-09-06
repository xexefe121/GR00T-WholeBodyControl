"""Record all original SONIC clips with actual captured C++ G1 observations.

Offline numerical-boundary experiment only. No SDK, DDS or robot controller
is created. Preserve the original reference, model, gains, action bounds and
all requested cases, including failures. Optional hand-collision ablation
uses the previously captured separate model; it does not alter native23.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys

import mujoco
import numpy as np
import onnxruntime as ort

from gear_sonic.scripts import simulate_g1_sonic_library_motions as legacy
from gear_sonic.scripts.record_g1_sonic_original29_baseline import (
    CLIPS,
    DECODER_SHA,
    ENCODER_SHA,
    dump,
    trace_metrics,
    validate_source_report,
)
from gear_sonic.utils.g1_sonic_cpp_observation_trace import (
    PROFILE,
    same_state_observation_comparison,
    trace_cpp_observation_motion,
)
from gear_sonic.utils.g1_sonic_cpp_observations import FLAGS, capture_cpp_observations
from gear_sonic.utils.g1_sonic_cpp_parameters import CppParameters, file_sha256
from gear_sonic.utils.g1_sonic_hand_collision_variant import inspect_hand_contacts
from gear_sonic.utils.g1_true23_reference_floor import compiled_model_sha256


def validate_baseline_rows(report, *, hand_collisions=False):
    kind = (
        "g1_sonic_original29_full_hand_collision_recorded_diagnostic_v1"
        if hand_collisions
        else "g1_sonic_original29_recorded_comparison_v1"
    )
    if report.get("kind") != kind or any(report.get(key) is not False for key in FLAGS):
        raise ValueError("requires the explicitly unaccepted recorded source comparison")
    rows = [
        row
        for row in report["records"]
        if hand_collisions or row.get("profile") == "cpp_parameters_and_float32_targets"
    ]
    if len(rows) != 3 or [row["name"] for row in rows] != list(CLIPS):
        raise ValueError("all three original clips must remain present and in order")
    if any(
        row.get("failure") is not None or not row.get("trace_path") or not row.get("trace_sha256") for row in rows
    ):
        raise ValueError("all baseline attempts require their retained recorded traces")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--hand-collision-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    baseline = args.baseline_dir.resolve(strict=True)
    assets = args.asset_root.resolve(strict=True)
    output = args.output_dir.resolve()
    inputs = {}

    def bind(path, expected=None):
        path = Path(path).resolve(strict=True)
        digest = file_sha256(path)
        if digest != inputs.get(str(path), digest) or (expected is not None and digest != expected):
            raise ValueError(f"recorded source identity changed: {path}")
        inputs[str(path)] = digest
        return path

    source = json.loads(bind(baseline / "report.json").read_text())
    rows = validate_baseline_rows(source)
    for path, digest in source["inputs"].items():
        bind(path, digest)
    models = [("original_masks", bind(baseline / "original29.mjb", source["compiled_model_sha256"]), rows)]
    if args.hand_collision_dir:
        directory = args.hand_collision_dir.resolve(strict=True)
        collision = json.loads(bind(directory / "report.json").read_text())
        collision_rows = validate_baseline_rows(collision, hand_collisions=True)
        for path, digest in collision["inputs"].items():
            bind(path, digest)
        variant = collision["model_variant"]
        if (
            variant["baseline_compiled_sha256"] != source["compiled_model_sha256"]
            or variant["variant_compiled_sha256"] == source["compiled_model_sha256"]
        ):
            raise ValueError("collision ablation must preserve the same baseline and a distinct model")
        models.append(
            (
                "hand_collisions",
                bind(directory / "original29_with_hand_collisions.mjb", variant["variant_compiled_sha256"]),
                collision_rows,
            )
        )
    parameters = CppParameters(json.loads(bind(baseline / "cpp_capture/parameters.json").read_text()))
    planner_dir = assets / "artifacts/g1_true23/sonic_library_motion_suite_v2"
    planner = validate_source_report(json.loads(bind(planner_dir / "report.json").read_text()))
    motions = {}
    for name in CLIPS:
        row = planner[name]
        with np.load(bind(planner_dir / row["npz"], row["npz_sha256"]), allow_pickle=False) as archive:
            if float(archive["fps"][0]) != 30 or int(archive["mode"][0]) != row["mode"]:
                raise ValueError("planner metadata drift")
            motions[name] = archive["qpos"].copy()
    encoder_path = bind(assets / "gear_sonic_deploy/policy/release/model_encoder.onnx", ENCODER_SHA)
    decoder_path = bind(assets / "gear_sonic_deploy/policy/release/model_decoder.onnx", DECODER_SHA)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    encoder = ort.InferenceSession(str(encoder_path), sess_options=options, providers=["CPUExecutionProvider"])
    decoder = ort.InferenceSession(str(decoder_path), sess_options=options, providers=["CPUExecutionProvider"])
    legacy._validate_policy_abi(encoder, decoder)
    output.mkdir(parents=True, exist_ok=False)
    capture = capture_cpp_observations(root, output / "cpp_capture")
    for path, digest in capture["inputs"].items():
        bind(path, digest)
    bind(output / "cpp_capture/report.json")
    bind(Path(__file__))
    for name, module in list(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if name.startswith("gear_sonic.") and path and Path(path).suffix == ".py":
            bind(path)
    started = {
        "kind": "g1_sonic_cpp_observation_full_source_replay_diagnostic_v1",
        "inputs": dict(inputs),
        "profile": PROFILE,
        "requested_clips": list(CLIPS),
        "requested_models": [name for name, _, _ in models],
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "mujoco": mujoco.__version__,
            "onnxruntime": ort.__version__,
        },
        "onnxruntime_providers": encoder.get_providers(),
        "onnxruntime_threads": 1,
        "complete_cpp_deployment_equivalence_proven": False,
        "live_receding_horizon_planner_executed": False,
        "physical_damping_cause_diagnosed": False,
        "native23_model_limits_controller_or_hardware_modified": False,
        **FLAGS,
    }
    dump(output / "started.json", started)
    records = []
    for model_name, model_path, previous_rows in models:
        model = mujoco.MjModel.from_binary_path(str(model_path))
        model_hash = compiled_model_sha256(model)
        if model_hash != inputs[str(model_path)]:
            raise ValueError("loaded source model does not match its captured MJB")
        for previous in previous_rows:
            name = previous["name"]
            key = f"{model_name}.{name}"
            record = {
                "name": name,
                "model_variant": model_name,
                "profile": PROFILE,
                "compiled_model_sha256": model_hash,
                "failure": None,
                **FLAGS,
            }
            print(json.dumps({"starting_cpp_observation_case": key}), flush=True)
            try:
                arrays, details = trace_cpp_observation_motion(
                    model, encoder, decoder, motions[name], parameters=parameters, capture=capture
                )
                path = output / f"{key}.npz"
                with path.open("xb") as stream:
                    np.savez_compressed(stream, **arrays)
                bind(path)
                record.update(
                    trace_path=str(path),
                    trace_sha256=inputs[str(path)],
                    details=details,
                    metrics=trace_metrics(model, arrays),
                    hand_contacts=inspect_hand_contacts(model, arrays["pre_qpos"]),
                )
                with np.load(
                    bind(previous["trace_path"], previous["trace_sha256"]), allow_pickle=False
                ) as archive:
                    old = {key: archive[key].copy() for key in archive.files}
                same_arrays, same_report = same_state_observation_comparison(
                    motions[name], old, encoder, decoder, capture=capture
                )
                same_path = output / f"{key}.same_state.npz"
                with same_path.open("xb") as stream:
                    np.savez_compressed(stream, **same_arrays)
                bind(same_path)
                record.update(
                    same_state_comparison=same_report,
                    same_state_trace_path=str(same_path),
                    same_state_trace_sha256=inputs[str(same_path)],
                    previous_trace_path=previous["trace_path"],
                    previous_trace_sha256=previous["trace_sha256"],
                    previous_details=previous["details"],
                    previous_metrics=previous["metrics"],
                )
            except Exception as exc:
                record["failure"] = f"{type(exc).__name__}: {exc}"
            records.append(record)
            dump(output / f"{key}.report.json", record)
            bind(output / f"{key}.report.json")
            print(
                json.dumps(
                    {
                        "name": key,
                        "failure": record["failure"],
                        "frames_completed": record.get("details", {}).get("frames_completed"),
                        "complete_clip": record.get("details", {}).get("complete_original_clip"),
                        "engine": record.get("details", {}).get("actual_engine_audit"),
                    }
                ),
                flush=True,
            )
        if compiled_model_sha256(model) != model_hash:
            raise ValueError("diagnostic changed a compiled source model")
    for path in list(inputs):
        bind(path)
    report = {
        **started,
        "inputs": inputs,
        "records": records,
        "all_requested_cases_recorded": len(records) == len(models) * 3
        and all(row["failure"] is None for row in records),
        "all_original_clips_completed": len(records) == len(models) * 3
        and all(
            row["failure"] is None and row.get("details", {}).get("complete_original_clip") is True
            for row in records
        ),
    }
    dump(output / "report.json", report)
    return 0 if report["all_requested_cases_recorded"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
