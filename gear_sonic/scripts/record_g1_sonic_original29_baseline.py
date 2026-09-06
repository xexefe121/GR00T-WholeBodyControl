"""Record all three original SONIC clips with legacy and C++ command parameters.

Offline only. This is a source-motion comparison, not native23 training,
deployment, complete C++ equivalence, or authorization to operate a robot.
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
from gear_sonic.utils.g1_sonic_cpp_parameters import capture_cpp_parameters, file_sha256
from gear_sonic.utils.g1_sonic_original29_trace import trace_original29_motion
from gear_sonic.utils.g1_true23_contact_geometry import audit_reset_contacts
from gear_sonic.utils.g1_true23_reference_floor import compiled_model_sha256
from gear_sonic.utils.g1_true23_reference_lineage import root_path_repair_bounds

CLIPS = ("hand_crawling", "elbow_crawling", "happy_dance")
ENCODER_SHA = "013ab0287236aa2721e13f1e936d699db982302d0de0bfcdae76d5c3245362d3"
DECODER_SHA = "c7241a123eaa36b5d64bad19540efde93cac1ad443bd4572fd12ca99898118ed"


def dump(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def validate_source_report(report):
    if report.get("kind") != "g1_released_sonic_planner_motion_suite" or report.get("fps") != 30:
        raise ValueError("requires the original 30-Hz SONIC planner report")
    records = report.get("records", [])
    if len({row["name"] for row in records}) != len(records):
        raise ValueError("duplicate source clip names")
    selected = {row["name"]: row for row in records if row["name"] in CLIPS}
    if set(selected) != set(CLIPS):
        raise ValueError("all three requested original clips must remain present")
    for name, row in selected.items():
        if row["mode"] != legacy.PLANNER_MODES[name] or row["npz"] != f"{name}.npz":
            raise ValueError("source motion mode or file identity drift")
    return selected


def trace_metrics(model, arrays):
    count = len(arrays["qpos"])
    source = arrays["planned_qpos50"][:count]
    contacts = audit_reset_contacts(model, arrays["pre_qpos"])
    rows = contacts["rows"]
    depths = np.asarray([max(0.0, -(row["minimum_floor_contact_distance_m"] or 0.0)) for row in rows])
    velocity = arrays["physics_post_qvel"][:, 6:]
    requested = arrays["physics_requested_torque_hardware29"]
    actual = arrays["physics_actual_actuated_generalized_force"]
    return {
        "pre_state_nominal_root_comparison": root_path_repair_bounds(
            source[:, :3], arrays["pre_qpos"][:, :3], maximum_offset_m=0
        ),
        "post_state_nominal_root_comparison_with_20ms_offset": root_path_repair_bounds(
            source[:, :3], arrays["qpos"][:, :3], maximum_offset_m=0
        ),
        "post_control_offset_not_hidden": True,
        "floor_overlap_pre_state_frames": int(np.sum(depths > 0)),
        "worst_floor_overlap_m": float(np.max(depths)),
        "worst_floor_overlap_frame": int(np.argmax(depths)),
        "maximum_physics_joint_velocity_rad_s": float(np.max(np.abs(velocity))),
        "maximum_requested_torque_Nm": float(np.max(np.abs(requested))),
        "maximum_actual_actuated_generalized_force_Nm": float(np.max(np.abs(actual))),
        "maximum_requested_vs_actual_actuated_force_difference_Nm": float(np.max(np.abs(requested - actual))),
        "native23_physical_limits_qualified": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    assets = args.asset_root.resolve(strict=True)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    inputs = {}

    def bind(path, expected=None):
        path = Path(path).resolve(strict=True)
        actual = file_sha256(path)
        if (expected is not None and actual != expected) or (str(path) in inputs and actual != inputs[str(path)]):
            raise ValueError(f"baseline source identity drift: {path}")
        inputs[str(path)] = actual
        return path

    planner_dir = assets / "artifacts/g1_true23/sonic_library_motion_suite_v2"
    source_report = json.loads(bind(planner_dir / "report.json").read_text())
    selected = validate_source_report(source_report)
    encoder_path = bind(assets / "gear_sonic_deploy/policy/release/model_encoder.onnx", ENCODER_SHA)
    decoder_path = bind(assets / "gear_sonic_deploy/policy/release/model_decoder.onnx", DECODER_SHA)
    scene = bind(root / "gear_sonic_deploy/g1/scene_29dof.xml")
    model = mujoco.MjModel.from_xml_path(str(scene))
    model.opt.timestep = 0.002
    model_hash = compiled_model_sha256(model)
    # The compiled MJB binds all included geometry, inertias, limits and options,
    # not just the scene wrapper. Retain it for exact offline reproduction.
    mujoco.mj_saveModel(model, filename=str(output / "original29.mjb"))
    bind(output / "original29.mjb", model_hash)
    parameters, capture = capture_cpp_parameters(
        root / "gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/policy_parameters.hpp", output / "cpp_capture"
    )
    for path, digest in capture["inputs"].items():
        bind(path, digest)
    dump(output / "cpp_capture/report.json", capture)
    bind(output / "cpp_capture/report.json")
    bind(root / "gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp")
    bind(Path(__file__))
    # All currently loaded local gear_sonic Python source dependencies, plus
    # the runtime versions, are bound before execution and rehashed afterward.
    for name, module in list(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if name.startswith("gear_sonic.") and path and Path(path).suffix == ".py":
            bind(path)
    motions = {}
    for name in CLIPS:
        row = selected[name]
        with np.load(bind(planner_dir / row["npz"], row["npz_sha256"]), allow_pickle=False) as archive:
            raw = {key: archive[key].copy() for key in archive.files}
        if float(raw["fps"][0]) != 30 or int(raw["mode"][0]) != row["mode"]:
            raise ValueError("raw planner metadata differs from its report")
        motions[name] = raw["qpos"]
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    encoder = ort.InferenceSession(str(encoder_path), sess_options=options, providers=["CPUExecutionProvider"])
    decoder = ort.InferenceSession(str(decoder_path), sess_options=options, providers=["CPUExecutionProvider"])
    legacy._validate_policy_abi(encoder, decoder)
    started = {
        "kind": "g1_sonic_original29_recorded_comparison_v1",
        "inputs": dict(inputs),
        "compiled_model_sha256": model_hash,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "mujoco": mujoco.__version__,
            "onnxruntime": ort.__version__,
        },
        "onnxruntime_providers": encoder.get_providers(),
        "onnxruntime_threads": 1,
        "source_report_passed": source_report["passed"],
        "source_report_is_not_physical_feasibility": True,
        "clips": list(CLIPS),
        "profiles": ["legacy_python", "cpp_parameters_and_float32_targets"],
        "original23_implementation_modified": False,
        "teacher_accepted": False,
        "hardware_authorized": False,
        "deployment_ready": False,
    }
    dump(output / "started.json", started)
    records = []
    for name, raw in motions.items():
        for profile_name, profile in (("legacy_python", None), ("cpp_parameters_and_float32_targets", parameters)):
            record = {"name": name, "profile": profile_name, "failure": None}
            try:
                arrays, details = trace_original29_motion(model, encoder, decoder, raw, parameters=profile)
                trace_path = output / f"{name}.{profile_name}.npz"
                with trace_path.open("xb") as stream:
                    np.savez_compressed(stream, **arrays)
                record.update(
                    details=details,
                    metrics=trace_metrics(model, arrays),
                    trace_path=str(trace_path),
                    trace_sha256=file_sha256(trace_path),
                )
                bind(trace_path)
            except Exception as exc:
                # Failed cases remain in the requested suite; never filter a
                # clip into a successful shorter baseline or accepted teacher.
                record["failure"] = f"{type(exc).__name__}: {exc}"
            dump(output / f"{name}.{profile_name}.report.json", record)
            records.append(record)
            print(
                json.dumps(
                    {
                        "name": name,
                        "profile": profile_name,
                        "failure": record["failure"],
                        "completed": record.get("details", {}).get("frames_completed"),
                    }
                ),
                flush=True,
            )
    for path, digest in inputs.items():
        if file_sha256(path) != digest:
            raise ValueError("baseline input or output changed during the run")
    if compiled_model_sha256(model) != model_hash:
        raise ValueError("recording or inspection changed the model")
    report = {
        **started,
        "inputs": inputs,
        "records": records,
        "all_requested_cases_recorded": len(records) == 6 and all(row["failure"] is None for row in records),
        "all_clips_completed": all(
            row.get("details", {}).get("complete_original_clip") is True for row in records
        ),
        "complete_cpp_deployment_equivalence_proven": False,
        "original29_policy_motion_parity_proven": False,
    }
    dump(output / "report.json", report)
    return 0 if report["all_requested_cases_recorded"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
