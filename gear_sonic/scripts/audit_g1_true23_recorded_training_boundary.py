"""Audit temporal training mappings against actual recorded full-request inputs.

Uses every recorded call, not only reset snapshots. Unexecuted clip tails and
the unavailable elbow request remain explicitly unqualified. No optimization,
physics stepping, DDS, hardware connection or deployment mutation occurs.
"""

import argparse
from dataclasses import replace
import importlib.metadata
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
import onnxruntime as ort

from gear_sonic.scripts.record_g1_sonic_original29_baseline import dump
from gear_sonic.utils.g1_sonic_cpp_parameters import file_sha256
from gear_sonic.utils.g1_true23_actuation_profile import SIM_CONFIG, NativeSupportActuationProfile
from gear_sonic.utils.g1_true23_recorded_training_boundary import (
    comparison,
    recorded_control_states,
    training_actuation,
    training_observations,
)
from gear_sonic.utils.g1_true23_reference_floor import compiled_model_sha256
from gear_sonic.utils.g1_true23_sim_acquisition import align_reference_xy_yaw
from gear_sonic.utils.g1_true23_training_precision import ieee_training_precision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-report", type=Path, required=True, action="append")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError("recorded boundary audit requires a fresh output directory")
    root = Path(__file__).resolve().parents[2]
    inputs = {}

    def bind(path, expected=None):
        path = Path(path).resolve(strict=True)
        digest = file_sha256(path)
        if digest != inputs.get(str(path), digest) or (expected is not None and digest != expected):
            raise ValueError(f"recorded boundary input changed: {path}")
        inputs[str(path)] = digest
        return path

    profile = replace(
        NativeSupportActuationProfile.from_sim_config(bind(root / SIM_CONFIG)), consistent_controller_state=True
    )
    model = mujoco.MjModel.from_binary_path(str(bind(args.model_path)))
    supplied_model_hash = compiled_model_sha256(model)
    # The saved earlier parity model has unset actuator force ranges; the
    # current envelope fills both tables from this exact profile before its
    # compiled-model hash. Apply that same metadata on this in-memory copy.
    # No model file, geometry, force limit or simulation state is changed.
    model.actuator_forcerange[:, 0] = -np.asarray(profile.effort)
    model.actuator_forcerange[:, 1] = profile.effort
    model.jnt_actfrcrange[1:, 0] = -np.asarray(profile.effort)
    model.jnt_actfrcrange[1:, 1] = profile.effort
    model_hash = compiled_model_sha256(model)
    body_names = [model.body(i).name for i in range(1, model.nbody)]
    if (model.nq, model.nv, model.nu) != (30, 29, 23):
        raise ValueError("boundary audit requires the native23 compiled model")
    output.mkdir(parents=True, exist_ok=False)
    bind(Path(__file__))
    flags = dict(hardware_authorized=False, deployment_ready=False, promotion_eligible=False)
    records = []
    with ieee_training_precision() as (precision, guard):
        for report_index, report_path in enumerate(args.evaluation_report):
            report_path = bind(report_path)
            source = json.loads(report_path.read_text())
            if (
                source.get("kind") != "g1_true23_motion_ppo_full_request_evaluation_v1"
                or source.get("original_eight_request_set_preserved") is not True
                or len(source.get("records", [])) != 11
            ):
                raise ValueError("boundary witness must preserve the original full-request evaluation")
            pair = source["pair"]
            options = ort.SessionOptions()
            options.intra_op_num_threads = options.inter_op_num_threads = 1
            sessions = {
                name: ort.InferenceSession(
                    str(bind(pair[name]["path"], pair[name]["sha256"])),
                    sess_options=options,
                    providers=["CPUExecutionProvider"],
                )
                for name in ("encoder", "decoder")
            }
            inputs_names = {name: session.get_inputs()[0].name for name, session in sessions.items()}
            for case in source["records"]:
                identity = dict(
                    evaluation_report=str(report_path), label=case["label"], source=case["source"], **flags
                )
                if case["unavailable"]:
                    records.append({**identity, "not_executed": True, "reason": case["reason"]})
                    continue
                result = case["result"]
                if result["compiled_native_model_sha256"] != model_hash:
                    raise ValueError("recorded motion body layout model does not match evaluator")
                trace_path = bind(case["trace_path"], source["inputs"][case["trace_path"]])
                with np.load(trace_path, allow_pickle=False) as archive:
                    arrays = {key: archive[key].copy() for key in archive.files}
                if not arrays["policy_inference_returned"].all():
                    raise ValueError("this witness requires actual returned actions for all recorded calls")
                motion_path = bind(case["source"], source["inputs"][case["source"]])
                with np.load(motion_path, allow_pickle=False) as archive:
                    motion = {key: archive[key].copy() for key in archive.files}
                states = recorded_control_states(arrays, result)
                if case["lifecycle"]:
                    motion, alignment = align_reference_xy_yaw(motion, states["qpos"][0])
                    if alignment != result["reference_alignment"]:
                        raise ValueError("one-time reference alignment differs from the actual recorded evaluator")
                devices, saved = {}, {}
                for device in ("cpu", "cuda:0"):
                    guard()
                    reconstructed = training_observations(motion, states, body_names, device=device)
                    actuation = training_actuation(arrays, result, profile, device=device)
                    guard()
                    semantic, history = reconstructed["encoder267"], reconstructed["history930"]
                    tokens, raw = [], []
                    for enc_row, history_row in zip(semantic, history):
                        token = (
                            sessions["encoder"].run(None, {inputs_names["encoder"]: enc_row[None]})[0].reshape(64)
                        )
                        decoded = (
                            sessions["decoder"]
                            .run(None, {inputs_names["decoder"]: np.concatenate((token, history_row))[None]})[0]
                            .reshape(23)
                        )
                        tokens.append(token)
                        raw.append(decoded)
                    from gear_sonic.utils.g1_23dof_safe_target_transform import safe_target_transform_numpy

                    target = np.asarray([safe_target_transform_numpy(row)[1] for row in raw])
                    captured_target = np.asarray(
                        [safe_target_transform_numpy(row)[1] for row in arrays["policy_raw23"]]
                    )
                    n = len(arrays["actuation_q"])
                    devices[device] = dict(
                        encoder267=comparison(semantic, arrays["policy_encoder267"], 1e-5),
                        encoder_terms={
                            name: comparison(semantic[:, lo:hi], arrays["policy_encoder267"][:, lo:hi], 1e-5)
                            for name, lo, hi in (
                                ("lower_body", 0, 240),
                                ("vr_position", 240, 249),
                                ("vr_orientation", 249, 261),
                                ("anchor_orientation", 261, 267),
                            )
                        },
                        history930=comparison(history, arrays["policy_history930"], 1e-5),
                        history_terms={
                            name: comparison(history[:, lo:hi], arrays["policy_history930"][:, lo:hi], 1e-5)
                            for name, lo, hi in (
                                ("gyro", 0, 30),
                                ("joint_pos_rel", 30, 320),
                                ("joint_vel", 320, 610),
                                ("previous_action", 610, 900),
                                ("gravity", 900, 930),
                            )
                        },
                        fsq64=comparison(np.asarray(tokens), arrays["policy_decoder994"][:, :64], 0),
                        raw23=comparison(np.asarray(raw), arrays["policy_raw23"], 1e-4),
                        safe_target=comparison(target, captured_target, 1e-5),
                        requested_target=comparison(
                            actuation["requested_per_control"][np.arange(n) // 10],
                            arrays["actuation_requested"],
                            2e-6,
                        ),
                        projected_target=comparison(actuation["target"], arrays["actuation_target"], 2e-6),
                        applied_effort=comparison(actuation["effort"], arrays["actuation_effort"], 2e-5),
                        invalid_successful_substeps=np.flatnonzero(actuation["invalid"]).tolist(),
                        terminal=actuation["terminal"],
                    )
                    prefix = device.replace(":", "_") + "_"
                    saved.update({prefix + key: value for key, value in reconstructed.items()})
                    saved.update(
                        {prefix + key: value for key, value in actuation.items() if isinstance(value, np.ndarray)}
                    )
                    saved[prefix + "fsq64"] = np.asarray(tokens)
                    saved[prefix + "raw23"] = np.asarray(raw)
                name = f"evaluation{report_index}.{case['label']}"
                array_path = output / (name + ".npz")
                with array_path.open("xb") as stream:
                    np.savez_compressed(stream, **saved)
                bind(array_path)
                row = {
                    **identity,
                    "recorded_trace": str(trace_path),
                    "witness_arrays": str(array_path),
                    "recorded_inference_calls": len(states["qpos"]),
                    "requested_transitions": result["requested_transitions"],
                    "completed_transitions": result["completed_transitions"],
                    "completed_active_substeps": len(arrays["actuation_q"]),
                    "stationary_prerequisite_only": case["stationary_prerequisite_only"],
                    "startup_target_reconstruction_rows": states["startup_target_reconstruction_rows"],
                    "devices": devices,
                }
                dump(output / (name + ".json"), row)
                bind(output / (name + ".json"))
                records.append(row)
                print(
                    json.dumps(
                        dict(
                            case=name,
                            calls=len(states["qpos"]),
                            cpu_fsq_differences=devices["cpu"]["fsq64"]["changed_elements"],
                            cuda_fsq_differences=devices["cuda:0"]["fsq64"]["changed_elements"],
                            history_max=devices["cuda:0"]["history930"]["maximum_absolute_difference"],
                            invalid_successful_substeps=len(devices["cuda:0"]["invalid_successful_substeps"]),
                        )
                    ),
                    flush=True,
                )
            del sessions
    for module_name, module in tuple(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if module_name.startswith(("gear_sonic.", "mjlab.")) and path and Path(path).suffix == ".py":
            bind(path)
    for path in list(inputs):
        bind(path)
    dump(
        output / "report.json",
        dict(
            kind="g1_true23_recorded_training_boundary_witness_v1",
            inputs=inputs,
            records=records,
            precision=precision,
            compiled_native_model_sha256=model_hash,
            supplied_model_sha256=supplied_model_hash,
            in_memory_force_range_metadata_matches_exact_recorded_evaluator=True,
            source_model_and_weights_unchanged=True,
            every_saved_inference_and_successful_active_substep_used=True,
            unexecuted_clip_tails_not_qualified=True,
            noise_and_reset_randomization_not_emulated=True,
            recorded_body_local_velocity_supplied_not_a_sensor_simulation=True,
            motion_time_steps_supplied_not_automatic_command_manager_parity=True,
            actual_training_observation_functions_entity_gravity_and_circular_buffer_executed=True,
            actual_training_pd_projection_function_executed=True,
            complete_training_environment_or_physics_executed=False,
            physical_damping_cause_established=False,
            packages={
                name: importlib.metadata.version(name)
                for name in ("torch", "numpy", "mujoco", "mjlab", "onnxruntime")
            },
            **flags,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
