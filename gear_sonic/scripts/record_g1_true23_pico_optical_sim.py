"""SIM-only full PICO optical-reference trial; preserve guard-failed attempts.

Reuses the existing audited policy, native plant, range guard and trace referees.
Neither a prior baseline report nor a successful source outcome is fabricated.
"""

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

import mujoco
import numpy as np
import torch

from gear_sonic.scripts.prepare_g1_true23_pico_optical_reference import save, write
from gear_sonic.scripts.record_g1_true23_original_intent import task_metrics
from gear_sonic.scripts.record_g1_true23_world_quality_lifecycle import verified_update_checkpoint
from gear_sonic.utils.g1_23dof_safe_target_transform import safe_target_transform_numpy
from gear_sonic.utils.g1_true23_generalist_benchmark import MODEL, PHYSICS, run_reference_diagnostic
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file
from gear_sonic.utils.g1_true23_generalist_lifecycle import assess_lifecycle_diagnostic
from gear_sonic.utils.g1_true23_original29_reference import build_original29_reference, verify_unmodified_native_pair
from gear_sonic.utils.g1_true23_root_feedback_campaign import validate_buffered_replay_arrays
from gear_sonic.utils.g1_true23_world_quality_checkpoint import load_cpu_actor

FLAGS = dict(hardware_authorized=False, deployment_ready=False, simulator_qualified=False,
             candidate_promoted=False, sensor_only_pico_reconstruction_tested=False)


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, inputs = Path(__file__).resolve().parents[2], {}
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("SIM trial refuses overwrite")

    def bind(path, expected=None):
        path = Path(path).resolve(strict=True)
        actual = sha256_file(path)
        if expected is not None and actual != expected:
            raise ValueError("changed trial input: " + str(path))
        if str(path) in inputs and inputs[str(path)] != actual:
            raise ValueError("input changed during load")
        inputs[str(path)] = actual
        return path

    def read(path, expected=None):
        return json.loads(bind(path, expected).read_text())

    def arrays(path, expected=None):
        with np.load(bind(path, expected), allow_pickle=False) as archive:
            return {key: archive[key].copy() for key in archive.files}

    preparation = read(args.reference / "report.json")
    if preparation["kind"] != "pico_freedancing_optical_reference_preparation_v1":
        raise ValueError("not this explicit optical-reference preparation")
    for path, expected in preparation["inputs"].items():
        bind(path, expected)
    for name, expected in preparation["outputs"].items():
        bind(args.reference / name, expected)
    timeline = preparation["timeline"]
    motion_path = bind(timeline["timeline_path"], timeline["timeline_sha256"])
    motion = arrays(motion_path)
    saved = arrays(args.reference / "original29.npz")
    native_path = bind(args.asset_root / MODEL)
    geometry_path = bind(args.asset_root / "gear_sonic/data/robots/g1/g1_29dof.xml")
    bind(root / PHYSICS)
    native = mujoco.MjModel.from_xml_path(str(native_path))
    geometry = mujoco.MjModel.from_xml_path(str(geometry_path))
    reference = build_original29_reference(geometry, saved["source_qpos29"])
    for key, value in reference.arrays().items():
        np.testing.assert_array_equal(value, saved[key])
    pairing = verify_unmodified_native_pair(reference, motion)
    guard_proof = read(args.base / "range_preview_v1/actual_v1/audit.json",
                       "e15c2fcc681249c4c938cee552878f6ce265d4374a9ab00c82b4f87a381e17c9")
    guard_path = args.base / "range_preview_v1/record.py"
    guard = import_file("pico_unchanged_range_wrapper", bind(guard_path, guard_proof["inputs"][str(guard_path)]))
    qrun = args.base / "world_quality_bonus_v1/regression_v1"
    update = read(qrun / "update_audit.json",
                  "4832a1ae70128bf1a838b8096bd2e710ba0fd59000b36444f7b34fcdc83a2342")
    path, digest = verified_update_checkpoint(qrun, update)
    bind(path, digest)
    # These helpers independently rerun recorded singleton inference and actual
    # fresh-state Euler physics. They do not invoke their original campaign mains.
    prior_audit = read(qrun / "cpu_audit.json",
                       "bc329ad913f577c3939e0128d775c6f9121ca36b5d74bc5444a7ea4ff4c48804")
    network_path = args.base / "world_quality_bonus_v1/audit_cpu.py"
    network = import_file("pico_existing_network_referee", bind(network_path, prior_audit["inputs"][str(network_path)]))
    physics_path = args.base / "saved_failure_localization_v1/run.py"
    physics = import_file("pico_existing_physics_referee", bind(physics_path, prior_audit["inputs"][str(physics_path)]))
    torch.set_num_threads(1)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    print(json.dumps(dict(stage="loading_audited_checkpoint", requested_controls=timeline["total_requested_controls"])), flush=True)
    policy, identity, semantics = load_cpu_actor(path,
        warm_start_path=bind(args.asset_root / "sonic_release/g1_23dof_rev_1_0_low_latency_init.pt"),
        source_checkpoint_path=bind(args.asset_root / "low_latency/last.pt"))
    for path, expected in semantics["reverified_repository_sources"].items():
        bind(path, expected)
    for name, module in list(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if name.startswith("gear_sonic.") and path and Path(path).suffix == ".py":
            bind(path)
    bind(__file__)
    write(output / "started.json", dict(inputs=inputs, identity=identity, full_lifecycle_requested=True, **FLAGS))

    class ReportingAdapter(guard.RangePreviewAdapter):
        def infer(self, *values, **state):
            result = super().infer(*values, **state)
            index = state["control_index"]
            if index % 500 == 0:
                print(json.dumps(dict(stage="actual_cpu_physics", control=index,
                                      elapsed_s=round(time.monotonic() - started, 1))), flush=True)
            return result

    adapter = ReportingAdapter(motion, reference.virtual_vr21)
    result, trace = run_reference_diagnostic(root=root, asset_root=args.asset_root, motion_path=motion_path,
                                              policy=policy, runtime_adapter=adapter)
    n = result["completed_controls"]
    attempted = dict(adapter.root_adapter.arrays())
    attempted.update(actual_policy_encoder267=np.asarray(adapter.actual_encoder_inputs),
        released_model_raw23=np.asarray(adapter.model_outputs),
        bounded_linear_projection_delta_rad=np.asarray(adapter.projection_delta_rad).reshape(-1, 23),
        source_emission_anchor_setpoint_timestamps_s=np.asarray(adapter.timestamps),
        preview_original_inverse_action23=np.asarray(adapter.original_inverse_actions),
        preview_accepted_post_qpos=np.asarray(adapter.preview.predictions),
        preview_intervened=np.asarray([r["intervened"] for r in adapter.preview.records]),
        preview_elapsed_s=np.asarray([r["elapsed_s"] for r in adapter.preview.records]),
        preview_calls=np.asarray([r["preview_calls"] for r in adapter.preview.records]))
    # Save original physical and attempted arrays before any evaluation. An extra
    # failed inference row must neither erase evidence nor count as integration.
    save(output / "physics.npz", **trace)
    save(output / "adapter_attempts.npz", **attempted)
    save(output / "received_source.npz", **adapter.source)
    write(output / "result.json", result)
    for key, value in attempted.items():
        if len(value) < n:
            raise ValueError("missing completed-control adapter telemetry: " + key)
        trace[key] = value[:n].copy()
    save(output / "completed_controls.npz", **trace)
    lifecycle = assess_lifecycle_diagnostic(timeline, result, trace)
    phase = next(p for p in timeline["phases"] if p["name"] == "source_motion")
    metrics = task_metrics(native, geometry, reference, motion, trace, phase)
    row = dict(result=result, lifecycle=lifecycle, policy_identity=identity,
               trace_path=str(output / "completed_controls.npz"), trace_sha256=sha256_file(output / "completed_controls.npz"))
    report = dict(kind="native23_pico_optical_reference_actual_cpu_trial_v1", records=[row], timeline=timeline,
        original_task_metrics=metrics, pairing=pairing, inputs=inputs,
        complete_recording_requested=True, completed_control_prefix_only=bool(result["failure"]),
        all_attempted_arrays_preserved=True, attempted_inference_rows=len(adapter.actual_encoder_inputs),
        actual_integrated_substeps=len(trace["physics_time"]), root_state_is_privileged_simulator_measurement=True,
        nominal_50hz_source_timestamps_not_real_capture_jitter=True, actual_sensor_stream_not_used_for_reference=True,
        range_guard_intervention_controls=np.flatnonzero(attempted["preview_intervened"]).tolist(),
        maximum_target_jump_rad=float(np.abs(np.diff(trace["target23"], axis=0)).max()) if n > 1 else None,
        **FLAGS)
    write(output / "report.json", report)
    print(json.dumps(dict(stage="rollout_finished", completed=n, requested=result["requested_controls"],
                          failure=result["failure"], tracking=lifecycle["source_motion_tracking"])), flush=True)
    if n:
        received_audit = validate_buffered_replay_arrays(adapter.source, trace)
        network_audit = network.verify_network(policy, trace)
        physics_audit, _ = physics.replay_physics(report, trace, [])
        _, decoded = safe_target_transform_numpy(trace["raw23"])
        np.testing.assert_array_equal(decoded, trace["target23"])
        predicted = trace["preview_accepted_post_qpos"].reshape(-1, 30)
        # Only completed control predictions participate; the raw physics file
        # still contains every substep if a future failure occurs inside control.
        preview_error = float(np.abs(predicted - trace["physics_post_qpos"][:10*n]).max())
        if preview_error > 2e-10:
            raise ValueError("preview prediction differs from real integrated plant")
        for path, expected in inputs.items():
            if sha256_file(Path(path)) != expected:
                raise ValueError("trial input changed during run: " + path)
        write(output / "audit.json", dict(received_inputs=received_audit, network=network_audit,
            physics=physics_audit, completed_controls_audited=n, maximum_preview_qpos_difference=preview_error,
            full_lifecycle_completed=result["failure"] is None and n == result["requested_controls"],
            failed_prefix_never_promoted=True, inputs=inputs,
            artifacts={p.name: sha256_file(p) for p in output.iterdir() if p.is_file()}, **FLAGS))
    print(json.dumps(dict(finished=True, output=str(output), elapsed_s=round(time.monotonic()-started, 1))), flush=True)


if __name__ == "__main__":
    main()
