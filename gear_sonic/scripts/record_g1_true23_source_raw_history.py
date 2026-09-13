"""Four full CPU history-only counterfactuals; no training or hardware transport."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import traceback

import mujoco
import numpy as np
import torch

from gear_sonic.scripts.record_g1_true23_original_intent import OriginalIntentAdapter, task_metrics
from gear_sonic.utils.g1_true23_generalist_benchmark import MODEL, PHYSICS, run_reference_diagnostic
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file
from gear_sonic.utils.g1_true23_generalist_lifecycle import assess_lifecycle_diagnostic
from gear_sonic.utils.g1_true23_original29_reference import (
    build_original29_reference,
    verify_unmodified_native_pair,
)
from gear_sonic.utils.g1_true23_range_preview import Native23RangePreview
from gear_sonic.utils.g1_true23_root_feedback_campaign import validate_buffered_replay_arrays
from gear_sonic.utils.g1_true23_source_raw_history import (
    replace_previous_action_history,
    source_raw_history_contract,
)
from gear_sonic.utils.g1_true23_world_tracking_checkpoint import load_cpu_actor

ROOT = Path("/mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof")
ASSETS = Path("/mnt/z/codex/GR00T-WholeBodyControl")
BASE = Path("/mnt/c/Users/camer/sonic23_sim_artifacts/internet_pico_20260909_v1")
WORLD = BASE / "world_tracking_termination_v1/regression_recovery_v1"
CHECKPOINT = WORLD / "train/checkpoints/world_tracking_model_100.pt"
CHECKPOINT_SHA = "5ba74a79c9b25fbb75eff797b77ab2bf2a362d53afa94020ef3432e060faad22"
FLAGS = dict(hardware_authorized=False, deployment_ready=False, candidate_promoted=False)


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def save_arrays(path, arrays):
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def actor_digest(policy):
    digest = hashlib.sha256()
    for name, value in sorted(policy.actor.state_dict().items()):
        digest.update(name.encode())
        digest.update(str((value.dtype, tuple(value.shape))).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class SourceRawHistoryAdapter(OriginalIntentAdapter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.preview = Native23RangePreview(model_path=ASSETS / MODEL, physics_path=ROOT / PHYSICS)
        self.original_inverse_actions = []
        self.effective_histories = []
        self.raw_histories = []

    def transform_history(self, history):
        if len(self.effective_histories) != len(self.model_outputs):
            raise ValueError("history transform requires exactly one completed previous inference")
        effective = super().transform_history(history)
        previous = np.asarray(self.model_outputs[-10:], dtype=np.float32).reshape(-1, 23)
        raw = replace_previous_action_history(effective, previous)
        self.effective_histories.append(effective.copy())
        self.raw_histories.append(raw.copy())
        return raw

    def infer(self, *args, **kwargs):
        raw, decoder = super().infer(*args, **kwargs)
        self.original_inverse_actions.append(raw.copy())
        return self.preview.filter(raw, kwargs["measured_qpos"], kwargs["measured_qvel"]), decoder

    def contract(self):
        result = super().contract()
        result.update(
            kind="native23_raw_history_guarded_CPU_counterfactual_v1",
            actual_training_contract_applied_without_change=False,
            previous_action_counterfactual=source_raw_history_contract(),
            joint_range_preview=self.preview.contract(),
            nominal_actor_only_replay_claimed=False,
        )
        # Keep original codec as the TARGET interface, but explicitly override
        # its historical history description for this distinct runtime.
        result["source_action_codec"] = {
            "target_codec": result["source_action_codec"],
            "consumed_previous_action": source_raw_history_contract(),
            "original_combined_codec_contract_claimed": False,
        }
        return result


def run_case(name, output, policy, identity, semantics, experiment):
    prior_dir = BASE / "range_preview_v1/actual_v1" / name
    inputs = copy.deepcopy(semantics["reverified_repository_sources"])

    def bind(path, expected=None):
        path = Path(path).resolve(strict=True)
        digest = sha256_file(path)
        if expected is not None and digest != expected:
            raise ValueError("counterfactual input differs: " + str(path))
        inputs[str(path)] = digest
        return path

    def read_arrays(path, expected=None):
        with np.load(bind(path, expected), allow_pickle=False) as archive:
            return {key: archive[key].copy() for key in archive.files}

    prior = json.loads(bind(prior_dir / "report.json").read_text())
    matched_command = json.loads(bind(WORLD / f"cpu100_{name}.command.json").read_text())["command"]
    reference_path = Path(matched_command[matched_command.index("--original-reference") + 1])
    timeline = prior["timeline"]
    motion_path = bind(timeline["timeline_path"], timeline["timeline_sha256"])
    motion = read_arrays(motion_path)
    saved = read_arrays(reference_path)
    geometry_path = bind(ASSETS / "gear_sonic/data/robots/g1/g1_29dof.xml")
    native_path = bind(ASSETS / MODEL)
    bind(ROOT / PHYSICS)
    bind(CHECKPOINT, CHECKPOINT_SHA)
    bind(experiment)
    bind(__file__)
    bind(ASSETS / "gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp")
    geometry = mujoco.MjModel.from_xml_path(str(geometry_path))
    model = mujoco.MjModel.from_xml_path(str(native_path))
    reference = build_original29_reference(geometry, saved["source_qpos29"])
    for key, value in reference.arrays().items():
        np.testing.assert_array_equal(saved[key], value)
    pair = verify_unmodified_native_pair(reference, motion)
    adapter = SourceRawHistoryAdapter(motion, reference.virtual_vr21)
    old_received = read_arrays(prior_dir / "nominal.received_source.npz")
    for key, value in adapter.source.items():
        np.testing.assert_array_equal(value, old_received[key])
    for module_name, module in list(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if module_name.startswith("gear_sonic.") and path and Path(path).suffix == ".py":
            bind(path)
    output.mkdir(exist_ok=False)
    write_json(output / "started.json", dict(inputs=inputs, checkpoint_identity=identity, **FLAGS))
    result, trace = run_reference_diagnostic(
        root=ROOT, asset_root=ASSETS, motion_path=motion_path, policy=policy, runtime_adapter=adapter
    )
    n = result["completed_controls"]
    attempted = dict(
        **adapter.root_adapter.arrays(),
        actual_policy_encoder267=np.asarray(adapter.actual_encoder_inputs),
        released_model_raw23=np.asarray(adapter.model_outputs).reshape(-1, 23),
        bounded_linear_projection_delta_rad=np.asarray(adapter.projection_delta_rad).reshape(-1, 23),
        source_emission_anchor_setpoint_timestamps_s=np.asarray(adapter.timestamps).reshape(-1, 3),
        preview_original_inverse_action23=np.asarray(adapter.original_inverse_actions).reshape(-1, 23),
        effective_source_history930=np.asarray(adapter.effective_histories).reshape(-1, 930),
        consumed_raw_history930=np.asarray(adapter.raw_histories).reshape(-1, 930),
    )
    # Persist ALL attempted rows separately, including any unapplied final
    # proposal. Only the first n rows accompany n physically executed controls.
    attempted_path = output / "attempted_inference.npz"
    save_arrays(attempted_path, attempted)
    trace.update({key: value[:n] for key, value in attempted.items()})
    trace.update(
        preview_accepted_post_qpos=np.asarray(adapter.preview.predictions[:n]),
        preview_intervened=np.asarray([r["intervened"] for r in adapter.preview.records[:n]]),
        preview_elapsed_s=np.asarray([r["elapsed_s"] for r in adapter.preview.records[:n]]),
        preview_calls=np.asarray([r["preview_calls"] for r in adapter.preview.records[:n]]),
    )
    # Save measured physics BEFORE postprocessing; a failed audit cannot erase it.
    trace_path = output / "nominal.npz"
    source_path = output / "nominal.received_source.npz"
    save_arrays(trace_path, trace)
    save_arrays(source_path, adapter.source)
    write_json(output / "integration_result.json", result)
    for key in (
        "compiled_model_sha256",
        "physics_config_sha256",
        "initial_state_and_history_sha256",
        "kp_hardware",
        "kd_hardware",
        "effort_limit_hardware_nm",
        "requested_controls",
    ):
        if result[key] != prior["records"][0]["result"][key]:
            raise ValueError("counterfactual altered referee setup: " + key)
    if result["compiled_model_sha256"] != adapter.preview.model_sha256:
        raise ValueError("preview and actual model differ")
    input_audit = validate_buffered_replay_arrays(adapter.source, trace) if n else {"controls_verified": 0}
    lifecycle = assess_lifecycle_diagnostic(timeline, result, trace)
    phase = next(p for p in timeline["phases"] if p["name"] == "source_motion")
    metrics = task_metrics(model, geometry, reference, motion, trace, phase)
    np.testing.assert_array_equal(trace["history930"], trace["consumed_raw_history930"])
    report = dict(
        kind="native23_source_raw_history_full_lifecycle_counterfactual_v1",
        records=[
            dict(
                case="nominal",
                policy_identity=identity,
                result=result,
                lifecycle=lifecycle,
                trace_path=str(trace_path),
                trace_sha256=sha256_file(trace_path),
            )
        ],
        timeline=timeline,
        original_task_metrics=metrics,
        pairing=pair,
        inputs=inputs,
        received_path=str(source_path),
        received_sha256=sha256_file(source_path),
        input_audit=input_audit,
        attempted_inference_path=str(attempted_path),
        attempted_inference_sha256=sha256_file(attempted_path),
        attempted_array_rows={key: len(value) for key, value in attempted.items()},
        unapplied_attempts_preserved_separately=True,
        runtime_counterfactual=source_raw_history_contract(),
        original_reference=str(reference_path),
        root_measurement_is_privileged_simulator_state=True,
        no_motion_retiming_or_trimming=True,
        **FLAGS,
    )
    for path, digest in inputs.items():
        if sha256_file(Path(path)) != digest:
            raise ValueError("counterfactual source changed during execution: " + path)
    write_json(output / "report.json", report)
    return dict(
        name=name,
        report=str(output / "report.json"),
        report_sha256=sha256_file(output / "report.json"),
        completed=n,
        requested=result["requested_controls"],
        failure=result["failure"],
        metrics=metrics,
        lifecycle=lifecycle,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    args = parser.parse_args()
    if args.output_directory.exists() or args.output_directory.is_symlink():
        raise FileExistsError("counterfactual refuses overwrite")
    experiment = args.experiment.resolve(strict=True)
    if sha256_file(CHECKPOINT) != CHECKPOINT_SHA:
        raise ValueError("counterfactual requires the audited rejected world100 checkpoint")
    proof = WORLD / "cpu_audit.json"
    if sha256_file(proof) != "66ce09ff9433b1b3381d2bb272dc312914617b078e8009aba00e9ee9872a4d2d":
        raise ValueError("matched CPU baseline audit changed")
    args.output_directory.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output_directory / "launch.json",
        dict(
            argv=sys.argv,
            checkpoint_sha256=CHECKPOINT_SHA,
            baseline_audit_sha256=sha256_file(proof),
            experiment_sha256=sha256_file(experiment),
            script_sha256=sha256_file(Path(__file__)),
            **FLAGS,
        ),
    )
    torch.set_num_threads(1)
    print("Loading one strict actor for four sequential independent lifecycles", flush=True)
    policy, identity, semantics = load_cpu_actor(
        CHECKPOINT,
        warm_start_path=ASSETS / "sonic_release/g1_23dof_rev_1_0_low_latency_init.pt",
        source_checkpoint_path=ASSETS / "low_latency/last.pt",
    )
    before = actor_digest(policy)
    cases = []
    for name in ("walk002", "walk003", "walk008", "dance"):
        print(json.dumps(dict(starting=name)), flush=True)
        try:
            row = run_case(name, args.output_directory / name, policy, identity, semantics, experiment)
        except Exception as error:
            row = dict(
                name=name, postprocessing_or_setup_error=dict(type=type(error).__name__, message=str(error))
            )
            traceback.print_exc()
        cases.append(row)
        write_json(args.output_directory / f"{name}.case.json", row)
        print(json.dumps(row, allow_nan=False), flush=True)
    if actor_digest(policy) != before:
        raise ValueError("counterfactual mutated actor state")
    write_json(
        args.output_directory / "trials.json",
        dict(
            kind="native23_source_raw_history_four_motion_counterfactual_v1",
            cases=cases,
            actor_state_sha256_before_and_after=before,
            checkpoint_identity=identity,
            runtime_counterfactual=source_raw_history_contract(),
            additional_training_updates=0,
            simulator_qualified=False,
            independent_audit_pending=True,
            **FLAGS,
        ),
    )
    print(json.dumps(dict(complete=True, output=str(args.output_directory / "trials.json"))), flush=True)


if __name__ == "__main__":
    main()
