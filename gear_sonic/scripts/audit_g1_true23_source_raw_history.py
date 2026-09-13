"""Independent history/network/physics audit of the four raw-history trials."""

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
import torch

from gear_sonic.scripts.record_g1_true23_original_intent import task_metrics
from gear_sonic.scripts.record_g1_true23_source_raw_history import (
    ASSETS,
    BASE,
    CHECKPOINT,
    CHECKPOINT_SHA,
    ROOT,
    WORLD,
    actor_digest,
)
from gear_sonic.utils.g1_23dof_safe_target_transform import safe_target_transform_numpy
from gear_sonic.utils.g1_true23_clean_mujoco_teleop import CleanTrue23MujocoController
from gear_sonic.utils.g1_true23_generalist_benchmark import MODEL, PHYSICS
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file
from gear_sonic.utils.g1_true23_generalist_lifecycle import assess_lifecycle_diagnostic
from gear_sonic.utils.g1_true23_original29_reference import build_original29_reference
from gear_sonic.utils.g1_true23_range_preview import RANGE_RESERVE_RAD, Native23RangePreview
from gear_sonic.utils.g1_true23_root_feedback_campaign import validate_buffered_replay_arrays
from gear_sonic.utils.g1_true23_source_action_codec import source_scaled_precompensation
from gear_sonic.utils.g1_true23_source_raw_history import source_raw_history_contract
from gear_sonic.utils.g1_true23_source_raw_history_audit import audit_consumed_histories
from gear_sonic.utils.g1_true23_step1b_mujoco import term_major_history
from gear_sonic.utils.g1_true23_world_tracking_checkpoint import load_cpu_actor

PHASE = "unchanged_referee_post_control_q2"


def verify_measured_histories(trace):
    """Fresh FK-only observer; never writes the independently replayed plant."""
    observer = CleanTrue23MujocoController(model_path=ASSETS / MODEL, physics_path=ROOT / PHYSICS, policy=None)
    n = len(trace["target23"])
    safe, _ = safe_target_transform_numpy(trace["raw23"])
    history = []
    for i in range(n):
        observer.data.qpos[:] = trace["qpos"][i]
        observer.data.qvel[:] = trace["qvel"][i]
        observer.previous_safe_native[:] = safe[i - 1] if i else 0
        frame = observer._policy_frame()
        history = [frame.copy() for _ in range(10)] if not history else [*history[1:], frame]
        np.testing.assert_array_equal(term_major_history(history), trace["native_precodec_history930"][i])
    return dict(measured_physical_history_rows_verified=n, observer_only_no_physical_state_rewrites=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve(strict=True)
    output = run / "audit.json"
    if output.exists():
        raise FileExistsError("raw-history audit refuses overwrite")
    inputs = {}

    def bind(path, expected=None):
        path = Path(path).resolve(strict=True)
        digest = sha256_file(path)
        if expected is not None and digest != expected:
            raise ValueError("audit input changed: " + str(path))
        if str(path) in inputs and inputs[str(path)] != digest:
            raise ValueError("audit input changed during read: " + str(path))
        inputs[str(path)] = digest
        return path

    def read(path, expected=None):
        return json.loads(bind(path, expected).read_text())

    def arrays(path, expected=None):
        with np.load(bind(path, expected), allow_pickle=False) as archive:
            return {key: archive[key].copy() for key in archive.files}

    def import_bound(name, path, expected):
        spec = importlib.util.spec_from_file_location(name, bind(path, expected))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    baseline = read(WORLD / "cpu_audit.json", "66ce09ff9433b1b3381d2bb272dc312914617b078e8009aba00e9ee9872a4d2d")
    network_helper = import_bound(
        "raw_history_pinned_network_audit",
        BASE / "world_tracking_termination_v1/audit_cpu.py",
        "5c330e3a3a73c85f247a10b2f8aa8ba59f3bba1f4927dfeacb97025a844271c3",
    )
    proof = read(
        BASE / "saved_failure_localization_v1/actual_v1/summary.json",
        "b2418cb792af704163a641052fd7fcb8b6a9aeb078a75e6ace2e41ea5355e089",
    )
    path = BASE / "saved_failure_localization_v1/run.py"
    physics_helper = import_bound("raw_history_pinned_physics_replay", path, proof["inputs"][str(path)])
    trials = read(run / "trials.json")
    assert [row["name"] for row in trials["cases"]] == ["walk002", "walk003", "walk008", "dance"]
    assert trials["runtime_counterfactual"] == source_raw_history_contract()
    assert trials["additional_training_updates"] == 0
    assert not any(trials[key] for key in ("hardware_authorized", "deployment_ready", "candidate_promoted"))
    bind(CHECKPOINT, CHECKPOINT_SHA)
    bind(run / "launch.json")
    bind(run.parent / "EXPERIMENT.md")
    bind(run.parent / "V2_EXPERIMENT.md")
    torch.set_num_threads(1)
    print("Loading one strict actor for singleton audit after all recording processes ended", flush=True)
    policy, identity, semantics = load_cpu_actor(
        CHECKPOINT,
        warm_start_path=ASSETS / "sonic_release/g1_23dof_rev_1_0_low_latency_init.pt",
        source_checkpoint_path=ASSETS / "low_latency/last.pt",
    )
    assert identity == trials["checkpoint_identity"]
    before_actor = actor_digest(policy)
    assert before_actor == trials["actor_state_sha256_before_and_after"]
    for path, digest in semantics["reverified_repository_sources"].items():
        bind(path, digest)
    rows = []
    for trial in trials["cases"]:
        name = trial["name"]
        report = read(trial["report"], trial["report_sha256"])
        record = report["records"][0]
        result = record["result"]
        assert record["policy_identity"] == identity
        assert report["runtime_counterfactual"] == source_raw_history_contract()
        assert result["previous_action_semantics"] == source_raw_history_contract()["previous_action"]
        for path, digest in report["inputs"].items():
            bind(path, digest)
        trace = arrays(record["trace_path"], record["trace_sha256"])
        source = arrays(report["received_path"], report["received_sha256"])
        attempted = arrays(report["attempted_inference_path"], report["attempted_inference_sha256"])
        n = result["completed_controls"]
        assert len(trace["target23"]) == n and len(trace["physics_time"]) == 10 * n
        assert result["state_pose_writes_after_reset"] == 0 and result["history_resets_during_motion"] == 0
        for key, value in attempted.items():
            assert len(value) == report["attempted_array_rows"][key]
            np.testing.assert_array_equal(value[:n], trace[key])
        histories = audit_consumed_histories(trace)
        measured = verify_measured_histories(trace)
        received = validate_buffered_replay_arrays(source, trace)
        network = network_helper.verify_network(policy, trace)
        _, targets = safe_target_transform_numpy(trace["raw23"])
        np.testing.assert_array_equal(targets, trace["target23"])
        nominal_raw, projection = zip(
            *(source_scaled_precompensation(row) for row in trace["released_model_raw23"]), strict=True
        )
        np.testing.assert_array_equal(nominal_raw, trace["preview_original_inverse_action23"])
        np.testing.assert_array_equal(projection, trace["bounded_linear_projection_delta_rad"])
        mask = trace["preview_intervened"]
        np.testing.assert_array_equal(trace["raw23"][~mask], np.asarray(nominal_raw)[~mask])
        model = mujoco.MjModel.from_xml_path(str(ASSETS / MODEL))
        q = trace["physics_post_qpos"][:, 7:]
        margin = np.minimum(q - model.jnt_range[1:, 0], model.jnt_range[1:, 1] - q)
        excess = np.maximum(-margin, 0)
        np.testing.assert_array_equal(excess, trace["physics_hard_limit_excess23"])
        predicted = trace["preview_accepted_post_qpos"]
        assert predicted.shape == (n, 10, 30)
        predicted_margin = np.minimum(
            predicted[:, :, 7:] - model.jnt_range[1:, 0], model.jnt_range[1:, 1] - predicted[:, :, 7:]
        )
        assert predicted_margin.min() >= RANGE_RESERVE_RAD - 1e-12
        prediction_error = float(np.max(np.abs(predicted.reshape(-1, 30) - trace["physics_post_qpos"])))
        assert prediction_error <= 2e-10
        failure_reproduced = None
        if result["failure"] is not None:
            assert len(attempted["released_model_raw23"]) == n + 1
            raw, _ = policy.infer(
                attempted["actual_policy_encoder267"][n],
                attempted["consumed_raw_history930"][n],
                attempted["root_feedback9"][n],
            )
            np.testing.assert_array_equal(raw, attempted["released_model_raw23"][n])
            guard = Native23RangePreview(model_path=ASSETS / MODEL, physics_path=ROOT / PHYSICS)
            try:
                guard.filter(
                    attempted["preview_original_inverse_action23"][n], trace["qpos"][-1], trace["qvel"][-1]
                )
            except ValueError as error:
                assert str(error) == result["failure"]["message"]
                failure_reproduced = True
            else:
                raise ValueError("failed range proposal no longer reproduces")
        replay, _ = physics_helper.replay_physics(report, trace, [])
        original = arrays(report["original_reference"])
        geometry = mujoco.MjModel.from_xml_path(str(ASSETS / "gear_sonic/data/robots/g1/g1_29dof.xml"))
        reference = build_original29_reference(geometry, original["source_qpos29"])
        for key, value in reference.arrays().items():
            np.testing.assert_array_equal(value, original[key])
        motion = arrays(report["timeline"]["timeline_path"], report["timeline"]["timeline_sha256"])
        phase = next(row for row in report["timeline"]["phases"] if row["name"] == "source_motion")
        metrics = task_metrics(model, geometry, reference, motion, trace, phase)
        assert json.loads(json.dumps(metrics)) == report["original_task_metrics"]
        lifecycle = assess_lifecycle_diagnostic(report["timeline"], result, trace)
        assert json.loads(json.dumps(lifecycle)) == record["lifecycle"]
        old_row = next(row for row in baseline["cases"] if row["name"] == name)
        old_dir = WORLD / ("cpu100_" + name)
        if old_row["full_lifecycle_completed"]:
            old_report = read(old_dir / "report.json")
            old_record = old_report["records"][0]
            old_trace = arrays(old_record["trace_path"], old_record["trace_sha256"])
            old_result = old_record["result"]
        else:
            old_result = read(old_dir / "failed_physics.json")["result"]
            old_trace = arrays(old_dir / "failed_physics.npz")
        for key in (
            "compiled_model_sha256",
            "physics_config_sha256",
            "initial_state_and_history_sha256",
            "actuator_contract",
            "observation_phase_contract",
            "kp_hardware",
            "kd_hardware",
            "effort_limit_hardware_nm",
            "requested_controls",
        ):
            assert old_result[key] == result[key], key
        for key in ("qpos", "qvel", "history930", "target23"):
            np.testing.assert_array_equal(old_trace[key][0], trace[key][0])
        common = min(n, old_result["completed_controls"])
        paired = {}
        for label, saved in (("projected_target_history", old_trace), ("raw_request_history", trace)):
            paired[label] = task_metrics(
                model, geometry, reference, motion, {"qpos": saved["qpos"][: common + 1]}, phase
            )
        row = dict(
            name=name,
            completed_controls=n,
            requested_controls=result["requested_controls"],
            failure=result["failure"],
            histories=histories,
            measured=measured,
            received=received,
            network=network,
            physics=replay,
            maximum_range_excess_rad=float(excess.max()),
            preview_maximum_qpos_difference=prediction_error,
            guard_intervention_controls=np.flatnonzero(mask).tolist(),
            failed_proposal_reproduced=failure_reproduced,
            maximum_target_jump_rad=float(np.abs(np.diff(targets, axis=0)).max()),
            full_source_and_lifecycle_metrics=metrics,
            lifecycle=lifecycle,
            previous_completed_controls=old_result["completed_controls"],
            previous_full_q2_metrics=old_row["original_q2_metrics_after"],
            common_prefix_controls=common,
            paired_common_prefix_metrics=paired,
        )
        rows.append(row)
        print(
            json.dumps(
                dict(
                    name=name,
                    completed=n,
                    requested=result["requested_controls"],
                    root_p95_m=metrics.get(PHASE, {}).get("root_world_position_p95_m"),
                    raw_history_verified=True,
                    independent_physics=replay,
                )
            ),
            flush=True,
        )
    assert actor_digest(policy) == before_actor
    bind(__file__)
    for name, module in list(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if name.startswith("gear_sonic.") and path and Path(path).suffix == ".py":
            bind(path)
    for path, digest in inputs.items():
        if sha256_file(Path(path)) != digest:
            raise ValueError("audit source changed during verification: " + path)
    result = dict(
        kind="native23_raw_history_independent_full_trace_audit_v1",
        passed=True,
        inputs=inputs,
        cases=rows,
        actor_state_sha256_before_and_after=before_actor,
        only_retained_previous_action_input_semantics_changed=True,
        training_runtime_distribution_change_not_compatibility=True,
        additional_training_updates=0,
        hardware_authorized=False,
        deployment_ready=False,
        candidate_promoted=False,
    )
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(passed=True, output=str(output))), flush=True)


if __name__ == "__main__":
    main()
