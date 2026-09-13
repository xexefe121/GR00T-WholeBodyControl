"""Independent mappings, weight/Adam resume, full-request and hash audit.

AUTHORITATIVE CORRECTION: the raw continuation driver's RNG-restored flag is
incorrect. Legacy frozen checkpoints store no global RNG or simulator state.
Do not rewrite that raw record: bind it and explicitly supersede that claim.
"""

import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from gear_sonic.scripts.record_g1_sonic_original29_baseline import dump
from gear_sonic.trl.mjlab.frozen_platform_lora_runner import load_frozen_platform_lora_checkpoint
from gear_sonic.utils.g1_23dof_mjlab_training import validate_mjlab_training_lineage
from gear_sonic.utils.g1_sonic_cpp_observation_trace import audit_engine_trace
from gear_sonic.utils.g1_sonic_cpp_parameters import file_sha256
from gear_sonic.utils.g1_true23_diagnostic_pair import load_diagnostic_pair
from gear_sonic.utils.g1_true23_training_precision import EXPECTED_STATE

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PARENT = HERE.parent / "ieee_motion_ppo_20260906_v2"
WITNESS = HERE.parent / "recorded_training_boundary_20260906_v2"
inputs, checkpoints, comparisons, traces = {}, [], [], []


def bind(path, expected=None):
    path = Path(path).resolve(strict=True)
    digest = file_sha256(path)
    if digest != inputs.get(str(path), digest) or (expected is not None and digest != expected):
        raise ValueError(f"continuation integrity mismatch: {path}")
    inputs[str(path)] = digest
    return path


def read(path):
    return json.loads(bind(path).read_text())


def merge(report):
    for path, digest in report["inputs"].items():
        bind(path, digest)


bind(Path(__file__))
old_audit = read(
    bind(PARENT / "integrity_report.json", "2b983c9b42dc20f2e441748dc98517e131f2cd15f450af0fc3fa2f302a03314a")
)
merge(old_audit)
assert len(old_audit["inputs"]) == 663
witness = read(WITNESS / "audit/report.json")
merge(witness)
for file in ("run_audit.py", "audit.log", "audit.stage.json"):
    bind(WITNESS / file)
assert read(WITNESS / "audit.stage.json")["exit_code"] == 0
failed = WITNESS.parent / "recorded_training_boundary_20260906_v1"
for file in ("run_audit.py", "audit.log", "audit.stage.json", "failed_audit_script.py"):
    bind(failed / file)
assert read(failed / "audit.stage.json")["exit_code"] != 0
assert "model does not match evaluator" in (failed / "audit.log").read_text()
assert not list((failed / "audit").iterdir())
executed = [row for row in witness["records"] if not row.get("not_executed")]
assert len(executed) == 20 and sum(row["recorded_inference_calls"] for row in executed) == 2675
maxima = {}
for row in executed:
    for device, report in row["devices"].items():
        for name in (
            "encoder267",
            "history930",
            "fsq64",
            "raw23",
            "safe_target",
            "requested_target",
            "projected_target",
            "applied_effort",
        ):
            assert report[name]["within_tolerance"], (row["label"], device, name)
            maxima[name] = max(maxima.get(name, 0), report[name]["maximum_absolute_difference"])
        assert report["invalid_successful_substeps"] == []
        if report["terminal"] is not None:
            assert report["terminal"]["rejected_by_training"]
            assert report["terminal"]["training_output_effort"] == [0.0] * 23
        with np.load(bind(row["witness_arrays"]), allow_pickle=False) as archive:
            prefix = device.replace(":", "_") + "_"
            assert len(archive[prefix + "encoder267"]) == row["recorded_inference_calls"]
            assert len(archive[prefix + "target"]) == row["completed_active_substeps"]
            assert not archive[prefix + "invalid"].any()
experiment, exports, regressions = map(
    read,
    (HERE / "experiment_report.json", HERE / "export_evaluation_report.json", HERE / "regression_report.json"),
)
for report in (experiment, exports, regressions):
    merge(report)
assert all(row["return_code"] == 0 for report in (experiment, exports) for row in report["stages"])
assert regressions["exit_code"] == 0
suite = ET.parse(bind(HERE / "regression.xml")).getroot().find("testsuite")
assert int(suite.attrib["tests"]) == 585
assert all(int(suite.attrib[key]) == 0 for key in ("errors", "failures", "skipped"))
config, lineage = read(HERE / "breadth/resolved_training.json"), read(HERE / "breadth/lineage.json")
assert config == read(PARENT / "breadth_serial/resolved_training.json")
assert lineage == read(PARENT / "breadth_serial/lineage.json")
validate_mjlab_training_lineage(lineage)
assert lineage["materials"]["source_files"]["file_count"] == 38
receipt = read(PARENT / "breadth_serial/standing_initialization.json")
precision = read(HERE / "breadth/training_precision.json")
assert precision["actual_state"] == EXPECTED_STATE == precision["contract"]["requested_state"]
assert precision == read(PARENT / "breadth_serial/training_precision.json")
retention = read(HERE / "breadth/standing_retention.json")
assert retention == read(PARENT / "breadth_serial/standing_retention.json")
bind(ROOT / "gear_sonic/trl/mjlab/runner.py")
for step, directory in ((100, PARENT / "breadth_serial"), (200, HERE / "breadth"), (300, HERE / "breadth")):
    path = bind(directory / f"checkpoints/frozen_lora_model_{step}.pt")
    state = load_frozen_platform_lora_checkpoint(
        path, expected_contract=receipt["frozen_platform_contract"], expected_lineage=lineage
    )
    assert state["update_count"] == step
    assert state["trainer_state"] == dict(
        algorithm_learning_rate=5e-6,
        completed_update_count=step,
        current_learning_iteration=step,
        env_common_step_counter=step * 16,
    )
    assert len(state["optimizer_state_dict"]["state"]) == 26
    assert {int(row["step"].item()) for row in state["optimizer_state_dict"]["state"].values()} == {step * 40}
    assert not any("rng" in name.lower() or "simulator_state" in name.lower() for name in state)
    checkpoints.append(
        dict(
            update_count=step,
            file_sha256=file_sha256(path),
            adapter_sha256=state["adapter_state_sha256"],
            critic_sha256=state["critic_state_sha256"],
            merged_policy_sha256=state["merged_true23_policy_sha256"],
            adam_step=step * 40,
            env_common_step_counter=step * 16,
            global_rng_state_saved=False,
            simulator_state_saved=False,
        )
    )
    del state
assert len({row["adapter_sha256"] for row in checkpoints}) == 3
assert len({row["critic_sha256"] for row in checkpoints}) == 3
events = []
for path in sorted((HERE / "breadth").glob("events.out.tfevents.*")):
    event = EventAccumulator(str(bind(path)), size_guidance={"scalars": 0}).Reload()
    events.append(
        {
            tag: [dict(step=item.step, value=item.value) for item in event.Scalars(tag)]
            for tag in event.Tags()["scalars"]
        }
    )
actual_updates = sorted(row["step"] for event in events for row in event["Loss/standing_retention"])
assert actual_updates == list(range(100, 300))
baseline = read(PARENT / "model_100/evaluation/report.json")
for step in (200, 300):
    directory = HERE / f"model_{step}"
    pair = load_diagnostic_pair(
        bind(directory / f"model_{step}.diagnostic.encoder.json"),
        bind(directory / f"model_{step}.diagnostic.decoder.json"),
    )
    checkpoint = next(row for row in checkpoints if row["update_count"] == step)
    assert pair["source"]["adapter_state_sha256"] == checkpoint["adapter_sha256"]
    assert pair["source"]["policy_state_sha256"] == checkpoint["merged_policy_sha256"]
    report = read(directory / "evaluation/report.json")
    merge(report)
    assert report["plan"] == baseline["plan"] and len(report["records"]) == 11
    assert pair["encoder"]["sha256"] == baseline["pair"]["encoder"]["sha256"]
    for old, new in zip(baseline["records"], report["records"], strict=True):
        assert old["label"] == new["label"]
        if new.get("not_executed"):
            assert old["not_executed"] and new["name"] == "elbow_crawling"
            continue
        result = new["result"]
        assert result["requested_transitions"] == old["result"]["requested_transitions"]
        for key in (
            "compiled_native_model_sha256",
            "gain_kp_hardware",
            "gain_kd_hardware",
            "effort_limit_hardware_nm",
            "previous_action_semantics",
            "observation_timing",
        ):
            assert result[key] == old["result"][key]
        with np.load(bind(new["trace_path"]), allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        audit = audit_engine_trace(arrays)
        assert audit["passed"]
        np.testing.assert_array_equal(arrays["physics_pre_qpos"][1:], arrays["physics_post_qpos"][:-1])
        np.testing.assert_array_equal(arrays["physics_pre_qvel"][1:], arrays["physics_post_qvel"][:-1])
        active = np.flatnonzero(arrays["physics_phase"] == 1)
        assert len(active) == result["completed_active_physics_steps"]
        assert len(active) // 10 == result["completed_transitions"]
        np.testing.assert_array_equal(arrays["physics_effort"][active], arrays["actuation_effort"])
        np.testing.assert_allclose(
            arrays["actuation_effort"],
            np.asarray(result["gain_kp_hardware"]) * (arrays["actuation_target"] - arrays["actuation_q"])
            - np.asarray(result["gain_kd_hardware"]) * arrays["actuation_dq"],
            atol=1e-12,
            rtol=0,
        )
        traces.append(
            dict(
                update_count=step,
                case=new["label"],
                actual_physics_steps=len(arrays["physics_phase"]),
                engine_audit=audit,
            )
        )
        comparisons.append(
            dict(
                update_count=step,
                case=new["label"],
                previous100=old["result"]["completed_transitions"],
                completed=result["completed_transitions"],
                requested=result["requested_transitions"],
                motion_fidelity=result["motion_fidelity"]["passed"],
                stationary_only=new["stationary_prerequisite_only"],
                return_completed=result["return_hold"].get("completed_transitions"),
                failure=result["failure"],
            )
        )
for path in (HERE / "breadth").glob("*.json"):
    bind(path)
for path in HERE.glob("*.stage.json"):
    bind(path)
for path in HERE.glob("*.py"):
    bind(path)
for path in list(inputs):
    bind(path)
correction = dict(
    raw_report=str(HERE / "experiment_report.json"),
    incorrect_field="actor_critic_adam_counters_rng_and_retention_cache_restored_by_checked_runner",
    raw_value=experiment["actor_critic_adam_counters_rng_and_retention_cache_restored_by_checked_runner"],
    corrected_global_rng_restored=False,
    optimizer_and_weight_resume_verified=True,
    standing_anchor_sampler_is_seed_plus_adam_step=True,
    global_training_rng_reinitialized=True,
    simulator_reinitialized=True,
    bit_exact_uninterrupted_training_equivalence=False,
    raw_evidence_preserved_not_relabelled=True,
)
corrected = {key: value for key, value in experiment.items() if key != correction["incorrect_field"]}
corrected.update(
    kind="g1_true23_ieee_motion_ppo_continuation_corrected_v1",
    correction_of_sha256=file_sha256(HERE / "experiment_report.json"),
    actor_critic_adam_counters_restored=True,
    global_rng_restored=False,
    simulator_state_restored=False,
    adaptive_motion_sampler_state_restored=False,
    standing_anchor_cache_recomputed_and_verified_identical=True,
    standing_anchor_sampler_resumes_from_seed_plus_adam_step=True,
    bit_exact_uninterrupted_training_equivalence=False,
    correction=correction,
)
dump(HERE / "experiment_report.corrected.json", corrected)
bind(HERE / "experiment_report.corrected.json")
dump(
    HERE / "integrity_report.json",
    dict(
        kind="g1_true23_recorded_boundary_and_ppo300_integrity_v1",
        inputs=inputs,
        previous_pinned_files=663,
        checked_files=len(inputs),
        mismatches=[],
        boundary_calls=2675,
        boundary_motion_prefix_calls=sum(
            row["recorded_inference_calls"] for row in executed if not row["stationary_prerequisite_only"]
        ),
        boundary_active_substeps=sum(row["completed_active_substeps"] for row in executed),
        boundary_terminal_rejections=sum(row["devices"]["cpu"]["terminal"] is not None for row in executed),
        boundary_maxima=maxima,
        checkpoints=checkpoints,
        additional_ppo_updates=200,
        additional_training_transitions=200 * 32 * 16,
        total_training_transitions=300 * 32 * 16,
        exact_global_rng_resume_claim_correction=correction,
        final_scalars={tag: rows[-1] for tag, rows in events[-1].items()},
        regression_tests=585,
        traces=traces,
        comparisons=comparisons,
        hardware_authorized=False,
        deployment_ready=False,
        full_motion_qualified=False,
    ),
)
print(
    json.dumps(
        dict(
            checked_files=len(inputs),
            mismatches=0,
            boundary_calls=2675,
            regression_tests=585,
            checkpoints=[100, 200, 300],
            trace_physics_steps=sum(row["actual_physics_steps"] for row in traces),
        )
    ),
    flush=True,
)
