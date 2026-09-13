"""Independent v14 initialization, optimizer, pair, trace and budget comparison."""

import gc
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import torch

from gear_sonic.scripts import train_g1_true23_v14_native_ieee as launcher
from gear_sonic.scripts.record_g1_sonic_original29_baseline import dump
from gear_sonic.trl.mjlab.sonic_recovery_blend_policy import load_hash_bound_policy_state
from gear_sonic.utils.g1_23dof_mjlab_training import load_mjlab_training_checkpoint, validate_mjlab_training_lineage
from gear_sonic.utils.g1_sonic_cpp_observation_trace import audit_engine_trace
from gear_sonic.utils.g1_sonic_cpp_parameters import file_sha256
from gear_sonic.utils.g1_true23_frozen_lora_artifact import load_frozen_lora_diagnostic_policy
from gear_sonic.utils.g1_true23_training_precision import EXPECTED_STATE

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ASSETS = ROOT.parent / "GR00T-WholeBodyControl"
PARENT = HERE.parent / "ieee_motion_ppo_resume300_20260906_v1"
LORA100 = HERE.parent / "ieee_motion_ppo_20260906_v2"
inputs, traces, comparisons = {}, [], []


def bind(path, expected=None):
    path = Path(path).resolve(strict=True)
    digest = file_sha256(path)
    if digest != inputs.get(str(path), digest) or (expected is not None and digest != expected):
        raise ValueError(f"v14 integrity mismatch: {path}")
    inputs[str(path)] = digest
    return path


def read(path):
    return json.loads(bind(path).read_text())


def merge(report):
    for path, digest in report["inputs"].items():
        bind(path, digest)


bind(Path(__file__))
previous = read(bind(PARENT / "integrity_report.json", "f5acb2b7994e95a034e7ac89c4a2031b3369c6d516e84692c828842ffbce73f5"))
assert previous["checked_files"] == 984 and previous["mismatches"] == []
merge(previous)
experiment = read(HERE / "experiment_report.json")
exports = read(HERE / "export_evaluation_report.json")
regressions = read(HERE / "regression_report.json")
for report in (experiment, exports, regressions):
    merge(report)
    assert report["hardware_authorized"] is False and report["deployment_ready"] is False
for report in (experiment, exports, read(HERE / "post_training_report.json")):
    assert all(stage["return_code"] == 0 for stage in report["stages"])
assert regressions["exit_code"] == 0 and len(regressions["modules"]) == 48
test_suite = ET.parse(bind(HERE / "regression.xml")).getroot().find("testsuite")
assert int(test_suite.attrib["tests"]) == 614
assert all(int(test_suite.attrib[key]) == 0 for key in ("errors", "failures", "skipped"))

resolved = read(HERE / "train100/resolved_training.json")
lineage = validate_mjlab_training_lineage(read(HERE / "train100/lineage.json"))
assert lineage["materials"]["source_files"]["file_count"] == 30
sources = launcher.v14.base._source_files()
for path in (*launcher.v14.base.CAUSAL_SOURCE_FILES, *launcher.v14._SOURCE_FILES, *launcher.SOURCE_FILES):
    sources[f"causal_recovery/{path.name}"] = path
source_rows = lineage["materials"]["source_files"]["files"]
assert set(sources) == {row["logical_path"] for row in source_rows}
for row in source_rows:
    bind(sources[row["logical_path"]], row["sha256"])
for row in lineage["materials"]["robot_assets"]["files"]:
    path = ASSETS / "external_dependencies/unitree_rl_mjlab/src/assets/robots" / row["logical_path"]
    bind(path, row["sha256"])
for row in lineage["materials"]["motion_dataset"]["files"]:
    bind(HERE.parent / "standing_motion_ppo_20260906_v1/corpus" / Path(row["logical_path"]).name, row["sha256"])
runtime = read(HERE / "train100/v14_native_ieee_runtime.json")
assert runtime["actual_precision"] == EXPECTED_STATE == resolved["training_precision"]["requested_state"]
assert runtime["trainable_actor_elements"] == 274455
assert runtime["fresh_critic_optimizer_counters"] is True
assert runtime["checkpoint_frozen_actor_guard"] is True
old_config = read(LORA100 / "breadth_serial/resolved_training.json")
for key in ("stage_one_actuation", "num_envs", "semantic_profile", "motion_filename", "seed"):
    assert resolved[key] == old_config[key], key
for key in ("num_steps_per_env", "max_iterations", "clip_actions", "critic"):
    assert resolved["agent"][key] == old_config["agent"][key], key
algorithm = resolved["agent"]["algorithm"]
old_algorithm = old_config["agent"]["algorithm"]
algorithm_differences = {key: dict(v14=value, lora100=old_algorithm[key])
                         for key, value in algorithm.items() if value != old_algorithm[key]}
assert set(algorithm_differences) == {"schedule", "desired_kl"}
assert algorithm["schedule"] == "adaptive" and old_algorithm["schedule"] == "fixed"

recovery = load_hash_bound_policy_state(
    bind(launcher.RECOVERY_CHECKPOINT_PATH, launcher.RECOVERY_CHECKPOINT_SHA256),
    expected_checkpoint_sha256=launcher.RECOVERY_CHECKPOINT_SHA256,
    expected_policy_sha256=launcher.RECOVERY_POLICY_SHA256,
)
initial = load_mjlab_training_checkpoint(bind(HERE / "train100/checkpoints/causal_model_0.pt"),
                                       expected_lineage_sha256=lineage["lineage_sha256"], map_location="cpu")
final_path = bind(HERE / "train100/checkpoints/causal_model_100.pt")
final = load_mjlab_training_checkpoint(final_path, expected_lineage_sha256=lineage["lineage_sha256"],
                                     map_location="cpu")
assert initial["update_count"] == 0 and final["update_count"] == 100
assert initial["optimizer_state_dict"]["state"] == {}
assert initial["trainer_state"]["env_common_step_counter"] == 0
assert final["trainer_state"]["env_common_step_counter"] == 1600
assert final["trainer_state"]["completed_update_count"] == final["trainer_state"]["current_learning_iteration"] == 100
assert initial["training_gate"]["simulation_candidate_review_allowed"] is False
assert final["training_gate"]["simulation_candidate_review_allowed"] is True
std_names = [name for name in recovery if name.rsplit(".", 1)[-1] in {"std", "log_std"}]
assert len(std_names) == 1 and std_names[0] == "std"
expected = {**recovery, "std": torch.full((23,), .10, dtype=torch.float32)}
assert set(expected) == set(initial["policy_state_dict"]) == set(final["policy_state_dict"])
for name, value in expected.items():
    assert torch.equal(value, initial["policy_state_dict"][name]), name
changed = sorted(name for name, value in initial["policy_state_dict"].items()
                 if not torch.equal(value, final["policy_state_dict"][name]))
expected_changed = sorted(name.removeprefix("core.") for name in runtime["trainable_actor_tensors"])
assert changed == expected_changed and len(changed) == 4
assert initial["critic_state_sha256"] != final["critic_state_sha256"]
adam = final["optimizer_state_dict"]["state"]
assert len(adam) == 12 and {int(row["step"].item()) for row in adam.values()} == {4000}
encoder = {name: value for name, value in final["policy_state_dict"].items()
           if name.startswith("actor_module.encoders.teleop.")}
assert encoder
weight_report = dict(
    initial_checkpoint_sha256=file_sha256(HERE / "train100/checkpoints/causal_model_0.pt"),
    final_checkpoint_sha256=file_sha256(final_path), initial_policy_sha256=initial["policy_state_sha256"],
    final_policy_sha256=final["policy_state_sha256"],
    initial_critic_sha256=initial["critic_state_sha256"], final_critic_sha256=final["critic_state_sha256"],
    policy_tensor_count=len(expected), changed_actor_tensors=changed,
    unchanged_actor_tensor_count=len(expected) - len(changed), trainable_actor_elements=274455,
    initial_matches_recovery_network_plus_exact_std_pin=True, final_std_unchanged=True,
    fresh_initial_adam_empty=True, final_adam_tensor_count=12, final_adam_step=4000,
    trainer_state=final["trainer_state"], initial_update_gate_false=True, final_update_gate_true=True,
    initial_mean_policy_not_exported_or_evaluated=True, resume_performed=False,
)
del initial, final, recovery, expected
gc.collect()
lora = load_frozen_lora_diagnostic_policy(bind(LORA100 / "model_100/model_100.diagnostic.pt"))
old_encoder = {name: value for name, value in lora["policy_state_dict"].items()
               if name.startswith("actor_module.encoders.teleop.")}
assert set(encoder) == set(old_encoder)
encoder_differences = {name: dict(max_abs=float((value - old_encoder[name]).abs().max()),
                                 changed_elements=int(torch.count_nonzero(value != old_encoder[name])))
                       for name, value in encoder.items()}
del lora, old_encoder, encoder
gc.collect()

events = []
for path in sorted((HERE / "train100").glob("events.out.tfevents.*")):
    event = EventAccumulator(str(bind(path)), size_guidance={"scalars": 0}).Reload()
    events.append({tag: [dict(step=item.step, value=item.value) for item in event.Scalars(tag)]
                   for tag in event.Tags()["scalars"]})
assert len(events) == 1
loss_tags = [tag for tag in events[0] if tag.startswith("Loss/")]
assert loss_tags
for tag in loss_tags:
    assert [row["step"] for row in events[0][tag]] == list(range(100)), tag
baseline = read(LORA100 / "model_100/evaluation/report.json")
current = read(HERE / "model_100/evaluation/report.json")
later = read(PARENT / "model_300/evaluation/report.json")
merge(current)
assert current["plan"] == baseline["plan"] == later["plan"] and len(current["records"]) == 11
assert current["pair"]["checkpoint"]["sha256"] == weight_report["final_checkpoint_sha256"]
assert current["pair"]["policy_state_sha256"] == weight_report["final_policy_sha256"]
assert current["pair"]["components_from_same_checkpoint"] is True
assert current["pair"]["released_frozen_lora_encoder_substituted"] is False
for old, new, last in zip(baseline["records"], current["records"], later["records"], strict=True):
    assert old["label"] == new["label"] == last["label"]
    if new.get("not_executed"):
        assert old["not_executed"] and last["not_executed"] and new["name"] == "elbow_crawling"
        continue
    result = new["result"]
    for key in ("requested_transitions", "compiled_native_model_sha256", "gain_kp_hardware",
                "gain_kd_hardware", "effort_limit_hardware_nm", "previous_action_semantics", "observation_timing"):
        assert result[key] == old["result"][key] == last["result"][key], (new["label"], key)
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
    effort = np.asarray(result["gain_kp_hardware"]) * (arrays["actuation_target"] - arrays["actuation_q"]) - np.asarray(result["gain_kd_hardware"]) * arrays["actuation_dq"]
    np.testing.assert_allclose(arrays["actuation_effort"], effort, atol=1e-12, rtol=0)
    assert np.max(np.abs(effort) / np.asarray(result["effort_limit_hardware_nm"])) <= .2375 + 1e-12
    traces.append(dict(case=new["label"], physics_steps=len(arrays["physics_phase"]), active_steps=len(active), engine_audit=audit))
    comparisons.append(dict(case=new["label"], v14_100=result["completed_transitions"],
                            lora_100=old["result"]["completed_transitions"], lora_300=last["result"]["completed_transitions"],
                            requested=result["requested_transitions"], stationary_only=new["stationary_prerequisite_only"],
                            v14_motion_fidelity=result["motion_fidelity"]["passed"],
                            v14_return_completed=result["return_hold"].get("completed_transitions"), failure=result["failure"]))
for directory in (HERE, HERE / "smoke", HERE / "train100"):
    for path in directory.glob("*.json"):
        bind(path)
for path in HERE.glob("*.py"):
    bind(path)
for path in list(inputs):
    bind(path)
dump(HERE / "integrity_report.json", dict(
    kind="g1_true23_original_v14_native_ieee_integrity_v1", inputs=inputs, checked_files=len(inputs),
    previous_pinned_files=984, mismatches=[], source_manifest_files=30, regression_tests=614,
    weights=weight_report, encoder_differences_vs_frozen_lora=encoder_differences,
    all_encoder_tensors_equal_frozen_lora=not any(row["changed_elements"] for row in encoder_differences.values()),
    additional_v14_ppo_updates=100, additional_v14_training_transitions=51200,
    planned_v14_updates=1000, remaining_v14_updates_not_run=900,
    budget_match_is_new_update_and_transition_count_not_equal_actor_pretraining=True,
    one_variable_lora_ablation=False, algorithm_config_differences=algorithm_differences,
    other_method_differences=["recovery_actor_vs_standing_lora_initialization", "four_last_block_tensors_vs_all_decoder_lora",
                              "fixed_std_0p1_vs_source_std", "clip_uniform_vs_failure_adaptive", "no_standing_retention_vs_weight10"],
    final_scalars={tag: rows[-1] for tag, rows in events[0].items()}, traces=traces, comparisons=comparisons,
    hardware_authorized=False, deployment_ready=False, full_motion_qualified=False,
    original_v14_selected_for_deployment=False, physical_bit30_failure_explained=False))
print(json.dumps(dict(checked_files=len(inputs), mismatches=0, regression_tests=614,
                       v14_ppo_updates=100, trace_physics_steps=sum(row["physics_steps"] for row in traces))), flush=True)
