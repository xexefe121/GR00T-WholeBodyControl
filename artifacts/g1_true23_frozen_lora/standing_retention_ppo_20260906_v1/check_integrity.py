"""Independent exact-state, lineage, source and saved-physics retention audit."""

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
from gear_sonic.utils.g1_true23_actuation_profile import NativeSupportActuationProfile
from gear_sonic.utils.g1_true23_diagnostic_pair import load_diagnostic_pair

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ASSETS = ROOT.parent / "GR00T-WholeBodyControl"
PREVIOUS = HERE.parent / "standing_motion_ppo_20260906_v1"
inputs = {}


def bind(path, expected=None):
    path = Path(path).resolve(strict=True)
    digest = file_sha256(path)
    if digest != inputs.get(str(path), digest) or (expected is not None and digest != expected):
        raise ValueError(f"retention audit input changed: {path}")
    inputs[str(path)] = digest
    return path


def read(path):
    value = json.loads(bind(path).read_text())
    materials = value.get("inputs", {})
    if materials.get("kind") == "g1_true23_standing_only_retention_inputs_v1":
        bind(materials["teacher_report_path"], materials["teacher_report_sha256"])
        for episode in materials["episodes"]:
            bind(episode["path"], episode["sha256"])
    else:
        for source, digest in materials.items():
            bind(source, digest)
    return value


def resolve_material(logical):
    prefix, remainder = logical.split("/", 1)
    if prefix == "causal_recovery":
        candidates = [
            ROOT / "gear_sonic" / folder / remainder
            for folder in ("scripts", "envs/mjlab", "trl/mjlab", "utils", "config/sim_validation")
        ]
        existing = [path for path in candidates if path.is_file()]
        assert len(existing) == 1, logical
        return existing[0]
    if prefix == "gear_sonic":
        return ROOT / logical
    if prefix == "unitree_rl_mjlab":
        return ASSETS / "external_dependencies/unitree_rl_mjlab" / remainder
    if prefix == "unitree_g1":
        return ASSETS / "external_dependencies/unitree_rl_mjlab/src/assets/robots/unitree_g1" / remainder
    if prefix == "motions":
        return PREVIOUS / "corpus" / remainder
    raise ValueError(f"unmapped retention material: {logical}")


bind(Path(__file__))
old_audit = read(PREVIOUS / "integrity_report.json")
experiment = read(HERE / "experiment_report.json")
export = read(HERE / "export_evaluation_report.json")
assert all(row["return_code"] == 0 for row in experiment["stages"])
assert all(row["return_code"] == 0 for row in export["stages"])
assert experiment["failed_attempt_training_updates"] == 0
assert len(experiment["failed_attempts"]) == 1
assert "OSError: [Errno 12] Cannot allocate memory" in Path(experiment["failed_attempts"][0]["log"]).read_text()
assert not (HERE / "breadth/checkpoints").exists()
directory = Path(experiment["training_directory"]).resolve(strict=True)
assert directory.parent == HERE
old_config = read(PREVIOUS / "breadth/resolved_training.json")
old_receipt = read(PREVIOUS / "breadth/standing_initialization.json")
old_zero = load_frozen_platform_lora_checkpoint(
    bind(PREVIOUS / "breadth/checkpoints/frozen_lora_model_0.pt"),
    expected_contract=old_receipt["frozen_platform_contract"],
)
runs, checkpoints = [], {}
for path, steps in ((HERE / "smoke", (0, 2, 4)), (directory, (0, 100))):
    config, lineage = read(path / "resolved_training.json"), read(path / "lineage.json")
    validate_mjlab_training_lineage(lineage)
    assert lineage["materials"]["resolved_config"]["payload"] == config
    if path == directory:
        assert {key: value for key, value in config.items() if key != "standing_output_retention"} == old_config
    for key in ("source_files", "robot_assets", "motion_dataset"):
        material = lineage["materials"][key]
        for row in material["files"]:
            assert bind(resolve_material(row["logical_path"]), row["sha256"]).stat().st_size == row["size_bytes"]
    receipt, runtime = read(path / "standing_initialization.json"), read(path / "standing_retention.json")
    assert receipt == old_receipt
    retention = config["standing_output_retention"]
    assert all(runtime[key] == value for key, value in retention.items())
    assert retention["weight"] == 10 and retention["held_out_training_rows"] == 0
    assert runtime["runtime"]["optimizer_steps_per_minibatch"] == 1
    bind(runtime["runtime"]["upstream_path"], runtime["runtime"]["upstream_sha256"])
    teacher_inputs = runtime["inputs"]
    teacher = read(bind(teacher_inputs["teacher_report_path"], teacher_inputs["teacher_report_sha256"]))
    assert teacher_inputs["held_out_rows_loaded"] == 0 and teacher_inputs["train_rows"] == 1500
    assert [row["name"] for row in teacher_inputs["episodes"]] == [
        "bounded_nominal",
        "bounded_plus",
        "bounded_minus",
    ]
    held = next(row for row in teacher["records"] if row["role"] == "held_out_episode")
    assert held["arrays"] not in [row["path"] for row in teacher_inputs["episodes"]]
    records = []
    for step in steps:
        target = bind(path / f"checkpoints/frozen_lora_model_{step}.pt")
        checkpoint = load_frozen_platform_lora_checkpoint(
            target, expected_contract=receipt["frozen_platform_contract"], expected_lineage=lineage
        )
        assert checkpoint["update_count"] == step
        trainer = checkpoint["trainer_state"]
        assert trainer["env_common_step_counter"] == step * config["agent"]["num_steps_per_env"]
        adam = checkpoint["optimizer_state_dict"]["state"]
        counts = sorted({int(value["step"].item()) for value in adam.values()})
        if step == 0:
            assert adam == {}
            for key in ("adapter_state_sha256", "critic_state_sha256", "merged_true23_policy_sha256"):
                assert checkpoint[key] == old_zero[key]
        else:
            assert counts == [
                step
                * config["agent"]["algorithm"]["num_learning_epochs"]
                * config["agent"]["algorithm"]["num_mini_batches"]
            ]
            assert len(adam) == 26 and checkpoint["adapter_state_sha256"] != old_zero["adapter_state_sha256"]
        records.append(
            dict(
                step=step,
                sha256=file_sha256(target),
                adapter_sha256=checkpoint["adapter_state_sha256"],
                critic_sha256=checkpoint["critic_state_sha256"],
                adam_steps=counts,
                env_common_step_counter=trainer["env_common_step_counter"],
            )
        )
        checkpoints[(path.name, step)] = checkpoint
    events = []
    for event_path in sorted(path.glob("events.out.tfevents.*")):
        event = EventAccumulator(str(bind(event_path)), size_guidance={"scalars": 0}).Reload()
        events.append(
            {
                tag: [dict(step=value.step, value=value.value) for value in event.Scalars(tag)]
                for tag in event.Tags()["scalars"]
            }
        )
    retention_events = sorted(
        [row for event in events for row in event["Loss/standing_retention"]], key=lambda row: row["step"]
    )
    assert [row["step"] for row in retention_events] == list(range(steps[-1]))
    assert all(np.isfinite(row["value"]) and row["value"] >= 0 for row in retention_events)
    assert len(events) == (2 if path.name == "smoke" else 1)
    prime = read(path / "environment_prime_standing_lora.json")
    assert prime["physics_steps"] == 0 and prime["common_step_counter"] == 0
    final_scalars = {tag: rows[-1] for tag, rows in events[-1].items()}
    runs.append(
        dict(
            run=path.name,
            checkpoints=records,
            final_scalars=final_scalars,
            source_files=lineage["materials"]["source_files"]["file_count"],
            training_transitions=steps[-1] * config["num_envs"] * config["agent"]["num_steps_per_env"],
            retention_minibatches=steps[-1]
            * config["agent"]["algorithm"]["num_learning_epochs"]
            * config["agent"]["algorithm"]["num_mini_batches"],
        )
    )

pair = load_diagnostic_pair(
    bind(HERE / "model_100/model_100.diagnostic.encoder.json"),
    bind(HERE / "model_100/model_100.diagnostic.decoder.json"),
)
trained = checkpoints[(directory.name, 100)]
for part in ("encoder", "decoder"):
    bind(pair[part]["path"], pair[part]["sha256"])
    component = read(HERE / f"model_100/model_100.diagnostic.{part}.json")
    assert component["source"]["policy_state_sha256"] == trained["merged_true23_policy_sha256"]
    assert component["source"]["adapter_state_sha256"] == trained["adapter_state_sha256"]
    assert component["source"]["update_count"] == 100
baseline = read(PREVIOUS / "model_100/evaluation/report.json")
report = read(HERE / "model_100/evaluation/report.json")
assert pair["paired_encoder_state_sha256"] == baseline["pair"]["paired_encoder_state_sha256"]
assert pair["encoder"]["sha256"] == baseline["pair"]["encoder"]["sha256"]
assert report["plan"] == baseline["plan"]
assert len(report["records"]) == 11 and sum(bool(row.get("not_executed")) for row in report["records"]) == 1
profile = NativeSupportActuationProfile.from_sim_config(
    bind(ROOT / "gear_sonic/config/sim_validation/g1_23dof_mujoco_sim2sim.json")
)
traces = []
for old, row in zip(baseline["records"], report["records"], strict=True):
    assert old["label"] == row["label"]
    if row.get("not_executed"):
        assert row["name"] == "elbow_crawling" and old["not_executed"] is True
        continue
    with np.load(bind(row["trace_path"]), allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    result = row["result"]
    with np.load(bind(row["source"], row["source_sha256"]), allow_pickle=False) as source:
        assert result["requested_transitions"] == len(source["joint_pos"]) - 11
    assert result["compiled_native_model_sha256"] == old["result"]["compiled_native_model_sha256"]
    engine = audit_engine_trace(arrays)
    assert engine["passed"]
    for key in ("qpos", "qvel"):
        before, after = arrays[f"physics_pre_{key}"], arrays[f"physics_post_{key}"]
        assert np.isfinite(before).all() and np.isfinite(after).all()
        np.testing.assert_array_equal(before[1:], after[:-1])
    np.testing.assert_array_equal(arrays["physics_effort"], arrays["physics_generalized_actuator_force"])
    ratio = float(np.max(np.abs(arrays["physics_effort"]) / (0.95 * 0.25 * np.asarray(profile.effort))))
    assert ratio <= 1 + 1e-10
    assert int(np.sum(arrays["physics_phase"] == 1)) == result["completed_active_physics_steps"]
    traces.append(
        dict(
            case=row["label"],
            baseline=old["result"]["completed_transitions"],
            retention=result["completed_transitions"],
            requested=result["requested_transitions"],
            motion_fidelity=result["motion_fidelity"]["passed"],
            return_completed=result["return_hold"].get("completed_transitions"),
            actual_physics_steps=engine["physics_steps_checked"],
            actual_time_s=engine["actual_final_engine_time_s"],
            maximum_outer_effort_ratio=ratio,
        )
    )
before_drift = read(HERE / "previous_drift.json")
after_drift = read(HERE / "retained_drift.json")
assert after_drift["records"][0]["metrics"] == before_drift["records"][0]["metrics"]
assert after_drift["backend"]["actual_training_gpu_cache_reconstructed"] is True
assert after_drift["actual_training_precision"] == "tf32"
assert after_drift["backend_comparisons"][1]["matches_recorded_training_cache"] is True
assert after_drift["held_out_arrays_loaded"] is before_drift["held_out_arrays_loaded"] is False
regression = read(HERE / "regression_report.json")
assert regression["exit_code"] == 0
tests = ET.parse(bind(HERE / "regression.xml")).getroot().find("testsuite").attrib
assert tests["tests"] == "551" and tests["failures"] == tests["errors"] == tests["skipped"] == "0"
for path in list(inputs):
    bind(path)
dump(
    HERE / "integrity_report.json",
    dict(
        kind="g1_true23_standing_retention_training_evaluation_integrity_v1",
        inputs=inputs,
        bound_file_count=len(inputs),
        mismatches=[],
        runs=runs,
        traces=traces,
        regression=tests,
        prior_inputs_preserved=len(old_audit["inputs"]),
        unchanged_original_motion_config_and_requests=True,
        zero_update_initial_actor_and_critic_exact=True,
        full_motion_qualified=False,
        hardware_authorized=False,
        deployment_ready=False,
        candidate_selected=False,
    ),
)
print(
    json.dumps(dict(bound_files=len(inputs), traces=len(traces), tests=tests["tests"], mismatches=0)), flush=True
)
