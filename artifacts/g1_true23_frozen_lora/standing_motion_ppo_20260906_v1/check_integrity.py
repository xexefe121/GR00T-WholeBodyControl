"""Independent audit of source lineage, real PPO updates and saved physics."""

import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from gear_sonic.scripts.record_g1_sonic_original29_baseline import dump
from gear_sonic.trl.mjlab.frozen_platform_lora_runner import load_frozen_platform_lora_checkpoint
from gear_sonic.utils.g1_23dof_mjlab_training import validate_mjlab_training_lineage
from gear_sonic.utils.g1_23dof_multi_motion import MOTION_ARRAY_NAMES
from gear_sonic.utils.g1_sonic_cpp_observation_trace import audit_engine_trace
from gear_sonic.utils.g1_sonic_cpp_parameters import file_sha256
from gear_sonic.utils.g1_true23_actuation_profile import NativeSupportActuationProfile
from gear_sonic.utils.g1_true23_diagnostic_pair import load_diagnostic_pair

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ASSETS = ROOT.parent / "GR00T-WholeBodyControl"
OLD = HERE.parent / "interior_effort_20260906_v1"
inputs = {}


def bind(path, expected=None):
    path = Path(path).resolve(strict=True)
    digest = file_sha256(path)
    if digest != inputs.get(str(path), digest) or (expected is not None and digest != expected):
        raise ValueError(f"motion PPO audit evidence changed: {path}")
    inputs[str(path)] = digest
    return path


def read(path):
    value = json.loads(bind(path).read_text())
    for source, digest in value.get("inputs", {}).items():
        bind(source, digest)
    return value


bind(Path(__file__))
corpus = read(HERE / "corpus/report.json")
spans = read(HERE / "corpus/corpus.spans.json")
assert spans["total_frames"] == 5940 and spans["clip_count"] == 8
assert spans["dropped_short"] == spans["dropped_bad"] == 0
assert corpus["action_teacher_or_behavior_bank"] is False
unavailable = [row for row in corpus["source_request_ledger"] if not row["available"]]
assert len(unavailable) == 1 and unavailable[0]["name"] == "elbow_crawling"
assert unavailable[0]["replacement"] is None
sources = {row["name"]: row for row in corpus["source_request_ledger"] if row["available"]}
sources["synthetic_standing_prerequisite"] = dict(path=str(OLD / "stationary/stationary_reference.npz"))
with np.load(bind(HERE / "corpus/corpus.npz"), allow_pickle=False) as joined:
    assert float(joined["fps"].item()) == 50.0
    cursor = 0
    for span in spans["spans"]:
        row = sources[span["name"]]
        assert span["start"] == cursor
        with np.load(bind(row["path"], row.get("sha256")), allow_pickle=False) as source:
            assert len(source["joint_pos"]) == span["length"]
            for key in MOTION_ARRAY_NAMES:
                assert joined[key].dtype == np.float32
                np.testing.assert_array_equal(
                    joined[key][cursor : cursor + span["length"]], source[key].astype(np.float32)
                )
        cursor += span["length"]
    assert cursor == 5940


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
        return HERE / "corpus" / remainder
    raise ValueError(f"unmapped material: {logical}")


checkpoints, runs = {}, []
for run, updates in (("smoke", 2), ("breadth", 100)):
    directory = HERE / run
    lineage = read(directory / "lineage.json")
    validate_mjlab_training_lineage(lineage)
    config = read(directory / "resolved_training.json")
    assert lineage["materials"]["resolved_config"]["payload"] == config
    for key in ("source_files", "robot_assets", "motion_dataset"):
        material = lineage["materials"][key]
        assert len(material["files"]) == material["file_count"]
        for row in material["files"]:
            assert bind(resolve_material(row["logical_path"]), row["sha256"]).stat().st_size == row["size_bytes"]
    receipt = read(directory / "standing_initialization.json")
    descriptor = config["standing_training_initialization"]
    assert all(receipt[key] == value for key, value in descriptor.items())
    assert all(receipt[key] is False for key in ("critic_reused", "optimizer_reused", "counters_reused"))
    pair = []
    for step in (0, updates):
        path = bind(directory / f"checkpoints/frozen_lora_model_{step}.pt")
        checkpoint = load_frozen_platform_lora_checkpoint(
            path, expected_contract=receipt["frozen_platform_contract"], expected_lineage=lineage
        )
        assert checkpoint["update_count"] == step
        trainer = checkpoint["trainer_state"]
        assert trainer["env_common_step_counter"] == step * config["agent"]["num_steps_per_env"]
        assert trainer["algorithm_learning_rate"] == config["agent"]["algorithm"]["learning_rate"]
        state = checkpoint["optimizer_state_dict"]["state"]
        adam_steps = sorted({int(value["step"].item()) for value in state.values()})
        if step == 0:
            assert state == {}
            assert checkpoint["adapter_state_sha256"] == receipt["adapter_state_sha256"]
            assert checkpoint["merged_true23_policy_sha256"] == receipt["merged_true23_policy_sha256"]
        else:
            assert adam_steps == [
                step
                * config["agent"]["algorithm"]["num_learning_epochs"]
                * config["agent"]["algorithm"]["num_mini_batches"]
            ]
            assert checkpoint["adapter_state_sha256"] != receipt["adapter_state_sha256"]
        row = dict(
            path=str(path),
            sha256=file_sha256(path),
            update_count=step,
            trainer_state=trainer,
            adapter_state_sha256=checkpoint["adapter_state_sha256"],
            critic_state_sha256=checkpoint["critic_state_sha256"],
            merged_true23_policy_sha256=checkpoint["merged_true23_policy_sha256"],
            optimizer_tensor_count=len(state),
            unique_adam_steps=adam_steps,
        )
        pair.append(row)
        checkpoints[(run, step)] = checkpoint
    assert pair[0]["critic_state_sha256"] != pair[1]["critic_state_sha256"]
    prime = read(directory / "environment_prime_standing_lora.json")
    assert prime["physics_steps"] == 0
    assert prime["common_step_counter"] == prime["simulation_step_counter"] == 0
    event_paths = list(directory.glob("events.out.tfevents.*"))
    assert len(event_paths) == 1
    event = EventAccumulator(str(bind(event_paths[0])), size_guidance={"scalars": 0}).Reload()
    final_scalars = {}
    for tag in event.Tags()["scalars"]:
        values = event.Scalars(tag)
        final_scalars[tag] = dict(count=len(values), step=values[-1].step, value=values[-1].value)
    runs.append(
        dict(
            run=run,
            checkpoints=pair,
            source_file_count=lineage["materials"]["source_files"]["file_count"],
            transitions=updates * config["num_envs"] * config["agent"]["num_steps_per_env"],
            planned_updates=config["planned_updates"],
            final_scalars=final_scalars,
            full_batch_reset_retries=prime["full_batch_reset_retries"],
        )
    )

pipeline = read(HERE / "trained_pipeline_report.json")
assert len(pipeline["stages"]) == 4 and all(row["return_code"] == 0 for row in pipeline["stages"])
assert pipeline["zero_update_diagnostic_export_rejected"] is True
pair = load_diagnostic_pair(
    bind(HERE / "model_100/model_100.diagnostic.encoder.json"),
    bind(HERE / "model_100/model_100.diagnostic.decoder.json"),
)
for part in ("encoder", "decoder"):
    bind(pair[part]["path"], pair[part]["sha256"])
    component = read(HERE / f"model_100/model_100.diagnostic.{part}.json")
    assert component["source"]["update_count"] == 100
    assert (
        component["source"]["policy_state_sha256"] == checkpoints[("breadth", 100)]["merged_true23_policy_sha256"]
    )
    assert component["source"]["adapter_state_sha256"] == checkpoints[("breadth", 100)]["adapter_state_sha256"]
assert pair["paired_encoder_state_sha256"] == "3625edb10aabd266196702aefd464ad07c93847f2d1722a977e18ef2a0143990"

reports = [
    read(HERE / path)
    for path in ("torch_0/report.json", "torch_100/report.json", "model_100/evaluation/report.json")
]
read(HERE / "matched_torch_comparison.json")
profile = NativeSupportActuationProfile.from_sim_config(
    bind(ROOT / "gear_sonic/config/sim_validation/g1_23dof_mujoco_sim2sim.json")
)
traces, results = [], []
for index, report in enumerate(reports):
    assert len(report["records"]) == 11
    assert (
        report["hardware_authorized"]
        is report["deployment_ready"]
        is report["full_eight_clip_qualification"]
        is False
    )
    assert sum(bool(row.get("not_executed")) for row in report["records"]) == 1
    result_map = {}
    for row in report["records"]:
        if row.get("not_executed"):
            assert row["name"] == "elbow_crawling"
            continue
        result_map[row["label"]] = row
        with np.load(bind(row["trace_path"]), allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        result = row["result"]
        with np.load(bind(row["source"], row["source_sha256"]), allow_pickle=False) as source:
            assert result["requested_transitions"] == len(source["joint_pos"]) - 11
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
        assert (
            result["compiled_native_model_sha256"]
            == "1f616be811e72988f4764f74b8c2eb7f57b5ee87873ee36204bd882ff9cf96d7"
        )
        traces.append(
            dict(
                report=index,
                case=row["label"],
                actual_physics_steps=engine["physics_steps_checked"],
                actual_time_s=engine["actual_final_engine_time_s"],
                outer_effort_peak_ratio=ratio,
            )
        )
    results.append(result_map)

comparisons = []
for label in results[0]:
    rows = [result[label] for result in results]
    for key in ("completed_transitions", "requested_transitions"):
        assert rows[1]["result"][key] == rows[2]["result"][key]
    comparisons.append(
        dict(
            case=label,
            initial_cpu=rows[0]["result"]["completed_transitions"],
            ppo100_cpu=rows[1]["result"]["completed_transitions"],
            ppo100_onnx=rows[2]["result"]["completed_transitions"],
            requested=rows[0]["result"]["requested_transitions"],
        )
    )
for label in ("standing.reference", "standing.acquired"):
    assert results[0][label]["result"]["motion_fidelity"]["passed"] is True
    assert results[1][label]["result"]["motion_fidelity"]["passed"] is False
assert results[0]["standing.acquired"]["result"]["return_hold"]["completed_transitions"] == 250
assert results[1]["standing.acquired"]["result"]["return_hold"]["completed_transitions"] == 0
old_motion = read(OLD / "standing_lora_full_motion/report.json")
old_standing = read(OLD / "standing_lora500/report.json")
equivalent_arrays = 0
for row in old_motion["records"]:
    if row.get("not_executed"):
        continue
    label = row["name"] + (".historical" if row["historical_start"] else ".reference")
    with (
        np.load(bind(row["trace_path"]), allow_pickle=False) as old,
        np.load(results[0][label]["trace_path"], allow_pickle=False) as new,
    ):
        for key in old.files:
            np.testing.assert_array_equal(old[key], new[key])
            equivalent_arrays += 1
for old_name, label in (
    ("standing_synthetic_start", "standing.reference"),
    ("standing_after_acquisition", "standing.acquired"),
):
    with (
        np.load(bind(OLD / f"standing_lora500/{old_name}.npz"), allow_pickle=False) as old,
        np.load(results[0][label]["trace_path"], allow_pickle=False) as new,
    ):
        for key in old.files:
            np.testing.assert_array_equal(old[key], new[key])
            equivalent_arrays += 1
regression = read(HERE / "regression_report.json")
assert regression["exit_code"] == 0
tests = ET.parse(bind(HERE / "regression.xml")).getroot().find("testsuite").attrib
assert tests["tests"] == "511" and tests["failures"] == tests["errors"] == tests["skipped"] == "0"
for path in list(inputs):
    bind(path)
dump(
    HERE / "integrity_report.json",
    dict(
        kind="g1_true23_standing_initialized_motion_ppo_independent_integrity_v1",
        inputs=inputs,
        bound_file_count=len(inputs),
        mismatches=[],
        runs=runs,
        traces=traces,
        comparisons=comparisons,
        source_spans_bit_identical_after_declared_float32_cast=True,
        original_eight_requests_preserved=True,
        initial_cpu_arrays_exactly_reproduced=equivalent_arrays,
        regression=tests,
        standing_regression_confirmed_same_inference_backend=True,
        candidate_selected=False,
        full_motion_qualified=False,
        hardware_authorized=False,
        deployment_ready=False,
    ),
)
print(
    json.dumps(
        dict(
            bound_files=len(inputs),
            traces=len(traces),
            equivalent_initial_arrays=equivalent_arrays,
            tests=tests["tests"],
            mismatches=0,
        )
    ),
    flush=True,
)
