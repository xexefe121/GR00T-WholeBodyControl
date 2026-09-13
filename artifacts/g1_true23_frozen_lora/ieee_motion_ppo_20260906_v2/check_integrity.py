"""Independent IEEE lineage/counter audit and recorder physics equivalence."""

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
from gear_sonic.utils.g1_true23_training_precision import EXPECTED_STATE

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ASSETS = ROOT.parent / "GR00T-WholeBodyControl"
PREVIOUS = HERE.parent / "standing_retention_ppo_20260906_v1"
CORPUS = HERE.parent / "standing_motion_ppo_20260906_v1/corpus"
inputs, runs, traces = {}, [], []


def bind(path, expected=None):
    path = Path(path).resolve(strict=True)
    actual = file_sha256(path)
    if actual != inputs.get(str(path), actual) or (expected is not None and actual != expected):
        raise ValueError(f"IEEE integrity input changed: {path}")
    inputs[str(path)] = actual
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
        return CORPUS / remainder
    raise ValueError(f"unknown IEEE material: {logical}")


bind(Path(__file__))
old_audit = read(PREVIOUS / "integrity_report.json")
experiment = read(HERE / "experiment_report.json")
post = read(HERE / "post_training_report.json")
export = read(HERE / "export_evaluation_report.json")
for report in (experiment, post, export):
    assert all(row["return_code"] == 0 for row in report["stages"])
failed = HERE.parent / "ieee_motion_ppo_20260906_v1"
failure = read(failed / "smoke_initial.stage.json")
assert failure["return_code"] != 0 and not (failed / "smoke").exists()
assert "IEEE training precision or backend contract changed" in bind(failed / "smoke_initial.log").read_text()
bind(failed / "failed_precision_helper.py")
old_config = read(PREVIOUS / "breadth_serial/resolved_training.json")
old_receipt = read(PREVIOUS / "breadth_serial/standing_initialization.json")
old_zero = load_frozen_platform_lora_checkpoint(
    bind(PREVIOUS / "breadth_serial/checkpoints/frozen_lora_model_0.pt"),
    expected_contract=old_receipt["frozen_platform_contract"],
)
trained = None
for directory, steps in ((HERE / "smoke", (0, 2, 4)), (HERE / "breadth_serial", (0, 100))):
    config, lineage = read(directory / "resolved_training.json"), read(directory / "lineage.json")
    validate_mjlab_training_lineage(lineage)
    assert lineage["materials"]["resolved_config"]["payload"] == config
    if directory.name == "breadth_serial":
        assert {key: value for key, value in config.items() if key != "training_precision"} == old_config
    for kind in ("source_files", "robot_assets", "motion_dataset"):
        for row in lineage["materials"][kind]["files"]:
            assert bind(resolve_material(row["logical_path"]), row["sha256"]).stat().st_size == row["size_bytes"]
    assert lineage["materials"]["source_files"]["file_count"] == 38
    receipt, retention = (
        read(directory / "standing_initialization.json"),
        read(directory / "standing_retention.json"),
    )
    assert receipt == old_receipt
    precision = read(directory / "training_precision.json")
    contract = config["training_precision"]
    assert (
        precision["contract"] == contract
        and precision["actual_state"] == contract["requested_state"] == EXPECTED_STATE
    )
    bind(contract["mjlab_helper_path"], contract["mjlab_helper_sha256"])
    assert contract["existing_tf32_checkpoint_relabelled"] is False
    assert retention["weight"] == 10 and retention["inputs"]["train_rows"] == 1500
    assert retention["inputs"]["held_out_rows_loaded"] == 0
    for key, value in config["standing_output_retention"].items():
        assert retention[key] == value
    bind(retention["runtime"]["upstream_path"], retention["runtime"]["upstream_sha256"])
    checkpoint_rows = []
    for step in steps:
        path = bind(directory / f"checkpoints/frozen_lora_model_{step}.pt")
        checkpoint = load_frozen_platform_lora_checkpoint(
            path, expected_contract=receipt["frozen_platform_contract"], expected_lineage=lineage
        )
        assert checkpoint["update_count"] == step
        assert (
            checkpoint["trainer_state"]["env_common_step_counter"] == step * config["agent"]["num_steps_per_env"]
        )
        adam = checkpoint["optimizer_state_dict"]["state"]
        adam_steps = sorted({int(value["step"].item()) for value in adam.values()})
        expected_steps = (
            step
            * config["agent"]["algorithm"]["num_learning_epochs"]
            * config["agent"]["algorithm"]["num_mini_batches"]
        )
        if step == 0:
            assert adam == {}
            for key in ("adapter_state_sha256", "critic_state_sha256", "merged_true23_policy_sha256"):
                assert checkpoint[key] == old_zero[key]
        else:
            assert adam_steps == [expected_steps] and len(adam) == 26
        checkpoint_rows.append(
            dict(
                update_count=step,
                file_sha256=file_sha256(path),
                adam_steps=adam_steps,
                adapter_sha256=checkpoint["adapter_state_sha256"],
                critic_sha256=checkpoint["critic_state_sha256"],
                env_common_step_counter=checkpoint["trainer_state"]["env_common_step_counter"],
            )
        )
        if step == 100:
            trained = checkpoint
    events = []
    for path in directory.glob("events.out.tfevents.*"):
        event = EventAccumulator(str(bind(path)), size_guidance={"scalars": 0}).Reload()
        events.append(
            {
                tag: [dict(step=item.step, value=item.value) for item in event.Scalars(tag)]
                for tag in event.Tags()["scalars"]
            }
        )
    actual_updates = sorted(row["step"] for event in events for row in event["Loss/standing_retention"])
    assert actual_updates == list(range(steps[-1]))
    prime = read(directory / "environment_prime_standing_lora.json")
    assert prime["physics_steps"] == prime["common_step_counter"] == 0
    runs.append(
        dict(
            run=directory.name,
            checkpoints=checkpoint_rows,
            source_file_count=38,
            actual_training_transitions=steps[-1] * config["num_envs"] * config["agent"]["num_steps_per_env"],
            anchor_cache_sha256=retention["anchor_cache_sha256"],
            final_scalars={
                tag: rows[-1]
                for tag, rows in sorted(events, key=lambda event: event["Loss/standing_retention"][0]["step"])[
                    -1
                ].items()
            },
        )
    )
pair = load_diagnostic_pair(
    bind(HERE / "model_100/model_100.diagnostic.encoder.json"),
    bind(HERE / "model_100/model_100.diagnostic.decoder.json"),
)
assert pair["source"]["adapter_state_sha256"] == trained["adapter_state_sha256"]
assert pair["source"]["policy_state_sha256"] == trained["merged_true23_policy_sha256"]
profile = NativeSupportActuationProfile.from_sim_config(
    bind(ROOT / "gear_sonic/config/sim_validation/g1_23dof_mujoco_sim2sim.json")
)
baseline = read(PREVIOUS / "model_100/evaluation/report.json")
current = read(HERE / "model_100/evaluation/report.json")
assert current["plan"] == baseline["plan"] and pair["encoder"]["sha256"] == baseline["pair"]["encoder"]["sha256"]
comparisons = []
for old, new in zip(baseline["records"], current["records"], strict=True):
    assert old["label"] == new["label"]
    if new.get("not_executed"):
        assert old["not_executed"] and new["name"] == "elbow_crawling"
        continue
    comparisons.append(
        dict(
            case=new["label"],
            tf32=old["result"]["completed_transitions"],
            ieee=new["result"]["completed_transitions"],
            requested=new["result"]["requested_transitions"],
            fidelity=new["result"]["motion_fidelity"]["passed"],
            return_completed=new["result"]["return_hold"].get("completed_transitions"),
        )
    )
array_comparisons = 0
for label, report, reference in (
    ("ieee_base", current, None),
    ("ieee_recorded", read(HERE / "model_100/recorded_evaluation/report.json"), current),
    ("tf32_recorded", read(HERE / "tf32_recorded_evaluation/report.json"), baseline),
):
    assert report["plan"] == baseline["plan"] and len(report["records"]) == 11
    for index, row in enumerate(report["records"]):
        if row.get("not_executed"):
            continue
        with np.load(bind(row["trace_path"]), allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        if reference is not None:
            original = reference["records"][index]
            assert {key: value for key, value in row["result"].items() if key != "policy_input_trace"} == original[
                "result"
            ]
            with np.load(bind(original["trace_path"]), allow_pickle=False) as archive:
                assert set(arrays) == set(archive.files) | {
                    "policy_encoder267",
                    "policy_history930",
                    "policy_raw23",
                    "policy_decoder994",
                    "policy_inference_returned",
                }
                for key in archive.files:
                    np.testing.assert_array_equal(arrays[key], archive[key])
                    array_comparisons += 1
            assert arrays["policy_inference_returned"].all()
            for key in ("policy_encoder267", "policy_history930", "policy_raw23", "policy_decoder994"):
                assert arrays[key].dtype == np.float32 and np.isfinite(arrays[key]).all()
        with np.load(bind(row["source"], row["source_sha256"]), allow_pickle=False) as source:
            assert len(source["joint_pos"]) - 11 == row["result"]["requested_transitions"]
        assert (
            row["result"]["compiled_native_model_sha256"]
            == baseline["records"][index]["result"]["compiled_native_model_sha256"]
        )
        engine = audit_engine_trace(arrays)
        assert engine["passed"]
        for key in ("qpos", "qvel"):
            before, after = arrays[f"physics_pre_{key}"], arrays[f"physics_post_{key}"]
            assert np.isfinite(before).all() and np.isfinite(after).all()
            np.testing.assert_array_equal(before[1:], after[:-1])
        np.testing.assert_array_equal(arrays["physics_effort"], arrays["physics_generalized_actuator_force"])
        ratio = float(np.max(np.abs(arrays["physics_effort"]) / (0.95 * 0.25 * np.asarray(profile.effort))))
        assert ratio <= 1 + 1e-10
        traces.append(
            dict(
                run=label,
                case=row["label"],
                physics_steps=engine["physics_steps_checked"],
                max_outer_effort_ratio=ratio,
            )
        )
stream_comparison = read(HERE / "recorded_stream_precision.json")
assert stream_comparison["ieee_anchor_reconstruction"]["matches_actual_training"] is True
assert stream_comparison["ieee_anchor_reconstruction"]["cache_sha256"] == runs[1]["anchor_cache_sha256"]
assert len(stream_comparison["records"]) == 6
regression = read(HERE / "regression_report.json")
assert regression["exit_code"] == 0
tests = ET.parse(bind(HERE / "regression.xml")).getroot().find("testsuite").attrib
assert tests["tests"] == "572" and tests["failures"] == tests["errors"] == tests["skipped"] == "0"
for path in list(inputs):
    bind(path)
dump(
    HERE / "integrity_report.json",
    dict(
        kind="g1_true23_ieee_training_recorded_stream_integrity_v1",
        inputs=inputs,
        bound_file_count=len(inputs),
        prior_inputs_preserved=len(old_audit["inputs"]),
        mismatches=[],
        runs=runs,
        comparisons=comparisons,
        traces=traces,
        original_simulator_arrays_exactly_preserved=array_comparisons,
        regression=tests,
        original_training_configuration_changed_only_by_explicit_precision=True,
        initial_actor_and_critic_exactly_preserved=True,
        failed_attempt_training_updates=0,
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
            exact_original_arrays=array_comparisons,
            tests=tests["tests"],
            mismatches=0,
        )
    ),
    flush=True,
)
