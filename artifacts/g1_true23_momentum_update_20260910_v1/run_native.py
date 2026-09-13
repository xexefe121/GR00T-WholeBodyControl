"""One full native23 trial of momentum-preserving internal source assimilation."""

import importlib.util
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
import torch

from gear_sonic.scripts.record_g1_true23_original_intent import task_metrics
from gear_sonic.utils.g1_23dof_contract import REFERENCE_PROFILE_NORMAL
from gear_sonic.utils.g1_sonic_cpp_parameters import CppParameters
from gear_sonic.utils.g1_true23_generalist_benchmark import MODEL, PHYSICS, run_reference_diagnostic
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file
from gear_sonic.utils.g1_true23_generalist_lifecycle import assess_lifecycle_diagnostic
from gear_sonic.utils.g1_true23_original29_reference import (
    build_original29_reference,
    verify_unmodified_native_pair,
)
from gear_sonic.utils.g1_true23_virtual_source_history import (
    VirtualSourceHistoryAdapter,
    VirtualSourceHistoryPolicy,
)
from gear_sonic.utils.g1_true23_virtual_source_model import MISSING_IL
from gear_sonic.utils.g1_true23_momentum_source_model import MomentumSourceModel as VirtualSourceModel

HERE = Path(__file__).resolve().parent
BASE = Path("/mnt/c/Users/camer/sonic23_sim_artifacts/internet_pico_20260909_v1")
OLD = BASE / "normal_core_pico_v1"
OUTPUT = HERE / "native_actual_v1"
ROOT = Path("/mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof")
ASSETS = Path("/mnt/z/codex/GR00T-WholeBodyControl")
PHASE = "unchanged_referee_post_control_q2"
FLAGS = dict(deployment_ready=False, hardware_authorized=False, candidate_promoted=False)


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def main():
    if OUTPUT.exists():
        raise FileExistsError("one native trial only; refuses implicit retry")
    inputs = {}

    def bind(path, expected=None):
        path = Path(path).resolve(strict=True)
        digest = sha256_file(path)
        if expected is not None and digest != expected:
            raise ValueError("input changed: " + str(path))
        if str(path) in inputs:
            assert inputs[str(path)] == digest
        inputs[str(path)] = digest
        return path

    def arrays(path, expected=None):
        with np.load(bind(path, expected), allow_pickle=False) as archive:
            return {key: archive[key].copy() for key in archive.files}

    def read(path, expected=None):
        return json.loads(bind(path, expected).read_text())

    bind(__file__)
    bind(HERE / "EXPERIMENT.md")
    preflight = read(HERE / "physics_preflight_v1/report.json")
    assert preflight["passed"] and not preflight["native23_dynamics_performed"]
    for path, digest in preflight["inputs"].items():
        bind(path, digest)
    old = read(OLD / "report.json", "112f4f5163ffe165c03910682c7e71226f805322a6535fea12d507d748e3b1d1")
    for path, value in old["inputs"].items():
        bind(path, value)
    old_trace = arrays(OLD / "trace.npz", "2c06267ebc3a5217485fe423eabb978075334665fda7dbc3b0fc98be7754f91e")
    old_attempts = arrays(OLD / "attempts.npz", "a3cd802e98d93b69c3005498402b4e18c9124972ca54467a1838daa031715db8")
    timeline = old["timeline"]
    motion_path = bind(timeline["timeline_path"], timeline["timeline_sha256"])
    motion = arrays(motion_path)
    original = arrays(
        BASE / "pico_freedancing_v1/optical_reference_v2/original29.npz",
        "58c65ea2805cfe740989661fe26d09e9b8d9b5f1e2c5ac0697167074fef82c5e",
    )
    geometry = mujoco.MjModel.from_xml_path(str(bind(ASSETS / "gear_sonic/data/robots/g1/g1_29dof.xml")))
    native = mujoco.MjModel.from_xml_path(str(bind(ASSETS / MODEL)))
    bind(ROOT / PHYSICS)
    reference = build_original29_reference(geometry, original["source_qpos29"])
    for key, value in reference.arrays().items():
        np.testing.assert_array_equal(value, original[key])
    pair = verify_unmodified_native_pair(reference, motion)
    source = read(BASE / "normal29_upstream_action_v1/actual_v1/frozen29_2ms/pico/report.json")
    parameters = CppParameters(read(source["parameter_path"]))
    predictor = VirtualSourceModel(
        bind(source["model_path"], source["model_sha256"]),
        parameters,
        source_effort29=source["actuation"]["effective_sim_effort29"],
    )
    torch.set_num_threads(1)
    print(json.dumps(dict(loading_frozen_normal_core=True, native_physics_started=False)), flush=True)
    policy = VirtualSourceHistoryPolicy(
        warm_start_path=bind(ASSETS / "sonic_release/g1_23dof_rev_1_0_init.pt"),
        source_checkpoint_path=bind(ASSETS / "sonic_release/last.pt"),
        profile=REFERENCE_PROFILE_NORMAL,
    )
    identity = policy.identity()
    assert identity["frozen_parent"] == old["identity"]
    selected = np.linspace(0, len(old_attempts["released_raw23"]) - 1, 64, dtype=int)
    for control in selected:
        raw, decoder = policy.infer(old_attempts["encoder267"][control], old_attempts["history930"][control])
        np.testing.assert_array_equal(raw, old_attempts["released_raw23"][control])
        np.testing.assert_array_equal(decoder, old_attempts["decoder994"][control])
    adapter = VirtualSourceHistoryAdapter(
        motion,
        reference.virtual_vr21,
        predictor=predictor,
        profile=REFERENCE_PROFILE_NORMAL,
        root=ROOT,
        assets=ASSETS,
    )
    for name, module in tuple(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if name.startswith("gear_sonic.") and path and str(path).endswith(".py"):
            bind(path)
    OUTPUT.mkdir()
    write(
        OUTPUT / "started.json",
        dict(
            inputs=inputs.copy(),
            identity=identity,
            zero_model_history_reinference_observations=len(selected),
            baseline_raw23_and_decoder994_bit_exact=True,
            **FLAGS,
        ),
    )
    print(json.dumps(dict(starting_full_lifecycle=True, requested_controls=6530)), flush=True)
    result, trace = run_reference_diagnostic(
        root=ROOT,
        asset_root=ASSETS,
        motion_path=motion_path,
        policy=policy,
        runtime_adapter=adapter,
    )
    attempts = adapter.arrays()
    for filename, value in (
        ("trace.npz", trace),
        ("attempts.npz", attempts),
        ("received_source.npz", adapter.source),
        ("assimilation.npz", predictor.assimilation_arrays()),
    ):
        with (OUTPUT / filename).open("xb") as stream:
            np.savez_compressed(stream, **value)
        bind(OUTPUT / filename)
    write(OUTPUT / "physics_result.json", result)
    bind(OUTPUT / "physics_result.json")
    print(json.dumps(dict(completed=result["completed_controls"], failure=result["failure"])), flush=True)
    for key in (
        "compiled_model_sha256",
        "physics_config_sha256",
        "initial_state_and_history_sha256",
        "kp_hardware",
        "kd_hardware",
        "effort_limit_hardware_nm",
        "requested_controls",
        "state_pose_writes_after_reset",
        "history_resets_during_motion",
    ):
        assert result[key] == old["result"][key], key
    assert result["compiled_model_sha256"] == adapter.preview.model_sha256
    assert policy.identity() == identity
    np.testing.assert_array_equal(attempts["history930"][0], old_attempts["history930"][0])
    np.testing.assert_array_equal(attempts["released_raw23"][0], old_attempts["released_raw23"][0])
    np.testing.assert_array_equal(trace["physics_post_qpos"][:10], old_trace["physics_post_qpos"][:10])
    phase = next(p for p in timeline["phases"] if p["name"] == "source_motion")
    metrics = task_metrics(native, geometry, reference, motion, trace, phase)
    lifecycle = assess_lifecycle_diagnostic(timeline, result, trace)
    completed = result["completed_controls"]
    common = min(completed, old["result"]["completed_controls"])
    common_metrics = {
        family: task_metrics(native, geometry, reference, motion, {"qpos": saved["qpos"][: common + 1]}, phase)
        for family, saved in (("internal_source_model", trace), ("zero_model_baseline", old_trace))
    }
    q = trace["physics_post_qpos"][:, 7:]
    excess = np.maximum(np.maximum(native.jnt_range[1:, 0] - q, q - native.jnt_range[1:, 1]), 0)
    np.testing.assert_array_equal(excess, trace["physics_hard_limit_excess23"])
    effort = np.asarray(result["effort_limit_hardware_nm"])
    evidence = dict(
        kind="native23_momentum_preserving_source_model_full_pico_v1",
        identity=identity,
        result=result,
        timeline=timeline,
        original_reference_pair=pair,
        lifecycle=lifecycle,
        metrics=metrics,
        common_prefix_controls=common,
        common_prefix_only_metrics=common_metrics,
        common_prefix_not_full_motion_score=True,
        actual_hard_range_excess_max_rad=float(excess.max()) if len(q) else None,
        actual_effort_excess_max_nm=max(
            float(np.maximum(np.abs(trace[key]) - effort, 0).max())
            for key in ("applied_torque23", "engine_actuator_force23")
        )
        if len(q)
        else None,
        discarded_internal_raw_action_abs_max=float(np.abs(attempts["released_raw29"][:, MISSING_IL]).max()),
        target_jump_max_rad=float(np.max(np.abs(np.diff(trace["target23"], axis=0)))) if completed > 1 else None,
        source_predictor=predictor.descriptor(),
        inputs=inputs.copy(),
        training_updates=0,
        **FLAGS,
    )
    write(OUTPUT / "report.json", evidence)
    bind(OUTPUT / "report.json")
    helper_path = bind(
        BASE / "saved_failure_localization_v1/run.py",
        "470d207e4076642f224300a9586b9d72965563b77206b62c3536710ec6f32b8b",
    )
    spec = importlib.util.spec_from_file_location("internal_model_saved_physics", helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    physics, _ = helper.replay_physics({"records": [{"result": result}]}, trace, [])
    for path, value in inputs.items():
        assert sha256_file(Path(path)) == value, path
    write(
        OUTPUT / "physics_audit.json",
        dict(
            physics_reintegration=physics,
            saved_applied_target_physics_verified=True,
            fresh_independent_network_reinference=False,
            force_decomposition_performed=False,
            new_policy_trials_during_audit=0,
            inputs=inputs,
            **FLAGS,
        ),
    )
    print(
        json.dumps(
            dict(
                complete=True,
                completed=completed,
                requested=result["requested_controls"],
                source_controls=metrics["source_controls_completed"],
                actual_source_metrics=metrics.get(PHASE),
                common_prefix_metrics={key: value.get(PHASE) for key, value in common_metrics.items()},
                failure=result["failure"],
                physics_verified=True,
                report_sha256=sha256_file(OUTPUT / "report.json"),
                **FLAGS,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
