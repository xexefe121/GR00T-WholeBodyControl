"""Bounded offline adaptation with real native23 and original29 MuJoCo models."""

from __future__ import annotations

from pathlib import Path
import json

import mujoco
import numpy as np
import pytest

from gear_sonic.utils import g1_23dof_task_space_retarget as ik
from gear_sonic.utils.g1_23dof_contract import HARDWARE_23_JOINT_NAMES
from gear_sonic.utils.g1_23dof_safe_target_transform import SAFE_TARGET_DEFAULT_Q_HARDWARE
from gear_sonic.utils.g1_true23_generalist_retarget import (
    AdaptationLimits,
    _candidate_assessment,
    _refine_root_reference,
    _reduce_excursion,
    _resample,
    adapt_offline_motion,
    task_space_fidelity,
    validate_named_motion,
)
from gear_sonic.scripts.retarget_g1_true23_generalist_offline import main
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file


@pytest.fixture(scope="module")
def models():
    root = Path(__file__).resolve().parents[2]
    return ik.load_models(root / ik.DEFAULT_SOURCE_MODEL, root / ik.DEFAULT_TARGET_MODEL)


def _source(source_model: mujoco.MjModel, frames: int = 4) -> dict[str, np.ndarray]:
    names = ik._model_layout(source_model).joint_names
    defaults = dict(zip(HARDWARE_23_JOINT_NAMES, SAFE_TARGET_DEFAULT_Q_HARDWARE, strict=True))
    pose = np.asarray([defaults.get(name, 0.0) for name in names])
    pose[names.index("waist_roll_joint")] = 0.15
    pose[names.index("waist_pitch_joint")] = -0.10
    pose[names.index("left_wrist_pitch_joint")] = 0.20
    pose[names.index("right_wrist_yaw_joint")] = -0.20
    return {
        "joint_names": np.asarray(names),
        "joint_pos": np.repeat(pose[None], frames, axis=0),
        "root_pos_w": np.tile([0.0, 0.0, 0.8], (frames, 1)),
        "root_quat_wxyz": np.tile([1.0, 0.0, 0.0, 0.0], (frames, 1)),
        "fps": np.asarray([50.0]),
        "timestamps_s": 3.0 + np.arange(frames) / 50,
        "contact_flags": np.ones((frames, 2), dtype=bool),
    }


def test_named_reordering_preserves_all_29_axes(models):
    source = _source(models[0])
    reversed_source = {
        **source,
        "joint_names": source["joint_names"][::-1],
        "joint_pos": source["joint_pos"][:, ::-1],
    }
    normalized = validate_named_motion(reversed_source, models[0])
    np.testing.assert_array_equal(normalized["joint_pos"], source["joint_pos"])
    assert normalized["joint_pos"].shape[1] == 29


def test_requested_choreography_preserves_source_limit_excess_without_clipping(models):
    source = _source(models[0])
    layout = ik._model_layout(models[0])
    index = layout.joint_names.index("waist_roll_joint")
    source["joint_pos"][:, index] = layout.upper[index] + 0.04
    with pytest.raises(ValueError, match="source-model joint limits"):
        validate_named_motion(source, models[0])
    requested = validate_named_motion(source, models[0], allow_source_limit_excess=True)
    np.testing.assert_array_equal(requested["joint_pos"], source["joint_pos"])
    result = adapt_offline_motion(
        source_model=models[0],
        target_model=models[1],
        arrays=source,
        source_role="requested_choreography",
        limits=AdaptationLimits(duration_scales=(1.0,), excursion_scales=(1.0,), ik_iterations=1),
    )
    assert not result.report["source_joint_targets_clipped"]
    assert result.report["source_joint_limit_excess_frame_count"] == 4
    assert result.report["source_joint_limit_excess_max_rad"]["waist_roll_joint"] == pytest.approx(0.04)
    assert not result.report["dynamic_feasibility_verified"]


def test_planned_trace_never_substitutes_recorded_policy_poses(models):
    from gear_sonic.scripts.retarget_g1_true23_generalist_planned_trace import planned_named_source

    source = _source(models[0])
    planned = np.concatenate((source["root_pos_w"], source["root_quat_wxyz"], source["joint_pos"]), axis=1)
    result = planned_named_source(
        {"planned_qpos50": planned, "pre_qpos": planned + 100, "command_time_s": np.arange(4) * 0.02}, models[0]
    )
    np.testing.assert_array_equal(result["joint_pos"], planned[:, 7:])
    with pytest.raises(KeyError):
        planned_named_source({"pre_qpos": planned, "command_time_s": np.arange(4) * 0.02}, models[0])


@pytest.mark.parametrize(
    "mutation,error",
    [
        (lambda a: a.update(joint_names=a["joint_names"][:23]), "29"),
        (lambda a: a.update(joint_names=np.repeat("knee", 29)), "29"),
        (lambda a: a.update(joint_pos=a["joint_pos"][:1]), "frames"),
        (lambda a: a.update(fps=np.asarray([0.0])), "fps"),
        (lambda a: a.update(root_pos_w=np.zeros((1, 3))), "shapes"),
        (lambda a: a["root_pos_w"].__setitem__((0, 0), np.nan), "nonfinite"),
        (lambda a: a["root_quat_wxyz"].__setitem__((0, 0), 2.0), "normalized"),
        (lambda a: a["joint_pos"].__setitem__((0, 0), 100.0), "joint limits"),
        (lambda a: a.update(timestamps_s=np.asarray([0.0, 0.02, 0.01, 0.06])), "timestamps"),
        (lambda a: a.update(contact_flags=a["contact_flags"].astype(float)), "boolean"),
    ],
)
def test_boundary_rejects_ambiguous_or_incomplete_source(models, mutation, error):
    source = _source(models[0])
    mutation(source)
    with pytest.raises(ValueError, match=error):
        validate_named_motion(source, models[0])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_duration_scale": 2.01},
        {"max_excursion_reduction": 0.21},
        {"duration_scales": (0.5,)},
        {"excursion_scales": (0.79,)},
        {"target_fps": float("nan")},
        {"maximum_output_frames": 1},
        {"ik_iterations": True},
    ],
)
def test_configuration_cannot_silently_expand_adaptation(kwargs):
    with pytest.raises(ValueError):
        AdaptationLimits(**kwargs)


def test_full_source_endpoints_and_contact_timeline_survive_retime(models):
    source = validate_named_motion(_source(models[0]), models[0])
    source["contact_flags"][2:, 0] = False
    retimed, time_map, scale = _resample(source, 2.0, AdaptationLimits())
    assert len(time_map) == 7
    assert scale == pytest.approx(2.0)
    assert time_map[0] == source["timestamps_s"][0]
    assert time_map[-1] == source["timestamps_s"][-1]
    np.testing.assert_array_equal(retimed["joint_pos"][[0, -1]], source["joint_pos"][[0, -1]])
    assert retimed["contact_flags"][:, 0].tolist() == [True, True, True, True, False, False, False]


def test_joint_angle_reduction_is_not_treated_as_task_space_proof():
    count = len(ik.DEFAULT_TASKS)
    original = np.zeros((3, count, 3))
    original[:, :, 0] = np.asarray([0.0, 0.5, 1.0])[:, None]
    adapted = original * 0.7
    fidelity = task_space_fidelity(original, adapted, adapted)
    assert all(value["adaptation_distortion_fraction"] == pytest.approx(0.3) for value in fidelity.values())
    assert all(value["achieved_to_adapted_m"]["max"] == 0 for value in fidelity.values())
    assert all(value["achieved_to_original_m"]["max"] == pytest.approx(0.3) for value in fidelity.values())


def test_static_task_cannot_drift_under_excursion_percentage():
    original = np.zeros((3, len(ik.DEFAULT_TASKS), 3))
    adapted = original + 0.01
    fidelity = task_space_fidelity(original, adapted, adapted)
    assert all(value["adaptation_distortion_fraction"] > 0.2 for value in fidelity.values())


def test_reduction_keeps_initial_pose_and_does_not_mutate_source(models):
    original = _source(models[0])
    original["joint_pos"][1:, 0] += 0.05
    before = original["joint_pos"].copy()
    candidate = _reduce_excursion(original, 0.8)
    np.testing.assert_array_equal(original["joint_pos"], before)
    np.testing.assert_array_equal(candidate["joint_pos"][0], before[0])
    assert candidate["joint_pos"][1, 0] - before[0, 0] == pytest.approx(0.04)


def test_offline_lookahead_explicitly_rejected_for_live_input(models):
    with pytest.raises(ValueError, match="causal live adapter"):
        adapt_offline_motion(
            source_model=models[0], target_model=models[1], arrays=_source(models[0]), input_mode="live"
        )


def test_real_models_use_all_23_actuators_and_preserve_complete_request(models):
    source_model, target_model = models
    source = _source(source_model)
    missing = [index for index, name in enumerate(source["joint_names"]) if name not in HARDWARE_23_JOINT_NAMES]
    source["joint_pos"][:, missing] *= 0.2
    result = adapt_offline_motion(
        source_model=source_model,
        target_model=target_model,
        arrays=source,
        limits=AdaptationLimits(duration_scales=(1.0,), excursion_scales=(1.0,), ik_iterations=12),
    )
    assert result.report["source_frame_count"] == 4
    assert len(result.report["attempts"]) == 1
    attempt = result.report["attempts"][0]
    assert "ik_summary" in attempt, attempt
    assert attempt["ik_summary"]["frame_count"] == 4
    assert attempt["output_frames"] == 4
    assert attempt["ik_summary"]["hard_position_limit_violation_count"] == 0
    assert attempt["ik_summary"]["trajectory_constraint_relaxation_count"] == 0
    assert not result.report["dynamic_feasibility_verified"]
    assert not result.report["causal_live_adapter"]
    assert result.report["full_clip_future_access"]
    assert result.accepted, attempt["failures"]
    assert result.arrays["joint_pos"].shape == (4, 23)
    assert result.arrays["source_joint_pos_resampled"].shape == (4, 29)
    assert tuple(result.arrays["joint_names"]) == tuple(HARDWARE_23_JOINT_NAMES)
    assert result.arrays["source_time_map_s"][-1] == pytest.approx(3.06)
    np.testing.assert_array_equal(result.arrays["contact_flags"], np.ones((4, 2), dtype=bool))


def test_already_feasible_pose_does_not_require_artificial_improvement(models):
    source = _source(models[0])
    missing = [index for index, name in enumerate(source["joint_names"]) if name not in HARDWARE_23_JOINT_NAMES]
    source["joint_pos"][:, missing] = 0
    result = adapt_offline_motion(
        source_model=models[0],
        target_model=models[1],
        arrays=source,
        limits=AdaptationLimits(duration_scales=(1.0,), excursion_scales=(1.0,)),
    )
    assert result.accepted, result.report["attempts"]
    attempt = result.report["attempts"][0]
    assert "no mean task-space improvement" in attempt["ik_summary"]["kinematic_gate_failures"]
    assert attempt["no_mean_improvement_does_not_disqualify_already_feasible_motion"]
    assert not attempt["failures"]
    assert attempt["fidelity"]["right_hand"]["achieved_to_original_m"]["p95"] == pytest.approx(0.084)


def test_real_unreachable_candidate_retains_failed_hand_metric(models):
    result = adapt_offline_motion(
        source_model=models[0],
        target_model=models[1],
        arrays=_source(models[0]),
        limits=AdaptationLimits(duration_scales=(1.0,), excursion_scales=(1.0,), ik_iterations=12),
    )
    assert not result.accepted and result.arrays is None
    assert any("right_hand" in failure for failure in result.report["attempts"][0]["failures"])
    assert result.report["attempts"][0]["fidelity"]["right_hand"]["achieved_to_adapted_m"]["p95"] > 0.10


def test_rejected_candidate_never_truncates_to_successful_prefix(models):
    source = _source(models[0], frames=6)
    result = adapt_offline_motion(
        source_model=models[0],
        target_model=models[1],
        arrays=source,
        limits=AdaptationLimits(duration_scales=(1.0,), excursion_scales=(1.0,), maximum_output_frames=4),
    )
    assert not result.accepted and result.arrays is None
    assert result.report["source_frame_count"] == 6
    assert "bounded output frame count" in result.report["attempts"][0]["failures"][0]


def test_cli_writes_hash_bound_named_motion_exclusively(tmp_path, models):
    source = _source(models[0])
    missing = [index for index, name in enumerate(source["joint_names"]) if name not in HARDWARE_23_JOINT_NAMES]
    source["joint_pos"][:, missing] = 0
    source_path = tmp_path / "test-only-source.npz"
    np.savez_compressed(source_path, **source)
    root = Path(__file__).resolve().parents[2]
    output = tmp_path / "adaptation"
    args = [
        "--input",
        str(source_path),
        "--source-model",
        str(root / ik.DEFAULT_SOURCE_MODEL),
        "--target-model",
        str(root / ik.DEFAULT_TARGET_MODEL),
        "--output-directory",
        str(output),
    ]
    assert main(args) == 0
    report = json.loads((output / "report.json").read_text())
    assert report["accepted"]
    assert report["input_bindings"][str(source_path)] == sha256_file(source_path)
    motion_path = output / report["output"]["path"]
    assert report["output"]["sha256"] == sha256_file(motion_path)
    with np.load(motion_path, allow_pickle=False) as archive:
        assert archive["joint_pos"].shape == (4, 23)
        assert archive["source_time_map_s"][-1] == pytest.approx(3.06)
    with pytest.raises(FileExistsError):
        main(args)


def test_cli_rejects_missing_named_joints_without_motion_output(tmp_path, models):
    source = _source(models[0])
    source.pop("joint_names")
    source_path = tmp_path / "test-only-incomplete.npz"
    np.savez_compressed(source_path, **source)
    root = Path(__file__).resolve().parents[2]
    output = tmp_path / "rejected"
    assert (
        main(
            [
                "--input",
                str(source_path),
                "--source-model",
                str(root / ik.DEFAULT_SOURCE_MODEL),
                "--target-model",
                str(root / ik.DEFAULT_TARGET_MODEL),
                "--output-directory",
                str(output),
            ]
        )
        == 2
    )
    report = json.loads((output / "report.json").read_text())
    assert not report["accepted"]
    assert "joint_names" in report["input_error"]
    assert not list(output.glob("*.npz"))


def test_missing_contacts_inferred_once_before_retiming(models, monkeypatch):
    source = _source(models[0])
    source.pop("contact_flags")
    real_infer = ik.infer_foot_contacts
    calls = []

    def observed_infer(feet, **kwargs):
        calls.append((len(feet), kwargs["fps"]))
        return real_infer(feet, **kwargs)

    monkeypatch.setattr(ik, "infer_foot_contacts", observed_infer)
    result = adapt_offline_motion(
        source_model=models[0],
        target_model=models[1],
        arrays=source,
        limits=AdaptationLimits(duration_scales=(1.0, 2.0), excursion_scales=(1.0,), maximum_output_frames=3),
    )
    assert calls == [(4, 50.0)]
    assert not result.report["contact_flags_supplied"]
    assert not result.report["contact_schedule_reinferred_after_adaptation"]
    assert result.report["contact_schedule_source"] == "original29_height_velocity_heuristic_before_adaptation"


def test_specific_whole_horizon_seed_failure_uses_full_all23_alternative(models, monkeypatch):
    source = _source(models[0])
    missing = [index for index, name in enumerate(source["joint_names"]) if name not in HARDWARE_23_JOINT_NAMES]
    source["joint_pos"][:, missing] = 0
    original_solver = ik.retarget_trajectory
    calls = []

    def fail_only_whole_horizon(**kwargs):
        calls.append({key: value.copy() for key, value in kwargs.items() if isinstance(value, np.ndarray)})
        if kwargs["config"].enable_lower_root_feasibility:
            raise RuntimeError("whole-horizon lower/root seed creates new invalid frames; test-only seed fault")
        assert kwargs["config"].optimize_lower_body
        assert not kwargs["config"].allow_acceleration_constraint_relaxation
        return original_solver(**kwargs)

    monkeypatch.setattr(ik, "retarget_trajectory", fail_only_whole_horizon)
    result = adapt_offline_motion(
        source_model=models[0],
        target_model=models[1],
        arrays=source,
        limits=AdaptationLimits(duration_scales=(1.0,), excursion_scales=(1.0,)),
    )
    assert result.accepted, result.report
    assert result.arrays["joint_pos"].shape == (4, 23)
    assert len(calls) == 2
    for key in calls[0]:
        np.testing.assert_array_equal(calls[0][key], calls[1][key])
    attempts = result.report["attempts"][0]["solver_attempts"]
    assert not attempts[0]["completed"] and "seed fault" in attempts[0]["error"]
    assert attempts[1]["completed"] and attempts[1]["completed_frames"] == 4
    first_config, second_config = attempts[0]["config"], attempts[1]["config"]
    assert {key for key in first_config if first_config[key] != second_config[key]} == {
        "enable_lower_root_feasibility"
    }
    assert result.report["attempts"][0]["solver_strategy"] == "sequential_all23_fixed_root"


@pytest.mark.parametrize(
    "message",
    [
        "acceleration constraint became infeasible",
        "retarget result violates configured joint velocity limit",
        "lower/root whole-horizon projection failed independent audit",
    ],
)
def test_other_failures_do_not_trigger_strategy_fallback(models, monkeypatch, message):
    calls = []

    def fail(**kwargs):
        calls.append(kwargs["config"])
        raise RuntimeError(message)

    monkeypatch.setattr(ik, "retarget_trajectory", fail)
    result = adapt_offline_motion(
        source_model=models[0],
        target_model=models[1],
        arrays=_source(models[0]),
        limits=AdaptationLimits(duration_scales=(1.0,), excursion_scales=(1.0,)),
    )
    assert not result.accepted and result.arrays is None
    assert len(calls) == 1
    assert message in result.report["attempts"][0]["failures"][0]


def test_alternative_strategy_does_not_bypass_hand_fidelity_gate(models, monkeypatch):
    original_solver = ik.retarget_trajectory

    def fail_only_whole_horizon(**kwargs):
        if kwargs["config"].enable_lower_root_feasibility:
            raise RuntimeError("whole-horizon lower/root seed creates new invalid frames; test-only seed fault")
        return original_solver(**kwargs)

    monkeypatch.setattr(ik, "retarget_trajectory", fail_only_whole_horizon)
    source = _source(models[0])
    source["joint_pos"][:, list(source["joint_names"]).index("waist_roll_joint")] = 0.4
    result = adapt_offline_motion(
        source_model=models[0],
        target_model=models[1],
        arrays=source,
        limits=AdaptationLimits(duration_scales=(1.0,), excursion_scales=(1.0,)),
    )
    assert not result.accepted and result.arrays is None
    attempt = result.report["attempts"][0]
    assert len(attempt["solver_attempts"]) == 2
    assert attempt["solver_attempts"][1]["completed"]
    assert "fidelity" in attempt
    assert any("hand" in failure for failure in attempt["failures"])


def test_native23_head_needs_root_reference_not_joint_only_weights(models):
    target = models[1]
    layout, data = ik._model_layout(target), mujoco.MjData(target)
    ik._set_configuration(
        target,
        data,
        layout,
        np.array([0, 0, 0.8]),
        np.array([1, 0, 0, 0]),
        np.asarray(SAFE_TARGET_DEFAULT_Q_HARDWARE),
    )
    task = next(task for task in ik.DEFAULT_TASKS if task.name == "head_proxy")
    body = target.body(task.target_body).id
    position, _ = ik._point_pose(data, body, task.target_point)
    translation, rotation = np.zeros((3, target.nv)), np.zeros((3, target.nv))
    mujoco.mj_jac(target, data, translation, rotation, position, body)
    assert np.max(np.abs(translation[:, layout.dof_addresses])) == 0.0
    assert np.max(np.abs(translation[:, 3:6])) > 0.3


def test_root_refinement_is_explicit_single_candidate_only(models):
    with pytest.raises(ValueError, match="exactly one bounded candidate"):
        adapt_offline_motion(
            source_model=models[0],
            target_model=models[1],
            arrays=_source(models[0]),
            root_reference_refinement=True,
        )
    with pytest.raises(ValueError, match="explicit boolean"):
        adapt_offline_motion(
            source_model=models[0], target_model=models[1], arrays=_source(models[0]), root_reference_refinement=1
        )


def test_real_root_refinement_repairs_head_and_reaudits_serialized_full_path(models):
    source = _source(models[0], frames=4)
    source["joint_pos"][:, list(source["joint_names"]).index("waist_roll_joint")] = 0.3
    previous = ik.retarget_trajectory(
        source_model=models[0],
        target_model=models[1],
        root_pos_w=source["root_pos_w"],
        root_quat_wxyz=source["root_quat_wxyz"],
        source_joint_pos_hardware=source["joint_pos"],
        fps=50,
        contact_flags=source["contact_flags"],
        config=ik.RetargetConfig(enable_lower_root_feasibility=False, optimize_lower_body=True),
    )
    before = {key: values.copy() for key, values in previous.diagnostics.items()}
    original_joints = source["joint_pos"].copy()
    receipt = {}
    refined = _refine_root_reference(models[0], models[1], source, previous, receipt)
    assert refined.joint_pos_hardware.shape == (4, 23)
    report = receipt["fit_report"]
    assert report["path_constraints"]["passed"] and report["serialized_path_constraints"]["passed"]
    assert report["frames_dropped"] == 0
    assert not report["source_goals_replaced"] and not report["contacts_reinferred"]
    assert not report["root_reference_variables_are_actuators"]
    for key, values in before.items():
        np.testing.assert_array_equal(previous.diagnostics[key], values)
        if key.endswith("_before_m") or key.endswith("_before_rad") or key == "weighted_task_error_before":
            np.testing.assert_array_equal(refined.diagnostics[key], values)
    np.testing.assert_array_equal(source["joint_pos"], original_joints)
    np.testing.assert_array_equal(refined.desired_task_pos_w, previous.desired_task_pos_w)
    np.testing.assert_array_equal(refined.contact_flags, source["contact_flags"])
    assert np.mean(refined.diagnostics["task_head_proxy_position_error_after_m"]) < (
        np.mean(previous.diagnostics["task_head_proxy_position_error_after_m"]) * 0.5
    )
    summary, _, failures = _candidate_assessment(refined, refined.desired_task_pos_w, AdaptationLimits())
    assert summary["constraints"]["measured_velocity_abs_max_rad_s"] <= 5.0 + 1e-6
    assert summary["constraints"]["measured_acceleration_abs_max_rad_s2"] <= 80.0 + 1e-6
    assert summary["hard_position_limit_violation_count"] == 0
    # Passing a numerical fit never overrides the original per-frame gate.
    if not np.all(refined.expert_valid_mask()):
        assert "not every requested frame passes protected-task and trajectory bounds" in failures


def test_failed_root_refinement_preserves_original_rejection_evidence(models, monkeypatch):
    from gear_sonic.utils import g1_true23_generalist_retarget as adapter

    def failed_refinement(*args):
        raise RuntimeError("test-only strict QP failure")

    monkeypatch.setattr(adapter, "_refine_root_reference", failed_refinement)
    source = _source(models[0])
    source["joint_pos"][:, list(source["joint_names"]).index("waist_roll_joint")] = 0.4
    result = adapt_offline_motion(
        source_model=models[0],
        target_model=models[1],
        arrays=source,
        limits=AdaptationLimits(duration_scales=(1.0,), excursion_scales=(1.0,)),
        root_reference_refinement=True,
    )
    assert not result.accepted and result.arrays is None
    attempt = result.report["attempts"][0]
    assert attempt["before_root_refinement"]["failures"]
    assert attempt["before_root_refinement"]["fidelity"] == attempt["fidelity"]
    assert not attempt["solver_attempts"][-1]["completed"]
    assert "strict QP failure" in attempt["solver_attempts"][-1]["error"]


def test_root_refinement_never_runs_on_already_forbidden_task_distortion(models, monkeypatch):
    from gear_sonic.utils import g1_true23_generalist_retarget as adapter

    real_assessment = adapter._candidate_assessment

    def forbidden_assessment(*args):
        summary, fidelity, failures = real_assessment(*args)
        return (
            summary,
            fidelity,
            [*failures, "right_foot: adaptation exceeds 20% source task-space excursion bound"],
        )

    def forbidden_refinement(*args):
        pytest.fail("forbidden source adaptation must not reach root refinement")

    monkeypatch.setattr(adapter, "_candidate_assessment", forbidden_assessment)
    monkeypatch.setattr(adapter, "_refine_root_reference", forbidden_refinement)
    result = adapt_offline_motion(
        source_model=models[0],
        target_model=models[1],
        arrays=_source(models[0]),
        limits=AdaptationLimits(duration_scales=(1.0,), excursion_scales=(0.8,)),
        root_reference_refinement=True,
    )
    assert not result.accepted and result.arrays is None
    assert "already exceeds bound" in result.report["attempts"][0]["root_refinement_skipped"]
