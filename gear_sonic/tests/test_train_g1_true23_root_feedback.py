from pathlib import Path

import pytest

from gear_sonic.scripts.train_g1_true23_root_feedback import (
    feedback_training_contract,
    make_parser,
    validate_bounds,
    validate_campaign_reference_count,
)


def arguments(mode="regression", *extra):
    return make_parser().parse_args(
        [
            mode,
            "--source-checkpoint",
            "unused.pt",
            "--spans",
            "spans.json",
            "--curriculum-stage",
            "acquisition",
            "--curriculum-directory",
            "inputs",
            "--run-dir",
            "train",
            *extra,
        ]
    )


def test_regression_explicitly_distinct_from_smoke_and_production():
    args = arguments()
    validate_bounds(args)
    assert args.iterations == 100 and args.num_envs == 16
    contract = feedback_training_contract(args, {"stage": "acquisition"})
    assert contract["experiment_class"] == "local_regression_only"
    assert contract["deployment_ready"] is False
    assert contract["physical_estimator_qualified"] is False
    assert contract["reward_task_target_frame"] == "q10_fixed_world"
    assert contract["critic_observes_root_feedback"] is True


@pytest.mark.parametrize(
    "profile, weight",
    [
        ("legacy_root_tracking", -10.0),
        ("root_and_posture_v1", -10.0),
        ("root_and_upper_posture_v2", -10.0),
        ("root_and_upper_world_priority_v3", -30.0),
        ("root_and_upper_feet_sole_world_v5", -10.0),
    ],
)
def test_top_level_reward_metadata_matches_constructed_objective(profile, weight):
    args = arguments("regression", "--objective-profile", profile)
    contract = feedback_training_contract(args, {})
    assert contract["root_world_tracking_error_weight"] == weight
    assert contract["root_world_tracking_error_weight"] == contract["objective_profile_contract"].get(
        "root_world_tracking_error_weight", -10.0
    )


def test_nominal_scene_is_opt_in_and_explicitly_bound_to_training():
    args = arguments()
    assert args.training_physics_profile == "legacy_training_asset"
    assert "training_physics" not in feedback_training_contract(args, {})
    args = arguments("regression", "--training-physics-profile", "pinned_cpu_referee_scene_v1")
    validate_bounds(args)
    args.training_physics_contract = {"name": args.training_physics_profile, "contract_sha256": "test-binding"}
    assert feedback_training_contract(args, {})["training_physics"] == args.training_physics_contract


@pytest.mark.parametrize(
    "extra",
    [
        ("--iterations", "101"),
        ("--num-envs", "33"),
        ("--session-updates", "101"),
        ("--save-interval", "101"),
        ("--ppo-epochs", "0"),
        ("--ppo-minibatches", "9"),
        ("--reset-position-range-m", "0.16"),
        ("--reset-velocity-range-m-s", "nan"),
        ("--resume", "old.pt", "--initialize-actor-from", "parent.pt"),
    ],
)
def test_regression_bounds(extra):
    with pytest.raises(ValueError):
        validate_bounds(arguments("regression", *extra))


def test_smoke_retains_two_update_bound():
    validate_bounds(arguments("smoke"))
    with pytest.raises(ValueError):
        validate_bounds(arguments("smoke", "--iterations", "3"))


def test_generated_ramp_repair_index_cannot_bypass_evaluated_bank_mode():
    with pytest.raises(ValueError, match="evaluated --reference-bank"):
        validate_bounds(arguments("regression", "--lifecycle-reference-repairs", "repairs.json"))


def test_source_start_registration_requires_evaluated_bank_and_repair_index():
    with pytest.raises(ValueError, match="evaluated bank"):
        validate_bounds(arguments("regression", "--source-start-registration", "once_only_source_start_se2_v1"))
    with pytest.raises(ValueError, match="own explicit"):
        validate_bounds(arguments("campaign", "--allow-source-start-registration-transition"))
    args = arguments(
        "campaign",
        "--reference-bank",
        "--curriculum-stage",
        "lifecycle",
        "--reference-timing",
        "received_source_horizon_200ms_v1",
        "--training-physics-profile",
        "pinned_cpu_referee_scene_v1",
        "--release-source-geometry",
        "source.xml",
        "--release-action-convention",
        "released29_scale_bounded_linear_v2",
        "--continue-from",
        "parent.pt",
        "--continuation-evaluation",
        "evaluations.json",
        "--lifecycle-reference-repairs",
        "repairs.json",
        "--source-start-registration",
        "once_only_source_start_se2_v1",
        "--allow-source-start-registration-transition",
    )
    validate_bounds(args)
    args.allow_reference_bank_transition = True
    with pytest.raises(ValueError, match="own explicit"):
        validate_bounds(args)


def test_source_projection_objective_cannot_be_added_to_legacy_action_semantics():
    with pytest.raises(ValueError, match="source-scaled"):
        validate_bounds(arguments("regression", "--ppo-auxiliary-objective", "source_target_projection_l2_v1"))
    with pytest.raises(ValueError, match="evaluated campaign"):
        validate_bounds(arguments("regression", "--allow-ppo-objective-transition"))


def test_training_retains_audited_mode_and_separate_outputs():
    args = arguments("train")
    assert args.mode == "train"
    args.curriculum_directory = Path("train")
    with pytest.raises(ValueError, match="separate"):
        validate_bounds(args)


def test_campaign_requires_evaluated_parent_and_retains_short_sessions():
    with pytest.raises(ValueError, match="evaluated"):
        validate_bounds(arguments("campaign"))
    args = arguments("campaign", "--continue-from", "parent.pt", "--continuation-evaluation", "report.json")
    validate_bounds(args)
    assert args.iterations == 2000 and args.session_updates == 100
    contract = feedback_training_contract(args, {"stage": "lifecycle"})
    assert contract["experiment_class"] == "evaluated_local_training_campaign"
    assert contract["held_out_policy_generalization_verified"] is False
    assert contract["hardware_authorized"] is False
    args.session_updates = 101
    with pytest.raises(ValueError, match="CPU evaluation"):
        validate_bounds(args)


@pytest.mark.parametrize("extra", [("--iterations", "5001"), ("--num-envs", "33"), ("--resume", "old.pt")])
def test_campaign_limits_and_explicit_continuation(extra):
    with pytest.raises(ValueError):
        validate_bounds(
            arguments(
                "campaign", "--continue-from", "parent.pt", "--continuation-evaluation", "report.json", *extra
            )
        )


def test_production_cannot_use_local_continuation_switch():
    with pytest.raises(ValueError, match="only in explicit campaign"):
        validate_bounds(arguments("train", "--continue-from", "parent.pt"))


def test_bank_count_does_not_weaken_the_legacy_single_reference_guard():
    args = arguments("campaign")
    validate_campaign_reference_count(args, 1)
    with pytest.raises(ValueError, match="explicit audited"):
        validate_campaign_reference_count(args, 4)
    args.reference_bank = True
    for count in (2, 4, 16):
        validate_campaign_reference_count(args, count)
    for count in (0, 1, 17, True, 4.0):
        with pytest.raises(ValueError):
            validate_campaign_reference_count(args, count)


@pytest.mark.parametrize("mode", ["smoke", "regression", "train"])
def test_bank_is_only_available_in_explicit_evaluated_campaign(mode):
    with pytest.raises(ValueError, match="evaluated buffered nominal lifecycle"):
        validate_bounds(arguments(mode, "--reference-bank"))


def test_bank_expansion_without_bank_and_conflicting_reference_changes_are_rejected():
    with pytest.raises(ValueError, match="explicit --reference-bank"):
        validate_bounds(arguments("campaign", "--allow-reference-bank-transition"))
    with pytest.raises(ValueError, match="without another reference change"):
        validate_bounds(arguments("campaign", "--reference-bank", "--allow-reference-transition"))


@pytest.mark.parametrize("mode", ["smoke", "regression", "train"])
def test_reference_transition_cannot_bypass_evaluated_campaign(mode):
    with pytest.raises(ValueError, match="evaluated buffered nominal lifecycle"):
        validate_bounds(arguments(mode, "--allow-reference-transition"))


def test_reference_proof_files_cannot_be_silently_ignored():
    with pytest.raises(ValueError, match="explicit --allow-reference-transition"):
        validate_bounds(arguments("regression", "--previous-reference-report", "old.json"))


def test_reference_transition_requires_both_proofs_and_unchanged_buffered_physics():
    args = arguments(
        "campaign",
        "--continue-from",
        "parent.pt",
        "--continuation-evaluation",
        "new_evaluation.json",
        "--curriculum-stage",
        "lifecycle",
        "--reference-timing",
        "received_source_horizon_200ms_v1",
        "--training-physics-profile",
        "pinned_cpu_referee_scene_v1",
        "--release-source-geometry",
        "source.xml",
        "--release-action-convention",
        "released29_scale_bounded_linear_v2",
        "--allow-reference-transition",
        "--previous-reference-report",
        "old.json",
        "--reference-original-time-audit",
        "new_audit.json",
    )
    validate_bounds(args)
    args.reference_original_time_audit = None
    with pytest.raises(ValueError, match="both proof files"):
        validate_bounds(args)


def test_parallel_batching_is_opt_in_and_changes_only_recorded_simulation_resources():
    args = arguments(
        "campaign",
        "--continue-from",
        "parent.pt",
        "--continuation-evaluation",
        "report.json",
        "--campaign-batching",
        "parallel128_v1",
        "--num-envs",
        "128",
        "--rollout-steps",
        "64",
        "--reference-timing",
        "received_source_horizon_200ms_v1",
        "--curriculum-stage",
        "lifecycle",
        "--training-physics-profile",
        "pinned_cpu_referee_scene_v1",
        "--release-source-geometry",
        "source.xml",
        "--release-action-convention",
        "released29_scale_bounded_linear_v2",
    )
    validate_bounds(args)
    recorded = feedback_training_contract(args, {})["simulation_batching"]
    assert recorded["transitions_per_update"] == 8192
    assert recorded["maximum_updates_between_cpu_evaluations"] == 100
    assert recorded["robot_physics_or_acceptance_limits_changed"] is False
    args.num_envs = 132
    with pytest.raises(ValueError, match="128 environments"):
        validate_bounds(args)
    args.num_envs = 128
    args.session_updates = 101
    with pytest.raises(ValueError, match="CPU evaluation"):
        validate_bounds(args)


@pytest.mark.parametrize("mode", ["regression", "smoke", "train"])
def test_parallel_batching_cannot_bypass_other_training_modes(mode):
    with pytest.raises(ValueError, match="evaluated buffered"):
        validate_bounds(arguments(mode, "--campaign-batching", "parallel128_v1"))


def test_staggered_schedule_is_bound_but_does_not_waive_campaign_requirements():
    args = arguments("regression", "--start-schedule", "staggered_standing_start_v1")
    validate_bounds(args)
    contract = feedback_training_contract(args, {"stage": "lifecycle"})
    assert contract["start_schedule"]["name"] == "staggered_standing_start_v1"
    assert not contract["start_schedule"]["evaluation_schedule_changed"]
    args.allow_start_schedule_transition = True
    with pytest.raises(ValueError, match="evaluated campaign"):
        validate_bounds(args)
