from pathlib import Path

import pytest

from gear_sonic.scripts.train_g1_true23_root_feedback import (
    feedback_training_contract,
    make_parser,
    validate_bounds,
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


def test_training_retains_audited_mode_and_separate_outputs():
    args = arguments("train")
    assert args.mode == "train"
    args.curriculum_directory = Path("train")
    with pytest.raises(ValueError, match="separate"):
        validate_bounds(args)
