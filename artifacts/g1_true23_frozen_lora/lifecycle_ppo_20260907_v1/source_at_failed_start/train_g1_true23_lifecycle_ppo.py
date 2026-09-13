"""Experimental full-lifecycle PPO in the exact native CPU evaluation simulator.

Preserve SONIC's frozen encoder/base decoder and train only decoder LoRA plus
a fresh critic. This is not MJLab resume, does not emit deployment artifacts,
and does not reproduce native Unitree FSM ownership or physical gantry forces.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import onnxruntime as ort
import torch

from gear_sonic.scripts import evaluate_g1_true23_deployment_envelope as envelope
from gear_sonic.scripts.evaluate_g1_true23_motion_ppo import evaluation_plan
from gear_sonic.scripts.record_g1_sonic_original29_baseline import dump
from gear_sonic.trl.mjlab.frozen_platform_lora_actor import FrozenPlatformTrue23Core, _tensor_state_sha256
from gear_sonic.trl.mjlab.frozen_platform_lora_runner import load_frozen_platform_lora_checkpoint
from gear_sonic.utils.g1_sonic_cpp_parameters import file_sha256
from gear_sonic.utils.g1_true23_actuation_profile import NativeSupportActuationProfile
from gear_sonic.utils.g1_true23_clean_mujoco_teleop import UnitreeZeroVelocityFallbackPolicy
from gear_sonic.utils.g1_true23_lifecycle_ppo import LifecyclePPO, LifecyclePolicy, make_critic
from gear_sonic.utils.g1_true23_lifecycle_rollout import case_options, collect_attempt, lifecycle_training_plan
from gear_sonic.utils.g1_true23_policy_input_trace import run_recorded_case
from gear_sonic.utils.g1_true23_standing_initialization import read_standing_initialization
from gear_sonic.utils.g1_true23_standing_retention import StandingOutputAnchor, load_training_states
from gear_sonic.utils.g1_true23_training_precision import ieee_training_precision

FLAGS = dict(hardware_authorized=False, deployment_ready=False, promotion_eligible=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "asset-root",
        "full-request-report",
        "stationary-report",
        "motor-health-snapshot",
        "warm-start",
        "source-checkpoint",
        "actor-checkpoint",
        "standing-bootstrap-report",
        "standing-teacher-report",
        "output-dir",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--expected-actor-sha256", required=True)
    parser.add_argument("--updates", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args(argv)
    if not 1 <= args.updates <= 1000 or not 0 <= args.seed < 2**31:
        parser.error("requires 1..1000 real updates and a bounded integer seed")
    root = Path(__file__).resolve().parents[2]
    assets, output = args.asset_root.resolve(strict=True), args.output_dir.resolve()
    if args.output_dir.is_symlink() or output.exists():
        raise ValueError("lifecycle training requires a separate new output directory")
    inputs = {}

    def bind(path, expected=None):
        path = Path(path).resolve(strict=True)
        digest = file_sha256(path)
        if digest != inputs.get(str(path), digest) or (expected is not None and digest != expected):
            raise ValueError(f"lifecycle input changed: {path}")
        inputs[str(path)] = digest
        return path

    source_checkpoint = bind(args.actor_checkpoint, args.expected_actor_sha256)
    prior = load_frozen_platform_lora_checkpoint(source_checkpoint)
    suite = json.loads(bind(args.full_request_report).read_text())
    stationary = json.loads(bind(args.stationary_report).read_text())
    if stationary.get("stationary_only") is not True:
        raise ValueError("stationary reference must remain explicitly separate")
    for report in (suite, stationary):
        for path, expected in report["inputs"].items():
            bind(path, expected)
    plan = evaluation_plan(suite, bind(args.stationary_report.parent / "stationary_reference.npz"))
    training_plan = lifecycle_training_plan(plan)
    for case in plan:
        if not case["unavailable"]:
            bind(case["source"], case["source_sha256"])
    measured = envelope.load_measured_initial_state(bind(args.motor_health_snapshot))
    profile = NativeSupportActuationProfile.from_sim_config(bind(root / envelope.PHYSICS))
    model_sha = stationary["records"][0]["result"]["compiled_native_model_sha256"]
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    balance = UnitreeZeroVelocityFallbackPolicy(
        bind(
            assets
            / "artifacts/external/unitree_rl_mjlab/deploy/robots/g1/config/policy/velocity/v0/exported/policy.onnx"
        ),
        session_options=options,
    )
    standing_payload, standing_descriptor = read_standing_initialization(bind(args.standing_bootstrap_report))
    standing_data, standing_inputs = load_training_states(bind(args.standing_teacher_report), standing_descriptor)
    for path, expected in standing_inputs.items():
        bind(path, expected)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    with ieee_training_precision() as (precision, guard):
        core = FrozenPlatformTrue23Core(
            warm_start_path=bind(args.warm_start),
            source_checkpoint_path=bind(args.source_checkpoint),
            lora_rank=prior["adapter_contract"]["lora_rank"],
            lora_alpha=prior["adapter_contract"]["lora_alpha"],
        )
        if core.adapter_contract() != prior["adapter_contract"]:
            raise ValueError("lifecycle actor differs from checked prior frozen platform")
        core.load_lora_state_dict(prior["adapter_state_dict"], strict=True)
        if core.merged_true23_policy_sha256(core.initial_std) != prior["merged_true23_policy_sha256"]:
            raise ValueError("lifecycle actor initialization differs from prior policy")
        core = core.to(args.device)
        critic = make_critic(args.device)
        anchor = StandingOutputAnchor(core, standing_data, standing_payload, batch_size=128, seed=args.seed)
        learner = LifecyclePPO(core, critic, anchor, guard=guard, seed=args.seed)
        for name, module in list(sys.modules.items()):
            path = getattr(module, "__file__", None)
            if name.startswith("gear_sonic.") and path and Path(path).suffix == ".py":
                bind(path)
        bind(Path(__file__))
        output.mkdir(parents=True, exist_ok=False)
        contract = dict(
            kind="g1_true23_native_cpu_full_lifecycle_ppo_v1",
            simulator="native CPU MuJoCo acceptance-test implementation",
            compiled_native_model_sha256=model_sha,
            control_hz=50,
            physics_hz=500,
            complete_original_evaluation_plan=plan,
            training_plan=training_plan,
            training_source_frames_trimmed=False,
            training_reset=(
                "historical joint/IMU snapshot with inferred foot contact height, then 250 actual balance controls"
            ),
            mid_motion_state_or_history_reset=False,
            active_policy_controls_all_native_joints=23,
            acquisition_and_return_controller=(
                "existing 29-to-23 zero-velocity compatibility actor, not native Unitree FSM"
            ),
            return_controls_requested=250,
            actor_action_noise="unchanged released diagonal Gaussian std",
            added_sensor_noise=False,
            sensor_noise_differs_from_mjlab_training=True,
            reset_sampling_and_rewards_differ_from_prior_mjlab_training=True,
            ppo_only_receives_actual_active_policy_actions=True,
            return_outcome_is_terminal_assessment_not_fictitious_policy_actions=True,
            per_completed_control_reward=(
                "1 + mean(exp(-(four measured tracking errors / existing fidelity scales)^2))"
            ),
            terminal_reward=(
                "base 25 for full-motion fidelity, otherwise -50; always add 10 * completed_return/250; "
                "add 15 only for full-motion fidelity plus full guarded return"
            ),
            ppo=dict(
                learning_rate=5e-6,
                epochs=4,
                minibatch_size=128,
                gamma=0.99,
                gae_lambda=0.95,
                clip=0.2,
                value_coefficient=1,
                standing_retention_weight=10,
                gradient_norm_clip_per_network=1,
            ),
            initial_actor=dict(
                path=str(source_checkpoint),
                sha256=args.expected_actor_sha256,
                prior_update_count=prior["update_count"],
                prior_lineage_sha256=prior["lineage_sha256"],
            ),
            prior_updates_counted_as_new=False,
            fresh_critic_and_adam=True,
            no_exact_resume_claim=True,
            hardware_telemetry_is_historical_not_fresh=True,
            gantry_forces_modeled=False,
            **FLAGS,
        )
        dump(output / "started.json", dict(inputs=dict(inputs), precision=precision, contract=contract))
        initial_actor_hash = _tensor_state_sha256(core.lora_state_dict())
        initial_critic_hash = _tensor_state_sha256(critic.state_dict())
        evaluation_records, learning_records = [], []

        def save_checkpoint():
            guard()
            core.assert_frozen_platform_unchanged()
            if learner.update_count == 0 and learner.optimizer.state:
                raise ValueError("initial lifecycle checkpoint must have fresh empty Adam")
            if learner.minibatch_count:
                counts = {int(state["step"].item()) for state in learner.optimizer.state.values()}
                if counts != {learner.minibatch_count} or len(learner.optimizer.state) != len(learner.parameters):
                    raise ValueError("lifecycle optimizer counters/parameter scope disagree")
            path = output / f"lifecycle_ppo_model_{learner.update_count}.pt"
            payload = dict(
                kind="g1_true23_cpu_lifecycle_lora_ppo_checkpoint_v1",
                contract=contract,
                adapter_contract=core.adapter_contract(),
                adapter_state_dict=core.lora_state_dict(),
                adapter_state_sha256=_tensor_state_sha256(core.lora_state_dict()),
                critic_state_dict={key: value.detach().cpu() for key, value in critic.state_dict().items()},
                critic_state_sha256=_tensor_state_sha256(critic.state_dict()),
                optimizer_state_dict=learner.optimizer.state_dict(),
                new_update_count=learner.update_count,
                actual_minibatches=learner.minibatch_count,
                actual_active_actions=learner.active_action_count,
                source_std=core.initial_std,
                merged_true23_policy_sha256=core.merged_true23_policy_sha256(core.initial_std),
                inputs=dict(inputs),
                mjlab_resume_checkpoint=False,
                **FLAGS,
            )
            with path.open("xb") as stream:
                torch.save(payload, stream)
            bind(path)

        def evaluate(label):
            destination = output / label
            destination.mkdir()
            rows = []
            for index, case in enumerate(plan):
                if case["unavailable"]:
                    rows.append({**case, "not_executed": True, **FLAGS})
                    continue
                policy = LifecyclePolicy(core, critic, guard=guard, seed=args.seed + index, stochastic=False)
                kwargs = case_options(
                    root=root,
                    assets=assets,
                    policy=policy,
                    profile=profile,
                    measured=measured,
                    balance=balance,
                    case=case,
                    training=False,
                )
                result, arrays = run_recorded_case(**kwargs)
                if (
                    result["requested_transitions"] != case["expected_transitions"]
                    or result["compiled_native_model_sha256"] != model_sha
                ):
                    raise ValueError("lifecycle evaluator changed full request or native physics")
                path = destination / f"{case['label']}.npz"
                with path.open("xb") as stream:
                    np.savez_compressed(stream, **arrays)
                bind(path)
                row = {**case, "result": result, "trace_path": str(path), **FLAGS}
                rows.append(row)
                dump(destination / f"{case['label']}.json", row)
                print(
                    json.dumps(
                        dict(
                            evaluation=label,
                            case=case["label"],
                            completed=result["completed_transitions"],
                            requested=result["requested_transitions"],
                            return_completed=result["return_hold"].get("completed_transitions"),
                        )
                    ),
                    flush=True,
                )
            report = dict(
                records=rows,
                new_update_count=learner.update_count,
                original_eight_request_set_preserved=True,
                additional_standing_not_counted_as_motion_success=True,
                torch_ieee_policy_evaluation_not_onnx_or_hardware=True,
                **FLAGS,
            )
            dump(destination / "report.json", report)
            bind(destination / "report.json")
            evaluation_records.append(dict(label=label, report=str(destination / "report.json")))
            return rows

        save_checkpoint()
        initial_evaluation = evaluate("initial_evaluation")
        standing = next(row for row in initial_evaluation if row["label"] == "standing.acquired")
        if not standing["result"]["lifecycle_simulator_screen_passed"]:
            raise ValueError("prior actor failed standing in new arithmetic/runtime; do not start learning")
        choices = [row for row in training_plan if not row["unavailable"]]
        rng = np.random.default_rng(args.seed)
        for update in range(args.updates):
            episodes, rows = [], []
            destination = output / f"rollout_{update}"
            destination.mkdir()
            for index in rng.permutation(len(choices)):
                case = choices[index]
                policy = LifecyclePolicy(
                    core, critic, guard=guard, seed=args.seed + 1000 * update + int(index), stochastic=True
                )
                kwargs = case_options(
                    root=root,
                    assets=assets,
                    policy=policy,
                    profile=profile,
                    measured=measured,
                    balance=balance,
                    case=case,
                    training=True,
                )
                episode, result, arrays = collect_attempt(
                    options=kwargs, case=case, policy_update=learner.update_count, compiled_model_sha256=model_sha
                )
                path = destination / f"{case['label']}.npz"
                with path.open("xb") as stream:
                    np.savez_compressed(stream, **arrays)
                bind(path)
                row = dict(case=case, result=result, trace_path=str(path), **FLAGS)
                dump(destination / f"{case['label']}.json", row)
                rows.append(row)
                episodes.append(episode)
            result = learner.update(episodes)
            learning_records.append({**result, "episodes": rows})
            dump(output / f"update_{learner.update_count}.json", learning_records[-1])
            print(json.dumps(result), flush=True)
            save_checkpoint()
        evaluate("final_evaluation")
        if (
            _tensor_state_sha256(core.lora_state_dict()) == initial_actor_hash
            or _tensor_state_sha256(critic.state_dict()) == initial_critic_hash
        ):
            raise ValueError("lifecycle run did not change both actor and fresh critic")
        for path in list(inputs):
            bind(path)
        dump(
            output / "report.json",
            dict(
                kind="g1_true23_cpu_lifecycle_ppo_experiment_v1",
                contract=contract,
                inputs=inputs,
                actual_new_updates=learner.update_count,
                actual_active_policy_actions=learner.active_action_count,
                actual_minibatches=learner.minibatch_count,
                evaluations=evaluation_records,
                learning=learning_records,
                initial_actor_sha256=initial_actor_hash,
                final_actor_sha256=_tensor_state_sha256(core.lora_state_dict()),
                frozen_platform_and_std_unchanged=True,
                deployment_artifacts_emitted=False,
                **FLAGS,
            ),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
