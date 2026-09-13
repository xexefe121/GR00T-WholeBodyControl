"""Bounded, fresh original-intent native23 SIM training; no legacy relabelling.

The existing trainer still owns physical action/optimizer/lineage mechanics.
Only this process installs the new source observations and task objectives.
Old export/live tools must reject this distinct actor until explicitly migrated.
"""

import argparse
import copy
from pathlib import Path

import numpy as np
import torch

from gear_sonic.envs.mjlab import sonic_true23_original_intent as intent
from gear_sonic.scripts import train_g1_true23_root_feedback as legacy
from gear_sonic.trl.mjlab.native23_original_intent_actor import original_intent_compatibility
from gear_sonic.utils.g1_true23_buffered_reference import BUFFERED_TIMING
from gear_sonic.utils.g1_true23_generalist_source_closure import collect_local_source_closure
from gear_sonic.utils.g1_true23_source_action_codec import SOURCE_ACTION_CONVENTION
from gear_sonic.utils.g1_true23_training_precision import write_runtime


def validate_fresh_recipe(args):
    if args.mode not in ("smoke", "regression") or any(
        getattr(args, name, None) is not None for name in ("resume", "continue_from", "initialize_actor_from")
    ):
        raise ValueError(
            "original-intent recipe requires a fresh bounded smoke/regression, not resume or promotion"
        )
    if (
        args.reference_timing != BUFFERED_TIMING
        or args.release_action_convention != SOURCE_ACTION_CONVENTION
        or args.release_source_geometry is None
        or args.training_physics_profile != "pinned_cpu_referee_scene_v1"
        or args.objective_profile != "root_and_upper_feet_world_v4"
        or args.curriculum_stage != "lifecycle"
    ):
        raise ValueError(
            "original-intent recipe requires the explicit buffered, source-scaled, nominal v4 predecessor"
        )


def verify_original_intent_environment(env, spec):
    """Check the actual new observations against separately fed received packets."""
    from gear_sonic.teleop.buffered_source_horizon import ReceivedSourceHorizon
    from gear_sonic.utils.g1_23dof_contract import HARDWARE_23_JOINT_NAMES
    from gear_sonic.utils.g1_true23_buffered_reference import continued_standing_source
    from gear_sonic.utils.g1_true23_original29_reference import task_targets_from_received_vr

    command = env.command_manager.get_term("motion")
    cache = intent._cache(command, spec)
    terms = env.cfg.observations["tokenizer"].terms
    arrays = {name: getattr(command.motion, name).cpu().numpy() for name in legacy.generalist.MOTION_KEYS}
    vr = torch.cat((intent.original_vr_position(env, spec), intent.original_vr_orientation(env, spec)), -1)
    lower = terms["received_source_horizon_lower_body"].func(env)
    orientation = terms["motion_anchor_ori_b"].func(env)
    target, quat, _, _ = intent.task_states(env, spec)
    max_encoder = max_world = max_quaternion = 0.0
    anchors = command.time_steps.cpu().tolist()
    for index, anchor in enumerate(anchors):
        row = next(
            row for row in command._curriculum_spans if row["start"] <= anchor < row["start"] + row["length"]
        )
        start, end = row["start"], row["start"] + row["length"]
        source = continued_standing_source(
            {key: value[start:end] for key, value in arrays.items()}, cache["vr21"][start:end].cpu().numpy()
        )
        buffer = ReceivedSourceHorizon()
        for i in range(anchor - start, anchor - start + 11):
            window = buffer.push(
                source_timestamp_s=i * 0.02,
                arrival_timestamp_s=i * 0.02,
                joint_names=HARDWARE_23_JOINT_NAMES,
                joint_position23=source["joint_pos"][i],
                root_position_w=source["root_position_w"][i],
                root_quaternion_wxyz=source["root_quaternion_wxyz"][i],
                virtual_source_vr21=source["virtual_vr21"][i],
            )
        expected = window.encoder267(command.robot_anchor_quat_w[index].cpu().numpy())
        actual = torch.cat((lower[index], vr[index], orientation[index])).cpu().numpy()
        np.testing.assert_array_equal(actual[:261], expected[:261])
        max_encoder = max(max_encoder, float(np.abs(actual - expected).max()))
        q1 = anchor - start + 1
        world_p, world_q = task_targets_from_received_vr(
            source["root_position_w"][q1 : q1 + 1],
            source["root_quaternion_wxyz"][q1 : q1 + 1],
            source["virtual_vr21"][q1 : q1 + 1],
        )
        world_p += command._env.scene.env_origins[index].cpu().numpy()
        max_world = max(max_world, float(np.abs(world_p[0] - target[index].cpu().numpy()).max()))
        max_quaternion = max(max_quaternion, float(np.abs(world_q[0] - quat[index].cpu().numpy()).max()))
    if max_encoder > 2e-6 or max_world > 2e-6 or max_quaternion > 2e-6:
        raise ValueError("original-intent observation/target phase diverges from received-source protocol")
    return dict(
        kind="native23_original_intent_executed_environment_parity_v1",
        reference_spec_sha256=spec["sha256"],
        environment_count=env.num_envs,
        anchors=anchors,
        encoder_max_abs_error=max_encoder,
        lower_and_VR_bit_exact=True,
        original_q1_world_target_component_max_error_m=max_world,
        original_q1_WXYZ_component_max_error=max_quaternion,
        pair=cache["pair"],
        hand_frame=cache["hand_frame"],
        objective=intent.objective_contract(),
        physics_steps=0,
        physical_state_mutated=False,
        tracking_improvement_proven=False,
        hardware_authorized=False,
        deployment_ready=False,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--original-reference", type=Path, required=True)
    extra, remaining = parser.parse_known_args(argv)
    source_reference = extra.original_reference.resolve(strict=True)
    prior_hooks = legacy.install_hooks
    prior_feedback = legacy.feedback_training_contract

    def install_hooks(args, inputs, curriculum, precision, guard):
        from gear_sonic.envs.mjlab import (
            sonic_true23 as original_task,
            sonic_true23_buffered_source as buffered,
            sonic_true23_causal_history as task,
            sonic_true23_release_compatible as release,
        )
        from gear_sonic.trl.mjlab import config

        validate_fresh_recipe(args)
        spec = intent.make_spec(
            original_reference=source_reference,
            native_motion=args.motion_file,
            source_model=args.release_source_geometry,
            native_model=args.native_model,
        )
        intent.load_reference(spec)  # Reject any dataset/lifecycle mismatch before constructing policy or physics.
        spec_file = args.curriculum_directory / "original_intent.spec.json"
        write_runtime(spec_file, spec)
        args.release_compatibility_contract = original_intent_compatibility(
            spec["files"]["source_model"]["sha256"], spec["files"]["native_model"]["sha256"]
        )

        def feedback(values, current_curriculum):
            result = prior_feedback(values, current_curriculum)
            previous_targets = result["objective_contract"]
            result.update(
                objective_profile=intent.RECIPE,
                objective_profile_contract=intent.objective_contract(),
                original_intent_spec=copy.deepcopy(spec),
                predecessor_objective_profile="root_and_upper_feet_world_v4",
                objective_contract=dict(
                    kind="native23_original_intent_received_q1_objectives_v1",
                    unchanged_lower_body_and_root=previous_targets,
                    upper_body_replacements=intent.objective_contract(),
                ),
                same_objective_or_reference_resume_claimed=False,
                old_live_or_export_compatibility_claimed=False,
                critic_observes_original_q1_vr21=True,
            )
            return result

        legacy.feedback_training_contract = feedback
        base = prior_hooks(args, {**inputs, "original_intent_spec": spec}, curriculum, precision, guard)
        previous_builder = task.make_causal_history_recovery_env_cfg
        task.make_causal_history_recovery_env_cfg = lambda **kw: intent.configure_original_intent_environment(
            previous_builder(**kw), spec
        )
        previous_agent = config.true23_mjlab_ppo_runner_cfg

        def agent():
            cfg = previous_agent()
            cfg.actor.class_name = (
                "gear_sonic.trl.mjlab.native23_original_intent_actor:True23OriginalIntentActorModel"
            )
            cfg.obs_groups["critic"] = ("critic", "root_feedback", "original_intent_value_reference")
            return cfg

        config.true23_mjlab_ppo_runner_cfg = agent
        previous_prime = original_task.prime_sonic_true23_training_environment

        def prime(wrapped, **kwargs):
            result = previous_prime(wrapped, **kwargs)
            env = wrapped.unwrapped
            error = float(intent.ee_height_error(env, spec).max().cpu())
            if error > 0.25:
                raise ValueError("original-intent prime accepted a terminal hand/foot target")
            return {
                **result,
                "legacy_ee_error_field_uses_native_body_origins_not_original_hand_points": True,
                "original_hand_and_foot_height_error_max_m": error,
                "original_intent_spec_sha256": spec["sha256"],
            }

        original_task.prime_sonic_true23_training_environment = prime
        previous_action_check = release.verify_executed_release_environment

        def verify_actions_and_reference(env, source_model_path):
            # Preserve the already tested action/history probes. Their auxiliary
            # old zero-absent cache is NOT the new observation's source proof.
            old = previous_action_check(env, source_model_path)
            action = {
                key: old[key]
                for key in (
                    "native_joint_count",
                    "probe_count",
                    "target_max_abs_error_rad",
                    "normalized_history_max_abs_error",
                    "action_convention",
                    "physical_state_mutated",
                    "physics_steps",
                )
            }
            return dict(
                kind="native23_original_intent_action_and_reference_parity_v1",
                unchanged_action_history_probes=action,
                actual_original_source_reference=verify_original_intent_environment(env, spec),
                legacy_zero_absent_reference_probe_not_used_as_new_source_evidence=True,
                deployment_ready=False,
                hardware_authorized=False,
            )

        release.verify_executed_release_environment = verify_actions_and_reference
        buffered.verify_buffered_source_environment = lambda env: verify_original_intent_environment(env, spec)
        previous_sources = base._source_files

        def sources():
            result = previous_sources()
            roots = [
                Path(__file__),
                Path(intent.__file__),
                legacy.generalist.ROOT / "gear_sonic/trl/mjlab/native23_original_intent_actor.py",
            ]
            result.update(
                collect_local_source_closure(legacy.generalist.ROOT, roots).as_source_files(legacy.generalist.ROOT)
            )
            result["native23_original_intent/reference.spec.json"] = spec_file
            result["native23_original_intent/original_reference.npz"] = source_reference
            return result

        base._source_files = sources
        return base

    legacy.install_hooks = install_hooks
    validate_fresh_recipe(legacy.make_parser().parse_args(remaining))
    return legacy.main(remaining)


if __name__ == "__main__":
    raise SystemExit(main())
