"""Additional SIM training failure for missed world-position intent.

This does not replace existing failures, change rewards, move the robot, or
relax a replay/hardware limit. The threshold is a training boundary, not an
acceptance threshold. Full-motion replay remains mandatory.
"""

import copy

import torch

from gear_sonic.envs.mjlab.sonic_true23_root_feedback import current_root_reference

THRESHOLD_M = 0.30
TERM_NAME = "world_root_tracking_failure"


def termination_contract():
    return dict(
        kind="native23_additive_world_root_failure_v1",
        term_name=TERM_NAME,
        threshold_m=THRESHOLD_M,
        metric="Euclidean_world_root_position_error_xyz",
        comparison="strict_greater_than",
        nonfinite_fails=True,
        timeout=False,
        reference="held_received_q1_plus_environment_origin",
        measured_phase="unchanged_pre_final_forward_reward_phase_2ms_stale",
        existing_termination_terms_preserved=True,
        reward_equations_weights_and_terminal_penalty_unchanged=True,
        reset_routine_and_reference_sampling_unchanged=True,
        acceptance_threshold_or_physical_limit_changed=False,
        threshold_is_not_motion_success=True,
        hardware_authorized=False,
        deployment_ready=False,
    )


def root_error_and_failure(desired, measured, *, threshold_m=THRESHOLD_M):
    if threshold_m != THRESHOLD_M:
        raise ValueError("this version requires the declared0.30m training boundary")
    if desired.shape != measured.shape or desired.ndim != 2 or desired.shape[-1] != 3:
        raise ValueError("world failure requires matching[env,3] positions")
    error = torch.linalg.vector_norm(desired - measured, dim=-1)
    failure = ~torch.isfinite(error) | (error > threshold_m)
    return error, failure


def world_root_tracking_failure(env, threshold_m=THRESHOLD_M):
    command = env.command_manager.get_term("motion")
    desired, _ = current_root_reference(command)
    measured = command.robot_anchor_pos_w
    error, failure = root_error_and_failure(desired, measured, threshold_m=threshold_m)
    capture = getattr(env, "_world_root_failure_capture", None)
    if capture is not None:
        step = int(env.common_step_counter)
        if capture and step <= capture[-1]["common_step_counter"]:
            raise ValueError("world failure captured twice or control counter regressed")
        capture.append(dict(
            common_step_counter=step,
            desired_position_w=desired.detach().cpu().clone(),
            measured_position_w=measured.detach().cpu().clone(),
            error_m=error.detach().cpu().clone(),
            failure=failure.detach().cpu().clone(),
            reference_q0=command.time_steps.detach().cpu().clone(),
        ))
    return failure


def configure_environment(cfg):
    from mjlab.managers.termination_manager import TerminationTermCfg

    if TERM_NAME in cfg.terminations:
        raise ValueError("world-root failure already installed")
    result = copy.deepcopy(cfg)
    result.terminations[TERM_NAME] = TerminationTermCfg(
        func=world_root_tracking_failure, time_out=False, params=dict(threshold_m=THRESHOLD_M)
    )
    return result


def verify_runtime(env):
    term = env.termination_manager.get_term_cfg(TERM_NAME)
    if term.func is not world_root_tracking_failure or term.time_out is not False:
        raise ValueError("world-root training failure function/timeout differs")
    if term.params != dict(threshold_m=THRESHOLD_M):
        raise ValueError("world-root training boundary differs")
    return termination_contract()
