"""SIM-only measured-leg feedback around an immutable SONIC policy.

This is an always-on hybrid controller experiment, not a new learned policy or
hardware configuration. It adds bounded reference-minus-measured leg error to
requested PD targets. Effective closed-loop joint feedback therefore changes;
physical gains, limits, projection, timing and reference poses do not change.
All actual network outputs, corrections and decoder inputs are recorded.
"""

import argparse
import copy
from pathlib import Path

import numpy as np

from gear_sonic.scripts.diagnose_g1_true23_root_error_scale import RootErrorScalePolicy, experiment_contract
from gear_sonic.utils.g1_23dof_contract import MUJOCO_TO_ISAACLAB_DOF
from gear_sonic.utils.g1_true23_source_action_codec import SOURCE_SCALE_NATIVE_IL23

KIND = "g1_true23_bounded_measured_leg_feedback_counterfactual_v1"
MAXIMUM_TARGET_CORRECTION_RAD = 0.1


def leg_contract(scale, root_scale):
    if type(scale) not in (float, int) or scale not in (0, 0.5, 1):
        raise ValueError("leg error scale must be one of the explicit SIM factors0,0.5,1")
    return dict(
        kind=KIND,
        leg_error_gain=float(scale),
        maximum_target_correction_rad=MAXIMUM_TARGET_CORRECTION_RAD,
        correction="clip(gain*(received_q1_leg_angles-current_measured_leg_angles),-0p1,+0p1)",
        hardware_joint_indices=list(range(12)),
        source_reference="already_received_horizon_q1_age180ms",
        measured_joint_source="current_simulator_native23_qpos_not_buffered_history",
        waist_and_arm_policy_outputs_unchanged=True,
        existing_target_and_motor_limits_preserved=True,
        physical_pd_gains_modified=False,
        effective_closed_loop_joint_feedback_modified=scale != 0,
        checkpoint_weights_modified=False,
        source_frames_or_timing_changed=False,
        robot_state_rewrites=False,
        controller_switching_or_fallback=False,
        always_on_hybrid_controller=scale != 0,
        root_input_counterfactual=experiment_contract(root_scale),
        inherited_released_model_raw23_trace_is_post_correction_source_units=True,
        unmodified_network_output_trace="uncorrected_model_raw23",
        eligible_parent_for_training_continuation=False,
        dynamic_feasibility_or_teleop_qualification=False,
        hardware_authorized=False,
        deployment_ready=False,
        simulator_qualified=False,
    )


class LegFeedbackPolicy(RootErrorScalePolicy):
    def __init__(self, policy, scale, root_scale):
        super().__init__(policy, root_scale)
        self.leg_contract = leg_contract(scale, root_scale)
        self.leg_scale = np.float32(scale)
        self.measured = None
        self.leg_records = []

    def set_measured_joints(self, qpos):
        qpos = np.asarray(qpos)
        if self.measured is not None or qpos.shape != (30,) or not np.isfinite(qpos).all():
            raise ValueError("leg feedback requires exactly one fresh finite native23 state per inference")
        self.measured = qpos[7:19].astype(np.float32).copy()

    def infer(self, encoder, history, feedback):
        if self.measured is None or encoder.shape != (267,) or encoder.dtype != np.float32:
            raise ValueError("leg feedback requires fresh state and exact received-source encoder267")
        measured, self.measured = self.measured, None
        reference = encoder[12:24].copy()  # q1, hardware left-six then right-six.
        if not np.isfinite(reference).all():
            raise ValueError("received leg reference must be finite")
        raw, decoder = super().infer(encoder, history, feedback)
        correction = np.clip(
            self.leg_scale * (reference - measured), -MAXIMUM_TARGET_CORRECTION_RAD, MAXIMUM_TARGET_CORRECTION_RAD
        )
        target_delta = np.zeros(23, dtype=np.float32)
        target_delta[:12] = correction
        order = np.asarray(MUJOCO_TO_ISAACLAB_DOF)
        mask = order < 12
        changed = raw.copy()
        if self.leg_scale != 0:
            changed[mask] += target_delta[order][mask] / np.asarray(SOURCE_SCALE_NATIVE_IL23, np.float32)[mask]
        self.leg_records.append((raw.copy(), measured, reference, correction))
        return changed, decoder

    def take_records(self):
        records = super().take_records()
        names = (
            "uncorrected_model_raw23",
            "leg_measured_position12",
            "leg_received_reference12",
            "leg_target_correction12",
        )
        for index, name in enumerate(names):
            records[name] = np.asarray([row[index] for row in self.leg_records], dtype=np.float32).reshape(
                -1, 23 if index == 0 else 12
            )
        self.leg_records = []
        return records


def main(argv=None):
    from gear_sonic.scripts import evaluate_g1_true23_root_feedback as evaluate
    from gear_sonic.teleop import buffered_source_simulation as buffered
    from gear_sonic.utils import g1_true23_generalist_benchmark as benchmark
    from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file

    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--leg-error-gain", type=float, required=True)
    parser.add_argument("--position-error-scale", type=float, required=True)
    parser.add_argument("--reference-timing", choices=("received_source_horizon_200ms_v1",), required=True)
    parser.add_argument("--maximum-controls", type=int)
    parser.add_argument("--training-model-counterfactual")
    args, rest = parser.parse_known_args(argv)
    if args.maximum_controls is not None or args.training_model_counterfactual is not None:
        parser.error("leg feedback experiment requires complete nominal-physics lifecycles")
    rest.extend(("--reference-timing", args.reference_timing))
    contract = leg_contract(args.leg_error_gain, args.position_error_scale)
    implementation = Path(__file__).resolve()
    implementation_sha = sha256_file(implementation)
    contract["implementation"] = {str(implementation): implementation_sha}
    original_load, original_write = evaluate.load_root_feedback_pair, evaluate.write_json
    original_run, original_adapter = benchmark.run_reference_diagnostic, buffered.BufferedSourceSimulationAdapter

    class Adapter(original_adapter):
        def infer(self, policy, encoder, history, *, control_index, **state):
            policy.set_measured_joints(state["measured_qpos"])
            result = super().infer(policy, encoder, history, control_index=control_index, **state)
            # The selected q1 is already received, never a later source sample.
            np.testing.assert_array_equal(
                policy.leg_records[-1][2], self.source["joint_pos"][10 + control_index, :12]
            )
            return result

        def contract(self):
            return {**super().contract(), "leg_feedback_counterfactual": contract}

    def load(*a, **kw):
        policy, identity = original_load(*a, **kw)
        return LegFeedbackPolicy(policy, args.leg_error_gain, args.position_error_scale), {
            **identity,
            "leg_feedback_counterfactual": contract,
        }

    def run(*a, **kw):
        result, arrays = original_run(*a, **kw)
        records = kw["policy"].take_records()
        calls = len(records["uncorrected_model_raw23"])
        if not result["completed_controls"] <= calls <= result["completed_controls"] + 1:
            raise ValueError("leg feedback must record all actual decoder calls and corrections")
        arrays.update(records)
        result.update(
            leg_feedback_counterfactual=contract,
            decoder_calls_recorded=calls,
            decoder_calls_not_followed_by_completed_control=calls - result["completed_controls"],
        )
        return result, arrays

    def write(path, value):
        if sha256_file(implementation) != implementation_sha:
            raise ValueError("leg feedback implementation changed during execution")
        value = copy.deepcopy(value)
        value["leg_feedback_counterfactual"] = contract
        if "kind" in value:
            value["unmodified_evaluator_report_kind"] = value["kind"]
            value["kind"] = KIND
        value["eligible_parent_for_training_continuation"] = False
        value["hardware_authorized"] = value["deployment_ready"] = value["simulator_qualified"] = False
        return original_write(path, value)

    evaluate.load_root_feedback_pair, evaluate.write_json = load, write
    benchmark.run_reference_diagnostic, buffered.BufferedSourceSimulationAdapter = run, Adapter
    try:
        return evaluate.main(rest)
    finally:
        evaluate.load_root_feedback_pair, evaluate.write_json = original_load, original_write
        benchmark.run_reference_diagnostic, buffered.BufferedSourceSimulationAdapter = (
            original_run,
            original_adapter,
        )


if __name__ == "__main__":
    raise SystemExit(main())
