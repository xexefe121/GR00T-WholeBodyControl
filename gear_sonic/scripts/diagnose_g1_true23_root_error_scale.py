"""Full-lifecycle SIM-only ablation of the learned position-feedback branch.

This is explicitly NOT the trained policy's normal input contract, a new trained
checkpoint, or an eligible continuation baseline. Physical PD gains, references,
encoder, history, velocity features and all 23 actions remain unchanged. Record
the actual modified decoder inputs separately from the raw measured feedback.
"""

import argparse
import copy
from pathlib import Path

import numpy as np

EXPERIMENT_KIND = "g1_true23_root_error_input_scale_counterfactual_v1"


def experiment_contract(scale):
    if type(scale) not in (float, int) or scale not in (0, 1, 2, 4):
        raise ValueError("root error scale must be one of the explicit SIM factors0,1,2,4")
    return dict(
        kind=EXPERIMENT_KIND,
        position_error_input_scale=float(scale),
        modified_decoder_root_input_columns=[0, 1, 2],
        velocity_input_columns_unchanged=[3, 4, 5, 6, 7, 8],
        encoder_and_proprioception_inputs_unchanged=True,
        checkpoint_weights_modified=False,
        actuator_pd_gains_or_limits_modified=False,
        source_reference_or_timing_modified=False,
        measured_robot_state_rewritten=False,
        normal_trained_decoder_input_contract_preserved=scale == 1,
        actual_decoder_root_inputs_separately_recorded=True,
        equivalent_first_affine_weight_column_scale_if_later_folded=True,
        eligible_parent_for_training_continuation=False,
        hardware_authorized=False,
        deployment_ready=False,
        simulator_qualified=False,
    )


class RootErrorScalePolicy:
    def __init__(self, policy, scale):
        self.policy = policy
        self.contract = experiment_contract(scale)
        self.scale = np.float32(scale)
        self.received, self.actual = [], []

    def infer(self, encoder, history, feedback):
        if feedback.shape != (9,) or feedback.dtype != np.float32 or not np.isfinite(feedback).all():
            raise ValueError("root input ablation requires finite float32 nine-component feedback")
        actual = feedback.copy()
        actual[:3] *= self.scale
        self.received.append(feedback.copy())
        self.actual.append(actual.copy())
        return self.policy.infer(encoder, history, actual)

    def take_records(self):
        result = dict(
            policy_received_root_feedback9=np.asarray(self.received, dtype=np.float32).reshape(-1, 9),
            actual_policy_root_feedback9=np.asarray(self.actual, dtype=np.float32).reshape(-1, 9),
        )
        self.received, self.actual = [], []
        return result


def disclosed_report(value, contract):
    result = copy.deepcopy(value)
    result["root_input_counterfactual"] = contract
    if "kind" in result:
        result["unmodified_evaluator_report_kind"] = result["kind"]
        result["kind"] = EXPERIMENT_KIND
    result["eligible_parent_for_training_continuation"] = False
    result["hardware_authorized"] = result["deployment_ready"] = result["simulator_qualified"] = False
    return result


def main(argv=None):
    from gear_sonic.scripts import evaluate_g1_true23_root_feedback as evaluate

    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--position-error-scale", type=float, required=True)
    parser.add_argument("--reference-timing", choices=("received_source_horizon_200ms_v1",), required=True)
    parser.add_argument("--maximum-controls", type=int)
    parser.add_argument("--training-model-counterfactual")
    args, rest = parser.parse_known_args(argv)
    contract = experiment_contract(args.position_error_scale)
    if args.maximum_controls is not None or args.training_model_counterfactual is not None:
        parser.error("root input ablation requires complete lifecycles on unchanged nominal physics")
    rest.extend(("--reference-timing", args.reference_timing))
    from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file

    implementation = Path(__file__).resolve()
    implementation_sha = sha256_file(implementation)
    contract["implementation"] = {str(implementation): implementation_sha}
    original_load, original_write = evaluate.load_root_feedback_pair, evaluate.write_json
    # run_reference_diagnostic is imported inside evaluate.main; intercept the
    # defining module temporarily before that import, within this SIM process.
    from gear_sonic.utils import g1_true23_generalist_benchmark as benchmark

    original_run = benchmark.run_reference_diagnostic

    def load(*a, **kw):
        policy, identity = original_load(*a, **kw)
        return RootErrorScalePolicy(policy, args.position_error_scale), {
            **identity,
            "root_input_counterfactual": contract,
        }

    def run(*a, **kw):
        result, arrays = original_run(*a, **kw)
        records = kw["policy"].take_records()
        calls = len(records["actual_policy_root_feedback9"])
        if not result["completed_controls"] <= calls <= result["completed_controls"] + 1:
            raise ValueError("root ablation must record every actual decoder input")
        arrays.update(records)
        result["root_input_counterfactual"] = contract
        result["decoder_calls_recorded"] = calls
        result["decoder_calls_not_followed_by_completed_control"] = calls - result["completed_controls"]
        result["runtime_adapter"]["actual_policy_root_feedback_input_modified"] = args.position_error_scale != 1
        return result, arrays

    def write(path, value):
        if sha256_file(implementation) != implementation_sha:
            raise ValueError("root input experiment implementation changed during execution")
        return original_write(path, disclosed_report(value, contract))

    evaluate.load_root_feedback_pair, evaluate.write_json = load, write
    benchmark.run_reference_diagnostic = run
    try:
        return evaluate.main(rest)
    finally:
        evaluate.load_root_feedback_pair, evaluate.write_json = original_load, original_write
        benchmark.run_reference_diagnostic = original_run


if __name__ == "__main__":
    raise SystemExit(main())
