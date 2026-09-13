"""SIM-only test of requested swing-leg posture, not a learned SONIC policy.

Blend only airborne-reference swing-leg targets during generated entry/return.
The reference is already received, all original motion frames still execute,
and physical gains, limits and collision geometry remain unchanged. A zero
strength control must reproduce the unmodified policy. No hardware transport.
"""

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from gear_sonic.utils.g1_23dof_contract import MUJOCO_TO_ISAACLAB_DOF
from gear_sonic.utils.g1_23dof_safe_target_transform import SAFE_TARGET_DEFAULT_Q_HARDWARE
from gear_sonic.utils.g1_true23_source_action_codec import SOURCE_SCALE_NATIVE_IL23

KIND = "g1_true23_generated_swing_reference_target_counterfactual_v1"


def swing_contract(strength):
    if type(strength) not in (int, float) or strength not in (0, 1):
        raise ValueError("swing target probe permits only zero control or full reference blend")
    return dict(
        kind=KIND,
        strength=float(strength),
        activation="smoothstep(clip((minimum_of_four_reference_sole_gaps_m-0p002)/0p018,0,1))",
        active_phases=["acquisition_ramp", "return_ramp"],
        target="blend_existing_source_raw_with_received_q1_swing_leg_pose_in_source_action_units",
        reference_sample="already_received_q1_age180ms",
        reference_only_airborne_gate_is_contact_or_dynamic_proof=False,
        original_source_standing_waist_and_arm_outputs_not_directly_overridden=True,
        actuator_gains_limits_and_existing_projection_unchanged=True,
        source_frames_timing_and_geometry_unchanged=True,
        checkpoint_weights_modified=False,
        modified_controller_not_pure_sonic_policy=strength != 0,
        no_midrun_state_resets=True,
        unmodified_network_output_trace="uncorrected_model_raw23",
        inherited_released_model_raw23_trace_is_post_blend=True,
        eligible_parent_for_training_continuation=False,
        hardware_authorized=False,
        deployment_ready=False,
        simulator_qualified=False,
    )


def swing_weights(gaps, timeline):
    gaps = np.asarray(gaps)
    if gaps.shape != (timeline["total_frames"], 2, 4) or not np.isfinite(gaps).all():
        raise ValueError("swing probe requires every finite reference sole corner")
    enabled = np.zeros(len(gaps), dtype=bool)
    for phase in timeline["phases"]:
        if phase["name"] in ("acquisition_ramp", "return_ramp"):
            enabled[phase["frame_start"] : phase["frame_stop"]] = True
    fraction = np.clip((gaps.min(axis=-1) - 0.002) / 0.018, 0, 1)
    return ((3 * fraction**2 - 2 * fraction**3) * enabled[:, None]).astype(np.float32)


class SwingReferencePolicy:
    def __init__(self, policy, strength):
        self.policy, self.strength = policy, np.float32(strength)
        self.contract, self.pending, self.records = swing_contract(strength), None, []

    def set_received_weights(self, weights):
        weights = np.asarray(weights)
        if (
            self.pending is not None
            or weights.shape != (2,)
            or not np.isfinite(weights).all()
            or np.any(weights < 0)
            or np.any(weights > 1)
        ):
            raise ValueError("swing probe requires one fresh finite reference activation per inference")
        self.pending = weights.astype(np.float32).copy()

    def infer(self, encoder, history, feedback):
        if self.pending is None or encoder.shape != (267,) or encoder.dtype != np.float32:
            raise ValueError("swing probe requires fresh received-source activation and encoder267")
        weights, self.pending = self.pending, None
        reference = encoder[12:24].copy()
        if not np.isfinite(reference).all():
            raise ValueError("swing reference q1 must be finite")
        raw, decoder = self.policy.infer(encoder, history, feedback)
        changed = raw.copy()
        order = np.asarray(MUJOCO_TO_ISAACLAB_DOF)
        leg = order < 12
        effective = np.repeat(weights * self.strength, 6)[order[leg]]
        active = effective > 0
        source_indices = np.flatnonzero(leg)[active]
        if len(source_indices):
            physical = order[source_indices]
            target = (
                reference[physical] - np.asarray(SAFE_TARGET_DEFAULT_Q_HARDWARE, np.float32)[physical]
            ) / np.asarray(SOURCE_SCALE_NATIVE_IL23, np.float32)[source_indices]
            amount = effective[active]
            changed[source_indices] = (1 - amount) * raw[source_indices] + amount * target
        self.records.append((raw.copy(), changed.copy(), reference, weights.copy()))
        return changed, decoder

    def take_records(self):
        result = {
            name: np.asarray([row[i] for row in self.records], dtype=np.float32).reshape(-1, width)
            for i, (name, width) in enumerate(
                (
                    ("uncorrected_model_raw23", 23),
                    ("actual_blended_source_raw23", 23),
                    ("swing_received_reference12", 12),
                    ("swing_reference_activation2", 2),
                )
            )
        }
        self.records = []
        return result


def main(argv=None):
    from gear_sonic.scripts import evaluate_g1_true23_root_feedback as evaluate
    from gear_sonic.scripts.diagnose_g1_true23_stance_foot_cleanup import sole_gaps
    from gear_sonic.teleop import buffered_source_simulation as buffered
    from gear_sonic.utils import g1_true23_generalist_benchmark as benchmark
    from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file
    from gear_sonic.utils.g1_true23_generalist_source_closure import collect_local_source_closure
    from gear_sonic.utils.g1_true23_reference_floor import motion_qpos

    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--swing-reference-strength", type=float, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--contact-step-reference", type=Path, required=True)
    parser.add_argument("--reference-timing", choices=("received_source_horizon_200ms_v1",), required=True)
    parser.add_argument("--maximum-controls", type=int)
    parser.add_argument("--training-model-counterfactual")
    args, rest = parser.parse_known_args(argv)
    if args.maximum_controls is not None or args.training_model_counterfactual is not None:
        parser.error("swing probe requires full lifecycles and unchanged nominal physics")
    rest.extend(
        (
            "--asset-root",
            str(args.asset_root),
            "--contact-step-reference",
            str(args.contact_step_reference),
            "--reference-timing",
            args.reference_timing,
        )
    )
    contract = swing_contract(args.swing_reference_strength)
    root = Path(__file__).resolve().parents[2]
    report_path = args.contact_step_reference.resolve(strict=True)
    report = json.loads(report_path.read_text())
    timeline = report["timeline"]
    _, model, _ = evaluate.prepare_true23_model(args.asset_root / benchmark.MODEL, root / benchmark.PHYSICS)
    paths = [report_path, *collect_local_source_closure(root, [Path(__file__)]).files]
    bindings = {str(path): sha256_file(path) for path in paths}
    contract["implementation_inputs"] = bindings
    original_load, original_write = evaluate.load_root_feedback_pair, evaluate.write_json
    original_run, original_adapter = benchmark.run_reference_diagnostic, buffered.BufferedSourceSimulationAdapter

    class Adapter(original_adapter):
        def __init__(self, motion, virtual_vr, pulses=()):
            super().__init__(motion, virtual_vr, pulses)
            self.swing = swing_weights(sole_gaps(model, motion_qpos(model, motion)), timeline)

        def infer(self, policy, encoder, history, *, control_index, **state):
            frame = 10 + control_index
            if frame > 19 + control_index or frame >= len(self.swing):
                raise ValueError("swing activation requested an unavailable reference frame")
            policy.set_received_weights(self.swing[frame])
            result = super().infer(policy, encoder, history, control_index=control_index, **state)
            np.testing.assert_array_equal(policy.records[-1][2], self.source["joint_pos"][frame, :12])
            return result

        def contract(self):
            return {**super().contract(), "swing_reference_counterfactual": contract}

    def load(*a, **kw):
        policy, identity = original_load(*a, **kw)
        return SwingReferencePolicy(policy, args.swing_reference_strength), {
            **identity,
            "swing_reference_counterfactual": contract,
        }

    def run(*a, **kw):
        result, arrays = original_run(*a, **kw)
        records = kw["policy"].take_records()
        calls = len(records["uncorrected_model_raw23"])
        if not result["completed_controls"] <= calls <= result["completed_controls"] + 1:
            raise ValueError("swing probe must record every decoder call")
        arrays.update(records)
        result.update(swing_reference_counterfactual=contract, decoder_calls_recorded=calls)
        return result, arrays

    def write(path, value):
        for source, digest in bindings.items():
            if sha256_file(Path(source)) != digest:
                raise ValueError("swing experiment implementation or reference changed")
        value = copy.deepcopy(value)
        value["swing_reference_counterfactual"] = contract
        if "kind" in value:
            value["unmodified_evaluator_report_kind"], value["kind"] = value["kind"], KIND
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
