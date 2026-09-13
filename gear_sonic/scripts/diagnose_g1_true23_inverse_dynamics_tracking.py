"""SIM-only received-reference full-body controller through the existing referee.

This is a privileged model-based diagnostic, not a learned SONIC policy. Zero
strength is an unchanged-policy control. Optional 250 controls test standing
only and cannot qualify a complete lifecycle. No hardware transport exists.
"""

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from gear_sonic.utils.g1_true23_inverse_dynamics_tracker import (
    KIND,
    PD_FEEDFORWARD,
    PD_ONLY,
    InverseDynamicsTracker,
    pd_plus_feedforward_numpy,
    tracker_contract,
)


class TrackingPolicy:
    def __init__(self, policy, tracker, strength, *, actuation_mode=PD_ONLY):
        self.policy, self.tracker, self.strength = policy, tracker, strength
        self.contract = tracker_contract(strength, actuation_mode=actuation_mode)
        self.actuation_mode = actuation_mode
        self.pending, self.records = None, []
        self.feedforward = np.zeros(23)
        self.remaining_substeps = 0
        self.physics_feedforward = []

    def set_received_state(self, qpos, qvel, poses):
        if self.pending is not None:
            raise ValueError("tracker requires exactly one fresh state per inference")
        if self.remaining_substeps:
            raise ValueError("tracker requires exactly ten physical substeps before next control")
        self.feedforward = np.zeros(23)
        self.pending = tuple(np.asarray(value).copy() for value in (qpos, qvel, poses))

    def infer(self, encoder, history, feedback):
        if self.pending is None:
            raise ValueError("tracker missing fresh copied state")
        state, self.pending = self.pending, None
        if encoder.shape != (267,) or encoder.dtype != np.float32:
            raise ValueError("tracker requires original encoder267 contract")
        np.testing.assert_array_equal(encoder[12:24], state[2][1, 7:19].astype(np.float32))
        raw, decoder = self.policy.infer(encoder, history, feedback)
        row = dict(uncorrected_model_raw23=raw.copy(), tracker_received_qpos3x30=state[2].copy())
        self.records.append(row)  # A rejected QP still records the exact network call.
        if self.strength:
            raw, diagnostics = self.tracker.infer(*state)
            row.update(diagnostics)
            feedforward = np.asarray(diagnostics["tracker_feedforward_torque23"], dtype=np.float64)
            if feedforward.shape != (23,) or not np.isfinite(feedforward).all():
                raise ValueError("tracker returned invalid joint feedforward")
            if self.actuation_mode == PD_ONLY and np.any(feedforward):
                raise ValueError("PD-only controller cannot use joint feedforward")
            self.feedforward = feedforward.copy()
        row["actual_tracker_source_raw23"] = raw.copy()
        self.remaining_substeps = 10
        return raw, decoder

    def apply_pd(self, target, q, dq, profile):
        if self.remaining_substeps <= 0:
            raise ValueError("tracker cannot reuse stale torque without fresh successful inference")
        result = pd_plus_feedforward_numpy(target, q, dq, profile, self.feedforward)
        self.physics_feedforward.append(self.feedforward.copy())
        self.remaining_substeps -= 1
        return result

    def take_records(self):
        result = {}
        keys = {key for row in self.records for key in row}
        for key in keys:
            rows = [(i, row[key]) for i, row in enumerate(self.records) if key in row]
            result[key] = np.asarray([value for _, value in rows])
            # Successful solver records can be shorter by one failed call.
            result[key + "_call_index"] = np.asarray([i for i, _ in rows], dtype=np.int64)
        self.records = []
        result["tracker_attempted_physics_feedforward_torque23"] = np.asarray(self.physics_feedforward).reshape(
            -1, 23
        )
        self.physics_feedforward = []
        # End this diagnostic case, including a failed partial case. Never reuse
        # held torque across independent simulator resets or QP failures.
        self.pending, self.remaining_substeps = None, 0
        self.feedforward = np.zeros(23)
        self.tracker.assert_original_unchanged()
        return result


def main(argv=None):
    from gear_sonic.scripts import evaluate_g1_true23_root_feedback as evaluate
    from gear_sonic.teleop import buffered_source_simulation as buffered
    from gear_sonic.utils import g1_true23_generalist_benchmark as benchmark
    from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file
    from gear_sonic.utils.g1_true23_generalist_source_closure import collect_local_source_closure
    from gear_sonic.utils.g1_true23_native_model_actuation import NativeModelActuationProfile
    from gear_sonic.utils.g1_true23_reference_floor import motion_qpos

    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--controller-strength", type=float, required=True)
    parser.add_argument("--actuation-mode", choices=(PD_ONLY, PD_FEEDFORWARD), default=PD_ONLY)
    parser.add_argument("--standing-smoke-controls", type=int, choices=(250,))
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--contact-step-reference", type=Path, required=True)
    parser.add_argument("--reference-timing", choices=("received_source_horizon_200ms_v1",), required=True)
    parser.add_argument("--maximum-controls", type=int)
    parser.add_argument("--training-model-counterfactual")
    args, rest = parser.parse_known_args(argv)
    if args.maximum_controls is not None or args.training_model_counterfactual is not None:
        parser.error("tracker permits only full nominal lifecycles or explicit standing smoke")
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
    contract = tracker_contract(args.controller_strength, args.standing_smoke_controls, args.actuation_mode)
    root = Path(__file__).resolve().parents[2]
    report_path = args.contact_step_reference.resolve(strict=True)
    report = json.loads(report_path.read_text())
    if args.standing_smoke_controls is not None:
        phase = next(row for row in report["timeline"]["phases"] if row["name"] == "initial_standing")
        if phase["frame_start"] != 11 or phase["frame_stop"] != 261:
            raise ValueError("standing smoke requires the unchanged 250-control initial-standing phase")
    _, model, _ = evaluate.prepare_true23_model(args.asset_root / benchmark.MODEL, root / benchmark.PHYSICS)
    actuation = NativeModelActuationProfile.from_sim_config(root / benchmark.PHYSICS)
    paths = [report_path, *collect_local_source_closure(root, [Path(__file__)]).files]
    bindings = {str(path): sha256_file(path) for path in paths}
    contract["implementation_inputs"] = bindings
    original_load, original_write = evaluate.load_root_feedback_pair, evaluate.write_json
    original_run, original_adapter = benchmark.run_reference_diagnostic, buffered.BufferedSourceSimulationAdapter

    class Adapter(original_adapter):
        def __init__(self, motion, virtual_vr, pulses=()):
            super().__init__(motion, virtual_vr, pulses)
            self.tracker_reference = motion_qpos(model, motion)

        def infer(self, policy, encoder, history, *, control_index, **state):
            frame = 10 + control_index
            if frame + 1 > 19 + control_index or frame + 1 >= len(self.tracker_reference):
                raise ValueError("tracker requested unavailable source poses")
            policy.set_received_state(
                state["measured_qpos"], state["measured_qvel"], self.tracker_reference[frame - 1 : frame + 2]
            )
            return super().infer(policy, encoder, history, control_index=control_index, **state)

        def contract(self):
            return {**super().contract(), "inverse_dynamics_counterfactual": contract}

    def load(*a, **kw):
        policy, identity = original_load(*a, **kw)
        tracker = InverseDynamicsTracker(model, actuation, actuation_mode=args.actuation_mode)
        return TrackingPolicy(policy, tracker, args.controller_strength, actuation_mode=args.actuation_mode), {
            **identity,
            "inverse_dynamics_counterfactual": contract,
        }

    def run(*a, **kw):
        if args.standing_smoke_controls is not None:
            if kw.get("maximum_controls") is not None:
                raise ValueError("standing smoke cannot override another prefix")
            kw["maximum_controls"] = args.standing_smoke_controls
        policy = kw["policy"]
        original_pd = benchmark.native_model_pd_numpy
        benchmark.native_model_pd_numpy = policy.apply_pd
        try:
            result, arrays = original_run(*a, **kw)
        finally:
            benchmark.native_model_pd_numpy = original_pd
        remaining = policy.remaining_substeps
        records = policy.take_records()
        attempts = records.pop("tracker_attempted_physics_feedforward_torque23")
        steps = len(arrays["physics_time"])
        if not steps <= len(attempts) <= steps + 1:
            raise ValueError("joint feedforward must record every physical torque request")
        if result["failure"] is None and (remaining or steps != 10 * result["completed_controls"]):
            raise ValueError("successful tracking requires exactly ten integrated substeps per control")
        arrays["physics_feedforward_torque23"] = attempts[:steps]
        arrays["tracker_unintegrated_feedforward_torque23"] = attempts[steps:]
        calls = len(records.get("uncorrected_model_raw23", []))
        if not result["completed_controls"] <= calls <= result["completed_controls"] + 1:
            raise ValueError("tracker must record every network call including solver rejection")
        arrays.update(records)
        result.update(inverse_dynamics_counterfactual=contract, decoder_calls_recorded=calls)
        if contract["actuator_command_law_changed"]:
            result["unchanged_baseline_actuator_contract"] = result["actuator_contract"]
            result["actuator_contract"] = {
                **result["actuator_contract"],
                "kind": "g1_true23_sim_only_pd_plus_bounded_joint_feedforward_v1",
                "pd": "kp*(held_target-measured_q)-kd*measured_dq+held_joint_feedforward",
                "feedforward_not_encoded_in_sonic_previous_action": True,
                "not_drop_in_sonic_or_hardware_compatible": True,
            }
        return result, arrays

    def write(path, value):
        for source, digest in bindings.items():
            if sha256_file(Path(source)) != digest:
                raise ValueError("tracker implementation or reference changed during experiment")
        value = copy.deepcopy(value)
        value["inverse_dynamics_counterfactual"] = contract
        if "kind" in value:
            value["unmodified_evaluator_report_kind"], value["kind"] = value["kind"], KIND
        if args.standing_smoke_controls is not None:
            value["maximum_controls"] = args.standing_smoke_controls
            value["diagnostic_standing_prefix_only"] = True
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
