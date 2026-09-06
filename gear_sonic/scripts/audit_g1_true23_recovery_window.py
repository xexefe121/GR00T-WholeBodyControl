"""Exhaustively test early standing return at observed full-dance boundaries.

Re-execute the original historical acquisition and same paired SONIC policy.
Every pre-return state/action must equal its saved full-request trace. Only
the explicit return time changes. No shorter source, synthetic state reset,
controller-limit change, physical FSM command or promotion is allowed here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import onnxruntime as ort

from gear_sonic.scripts import evaluate_g1_true23_deployment_envelope as envelope
from gear_sonic.scripts.record_g1_sonic_original29_baseline import dump
from gear_sonic.utils.g1_sonic_cpp_parameters import file_sha256
from gear_sonic.utils.g1_true23_actuation_profile import NativeSupportActuationProfile
from gear_sonic.utils.g1_true23_clean_mujoco_teleop import UnitreeZeroVelocityFallbackPolicy
from gear_sonic.utils.g1_true23_diagnostic_pair import load_diagnostic_pair
from gear_sonic.utils.g1_true23_interior_target_filter import run_interior_case
from gear_sonic.utils.g1_true23_recovery_window import (
    probe_outcome,
    probe_plan,
    require_full_replay_identity,
    require_prefix_identity,
    summarize_window,
)
from gear_sonic.utils.g1_true23_sim_acquisition import effort_target_interval


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-report", type=Path, required=True)
    parser.add_argument("--expected-parent-sha256", required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    assets, output = args.asset_root.resolve(strict=True), args.output_dir.resolve()
    inputs = {}

    def bind(path, expected=None):
        path = Path(path).expanduser()
        if path.is_symlink():
            raise ValueError("recovery evidence may not be a symlink")
        path = path.resolve(strict=True)
        digest = file_sha256(path)
        if digest != inputs.get(str(path), digest) or (expected is not None and digest != expected):
            raise ValueError(f"recovery source changed: {path}")
        inputs[str(path)] = digest
        return path

    parent = json.loads(bind(args.parent_report, args.expected_parent_sha256).read_text())
    if (
        parent.get("kind")
        not in {
            "g1_true23_motion_ppo_full_request_evaluation_v1",
            "g1_true23_original_v14_native_ieee_full_request_evaluation_v1",
        }
        or parent.get("hardware_authorized") is not False
        or parent.get("deployment_ready") is not False
        or parent.get("full_eight_clip_qualification") is not False
        or parent.get("original_eight_request_set_preserved") is not True
        or len(parent.get("records", [])) != 11
    ):
        raise ValueError("recovery parent must retain the complete unqualified original request set")
    for path, digest in parent["inputs"].items():
        bind(path, digest)
    cases = [row for row in parent["records"] if row["label"] == "happy_dance.historical"]
    if len(cases) != 1:
        raise ValueError("recovery parent lacks its unique historical full dance")
    case = cases[0]
    plan = probe_plan(case)
    source = bind(case["source"], case["source_sha256"])
    with np.load(source, allow_pickle=False) as archive:
        motion = {key: archive[key].copy() for key in archive.files}
    with np.load(bind(case["trace_path"]), allow_pickle=False) as archive:
        baseline = {key: archive[key].copy() for key in archive.files}
    pair = parent["pair"]
    if pair["kind"] == "g1_true23_diagnostic_pair_provenance_v1":
        checked = load_diagnostic_pair(
            bind(pair["encoder"]["report_path"], pair["encoder"]["report_sha256"]),
            bind(pair["decoder"]["report_path"], pair["decoder"]["report_sha256"]),
        )
    elif pair["kind"] == "g1_true23_original_v14_native_ieee_diagnostic_pair_v1":
        from gear_sonic.utils.g1_true23_v14_diagnostic_pair import load_v14_diagnostic_pair

        checked = load_v14_diagnostic_pair(
            bind(pair["checkpoint"]["path"], pair["checkpoint"]["sha256"]),
            bind(pair["metadata"]["path"], pair["metadata"]["sha256"]),
        )
    else:
        raise ValueError("unsupported recovery parent policy provenance")
    if json.loads(json.dumps(checked)) != pair:
        raise ValueError("recovery policy pair differs from full-request parent")
    policy = envelope._policy(
        bind(pair["encoder"]["path"], pair["encoder"]["sha256"]),
        bind(pair["decoder"]["path"], pair["decoder"]["sha256"]),
        pair["decoder"]["sha256"],
        encoder_hash=pair["encoder"]["sha256"],
    )
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    balance = UnitreeZeroVelocityFallbackPolicy(
        bind(
            assets
            / "artifacts/external/unitree_rl_mjlab/deploy/robots/g1/config/policy/velocity/v0/exported/policy.onnx"
        ),
        session_options=options,
    )
    snapshot = case["result"]["measured_initial_state"]
    measured = envelope.load_measured_initial_state(bind(snapshot["source"], snapshot["source_sha256"]))
    profile = NativeSupportActuationProfile.from_sim_config(bind(root / envelope.PHYSICS))
    bind(Path(__file__))

    def bind_imports():
        for name, module in list(sys.modules.items()):
            path = getattr(module, "__file__", None)
            if name.startswith("gear_sonic.") and path and Path(path).suffix == ".py":
                bind(path)

    bind_imports()
    output.mkdir(parents=True, exist_ok=False)
    flags = dict(hardware_authorized=False, deployment_ready=False, promotion_eligible=False)
    started = dict(
        kind="g1_true23_early_return_window_diagnostic_v1",
        inputs=dict(inputs),
        pair=pair,
        plan=plan,
        full_request_parent=str(args.parent_report.resolve()),
        full_request_parent_sha256=args.expected_parent_sha256,
        original_full_request_records=parent["records"],
        original_full_request_plan=parent["plan"],
        full_source_requested_controls=case["result"]["requested_transitions"],
        original_eight_request_set_preserved_as_failed_parent=True,
        early_abort_probes_are_not_complete_dances=True,
        historical_state_is_not_fresh_robot_state=True,
        compatibility_standing_actor_is_not_native_unitree_fsm=True,
        recovery_duration_s=5.0,
        no_training_or_policy_change=True,
        **flags,
    )
    dump(output / "started.json", started)
    records, outcomes = [], []
    for entry in plan:
        result, arrays = run_interior_case(
            fraction=1.0,
            root=root,
            asset_root=assets,
            policy=policy,
            motion={name: value.copy() for name, value in motion.items()},
            kp=np.asarray(profile.kp),
            kd=np.asarray(profile.kd),
            joint_scale=np.ones(23),
            ankle_effort=35.0,
            slew_rate=5.0,
            initial_state="measured",
            maximum_steps=entry["stop_after_controls"],
            measured_state=measured,
            startup_hold_s=5.0,
            return_hold_s=5.0,
            transition_policy=balance,
            align_reference_start=True,
            project_transition_effort=True,
            project_active_effort=True,
            stateful_native_controller=True,
            trace_active_actuation=True,
        )
        for key in (
            "compiled_native_model_sha256",
            "gain_kp_hardware",
            "gain_kd_hardware",
            "effort_limit_hardware_nm",
            "reference_alignment",
            "observation_timing",
            "previous_action_semantics",
            "startup_hold",
        ):
            if result[key] != case["result"][key]:
                raise ValueError(f"recovery replay changed full-run mechanics/acquisition: {key}")
        if entry["full_request_control"]:
            identity = require_full_replay_identity(arrays, baseline)
            if result != case["result"]:
                raise ValueError("full-request control report differs from saved parent")
            outcome = None
        else:
            count = entry["stop_after_controls"]
            identity = require_prefix_identity(arrays, baseline, count)
            outcome = probe_outcome(result, count, started["full_source_requested_controls"])
            returned = result["return_hold"]
            qpos, qvel = arrays["terminal_active_qpos"], arrays["terminal_active_qvel"]
            lower, upper = effort_target_interval(
                arrays["terminal_active_target"],
                qpos[7:],
                qvel[6:],
                np.asarray(returned["gain_kp_hardware"]),
                np.asarray(returned["gain_kd_hardware"]),
                np.asarray(result["effort_limit_hardware_nm"]),
                dt=0.002,
                slew_rate=5.0,
            )
            outcome["handoff_state"] = dict(
                pelvis_height_m=float(qpos[2]),
                tilt_rad=float(np.arccos(np.clip(-envelope._projected_gravity(qpos[3:7])[2], -1, 1))),
                maximum_joint_speed_rad_s=float(np.max(np.abs(qvel[6:]))),
                minimum_immediate_return_interval_width_rad=float(np.min(upper - lower)),
                immediate_return_interval_nonempty=bool(np.all(lower <= upper)),
                immediate_feasibility_is_not_future_recovery_proof=True,
            )
            outcomes.append(outcome)
        trace = output / f"{entry['label']}.npz"
        with trace.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        bind(trace)
        row = {
            **entry,
            "trace_path": str(trace),
            "result": result,
            "identity": identity,
            "recovery_outcome": outcome,
            **flags,
        }
        records.append(row)
        dump(output / f"{entry['label']}.json", row)
        bind(output / f"{entry['label']}.json")
        print(json.dumps(dict(probe=entry["label"], recovery=outcome, parent_prefix_exact=True)), flush=True)
    summary = summarize_window(outcomes, case["result"]["completed_transitions"])
    bind_imports()
    for path in list(inputs):
        bind(path)
    dump(
        output / "report.json",
        {
            **started,
            "inputs": inputs,
            "records": records,
            "summary": summary,
            "full_eight_clip_qualification": False,
            "default_candidate_selected": False,
        },
    )
    print(json.dumps(dict(summary=summary)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
