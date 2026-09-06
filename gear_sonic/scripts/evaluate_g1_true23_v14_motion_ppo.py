"""Compare original v14 on the unchanged full motion/standing request suite.

Only pair loading differs from the established LoRA evaluator. Use v14's own
checkpoint-bound encoder and raw decoder, with the same controller, timings,
frames, acquisitions, returns and unavailable-request accounting.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import onnxruntime as ort

from gear_sonic.scripts import evaluate_g1_true23_deployment_envelope as envelope
from gear_sonic.scripts.evaluate_g1_true23_motion_ppo import evaluation_plan
from gear_sonic.scripts.record_g1_sonic_original29_baseline import dump
from gear_sonic.utils.g1_sonic_cpp_parameters import file_sha256
from gear_sonic.utils.g1_true23_actuation_profile import NativeSupportActuationProfile
from gear_sonic.utils.g1_true23_clean_mujoco_teleop import UnitreeZeroVelocityFallbackPolicy
from gear_sonic.utils.g1_true23_v14_diagnostic_pair import load_v14_diagnostic_pair
from gear_sonic.utils.g1_true23_interior_target_filter import run_interior_case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--diagnostic-metadata", type=Path, required=True)
    parser.add_argument("--full-request-report", type=Path, required=True)
    parser.add_argument("--stationary-report", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--motor-health-snapshot", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root, assets, output = (
        Path(__file__).resolve().parents[2],
        args.asset_root.resolve(strict=True),
        args.output_dir.resolve(),
    )
    inputs = {}

    def bind(path, expected=None):
        path = Path(path).resolve(strict=True)
        digest = file_sha256(path)
        if digest != inputs.get(str(path), digest) or (expected is not None and digest != expected):
            raise ValueError(f"PPO evaluation source changed: {path}")
        inputs[str(path)] = digest
        return path

    suite = json.loads(bind(args.full_request_report).read_text())
    stationary = json.loads(bind(args.stationary_report).read_text())
    if (
        stationary.get("kind") != "g1_true23_stationary_actor_comparison_diagnostic_v1"
        or stationary.get("stationary_only") is not True
    ):
        raise ValueError("requires the separately labelled stationary diagnostic")
    for report in (suite, stationary):
        for path, digest in report["inputs"].items():
            bind(path, digest)
    plan = evaluation_plan(suite, bind(args.stationary_report.parent / "stationary_reference.npz"))
    for case in plan:
        if not case["unavailable"]:
            bind(case["source"], case["source_sha256"])
    pair = load_v14_diagnostic_pair(bind(args.checkpoint), bind(args.diagnostic_metadata))
    bind(pair["comparison"]["spans"]["path"], pair["comparison"]["spans"]["sha256"])
    for part in ("encoder", "decoder"):
        bind(pair[part]["path"], pair[part]["sha256"])
    policy = envelope._policy(
        Path(pair["encoder"]["path"]),
        Path(pair["decoder"]["path"]),
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
    measured = envelope.load_measured_initial_state(bind(args.motor_health_snapshot))
    profile = NativeSupportActuationProfile.from_sim_config(bind(root / envelope.PHYSICS))
    bind(Path(__file__))
    for name, module in list(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if name.startswith("gear_sonic.") and path and Path(path).suffix == ".py":
            bind(path)
    output.mkdir(parents=True, exist_ok=False)
    flags = dict(hardware_authorized=False, deployment_ready=False, promotion_eligible=False)
    started = {
        "kind": "g1_true23_original_v14_native_ieee_full_request_evaluation_v1",
        "inputs": dict(inputs),
        "pair": pair,
        "plan": plan,
        "original_eight_request_set_preserved": True,
        "original_method_not_one_variable_lora_ablation": True,
        "own_paired_encoder_used_not_borrowed_lora_encoder": True,
        "additional_standing_not_counted_as_motion_success": True,
        "full_body_sonic_controlled_joint_count": 23,
        "source_frames_tempo_gains_slew_effort_and_history_unchanged": True,
        "historical_start_is_not_fresh_robot_state": True,
        "standing_actor_is_not_native_unitree_fsm_handoff": True,
        **flags,
    }
    dump(output / "started.json", started)
    records = []
    for case in plan:
        if case["unavailable"]:
            records.append({**case, "not_executed": True, **flags})
            continue
        with np.load(case["source"], allow_pickle=False) as archive:
            motion = {key: archive[key].copy() for key in archive.files}
        lifecycle, historical = case["lifecycle"], case["historical_start"]
        result, arrays = run_interior_case(
            fraction=1.0,
            root=root,
            asset_root=assets,
            policy=policy,
            motion=motion,
            kp=np.asarray(profile.kp),
            kd=np.asarray(profile.kd),
            joint_scale=np.ones(23),
            ankle_effort=35.0,
            slew_rate=5.0,
            initial_state="measured" if historical else "reference",
            maximum_steps=None,
            measured_state=measured if historical else None,
            startup_hold_s=5.0 if lifecycle else 0.0,
            return_hold_s=5.0 if lifecycle else 0.0,
            transition_policy=balance if lifecycle else None,
            align_reference_start=lifecycle,
            project_transition_effort=lifecycle,
            project_active_effort=True,
            stateful_native_controller=True,
            trace_active_actuation=True,
        )
        if (
            result["requested_transitions"] != case["expected_transitions"]
            or result["compiled_native_model_sha256"]
            != stationary["records"][0]["result"]["compiled_native_model_sha256"]
        ):
            raise ValueError("PPO evaluation changed full requested frame count or compiled native model")
        path = output / f"{case['label']}.npz"
        with path.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        bind(path)
        row = {**case, "trace_path": str(path), "result": result, **flags}
        records.append(row)
        dump(output / f"{case['label']}.json", row)
        bind(output / f"{case['label']}.json")
        print(
            json.dumps(
                {
                    "case": case["label"],
                    "completed": result["completed_transitions"],
                    "requested": result["requested_transitions"],
                    "motion_fidelity": result["motion_fidelity"]["passed"],
                    "failure": None if result["failure"] is None else result["failure"]["type"],
                    "return_completed": result["return_hold"].get("completed_transitions"),
                }
            ),
            flush=True,
        )
    for path in list(inputs):
        bind(path)
    dump(
        output / "report.json",
        {
            **started,
            "inputs": inputs,
            "records": records,
            "full_eight_clip_qualification": False,
            "default_candidate_selected": False,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
