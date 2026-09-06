"""Retest a standing-only adapter on every original full-motion request.

Completion cannot replace choreography fidelity. Reuse the exact full request
set, source hashes, controller settings and historical-start lifecycle from
the preceding headroom comparison. No DDS, model export or robot deployment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import onnxruntime as ort
import torch

from gear_sonic.scripts import evaluate_g1_true23_deployment_envelope as envelope
from gear_sonic.scripts.evaluate_g1_true23_interior_effort import PICO_NAMES
from gear_sonic.scripts.record_g1_sonic_original29_baseline import CLIPS, dump
from gear_sonic.trl.mjlab.frozen_platform_lora_actor import FrozenPlatformTrue23Core, _tensor_state_sha256
from gear_sonic.utils.g1_sonic_cpp_parameters import file_sha256
from gear_sonic.utils.g1_true23_actuation_profile import NativeSupportActuationProfile
from gear_sonic.utils.g1_true23_clean_mujoco_teleop import UnitreeZeroVelocityFallbackPolicy
from gear_sonic.utils.g1_true23_interior_target_filter import run_interior_case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-report", type=Path, required=True)
    parser.add_argument("--original-suite", type=Path, required=True)
    parser.add_argument("--warm-start", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
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
            raise ValueError(f"standing motion comparison source changed: {path}")
        inputs[str(path)] = digest
        return path

    fit = json.loads(bind(args.fit_report).read_text())
    suite = json.loads(bind(args.original_suite).read_text())
    if (
        fit.get("kind") != "g1_true23_standing_only_lora_fit_diagnostic_v1"
        or suite.get("kind") != "g1_true23_interior_effort_full_request_comparison_v1"
    ):
        raise ValueError("requires standing-only fit and complete original request reports")
    for report in (fit, suite):
        if report.get("hardware_authorized") is not False or report.get("deployment_ready") is not False:
            raise ValueError("motion comparison inputs must remain unqualified diagnostics")
        for path, digest in report["inputs"].items():
            bind(path, digest)
    cases = [row for row in suite["records"] if row["inner_fraction"] == 1.0]
    if [row["name"] for row in cases if not row["historical_start"]] != [*CLIPS, *PICO_NAMES]:
        raise ValueError("full original SONIC/PICO request set changed")
    torch.set_num_threads(1)
    candidate = torch.load(
        bind(args.fit_report.parent / "standing_lora.pt"), map_location="cpu", weights_only=True
    )
    if candidate.get("kind") != "g1_true23_standing_only_lora_adapter_diagnostic_v1" or any(
        candidate.get(flag) is not False
        for flag in ("hardware_authorized", "deployment_ready", "promotion_eligible", "ppo_resume_checkpoint")
    ):
        raise ValueError("requires explicitly nondeployable standing-only adapter")
    if (
        _tensor_state_sha256(candidate["adapter_state_dict"]) != fit["adapter_state_sha256"]
        or candidate["adapter_state_sha256"] != fit["adapter_state_sha256"]
    ):
        raise ValueError("standing adapter tensor hash mismatch")
    contract = candidate["adapter_contract"]
    core = FrozenPlatformTrue23Core(
        warm_start_path=bind(args.warm_start),
        source_checkpoint_path=bind(args.source_checkpoint),
        lora_rank=contract["lora_rank"],
        lora_alpha=contract["lora_alpha"],
    ).eval()
    if core.adapter_contract() != contract or contract != fit["adapter_contract"]:
        raise ValueError("standing adapter frozen platform differs from fitting")
    core.load_lora_state_dict(candidate["adapter_state_dict"])
    core.assert_frozen_platform_unchanged()

    class Policy:
        def infer(self, semantic, history):
            with torch.no_grad():
                token = core.encode(torch.from_numpy(semantic[None]))
                encoded = core.codec.encode_proprioception(torch.from_numpy(history[None]))
                raw = core.codec.decode_action(core.decoder(torch.cat((token, encoded), dim=-1)))
            return raw.numpy()[0], token.numpy()[0]

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
        "kind": "g1_true23_standing_lora_full_motion_regression_v1",
        "inputs": dict(inputs),
        "requested_clips": [*CLIPS, *PICO_NAMES],
        "adapter_state_sha256": fit["adapter_state_sha256"],
        "frozen_encoder_unchanged": True,
        "active_sonic_controlled_joint_count": 23,
        "standing_reference_not_substituted_for_any_motion": True,
        "source_frames_tempo_gains_slew_effort_and_history_unchanged": True,
        "historical_start_is_not_fresh_robot_state": True,
        "torch_adapter_not_deployment_onnx": True,
        **flags,
    }
    dump(output / "started.json", started)
    records = []
    for old in cases:
        row = {"name": old["name"], "historical_start": old["historical_start"], **flags}
        if old.get("not_executed"):
            records.append({**row, "not_executed": True, "reason": old["reason"]})
            continue
        with np.load(bind(old["source_motion_path"], old["source_motion_sha256"]), allow_pickle=False) as archive:
            motion = {key: archive[key].copy() for key in archive.files}
        historical = old["historical_start"]
        name = old["name"] + (".historical" if historical else ".reference")
        print(json.dumps({"starting_full_motion": name}), flush=True)
        result, arrays = run_interior_case(
            fraction=1.0,
            root=root,
            asset_root=assets,
            policy=Policy(),
            motion=motion,
            kp=np.asarray(profile.kp),
            kd=np.asarray(profile.kd),
            joint_scale=np.ones(23),
            ankle_effort=35.0,
            slew_rate=5.0,
            initial_state="measured" if historical else "reference",
            maximum_steps=None,
            measured_state=measured if historical else None,
            startup_hold_s=5.0 if historical else 0.0,
            return_hold_s=5.0 if historical else 0.0,
            transition_policy=balance if historical else None,
            align_reference_start=historical,
            project_transition_effort=historical,
            project_active_effort=True,
            stateful_native_controller=True,
            trace_active_actuation=True,
        )
        if (
            result["requested_transitions"] != old["result"]["requested_transitions"]
            or result["compiled_native_model_sha256"] != old["result"]["compiled_native_model_sha256"]
        ):
            raise ValueError("motion request or compiled native controller changed")
        path = output / f"{name}.npz"
        with path.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        bind(path)
        row.update(
            result=result,
            trace_path=str(path),
            source_motion_path=old["source_motion_path"],
            source_motion_sha256=old["source_motion_sha256"],
            baseline_completed=old["result"]["completed_transitions"],
        )
        records.append(row)
        dump(output / f"{name}.json", row)
        bind(output / f"{name}.json")
        print(
            json.dumps(
                {
                    "case": name,
                    "baseline_completed": row["baseline_completed"],
                    "completed": result["completed_transitions"],
                    "requested": result["requested_transitions"],
                    "motion_fidelity": result["motion_fidelity"]["passed"],
                    "failure": result["failure"],
                    "return_completed": result["return_hold"].get("completed_transitions"),
                }
            ),
            flush=True,
        )
    core.assert_frozen_platform_unchanged()
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
