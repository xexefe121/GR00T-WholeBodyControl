"""Simulation-only, zero-training SONIC joint-surgery/reference ablation.

The preview case deliberately consumes a saved clip's future frames. It is not
a live causal policy and cannot qualify teleoperation. Both cases use identical
native23 physics, safety transforms, initialization and complete lifecycle.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from gear_sonic.trl.mjlab.native23_generalist_actor import Native23GeneralistCore
from gear_sonic.utils.g1_true23_clean_mujoco_teleop import sha256_file
from gear_sonic.utils.g1_true23_generalist_benchmark import FLAGS, MODEL, PHYSICS, run_reference_diagnostic
from gear_sonic.utils.g1_true23_generalist_lifecycle import assess_lifecycle_diagnostic, build_lifecycle_timeline
from gear_sonic.utils.g1_true23_training_precision import ieee_training_precision
from gear_sonic.utils.g1_true23_virtual_source_reference import virtual_source_vr_terms
from gear_sonic.utils.g1_true23_release_action_diagnostic import bounded_linear_precompensation

MODES = ("causal_past", "released_future_preview", "causal_past_virtual_source", "released_future_preview_virtual_source")


def preview_encoder(encoder, motion, control_index):
    """Change only the lower-body 240 values to released step-1 future semantics."""
    if encoder.shape != (267,) or encoder.dtype != np.float32:
        raise ValueError("requires float32 encoder267")
    if type(control_index) is not int or control_index < 0:
        raise ValueError("requires nonnegative control index")
    # Same buffered reference anchor as the causal referee. The last ten
    # lifecycle frames are verified constant, so terminal repeat adds no motion.
    q9 = 9 + control_index
    count = len(motion["joint_pos"])
    if q9 >= count:
        raise ValueError("reference anchor exceeds saved clip")
    indices = np.minimum(q9 + np.arange(10), count - 1)
    if q9 + 9 >= count and (
        not np.array_equal(motion["joint_pos"][-10:], np.tile(motion["joint_pos"][-1], (10, 1)))
        or np.any(motion["joint_vel"][-10:])
    ):
        raise ValueError("future preview requires genuine frames or a constant terminal hold")
    result = encoder.copy()
    result[:120] = motion["joint_pos"][indices, :12].reshape(-1)
    result[120:240] = motion["joint_vel"][indices, :12].reshape(-1)
    return result


class RowTrimmedRelease:
    def __init__(self, core):
        self.core = core

    def infer(self, encoder267, history930):
        with torch.inference_mode():
            semantic = torch.from_numpy(encoder267[None])
            history = torch.from_numpy(history930[None])
            padded = self.core.codec.encode_proprioception(history)
            if not torch.equal(padded, history):
                raise ValueError("missing-joint history slots must remain zero")
            decoder = torch.cat((self.core.encode(semantic), padded), dim=-1)
            raw = self.core.decoder(decoder)
        return raw[0].numpy().copy(), decoder[0].numpy().copy()


class ReferenceAblation:
    def __init__(self, motion, mode, virtual_vr=None, action_convention="native_tanh"):
        if mode not in MODES:
            raise ValueError("unknown reference ablation")
        self.motion, self.mode = motion, mode
        if action_convention not in ("native_tanh", "released_bounded_linear"):
            raise ValueError("unknown action convention")
        self.action_convention = action_convention
        self.virtual_vr = virtual_vr
        if mode.endswith("virtual_source") and (virtual_vr is None or virtual_vr.shape != (len(motion["joint_pos"]), 21)):
            raise ValueError("virtual source case requires explicitly constructed source-model VR terms")
        self.actual_encoder_inputs = []
        self.model_outputs = []
        self.projection_delta_rad = []

    def infer(self, policy, encoder, history, *, control_index, **unused):
        if self.mode.startswith("released_future_preview"):
            encoder = preview_encoder(encoder, self.motion, control_index)
        if self.mode.endswith("virtual_source"):
            encoder = encoder.copy()
            encoder[240:261] = self.virtual_vr[9 + control_index]
        self.actual_encoder_inputs.append(encoder.copy())
        raw, decoder = policy.infer(encoder, history)
        self.model_outputs.append(raw.copy())
        if self.action_convention == "released_bounded_linear":
            raw, projection = bounded_linear_precompensation(raw)
            self.projection_delta_rad.append(projection)
        return raw, decoder

    def external_force_world(self, step):
        return np.zeros(3)

    def contract(self):
        return dict(
            kind="zero_training_row_trimmed_sonic_reference_ablation_v1",
            mode=self.mode,
            reference_anchor="buffered_q9",
            decoder994_history_unchanged=True,
            actual_encoder_trace="actual_policy_encoder267",
            referee_encoder_trace="encoder267_is_unmodified_causal_reference_not_necessarily_consumed",
            future_reference_consumed=self.mode.startswith("released_future_preview"),
            reference_geometry="original29_zero_absent_fk" if self.mode.endswith("virtual_source") else "native23_fk",
            measured_robot_or_reference_metric_geometry_changed=False,
            action_convention=self.action_convention,
            existing_safe_target_transform_and_all_bounds_unchanged=True,
            terminal_reference_repeat="constant_standing_only",
            external_force_used=False,
            live_teleoperation_compatible=False,
            **FLAGS,
        )


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--modes", choices=MODES, nargs="+", default=list(MODES))
    parser.add_argument("--action-convention", choices=("native_tanh", "released_bounded_linear"), default="native_tanh")
    parser.add_argument("--profiles", choices=("native_model", "historical_released_gains"), nargs="+", default=["native_model"])
    args = parser.parse_args(argv)
    import mujoco

    root = Path(__file__).resolve().parents[2]
    assets, source = args.asset_root.resolve(strict=True), args.motion.resolve(strict=True)
    output = args.output_directory.resolve()
    if output.exists() or args.output_directory.is_symlink():
        raise ValueError("requires a new evidence directory")
    torch.set_num_threads(1)
    model = mujoco.MjModel.from_xml_path(str(assets / MODEL))
    with np.load(source, allow_pickle=False) as archive:
        source_motion = {key: archive[key].copy() for key in (
            "fps", "joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"
        )}
    motion, timeline = build_lifecycle_timeline(
        source_motion, model=model, simulation_config=root / PHYSICS, return_target="planned_endpoint"
    )
    virtual_vr = virtual_source_vr_terms(motion, assets / "gear_sonic/data/robots/g1/g1_29dof.xml")
    output.mkdir(parents=True, exist_ok=False)
    motion_path = output / "lifecycle_reference.npz"
    with motion_path.open("xb") as stream:
        np.savez_compressed(stream, **motion)
    inputs = {
        str(path): sha256_file(path)
        for path in (
            source, assets / MODEL, root / PHYSICS, Path(__file__),
            assets / "gear_sonic/data/robots/g1/g1_29dof.xml",
            root / "gear_sonic/utils/g1_true23_virtual_source_reference.py",
            root / "gear_sonic/utils/g1_true23_release_action_diagnostic.py",
            assets / "sonic_release/g1_23dof_rev_1_0_low_latency_init.pt", assets / "low_latency/last.pt",
        )
    }
    write_json(output / "started.json", dict(inputs=inputs, timeline=timeline, **FLAGS))
    rows = []
    with ieee_training_precision() as (precision, guard):
        core = Native23GeneralistCore(
            warm_start_path=assets / "sonic_release/g1_23dof_rev_1_0_low_latency_init.pt",
            source_checkpoint_path=assets / "low_latency/last.pt",
        ).cpu().eval()
        policy = RowTrimmedRelease(core)
        print("Loaded exact zero-training 23-row SONIC release; starting paired physics rollouts.", flush=True)
        for profile, mode in ((profile, mode) for profile in args.profiles for mode in args.modes):
            guard()
            adapter = ReferenceAblation(motion, mode, virtual_vr, args.action_convention)
            result, arrays = run_reference_diagnostic(
                root=root, asset_root=assets, motion_path=motion_path, policy=policy, runtime_adapter=adapter, profile=profile
            )
            arrays["actual_policy_encoder267"] = np.asarray(adapter.actual_encoder_inputs)
            arrays["released_model_raw23"] = np.asarray(adapter.model_outputs)
            arrays["bounded_linear_projection_delta_rad"] = np.asarray(adapter.projection_delta_rad).reshape(-1, 23)
            lifecycle = assess_lifecycle_diagnostic(timeline, result, arrays)
            label = f"{profile}.{mode}"
            trace = output / f"{label}.npz"
            with trace.open("xb") as stream:
                np.savez_compressed(stream, **arrays)
            projection = arrays["bounded_linear_projection_delta_rad"]
            row = dict(mode=mode, result=result, lifecycle=lifecycle, trace_sha256=sha256_file(trace),
                       linear_projection_max_rad=float(np.max(np.abs(projection))) if projection.size else 0.0,
                       linear_projection_joint_fraction=float(np.mean(np.abs(projection) > 1e-6)) if projection.size else 0.0)
            write_json(output / f"{label}.json", row)
            rows.append(row)
            print(json.dumps(dict(
                mode=mode, profile=profile, completed=result["completed_controls"], available=result["available_controls"],
                root_p95_m=result["pelvis_world_position_p95_m"],
                dance=lifecycle["source_motion_tracking"],
                final_joint_max_rad=lifecycle["final_proof_standing_joint_error_max_rad"],
                failure=result["failure"],
            )), flush=True)
        for field in ("initial_state_and_history_sha256", "compiled_model_sha256", "motion_sha256",
                      "available_controls", "effort_limit_hardware_nm"):
            if any(rows[0]["result"][field] != row["result"][field] for row in rows[1:]):
                raise ValueError(f"paired ablation differs at {field}")
        core.assert_frozen_encoder_unchanged()
        for path, digest in inputs.items():
            if sha256_file(Path(path)) != digest:
                raise ValueError(f"ablation input changed: {path}")
        write_json(output / "report.json", dict(
            kind="zero_training_native23_sonic_semantic_ablation_v1", rows=rows, inputs=inputs,
            policy=core.artifact_contract(), timeline=timeline, precision=precision,
            learned_parameter_updates=0, paired_initial_states_and_model_equal=True,
            paired_physics_except_explicit_gain_profile_equal=True,
            hardware_or_network_actuation_used=False, generalization_tested=False, **FLAGS,
        ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
