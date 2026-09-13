"""SIM-only matched G1-encoder counterfactual using the unchanged source29 loop."""

import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
import torch

from gear_sonic.scripts import record_g1_sonic_public29_baseline as baseline
from gear_sonic.teleop.buffered_source_horizon import ReceivedSourceHorizon
from gear_sonic.utils.g1_23dof_contract import SOURCE_MJ29_KEEP_INDICES
from gear_sonic.utils.g1_29dof_low_latency_g1_teacher import LowLatencyG1Teacher
from gear_sonic.utils.g1_sonic_cpp_parameters import CppParameters, file_sha256
from gear_sonic.utils.g1_sonic_full_pose_reference import full_pose_encoder640, full_pose_reference_contract
from gear_sonic.utils.g1_true23_reference_floor import compiled_model_sha256

BASELINE_REPORT_SHA256 = "1cb05c05b9d5051dc7c9f649f190dfd62be7a7dde4734557c77f4396d9b48fa0"
BASELINE_LOOP_SHA256 = "7cd54d5626895d7ca4f64e6f1a26179ad8128305f6b54eb61a5ad16ea0faec72"
CLIPS = ("0807_yanjie_walk_002", "0807_yanjie_walk_003", "0807_yanjie_walk_008")


def teacher_digest(teacher):
    digest = hashlib.sha256()
    for label, layers in (("encoder", teacher.encoder), ("decoder", teacher.decoder)):
        digest.update(label.encode())
        for layer in layers:
            for tensor in layer:
                value = tensor.detach().numpy()
                digest.update(str((value.dtype, value.shape)).encode())
                digest.update(value.tobytes())
    return digest.hexdigest()


class _Window:
    def __init__(self, bridge, original, poses):
        self.bridge, self.original, self.poses = bridge, original, poses

    def __getattr__(self, name):
        return getattr(self.original, name)

    def encoder267(self, measured):
        diagnostic = self.original.encoder267(measured)
        if self.bridge.pending is not None:
            raise RuntimeError("unconsumed G1 encoder input")
        self.bridge.pending = dict(
            encoder640=full_pose_encoder640(self.poses, measured),
            measured_wxyz=measured.copy(),
            anchor_sample_index=round(self.original.encoder_anchor_timestamp_s / 0.02),
        )
        return diagnostic


class FullPoseBridge:
    """Only source samples whose baseline push has succeeded enter the G1 buffer."""

    def __init__(self, teacher, poses):
        self.teacher = teacher
        self.poses = np.concatenate((poses, np.repeat(poses[-1:], 10, axis=0))).astype(np.float32)
        self.received29 = deque(maxlen=11)
        self.original_buffer = ReceivedSourceHorizon()
        self.push_count = 0
        self.factory_count = 0
        self.pending = None
        self.attempts = []

    def factory(self):
        self.factory_count += 1
        if self.factory_count != 1:
            raise RuntimeError("one fresh reference buffer required per physical lifecycle")
        return self

    def push(self, **sample):
        index = round(sample["source_timestamp_s"] / 0.02)
        if index != self.push_count or not 0 <= index < len(self.poses):
            raise ValueError("full-pose source is not the next received sample")
        pose = self.poses[index]
        np.testing.assert_array_equal(sample["joint_position23"], pose[7:][list(SOURCE_MJ29_KEEP_INDICES)])
        np.testing.assert_array_equal(sample["root_position_w"], pose[:3])
        np.testing.assert_array_equal(sample["root_quaternion_wxyz"], pose[3:7])
        original = self.original_buffer.push(**sample)
        self.received29.append(pose.copy())
        self.push_count += 1
        if original is None:
            return None
        return _Window(self, original, np.stack(self.received29))

    def infer(self, diagnostic267, history930):
        if self.pending is None:
            raise RuntimeError("G1 inference requires a fresh received-window input")
        attempt = self.pending
        self.pending = None
        attempt.update(history930=history930.copy(), output_present=False, failure=None)
        self.attempts.append(attempt)
        try:
            raw, token = self.teacher.infer(attempt["encoder640"], history930)
        except Exception as error:
            attempt["failure"] = dict(type=type(error).__name__, message=str(error))
            raise
        attempt.update(raw29=raw.copy(), token64=token.copy(), output_present=True)
        return raw

    def recorded_arrays(self, completed):
        if completed > len(self.attempts) or any(not row["output_present"] for row in self.attempts[:completed]):
            raise ValueError("executed control rows do not have matching G1 outputs")
        arrays = {}
        for key, width in (
            ("encoder640", 640),
            ("history930", 930),
            ("measured_wxyz", 4),
            ("raw29", 29),
            ("token64", 64),
        ):
            arrays["attempt_" + key] = np.asarray(
                [row.get(key, np.zeros(width, np.float32)) for row in self.attempts], dtype=np.float32
            ).reshape(-1, width)
        arrays["attempt_anchor_sample_index"] = np.asarray(
            [row["anchor_sample_index"] for row in self.attempts], dtype=np.int64
        )
        arrays["attempt_output_present"] = np.asarray([row["output_present"] for row in self.attempts], dtype=bool)
        arrays["encoder640"] = arrays["attempt_encoder640"][:completed].copy()
        arrays["token64"] = arrays["attempt_token64"][:completed].copy()
        return arrays


def matched_case(model, source_model, parameters, teacher, poses, phases):
    bridge = FullPoseBridge(teacher, poses)
    original = baseline.ReceivedSourceHorizon
    try:
        baseline.ReceivedSourceHorizon = bridge.factory
        arrays, received, metrics = baseline.run_case(model, source_model, parameters, bridge, poses, phases)
    finally:
        baseline.ReceivedSourceHorizon = original
    completed = metrics["completed_controls"]
    extra = bridge.recorded_arrays(completed)
    np.testing.assert_array_equal(extra["attempt_raw29"][:completed], arrays["raw29"][:completed])
    np.testing.assert_array_equal(extra["attempt_history930"][:completed], arrays["history930"][:completed])
    arrays.update(extra)
    attempts = [
        dict(index=i, output_present=row["output_present"], failure=row["failure"], applied=i < completed)
        for i, row in enumerate(bridge.attempts)
    ]
    return arrays, received, metrics, attempts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "baseline-report",
        "source-checkpoint",
        "source-model",
        "original-model",
        "cpp-parameters",
        "experiment",
        "output-directory",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output_directory.resolve()
    if output.exists():
        raise FileExistsError("full-pose diagnostic refuses overwrite or automatic retry")
    inputs = {}

    def bind(path, expected=None):
        path = Path(path).resolve(strict=True)
        digest = file_sha256(path)
        if expected is not None and expected != digest:
            raise ValueError("full-pose matched input hash mismatch: " + str(path))
        inputs[str(path)] = digest
        return path

    prior = json.loads(bind(args.baseline_report, BASELINE_REPORT_SHA256).read_text())
    bind(baseline.__file__, BASELINE_LOOP_SHA256)
    bind(args.experiment)
    if tuple(Path(row["source"]).stem for row in prior["records"]) != CLIPS:
        raise ValueError("requires the three predeclared original29 comparison cases")
    for path in (args.source_checkpoint, args.source_model, args.original_model, args.cpp_parameters):
        bind(path, prior["inputs"][str(path.resolve())])
    model = mujoco.MjModel.from_binary_path(str(args.original_model))
    source_model = mujoco.MjModel.from_xml_path(str(args.source_model))
    if compiled_model_sha256(model) != prior["physical_model_sha256"]:
        raise ValueError("matched original29 physical scene differs")
    if compiled_model_sha256(source_model) != prior["source_geometry_sha256"]:
        raise ValueError("matched source geometry differs")
    parameters = CppParameters(json.loads(args.cpp_parameters.read_text()))
    torch.set_num_threads(1)
    print("Loading one pinned CPU G1 encoder and unchanged original29 decoder", flush=True)
    teacher = LowLatencyG1Teacher(args.source_checkpoint)
    weights_before = teacher_digest(teacher)
    upstream = args.source_checkpoint.resolve().parents[1]
    for relative in (
        "low_latency/config.yaml",
        "low_latency/model_config.yaml",
        "gear_sonic/envs/manager_env/mdp/commands.py",
        "gear_sonic/envs/manager_env/mdp/observations.py",
        "gear_sonic/trl/modules/base_module.py",
        "gear_sonic/trl/modules/universal_token_modules.py",
    ):
        bind(upstream / relative)
    for name, module in list(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if name.startswith("gear_sonic.") and path and Path(path).suffix == ".py":
            bind(path)
    bind(__file__)
    output.mkdir(parents=True, exist_ok=False)
    print(json.dumps(teacher.descriptor()), flush=True)
    records = []
    for old in prior["records"]:
        with np.load(bind(old["trace"], old["trace_sha256"]), allow_pickle=False) as archive:
            poses = archive["reference_qpos29"].copy()
            old_received = {key: archive[key].copy() for key in archive.files if key.startswith("received_")}
            initial_q, initial_v = archive["qpos"][0].copy(), archive["qvel"][0].copy()
        arrays, received, metrics, attempts = matched_case(
            model, source_model, parameters, teacher, poses, old["phases"]
        )
        np.testing.assert_array_equal(initial_q, arrays["qpos"][0])
        np.testing.assert_array_equal(initial_v, arrays["qvel"][0])
        for name, value in received.items():
            np.testing.assert_array_equal(value, old_received["received_" + name])
        name = Path(old["source"]).stem
        path = output / (name + ".npz")
        with path.open("xb") as stream:
            np.savez_compressed(stream, **arrays, reference_qpos29=poses, **old_received)
        record = dict(
            source=old["source"],
            trace=str(path),
            trace_sha256=file_sha256(path),
            matched_baseline_trace=old["trace"],
            matched_baseline_trace_sha256=old["trace_sha256"],
            phases=old["phases"],
            metrics=metrics,
            attempts=attempts,
            source_speed_factor=old["source_speed_factor"],
            recorded_source_frames=old["recorded_source_frames"],
            recorded_source_fps=old["recorded_source_fps"],
            resampled_source_frames=old["resampled_source_frames"],
            unsampled_original_final_fraction_s=old["unsampled_original_final_fraction_s"],
        )
        with (output / (name + ".json")).open("x") as stream:
            json.dump(record, stream, indent=2, allow_nan=False)
        records.append(record)
        print(json.dumps(dict(clip=name, **metrics)), flush=True)
    if teacher_digest(teacher) != weights_before:
        raise ValueError("original G1 encoder/shared decoder changed during cases")
    for path, digest in inputs.items():
        if file_sha256(path) != digest:
            raise ValueError("matched experiment input changed during execution: " + path)
    report = dict(
        kind="sonic_public_original29_same_checkpoint_full_pose_encoder_v1",
        inputs=inputs,
        records=records,
        teacher=teacher.descriptor(),
        teacher_tensor_sha256=weights_before,
        reference_contract=full_pose_reference_contract(),
        original29_baseline_report=str(args.baseline_report.resolve()),
        original29_baseline_report_sha256=BASELINE_REPORT_SHA256,
        original_model=str(args.original_model.resolve()),
        source_model=str(args.source_model.resolve()),
        cpp_parameters=str(args.cpp_parameters.resolve()),
        physical_model_sha256=compiled_model_sha256(model),
        source_geometry_sha256=compiled_model_sha256(source_model),
        controlled_change="teleop267 to existing G1 encoder640; same original29 checkpoint/physics/history",
        unused_encoder267_saved_as_diagnostic_only=True,
        continuation_gate_applied_only_after_independent_audit=True,
        predeclared_gate=dict(
            all_lifecycles_complete=True, mean_root_p95_ratio_max=0.9, each_leg_rmse_ratio_max=1.05
        ),
        no_hardware_connection=True,
        training_updates=0,
        all29_joints_actuated=True,
        native23_physics_equivalent=False,
        hardware_authorized=False,
        deployment_ready=False,
        native23_qualified=False,
    )
    with (output / "report.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
