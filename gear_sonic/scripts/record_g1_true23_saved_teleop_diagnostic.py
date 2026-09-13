"""Record a bounded saved-input native23 rollout, including the failing state.

Uses the existing full-body controller and fallback with a validated model pair.
The historical mismatched live profile requires an explicit diagnostic flag.
Original V14 and virtual-source geometry are separately labeled SIM comparisons.
No transport, headset, DDS, hardware channel, policy update or relaxed gates.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from gear_sonic.scripts.replay_g1_true23_pico_packets_zmq import load_reference_packets
from gear_sonic.utils.g1_23dof_contract import NATIVE_IL23_JOINT_NAMES
from gear_sonic.utils.g1_true23_clean_mujoco_teleop import (
    NATIVE_TO_MJ,
    SupervisedCleanTrue23MujocoController,
    UnitreeZeroVelocityFallbackPolicy,
    sha256_file,
    validate_reference_terms,
)
from gear_sonic.utils.g1_true23_diagnostic_pair import load_diagnostic_pair
from gear_sonic.utils.g1_true23_frozen_lora_live_teleop import (
    ENCODER_RELATIVE_PATH,
    FALLBACK_RELATIVE_PATH,
    build_live_controller,
    initialize_live_controller,
    load_frozen_lora_live_profile,
    step_live_packet,
)
from gear_sonic.utils.g1_true23_generalist_benchmark import MODEL, PHYSICS
from gear_sonic.utils.g1_true23_sonic_library_replay import ExactHashSonicPolicy
from gear_sonic.utils.g1_true23_source_action_codec import (
    SOURCE_ACTION_CONVENTION,
    source_action_codec_contract,
    source_action_history_numpy,
    source_scaled_precompensation,
)
from gear_sonic.utils.g1_true23_step1b_mujoco import _projected_gravity
from gear_sonic.utils.g1_true23_virtual_source_reference import VIRTUAL_SOURCE_REFERENCE, virtual_source_vr_terms

ORIGINAL_WALK_DECODER = "artifacts/g1_true23/pico_internet_fullbody_v14_100_eval/model_100.decoder.onnx"
ORIGINAL_WALK_ENCODER_SHA256 = "733353148bef1eb8dd83a96416b7a89f0b5c3530ceb9e0cec9c25fdb04f56ff2"
ORIGINAL_WALK_DECODER_SHA256 = "f66408ae9a10720a3aff717269d0e2a4e07ab471e449a6fe8f5bae5e8607ef63"


class SourceActionDiagnosticPolicy:
    """Explicit output/history convention ablation, never a weight promotion.

    Existing source codec precompensates inside the unchanged native transform.
    Requests outside that envelope are projected and retained as diagnostics.
    Native23 physics, policy raw bound and the fallback policy are not modified.
    """

    def __init__(self, policy):
        self.policy = policy
        self.source_raw, self.native_raw, self.projections = [], [], []

    def infer(self, encoder, history):
        raw, decoder = self.policy.infer(encoder, source_action_history_numpy(history))
        try:
            native, projection = source_scaled_precompensation(raw)
        except ValueError as error:
            raise RuntimeError(f"source action diagnostic rejected: {error}") from error
        self.source_raw.append(raw.copy())
        self.native_raw.append(native.copy())
        self.projections.append(projection.copy())
        return native, decoder

    def trace_arrays(self):
        return {
            name: np.asarray(values, dtype=np.float32).reshape(-1, 23)
            for name, values in (
                ("source_raw_action23_native", self.source_raw),
                ("native_inverse_action23_native", self.native_raw),
                ("source_projection_delta_rad23_native", self.projections),
            )
        }

    def evidence(self):
        delta = self.trace_arrays()["source_projection_delta_rad23_native"]
        projected = np.abs(delta) > 1e-6
        return dict(
            action_convention=SOURCE_ACTION_CONVENTION,
            source_action_codec=source_action_codec_contract(),
            policy_queries_recorded=len(delta),
            projected_control_count=int(np.count_nonzero(projected.any(axis=1))),
            projected_coordinate_count=int(np.count_nonzero(projected)),
            projection_reporting_threshold_rad=1e-6,
            maximum_absolute_projection_rad=float(np.max(np.abs(delta), initial=0)),
            maximum_absolute_projection_by_joint_rad=dict(
                zip(NATIVE_IL23_JOINT_NAMES, np.max(np.abs(delta), axis=0, initial=0).tolist(), strict=True)
            ),
            action_trace_scope="accepted_SONIC_inferences_only_including_any_integrated_failure_not_fallback",
            training_runtime_semantics_match_claimed=False,
            existing_native_physics_and_bounds_unchanged=True,
            hardware_authorized=False,
            deployment_ready=False,
        )


def preserve_virtual_source_geometry(controller, packets, source_model_path):
    """Offline ablation: only current q9 joint angles determine each VR target.

    Precompute independent frames for this bounded recording, not a future
    preview. The source29 model supplies encoder reference geometry only; the
    measured robot, joints, orientation and tracking truth remain native23.
    """
    indices, references = [], []
    for packet in packets:
        summary = validate_reference_terms(packet)
        indices.append(summary["control_index"])
        references.append(np.asarray(packet["q_ref23_native"], dtype=np.float64))
    if len(set(indices)) != len(indices):
        raise ValueError("virtual-source recording requires unique control indices")
    references = np.asarray(references)
    terms = virtual_source_vr_terms(dict(joint_pos=references[:, NATIVE_TO_MJ]), source_model_path)
    lookup = {index: (q.copy(), vr.copy()) for index, q, vr in zip(indices, references, terms, strict=True)}
    original = controller.retarget_pico_reference_packet

    def retarget(packet):
        q, vr = lookup[packet["control_source_frame_index"]]
        if not np.array_equal(np.asarray(packet["q_ref23_native"]), q):
            raise ValueError("virtual-source reference changed after preparation")
        result = original(packet)
        result["vr_3point_local_target"] = vr[:9].tolist()
        result["vr_3point_local_orn_target"] = vr[9:].tolist()
        validate_reference_terms(result)
        return result

    controller.retarget_pico_reference_packet = retarget


def preserve_calibrated_source_orientation(controller, first_packet):
    """SIM experiment: retain source orientation after one initial yaw alignment.

    Local native23 hand/head FK, source joints, timestamps and all physical
    controller gates stay unchanged. Only the caller's SIM instance is changed.
    """
    initial = Rotation.from_quat(first_packet["reference_anchor_quaternion_xyzw"])
    matrix = initial.as_matrix()
    yaw = np.arctan2(matrix[1, 0], matrix[0, 0])
    alignment = Rotation.from_euler("z", -yaw)
    original = controller.retarget_pico_reference_packet

    def retarget(packet):
        result = original(packet)
        source = Rotation.from_quat(packet["reference_anchor_quaternion_xyzw"])
        result["reference_anchor_quaternion_xyzw"] = (alignment * source).as_quat().tolist()
        return result

    controller.retarget_pico_reference_packet = retarget


def record(controller, packets):
    initialize_live_controller(controller, packets[0], packets[1])
    positions = [controller.data.qpos.copy()]
    velocities = [controller.data.qvel.copy()]
    times = [float(controller.data.time)]
    error, failed_source_index, failed_attempt_integrated = None, None, None
    successful = 0
    for packet in packets:
        before = controller.completed
        try:
            step_live_packet(controller, packet)
            successful += 1
        except RuntimeError as failure:
            error = str(failure)
            failed_source_index = packet["control_source_frame_index"]
            failed_attempt_integrated = controller.completed > before
        # An inference/target rejection can happen before physics advances.
        # Its state is already the last saved state: retain the failure, but do
        # not fabricate another 20-ms physical sample for the rejected attempt.
        if controller.completed > before:
            positions.append(controller.data.qpos.copy())
            velocities.append(controller.data.qvel.copy())
            times.append(float(controller.data.time))
        if error is not None:
            break
    arrays = dict(qpos=np.asarray(positions), qvel=np.asarray(velocities), simulation_time=np.asarray(times))
    tilts = [float(np.arccos(np.clip(-_projected_gravity(p[3:7])[2], -1, 1))) for p in positions]
    result = dict(
        passed=error is None and successful == len(packets) and not controller.fallback_active,
        requested_controls=len(packets),
        successful_controls=successful,
        controller_completed_controls=controller.completed,
        recorded_post_attempt_states=len(positions) - 1,
        failure=error,
        failed_control_source_frame_index=failed_source_index,
        failed_attempt_integrated=failed_attempt_integrated,
        attempted_controls=successful + int(error is not None),
        recorded_state_scope="initial_state_and_completed_physics_transitions_including_integrated_failure",
        final_simulation_time_s=times[-1],
        fallback_active=controller.fallback_active,
        fallback_trigger=controller.fallback_trigger,
        fallback_first_transition=controller.fallback_transition,
        minimum_base_height_m=float(np.min(arrays["qpos"][:, 2])),
        maximum_base_tilt_rad=max(tilts),
    )
    return arrays, result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--decoder-report", type=Path)
    pair_mode = parser.add_mutually_exclusive_group(required=True)
    pair_mode.add_argument("--encoder-report", type=Path)
    pair_mode.add_argument("--legacy-unpaired-diagnostic", action="store_true")
    pair_mode.add_argument("--original-native23-v14-diagnostic", action="store_true")
    parser.add_argument("--candidate-summary", type=Path)
    parser.add_argument("--packets", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--preserve-calibrated-source-orientation", action="store_true")
    parser.add_argument("--virtual-source-reference-diagnostic", action="store_true")
    parser.add_argument("--source-action-units-diagnostic", action="store_true")
    args = parser.parse_args(argv)
    if bool(args.candidate_summary) != args.legacy_unpaired_diagnostic:
        parser.error("--candidate-summary belongs only to --legacy-unpaired-diagnostic")
    if args.original_native23_v14_diagnostic:
        if args.decoder_report is not None:
            parser.error("original native23 comparison uses pinned models, not --decoder-report")
    elif args.decoder_report is None:
        parser.error("paired and legacy modes require --decoder-report")
    if args.source_action_units_diagnostic and args.encoder_report is None:
        parser.error("source action convention ablation requires an explicit validated model pair")
    return args


def build_recording_controller(args):
    if args.legacy_unpaired_diagnostic:
        profile = load_frozen_lora_live_profile(
            decoder_report_path=args.decoder_report, candidate_summary_path=args.candidate_summary
        )
        return build_live_controller(repository_root=args.repository_root, profile=profile), dict(
            diagnostic_pair=None,
            legacy_unpaired_diagnostic=True,
            encoder_sha256=sha256_file(args.repository_root / ENCODER_RELATIVE_PATH),
            decoder_sha256=profile.decoder_sha256,
            decoder_report=str(profile.decoder_report_path),
            decoder_report_sha256=profile.decoder_report_sha256,
            candidate_summary=str(profile.candidate_summary_path),
            candidate_summary_sha256=profile.candidate_summary_sha256,
            checkpoint_update_count=profile.base_update_count,
            residual_alpha=profile.residual_alpha,
        )
    if getattr(args, "original_native23_v14_diagnostic", False):
        # Explicit comparison only: these are the original mode registry's
        # frozen encoder and V14-100 decoder, not a new LoRA checkpoint pair.
        encoder_path = args.repository_root / ENCODER_RELATIVE_PATH
        decoder_path = args.repository_root / ORIGINAL_WALK_DECODER
        identity = dict(
            diagnostic_pair=None,
            original_native23_v14_diagnostic=True,
            legacy_unpaired_diagnostic=False,
            encoder_sha256=ORIGINAL_WALK_ENCODER_SHA256,
            decoder_sha256=ORIGINAL_WALK_DECODER_SHA256,
            encoder_path=str(encoder_path.resolve()),
            decoder_path=str(decoder_path.resolve()),
            checkpoint_update_count=100,
            residual_alpha=None,
        )
    else:
        pair = load_diagnostic_pair(args.encoder_report, args.decoder_report)
        encoder_path, decoder_path = Path(pair["encoder"]["path"]), Path(pair["decoder"]["path"])
        identity = dict(
            diagnostic_pair=pair,
            original_native23_v14_diagnostic=False,
            legacy_unpaired_diagnostic=False,
            encoder_sha256=pair["encoder"]["sha256"],
            decoder_sha256=pair["decoder"]["sha256"],
            decoder_report=pair["decoder"]["report_path"],
            decoder_report_sha256=pair["decoder"]["report_sha256"],
            checkpoint_update_count=pair["source"]["update_count"],
            residual_alpha=None,
        )
    controller = SupervisedCleanTrue23MujocoController(
        model_path=args.repository_root / MODEL,
        physics_path=args.repository_root / PHYSICS,
        minimum_base_height_m=0.30,
        maximum_base_tilt_rad=1.0,
        fallback_tilt_trigger_rad=0.50,
        policy=ExactHashSonicPolicy(
            encoder_path=encoder_path,
            decoder_path=decoder_path,
            expected_encoder_sha256=identity["encoder_sha256"],
            expected_decoder_sha256=identity["decoder_sha256"],
        ),
        fallback_policy=UnitreeZeroVelocityFallbackPolicy(args.repository_root / FALLBACK_RELATIVE_PATH),
    )
    controller.use_released_retained_gains()
    return controller, identity


def main():
    args = parse_args()
    packets = load_reference_packets(args.packets)
    if not 2 <= len(packets) <= 10_000:
        raise ValueError("saved diagnostic requires 2 to 10000 packets")
    if args.output_directory.exists():
        raise FileExistsError("refusing to overwrite recorded SIM evidence")
    controller, policy_identity = build_recording_controller(args)
    args.output_directory.mkdir(parents=True, exist_ok=False)
    if args.preserve_calibrated_source_orientation:
        preserve_calibrated_source_orientation(controller, packets[0])
    source_model_path = args.repository_root / "gear_sonic/data/robots/g1/g1_29dof.xml"
    if args.virtual_source_reference_diagnostic:
        preserve_virtual_source_geometry(controller, packets, source_model_path)
    action_diagnostic = None
    if args.source_action_units_diagnostic:
        action_diagnostic = SourceActionDiagnosticPolicy(controller.policy)
        controller.policy = action_diagnostic
    arrays, report = record(controller, packets)
    if action_diagnostic is not None:
        arrays.update(action_diagnostic.trace_arrays())
    trace_path = args.output_directory / "measured_trace.npz"
    with trace_path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    report.update(
        kind="g1_true23_saved_fullbody_recorded_diagnostic_v1",
        packets=str(args.packets.resolve()),
        packets_sha256=sha256_file(args.packets),
        **policy_identity,
        model_sha256=sha256_file(args.repository_root / MODEL),
        trace_path=str(trace_path.resolve()),
        trace_sha256=sha256_file(trace_path),
        initialization="same_source_pose_initialization_as_live_SIM_consumer",
        reference_orientation=(
            "source_orientation_with_once_only_initial_yaw_alignment_SIM_experiment"
            if args.preserve_calibrated_source_orientation
            else "unchanged_live_consumer_identity_orientation"
        ),
        reference_geometry=(
            VIRTUAL_SOURCE_REFERENCE if args.virtual_source_reference_diagnostic else "native23_forward_kinematics"
        ),
        reference_only_source29_model_sha256=(
            sha256_file(source_model_path) if args.virtual_source_reference_diagnostic else None
        ),
        action_convention=(SOURCE_ACTION_CONVENTION if action_diagnostic is not None else "native_tanh"),
        action_unit_diagnostic=None if action_diagnostic is None else action_diagnostic.evidence(),
        source_29dof_physics_used=False,
        future_reference_consumed=False,
        live_consumer_modified=False,
        recorded_simulation_only=True,
        live_transport_proven=False,
        live_headset_source_proven=False,
        tracking_fidelity_qualified=False,
        hardware_authorized=False,
        dds_opened=False,
        robot_commands_published=False,
    )
    with (args.output_directory / "report.json").open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
