"""Localize existing paired-SONIC tracking errors without changing its runtime.

SIM only. Replay walk002 at its original sample rate, record all three
reference/input/action/plant boundaries, and fork three one-control experiments
from the first five-sample sustained leg error. Original29 task goals remain
available alongside the native joint reference. No training or qualification.
"""

import argparse
import copy
import json
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import onnxruntime as ort

from gear_sonic.scripts.record_g1_true23_saved_teleop_diagnostic import (
    build_recording_controller, initialize_live_controller, load_reference_packets,
    preserve_calibrated_source_orientation, step_live_packet,
)
from gear_sonic.teleop.cpu_paced_inference import prepare_nonspinning_sessions
from gear_sonic.utils.g1_23dof_contract import (
    HARDWARE_23_JOINT_NAMES, SOURCE_MJ29_KEEP_INDICES,
)
from gear_sonic.utils.g1_true23_clean_mujoco_teleop import (
    MJ_TO_NATIVE, encoder267_from_reference, safe_target_transform_numpy, sha256_file,
)
from gear_sonic.utils.g1_true23_sonic_library_replay import ReferenceActionPolicy


class CapturePolicy:
    def __init__(self, policy):
        self.policy = policy
        self.rows = []

    def infer(self, encoder, history):
        raw, decoder = self.policy.infer(encoder, history)
        self.rows.append(dict(encoder=encoder.copy(), history=history.copy(),
                              raw=raw.copy(), decoder=decoder.copy()))
        return raw, decoder


def integrate_target(controller, state, target):
    """Copy complete MjData; no mj_forward/solver reset before continuation."""
    probe = copy.copy(state)
    saturation = np.zeros(23, dtype=int)
    torques = []
    for _ in range(controller.physics.decimation):
        demand = controller.physics.kp * (target.astype(np.float64) - probe.qpos[7:])
        demand -= controller.physics.kd * probe.qvel[6:]
        saturation += np.abs(demand) > controller.physics.effort
        applied = np.clip(demand, -controller.physics.effort, controller.physics.effort)
        probe.ctrl[:] = applied
        torques.append(applied.copy())
        mujoco.mj_step(controller.model, probe)
    return probe, np.asarray(torques), saturation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository-root', type=Path, required=True)
    parser.add_argument('--recording-directory', type=Path, required=True)
    parser.add_argument('--output-directory', type=Path, required=True)
    args = parser.parse_args()
    root, source, output = args.repository_root, args.recording_directory, args.output_directory
    if output.exists():
        raise FileExistsError(output)
    pair = root / 'artifacts/g1_true23_frozen_lora/paired_encoder_20260905_v2/original_breadth25'
    controller, identity = build_recording_controller(SimpleNamespace(
        repository_root=root, encoder_report=pair / 'model_25.diagnostic.encoder.json',
        decoder_report=pair / 'model_25.diagnostic.decoder.json',
        original_native23_v14_diagnostic=False, legacy_unpaired_diagnostic=False,
    ))
    identity = prepare_nonspinning_sessions(controller, identity, root)
    policy = CapturePolicy(controller.policy)
    controller.policy = policy
    packets_path = source / 'causal_packets.json'
    packets = load_reference_packets(packets_path)
    motion_path = source / 'motion.native23.npz'
    original_path = source / 'original_source_bundle_v1/original_reference.npz'
    with np.load(motion_path, allow_pickle=False) as data:
        motion = {k: data[k] for k in data.files}
    with np.load(original_path, allow_pickle=False) as data:
        original = {k: data[k] for k in data.files}
    original_joints = original['source_qpos29'][:, 7 + np.asarray(SOURCE_MJ29_KEEP_INDICES)]
    # Bind the lifecycle section by exact joint identity, not a guessed time offset.
    starts = np.flatnonzero(np.max(np.abs(original_joints - motion['joint_pos'][0]), axis=1) == 0)
    matches = [int(i) for i in starts if i + len(motion['joint_pos']) <= len(original_joints)
               and np.array_equal(original_joints[i:i + len(motion['joint_pos'])], motion['joint_pos'])]
    if len(matches) != 1:
        raise ValueError('original task/reference lineage is ambiguous')
    offset = matches[0]
    preserve_calibrated_source_orientation(controller, packets[0])
    initialize_live_controller(controller, packets[0], packets[1])
    qpos, qvel = [controller.data.qpos.copy()], [controller.data.qvel.copy()]
    arrays = {k: [] for k in ('reference_joint', 'target', 'original_vr21', 'admitted_vr21',
                             'incoming_vr21', 'leg_rmse', 'source_frame')}
    recent = deque(maxlen=5)
    first = None
    failure = None
    for index, packet in enumerate(packets):
        control_frame = packet['control_source_frame_index']
        anchor_frame = packet['pico_anchor_source_frame_index']
        np.testing.assert_allclose(packet['q_ref23_native'],
                                   motion['joint_pos'][anchor_frame, MJ_TO_NATIVE], atol=1e-7, rtol=0)
        lower = np.asarray(packet['causal_history_lower_body']).reshape(2, 10, 12)
        np.testing.assert_allclose(lower[0], motion['joint_pos'][anchor_frame - 9:anchor_frame + 1, :12], atol=1e-7, rtol=0)
        reference = motion['joint_pos'][control_frame + 1]
        # Integrated state is at q(control+1), not at the received q9 anchor.
        admitted = controller.retarget_pico_reference_packet(packet)
        source_vr = original['virtual_vr21'][offset + anchor_frame]
        pre = copy.copy(controller.data)
        buffered = controller.buffered_robot_pelvis_q9.copy()
        before_rows = len(policy.rows)
        try:
            evidence = step_live_packet(controller, packet)
        except RuntimeError as error:
            failure = str(error)
            break
        if len(policy.rows) != before_rows + 1 or evidence['fallback_active']:
            failure = 'SONIC trace ended: fallback or missing actor command'
            break
        row = policy.rows[-1]
        _, target = safe_target_transform_numpy(row['raw'])
        error = float(np.sqrt(np.mean((controller.data.qpos[7:19] - reference[:12]) ** 2)))
        recent.append(dict(index=index, state=pre, buffered=buffered, packet=admitted,
                           source_vr=source_vr.copy(), row=row, target=target.copy(),
                           reference=reference.copy(), post=controller.data.qpos.copy(),
                           post_velocity=controller.data.qvel.copy(), error=error))
        if first is None and len(recent) == 5 and all(r['error'] > .15 for r in recent):
            first = recent[0]
        qpos.append(controller.data.qpos.copy())
        qvel.append(controller.data.qvel.copy())
        for name, value in dict(reference_joint=reference, target=target,
                                original_vr21=source_vr,
                                admitted_vr21=np.r_[admitted['vr_3point_local_target'], admitted['vr_3point_local_orn_target']],
                                incoming_vr21=np.r_[packet['vr_3point_local_target'], packet['vr_3point_local_orn_target']],
                                leg_rmse=error, source_frame=control_frame).items():
            arrays[name].append(value)
    arrays = {k: np.asarray(v) for k, v in arrays.items()}
    arrays.update(qpos=np.asarray(qpos), qvel=np.asarray(qvel))
    for key in ('encoder', 'history', 'raw', 'decoder'):
        arrays[key] = np.asarray([row[key] for row in policy.rows])
    position_loss = np.linalg.norm((arrays['admitted_vr21'][:, :9] - arrays['original_vr21'][:, :9]).reshape(-1, 3, 3), axis=2)
    incoming_loss = np.linalg.norm((arrays['incoming_vr21'][:, :9] - arrays['original_vr21'][:, :9]).reshape(-1, 3, 3), axis=2)
    report = dict(
        kind='genuine_sonic_three_boundary_diagnosis_v1', policy_identity=identity,
        reference_source_offset=offset, original_retained_joints_exact=True,
        completed_controls=len(arrays['leg_rmse']), failure=failure,
        first_sustained_definition='post-control leg RMSE > 0.15 rad for five consecutive 20ms samples; diagnostic localization, not a replacement acceptance gate',
        leg_rmse_all_samples_rad=float(np.sqrt(np.mean(arrays['leg_rmse'] ** 2))),
        admitted_original_task_position_loss_first_m=position_loss[0].tolist(),
        admitted_original_task_position_loss_p95_m=np.percentile(position_loss, 95, axis=0).tolist(),
        incoming_original_task_position_loss_p95_m=np.percentile(incoming_loss, 95, axis=0).tolist(),
        task_order=['left_hand', 'right_hand', 'head'],
        runtime_versions=dict(mujoco=mujoco.__version__, numpy=np.__version__, onnxruntime=ort.__version__),
        received_joint_mapping_and_past_position_samples_checked=True,
        dynamic_feasibility_proven=False, policy_insufficiency_proven=False,
        limitation='walk002 import is not dynamically qualified. Qualified prepared walk003 used a different waist gain. No feasibility or policy-capability claim transfers from it.',
        timing_qualified=False, standing_qualified=False, deployment_ready=False,
        counterfactual_scope='One 20ms command from identical full MjData; input change or target change only. Reference-target branch is a local plant probe, not a controller or feasibility proof.',
    )
    if first:
        probes = {}
        corrected_packet = copy.deepcopy(first['packet'])
        corrected_packet['vr_3point_local_target'] = first['source_vr'][:9].tolist()
        corrected_packet['vr_3point_local_orn_target'] = first['source_vr'][9:].tolist()
        corrected_encoder = encoder267_from_reference(corrected_packet, first['buffered'])
        corrected_raw, _ = policy.policy.infer(corrected_encoder, first['row']['history'])
        _, corrected_target = safe_target_transform_numpy(corrected_raw)
        reference_policy = ReferenceActionPolicy()
        try:
            reference_policy.set_target(first['reference'])
            _, reference_target = safe_target_transform_numpy(reference_policy.raw)
        except ValueError as error:
            reference_target = None
            probes['reference_target_unavailable'] = str(error)
        targets = dict(nominal=first['target'], original_task_input=corrected_target)
        if reference_target is not None:
            targets['reference_target'] = reference_target
        # A local causal probe, chosen after the initial trace: remove the
        # nominal command's extra knee flexion without altering other targets.
        # This is not reference projection or a proposed recovery controller.
        knee_hold = first['target'].astype(np.float64).copy()
        for joint in ('left_knee_joint', 'right_knee_joint'):
            j = list(HARDWARE_23_JOINT_NAMES).index(joint)
            knee_hold[j] = first['state'].qpos[7 + j]
        hold_policy = ReferenceActionPolicy()
        hold_policy.set_target(knee_hold)
        _, targets['knee_position_hold'] = safe_target_transform_numpy(hold_policy.raw)
        for name, target in targets.items():
            probe, torques, saturation = integrate_target(controller, first['state'], target)
            probes[name] = dict(
                post_leg_rmse_rad=float(np.sqrt(np.mean((probe.qpos[7:19] - first['reference'][:12]) ** 2))),
                target_hardware_rad=target.tolist(), post_joint_hardware_rad=probe.qpos[7:].tolist(),
                torque_saturated_substeps=saturation.tolist(), max_abs_torque_nm=np.max(abs(torques), axis=0).tolist(),
                baseline_post_qpos_max_abs_difference=float(np.max(abs(probe.qpos - first['post']))),
                baseline_post_qvel_max_abs_difference=float(np.max(abs(probe.qvel - first['post_velocity']))),
            )
        if probes['nominal']['baseline_post_qpos_max_abs_difference'] != 0 or probes['nominal']['baseline_post_qvel_max_abs_difference'] != 0:
            raise ValueError('copied simulator state did not reproduce nominal continuation exactly')
        report['first_sustained'] = dict(
            control_index=first['index'], post_time_s=(first['index'] + 1) * .02,
            source_control_frame=int(arrays['source_frame'][first['index']]),
            joint_names=list(HARDWARE_23_JOINT_NAMES),
            before_joint_hardware_rad=first['state'].qpos[7:].tolist(),
            before_velocity_hardware_rad_s=first['state'].qvel[6:].tolist(),
            reference_next_joint_hardware_rad=first['reference'].tolist(),
            raw_native23=first['row']['raw'].tolist(),
            corrected_raw_native23=corrected_raw.tolist(),
            original_goal_encoder_changed_coordinates=np.flatnonzero(corrected_encoder != first['row']['encoder']).tolist(),
            probes=probes,
        )
    else:
        report['first_sustained'] = None
    report['inputs_sha256'] = {str(p): sha256_file(p) for p in
                               (packets_path, motion_path, original_path, Path(__file__))}
    old_path = source / 'paired_orientation/measured_trace.npz'
    with np.load(old_path, allow_pickle=False) as old:
        report['archived_baseline_comparison'] = {
            k: dict(exact=bool(np.array_equal(arrays[k], old[k])),
                    max_abs_difference=float(np.max(abs(arrays[k] - old[k]))))
            for k in ('qpos', 'qvel') if arrays[k].shape == old[k].shape
        }
    report['inputs_sha256'][str(old_path)] = sha256_file(old_path)
    output.mkdir(parents=True)
    mujoco.mj_saveModel(controller.model, str(output / 'model.mjb'), None)
    report['model_sha256'] = sha256_file(output / 'model.mjb')
    report['kp_hardware'] = controller.physics.kp.tolist()
    report['kd_hardware'] = controller.physics.kd.tolist()
    report['effort_hardware'] = controller.physics.effort.tolist()
    np.savez_compressed(output / 'trace.npz', **arrays)
    report['trace_sha256'] = sha256_file(output / 'trace.npz')
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('policy_identity', 'inputs_sha256', 'first_sustained')}, indent=2))
    if first:
        print('FIRST', first['index'], 'PROBES', json.dumps({k: v.get('post_leg_rmse_rad') if isinstance(v, dict) else v for k, v in probes.items()}))


if __name__ == '__main__':
    main()
