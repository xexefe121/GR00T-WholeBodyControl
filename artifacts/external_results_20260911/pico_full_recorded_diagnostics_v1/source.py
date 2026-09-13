"""Full-source saved-state PICO diagnostics; no actor, optimizer or dynamics.

Run only after final trace and root physical/intent audits exist. Model operations
are constant preparation and kinematics of recorded poses, never mj_step/rollout.
All primary source and quiet values must reproduce the bound root intent report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path('Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof')
NAMES = ('left_hand', 'right_hand', 'head_proxy')
RECOVERY_SOURCE_SECONDS = 69.0
SOURCE_LIMITS = dict(root_m=.20, yaw_deg=15., hands_m=.15, head_m=.10, feet_m=.12, legs_rad=.15)
QUIET_LIMITS = dict(root_xy_p95_m=.05, original_yaw_p95_deg=5., root_linear_speed_p95_mps=.05,
    joint_speed_p95_radps=.5, joint_speed_max_radps=2., tilt_max_rad=.15)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_npz(path, fields=None):
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key].copy() for key in (saved.files if fields is None else fields)}


def yaw(q):
    w, x, y, z = np.asarray(q).T
    return np.arctan2(2 * (w*z+x*y), 1-2*(y*y+z*z))


def yaw_error(q, ref):
    delta = yaw(q)-yaw(ref)
    return np.abs(np.degrees(np.arctan2(np.sin(delta), np.cos(delta))))


def stats(values):
    values = np.asarray(values)
    assert len(values) and np.isfinite(values).all()
    return dict(p50=float(np.percentile(values, 50)), p95=float(np.percentile(values, 95)),
                maximum=float(np.max(values)))


def same(value, reference, name):
    np.testing.assert_array_equal(np.asarray(value), np.asarray(reference), err_msg=name)


def task_fk(model, data, pose, tasks, side):
    data.qpos[:] = pose
    mujoco.mj_kinematics(model, data)
    positions, rotations = [], []
    for task in tasks:
        body = model.body(task[side+'_body']).id
        matrix = data.xmat[body].reshape(3, 3)
        positions.append(data.xpos[body] + matrix @ np.asarray(task[side+'_point']))
        rotations.append(matrix.copy())
    return np.asarray(positions), np.asarray(rotations)


def main(args):
    assert args.trace.name == 'trace.npz', 'Final trace required; never use a rolling partial trace.'
    assert not args.output.exists(), 'Preserve an existing diagnostic result.'
    physics = json.loads(args.physical_report.read_text())
    intent = json.loads(args.intent_report.read_text())
    trace_sha, physical_sha = sha(args.trace), sha(args.physical_report)
    assert trace_sha in physics['input_hashes'].values()
    assert trace_sha in intent['hashes'].values() and physical_sha in intent['hashes'].values()
    assert physics['recorded_trace_reproduced_through_last_sample']
    assert intent['clip'] == 'pico' and intent['global_start'] == 0
    assert intent['requested_controls'] == physics['intended_segment_controls'] == 6530
    assert not intent['dynamics_executed']
    manifest_path, contract_path = args.bundle/'manifest.json', args.bundle/'contract.json'
    manifest, contract = [json.loads(p.read_text()) for p in (manifest_path, contract_path)]
    model_xml, model_arrays = args.bundle/'native_prepared.xml', args.bundle/'prepared_model_arrays.npz'
    assert sha(model_xml) == manifest['portable_xml_sha256']
    assert sha(model_arrays) == manifest['prepared_arrays_sha256']
    mesh_paths = []
    for name, digest in manifest['meshes'].items():
        path = args.bundle/'meshes'/name
        assert sha(path) == digest
        mesh_paths.append(path)
    source_path, timeline_path = args.bundle/'pico/original29.npz', args.bundle/'pico/timeline.json'
    assert sha(source_path) == manifest['cases']['pico']['original29.npz']
    receipt_path = args.reference.parent/'portable_receipt.json'
    receipt, timeline = [json.loads(p.read_text()) for p in (receipt_path, timeline_path)]
    assert receipt['clip'] == 'pico' and receipt['reference_sha256'] == sha(args.reference)
    assert receipt['original29_reference_sha256'] == sha(source_path)
    assert receipt['timeline_sha256'] == sha(timeline_path)
    for path in (args.reference, source_path, args.source_model, timeline_path, model_xml, model_arrays):
        assert sha(path) in intent['hashes'].values(), ('root intent input identity', str(path))
    phase = next(p for p in timeline['phases'] if p['name'] == 'source_motion')
    assert (phase['control_start'], phase['control_stop'], phase['requested_controls']) == (350, 6130, 5780)
    assert timeline['total_requested_controls'] == 6530
    fields = ('qpos', 'qvel', 'source_frame', 'physics_substeps', 'physics_qpos', 'physics_qvel', 'physics_time')
    t = load_npz(args.trace, fields)
    with np.load(args.trace, allow_pickle=False) as saved:
        saved_global = saved['global_control'].copy() if 'global_control' in saved else None
    original = load_npz(source_path, ('source_qpos29', 'source_task_position_w', 'source_task_quaternion_wxyz'))
    motion = load_npz(args.reference, ('joint_pos', 'body_pos_w'))
    count = len(t['source_frame'])
    controls = np.arange(count)
    if saved_global is not None:
        same(controls, saved_global, 'recorded global controls')
    frames = np.minimum(controls+11, len(motion['joint_pos'])-1)
    same(t['source_frame'], frames, 'original timing preserved')
    counts = t['physics_substeps']
    assert np.issubdtype(counts.dtype, np.integer) and np.all(counts[:-1] == 10)
    assert 1 <= counts[-1] <= 10 and count <= 6530
    boundaries = np.r_[0, np.cumsum(counts)]
    assert boundaries[-1] == physics['recorded_physics_steps']
    assert len(t['physics_qpos']) == len(t['physics_qvel']) == len(t['physics_time']) == boundaries[-1]+1
    for key in ('qpos', 'qvel'):
        same(t[key], t['physics_'+key][boundaries], 'control/substep boundary '+key)
    assert all(np.isfinite(t[k]).all() for k in fields)
    selected = (counts == 10) & (controls >= 350) & (controls < 6130)
    source_controls, source_frames = controls[selected], frames[selected]
    assert len(source_controls) == 5780, 'No cropped/partial source may be called a full-source diagnostic.'
    same(source_controls, np.arange(350, 6130), 'all source controls')
    poses = t['qpos'][1:][selected]
    ref_qpos = original['source_qpos29'][source_frames]
    ref_points = original['source_task_position_w'][source_frames]
    ref_quats = original['source_task_quaternion_wxyz'][source_frames]
    assert np.isfinite(ref_qpos).all() and np.isfinite(ref_points).all() and np.isfinite(ref_quats).all()
    native = mujoco.MjModel.from_xml_path(str(model_xml))
    prepared = load_npz(model_arrays)
    for key, value in prepared.items():
        getattr(native, key)[:] = value
    mujoco.mj_setConst(native, mujoco.MjData(native))
    for key, value in prepared.items():
        same(getattr(native, key), value, 'native prepared geometry '+key)
    source = mujoco.MjModel.from_xml_path(str(args.source_model))
    assert (native.nq, native.nv, native.nu, source.nq, source.nv) == (30, 29, 23, 36, 35)
    assert [native.joint(i).name for i in range(1, 24)] == contract['joint_names']
    convention = intent['convention']
    assert convention['task_point_convention'] == 'native23_source_neutral_wrist_hand_proxy_v1'
    tasks = [next(task for task in convention['tasks'] if task['name'] == name) for name in NAMES]
    native_data, source_data = mujoco.MjData(native), mujoco.MjData(source)
    # Re-derive each corrected neutral hand point from the actual source model.
    neutral_s, neutral_n = np.zeros(source.nq), np.zeros(native.nq)
    neutral_s[2:4] = neutral_n[2:4] = [.8, 1.]
    neutral_source_points, neutral_source_rotations = task_fk(source, source_data, neutral_s, tasks, 'source')
    task_fk(native, native_data, neutral_n, tasks, 'target')
    neutral_point_deltas = []
    for i, task in enumerate(tasks[:2]):
        body = native.body(task['target_body']).id
        matrix = native_data.xmat[body].reshape(3, 3)
        derived = matrix.T @ (neutral_source_points[i]-native_data.xpos[body])
        np.testing.assert_allclose(derived, task['target_point'], atol=1e-9, rtol=0)
        np.testing.assert_allclose(matrix, neutral_source_rotations[i], atol=1e-9, rtol=0)
        neutral_point_deltas.append(float(np.max(np.abs(derived-np.asarray(task['target_point'])))))
    actual_points, actual_rotations, derived_points, derived_rotations, feet = [], [], [], [], []
    foot_ids = [native.body(side+'_ankle_roll_link').id for side in ('left', 'right')]
    for actual, wanted in zip(poses, ref_qpos, strict=True):
        position, rotation = task_fk(native, native_data, actual, tasks, 'target')
        actual_points.append(position); actual_rotations.append(rotation)
        feet.append(native_data.xpos[foot_ids].copy())
        position, rotation = task_fk(source, source_data, wanted, tasks, 'source')
        derived_points.append(position); derived_rotations.append(rotation)
    actual_points, actual_rotations, feet, derived_points, derived_rotations = map(np.asarray,
        (actual_points, actual_rotations, feet, derived_points, derived_rotations))
    # Original task cache remains authoritative for exact root-audit parity.
    # Independent original29 FK verifies that cache without changing its values.
    source_position_delta = np.linalg.norm(derived_points-ref_points, axis=-1)
    cached_rotation = Rotation.from_quat(ref_quats.reshape(-1, 4)[:, [1, 2, 3, 0]])
    source_orientation_delta = (cached_rotation.inv()*Rotation.from_matrix(derived_rotations.reshape(-1, 3, 3))).magnitude().reshape(-1, 3)
    assert np.max(source_position_delta) < 1e-8, 'Original29 cached task point differs from source-model FK.'
    assert np.max(source_orientation_delta) < 1e-7, 'Original29 cached task orientation differs from source-model FK.'
    orientation_error = np.degrees((cached_rotation.inv()*Rotation.from_matrix(actual_rotations.reshape(-1, 3, 3))).magnitude().reshape(-1, 3))
    root = poses[:, :3]
    root_error = np.linalg.norm(root-ref_qpos[:, :3], axis=-1)
    heading_error = yaw_error(poses[:, 3:7], ref_qpos[:, 3:7])
    task_world_error = np.linalg.norm(actual_points-ref_points, axis=-1)
    task_relative_error = np.linalg.norm((actual_points-root[:, None])-(ref_points-ref_qpos[:, None, :3]), axis=-1)
    native_root = motion['body_pos_w'][source_frames, 0]
    native_feet = motion['body_pos_w'][source_frames][:, [6, 12]]
    foot_error = np.linalg.norm((feet-root[:, None])-(native_feet-native_root[:, None]), axis=-1)
    joint_error = poses[:, 7:]-motion['joint_pos'][source_frames]
    leg_error = poses[:, 7:19]-motion['joint_pos'][source_frames, :12]
    leg_frame_rms = np.sqrt(np.mean(leg_error**2, axis=1))
    metrics = dict(source_controls=len(poses), requested_source_controls=5780,
        original_root_p95_m=float(np.percentile(root_error, 95)),
        original_yaw_p95_deg=float(np.percentile(heading_error, 95)),
        original_relative_hand_head_p95_m=np.percentile(task_relative_error, 95, axis=0).tolist(),
        world_axis_relative_foot_p95_m=np.percentile(foot_error, 95, axis=0).tolist(),
        native_leg_rmse_rad=float(np.sqrt(np.mean(leg_error**2))))
    for key, value in metrics.items():
        same(value, intent['source_metrics'][key], 'aggregate source parity '+key)
    gates = dict(root=metrics['original_root_p95_m'] <= .20, yaw=metrics['original_yaw_p95_deg'] <= 15,
        hands=bool(np.all(np.asarray(metrics['original_relative_hand_head_p95_m'])[:2] <= .15)),
        head=bool(metrics['original_relative_hand_head_p95_m'][2] <= .10),
        feet=bool(np.all(np.asarray(metrics['world_axis_relative_foot_p95_m']) <= .12)),
        legs=metrics['native_leg_rmse_rad'] <= .15)
    assert gates == intent['source_metric_gates']
    q, dq = t['physics_qpos'][-1500:], t['physics_qvel'][-1500:]
    quiet_frames = np.repeat(frames, counts)[-1500:]
    quiet_ref = original['source_qpos29'][quiet_frames]
    speed = np.max(np.abs(dq[:, 6:]), axis=-1)
    quiet = dict(samples=1500, interval=[float(t['physics_time'][-1501]), float(t['physics_time'][-1])],
        root_xy_p95_m=float(np.percentile(np.linalg.norm(q[:, :2]-quiet_ref[:, :2], axis=-1), 95)),
        original_yaw_p95_deg=float(np.percentile(yaw_error(q[:, 3:7], quiet_ref[:, 3:7]), 95)),
        root_linear_speed_p95_mps=float(np.percentile(np.linalg.norm(dq[:, :3], axis=-1), 95)),
        joint_speed_p95_radps=float(np.percentile(speed, 95)), joint_speed_max_radps=float(speed.max()),
        tilt_max_rad=float(np.arccos(np.clip(1-2*np.sum(q[:, 4:6]**2, axis=-1), -1, 1)).max()))
    for key, value in quiet.items():
        same(value, intent['quiet_last_three_seconds'][key], 'quiet parity '+key)
    quiet['gates'] = {key: quiet[key] <= value for key, value in QUIET_LIMITS.items()}
    assert quiet['gates'] == intent['quiet_last_three_seconds']['gates']
    source_seconds = (source_controls-350)*.02
    assert source_seconds[3450] == RECOVERY_SOURCE_SECONDS and source_controls[3450] == 3800
    summary = dict(kind='full_source_saved_state_diagnostics_no_dynamics', source_controls=5780,
        source_reference_seconds=[0., 115.6], physical_report_pass=physics['independent_segment_pass'],
        full_lifecycle_source_intent_pass=intent['full_lifecycle_source_intent_pass'],
        requested_segment_quiet_pass=intent['requested_segment_quiet_pass'], aggregate_source_metrics=metrics,
        source_gates=gates, quiet_last_three_seconds=quiet, source_and_quiet_aggregates_bitexact_to_root=True,
        root_world_error_m=stats(root_error), original_yaw_error_deg=stats(heading_error),
        task_world_error_m={name: stats(task_world_error[:, i]) for i, name in enumerate(NAMES)},
        task_relative_error_m={name: stats(task_relative_error[:, i]) for i, name in enumerate(NAMES)},
        task_orientation_geodesic_deg={name: stats(orientation_error[:, i]) for i, name in enumerate(NAMES)},
        relative_foot_error_m={side: stats(foot_error[:, i]) for i, side in enumerate(('left', 'right'))},
        instantaneous_leg_rms_rad=stats(leg_frame_rms), original29_FK_point_cache_max_delta_m=float(source_position_delta.max()),
        original29_FK_orientation_cache_max_delta_rad=float(source_orientation_delta.max()),
        neutral_hand_point_derivation_max_deltas_m=neutral_point_deltas, task_convention=convention,
        recovery_marker=dict(source_reference_seconds=69., global_control=3800, changes_no_data=True),
        time_convention='x=(global_control-350)*.02; measurements are postcontrol states. Entire5780-source sample series retained.',
        thresholds=SOURCE_LIMITS, supplementary_world_orientation_metrics_have_no_new_acceptance_thresholds=True,
        policy_inference_calls=0, optimizer_calls=0, dynamics_steps=0, kinematics_only=True,
        hardware_authorized=False, timing_or_live_teleoperation_qualified=False,
        python=platform.python_version(), numpy=np.__version__, mujoco=mujoco.__version__)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 2, figsize=(15, 11), sharex=True, constrained_layout=True)
    panels = axes.ravel()
    panels[0].plot(source_seconds, root_error, color='#265d9b', lw=.9, label='Original root world error')
    panels[1].plot(source_seconds, heading_error, color='#744aa4', lw=.9, label='Original heading error')
    for i, color in enumerate(('#1d78b5', '#db6e22')):
        side = ('Left', 'Right')[i]
        panels[2].plot(source_seconds, task_relative_error[:, i], color=color, lw=.9, label=side+' relative')
        panels[2].plot(source_seconds, task_world_error[:, i], color=color, lw=.6, ls=':', alpha=.65, label=side+' world (diagnostic)')
        panels[4].plot(source_seconds, foot_error[:, i], color=color, lw=.9, label=side+' relative foot')
    panels[3].plot(source_seconds, task_relative_error[:, 2], color='#397d49', lw=.9, label='Relative head proxy')
    panels[3].plot(source_seconds, task_world_error[:, 2], color='#697f63', lw=.7, ls=':', label='World head proxy (diagnostic)')
    panels[5].plot(source_seconds, leg_frame_rms, color='#aa4040', lw=.9, label='Per-control 12-joint RMS')
    for ax, title, ylabel, limit, criterion in zip(panels,
        ('Root', 'Yaw', 'Hands', 'Head proxy', 'Feet', 'Leg joints'),
        ('Error (m)', 'Error (deg)', 'Error (m)', 'Error (m)', 'Error (m)', 'RMS error (rad)'),
        (.20, 15., .15, .10, .12, .15), ('p95', 'p95', 'relative p95', 'relative p95', 'p95', 'aggregate RMS'), strict=True):
        ax.axhline(limit, color='#a12222', ls='--', lw=.8, label=criterion+' criterion '+str(limit))
        ax.axvline(69., color='#222222', ls='-.', lw=.85)
        ax.set_xlim(0., 115.6); ax.set_ylim(bottom=0.)
        ax.set_title(title, loc='left'); ax.set_ylabel(ylabel); ax.grid(alpha=.18)
        ax.legend(loc='upper left', fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel('Source reference time (s)')
    verdict = 'PASS' if intent['full_lifecycle_source_intent_pass'] else 'FAIL'
    failed = ', '.join(key for key, value in gates.items() if not value) or 'none'
    fig.suptitle('PICO native23 - full 5,780-control source diagnostic\n'
        +'Source intent: '+verdict+'; failed groups: '+failed+'. Vertical marker: source69s / control3800.\n'
        +'Dashed criteria are aggregate tests, not per-sample limits. No cropping or dynamics replay.', fontsize=13)
    args.output.mkdir(parents=True, exist_ok=False)
    arrays_path, plot_path = args.output/'errors.npz', args.output/'full_source_six_panel.png'
    np.savez_compressed(arrays_path, global_control=source_controls, source_frame=source_frames,
        source_reference_seconds=source_seconds, measured_physics_boundary_time=t['physics_time'][boundaries[1:]][selected],
        actual_qpos=poses, original_qpos29=ref_qpos, actual_task_position_w=actual_points,
        original_task_position_w=ref_points, independently_derived_original_task_position_w=derived_points,
        actual_task_rotation_w=actual_rotations, independently_derived_original_task_rotation_w=derived_rotations,
        original_task_quaternion_wxyz=ref_quats, actual_root_position_w=root, original_root_position_w=ref_qpos[:, :3],
        actual_feet_position_w=feet, reference_native_root_position_w=native_root, reference_native_feet_position_w=native_feet,
        root_world_error_m=root_error, original_yaw_error_deg=heading_error, task_world_error_m=task_world_error,
        task_relative_error_m=task_relative_error, task_orientation_geodesic_deg=orientation_error,
        world_axis_relative_foot_error_m=foot_error, joint_error_rad=joint_error, leg_frame_rms_rad=leg_frame_rms,
        original29_FK_point_cache_delta_m=source_position_delta, original29_FK_orientation_cache_delta_rad=source_orientation_delta)
    fig.savefig(plot_path, dpi=170); plt.close(fig)
    inputs = [args.trace, args.physical_report, args.intent_report, args.reference, receipt_path, source_path,
        args.source_model, timeline_path, manifest_path, contract_path, model_xml, model_arrays, Path(__file__), *mesh_paths]
    summary['input_sha256'] = {str(path.resolve()): sha(path) for path in inputs}
    summary['output_sha256'] = {path.name: sha(path) for path in (arrays_path, plot_path)}
    (args.output/'report.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(dict(output=str(args.output), source_controls=5780, aggregates_match_root=True,
        source_intent_pass=summary['full_lifecycle_source_intent_pass'], quiet_pass=summary['requested_segment_quiet_pass'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('trace', 'physical-report', 'intent-report', 'reference'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--bundle', type=Path, default=ROOT/'artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1')
    parser.add_argument('--source-model', type=Path, default=ROOT.parent/'GR00T-WholeBodyControl/gear_sonic/data/robots/g1/g1_29dof.xml')
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent/'results')
    main(parser.parse_args())
