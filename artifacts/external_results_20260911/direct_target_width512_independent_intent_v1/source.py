"""Independent recorded-state intent/quiet metrics, bound to a separate physics audit."""

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from gear_sonic.utils.g1_true23_hand_frame_tasks import neutral_wrist_hand_tasks

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def yaw(q):
    w, x, y, z = np.asarray(q).T
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def yaw_error(q, ref):
    delta = yaw(q) - yaw(ref)
    return np.abs(np.degrees(np.arctan2(np.sin(delta), np.cos(delta))))


def main(args):
    assert args.requested_controls > 0 and args.global_start >= 0
    physics = json.loads(args.physical_audit.read_text())
    assert physics['intended_segment_controls'] == args.requested_controls
    assert sha(args.trace) in physics['input_hashes'].values()
    assert physics['recorded_trace_reproduced_through_last_sample']
    manifest = json.loads((args.bundle / 'manifest.json').read_text())
    assert sha(args.bundle / 'native_prepared.xml') == manifest['portable_xml_sha256']
    assert sha(args.bundle / 'prepared_model_arrays.npz') == manifest['prepared_arrays_sha256']
    for name, digest in manifest['meshes'].items():
        assert sha(args.bundle / 'meshes' / name) == digest
    native = mujoco.MjModel.from_xml_path(str(args.bundle / 'native_prepared.xml'))
    with np.load(args.bundle / 'prepared_model_arrays.npz', allow_pickle=False) as archive:
        for name in archive.files:
            getattr(native, name)[:] = archive[name]
        mujoco.mj_setConst(native, mujoco.MjData(native))
        for name in archive.files:
            np.testing.assert_array_equal(getattr(native, name), archive[name])
    contract = json.loads((args.bundle / 'contract.json').read_text())
    assert (native.nq, native.nv, native.nu) == (30, 29, 23)
    assert [native.joint(i).name for i in range(1, 24)] == contract['joint_names']
    original_path = args.bundle / args.clip / 'original29.npz'
    assert sha(original_path) == manifest['cases'][args.clip]['original29.npz']
    with np.load(original_path, allow_pickle=False) as a:
        original = {k: a[k].copy() for k in ('source_qpos29', 'source_task_position_w')}
    with np.load(args.reference, allow_pickle=False) as a:
        motion = {k: a[k].copy() for k in ('body_pos_w', 'body_quat_w', 'joint_pos',
            'body_lin_vel_w', 'body_ang_vel_w', 'joint_vel')}
    receipt_path = args.reference.parent / 'portable_receipt.json'
    receipt = json.loads(receipt_path.read_text())
    assert receipt['reference_sha256'] == sha(args.reference)
    assert receipt['original29_reference_sha256'] == sha(original_path)
    assert receipt['clip'] == args.clip
    timeline_path = args.bundle / args.clip / 'timeline.json'
    timeline = json.loads(timeline_path.read_text())
    assert receipt['timeline_sha256'] == sha(timeline_path)
    phase = next(p for p in timeline['phases'] if p['name'] == 'source_motion')
    fields = ('qpos', 'qvel', 'source_frame', 'physics_substeps',
              'physics_qpos', 'physics_qvel', 'physics_time')
    with np.load(args.trace, allow_pickle=False) as a:
        t = {k: a[k].copy() for k in fields}
        saved_global = a['global_control'].copy() if 'global_control' in a else None
        initial_integration = a['initial_integration'].copy() if 'initial_integration' in a else None
        integration_spec = int(a['integration_state_spec']) if 'integration_state_spec' in a else None
        warning_name = 'physics_warning_number' if 'physics_warning_number' in a else 'physics_warning_counts'
        initial_warnings = a[warning_name][0].copy() if warning_name in a else None
        initial_warning_lastinfo = a['physics_warning_lastinfo'][0].copy() if 'physics_warning_lastinfo' in a else None
    count = len(t['source_frame'])
    assert 0 < count <= args.requested_controls
    controls = args.global_start + np.arange(count)
    if saved_global is not None:
        np.testing.assert_array_equal(controls, saved_global)
    frames = np.minimum(controls + 11, len(motion['joint_pos']) - 1)
    np.testing.assert_array_equal(t['source_frame'], frames)
    counts = t['physics_substeps']
    assert np.issubdtype(counts.dtype, np.integer) and np.all(counts[:-1] == 10)
    assert 1 <= counts[-1] <= 10
    boundaries = np.r_[0, np.cumsum(counts)]
    assert len(t['physics_qpos']) == boundaries[-1] + 1
    assert len(t['physics_time']) == len(t['physics_qpos'])
    for key in ('qpos', 'qvel'):
        np.testing.assert_array_equal(t[key], t['physics_' + key][boundaries])
    assert all(np.isfinite(t[k]).all() for k in fields)
    initialization = 'bounded recorded-state segment; full lifecycle cannot be credited'
    if args.global_start == 0:
        q = motion['body_quat_w'][10, 0]
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, q)
        np.testing.assert_array_equal(t['qpos'][0], np.r_[motion['body_pos_w'][10, 0], q, motion['joint_pos'][10]])
        np.testing.assert_array_equal(t['qvel'][0], np.r_[motion['body_lin_vel_w'][10, 0],
            rotation.reshape(3, 3).T @ motion['body_ang_vel_w'][10, 0], motion['joint_vel'][10]])
        assert t['physics_time'][0] == 0
        initialization = 'canonical declared reference frame10 checked exactly'
    elif args.global_start == timeline['total_requested_controls']:
        if args.preceding_trace is None or initial_integration is None:
            raise ValueError('separate hold requires preceding full trace and exact full integration boundary')
        with np.load(args.preceding_trace, allow_pickle=False) as previous:
            assert len(previous['source_frame']) == args.global_start
            assert np.all(previous['physics_substeps'] == 10)
            if 'final_integration' in previous:
                np.testing.assert_array_equal(initial_integration, previous['final_integration'])
            elif args.preceding_endpoint is None:
                raise ValueError('preceding trace lacks full integration; require independently reconstructed endpoint')
            np.testing.assert_array_equal(t['qpos'][0], previous['qpos'][-1])
            np.testing.assert_array_equal(t['qvel'][0], previous['qvel'][-1])
            assert t['physics_time'][0] == previous['physics_time'][-1]
            if args.preceding_endpoint is not None:
                warning_name = ('physics_warning_number' if 'physics_warning_number' in previous
                                else 'physics_warning_counts')
                np.testing.assert_array_equal(initial_warnings, previous[warning_name][-1])
                np.testing.assert_array_equal(initial_warning_lastinfo, previous['physics_warning_lastinfo'][-1])
        if args.preceding_endpoint is not None:
            endpoint_report_path = args.preceding_endpoint.parent / 'report.json'
            endpoint_report = json.loads(endpoint_report_path.read_text())
            assert endpoint_report['kind'] == 'independent_native_full_integration_endpoint_reconstruction'
            assert endpoint_report['original_trace_sha256'] == sha(args.preceding_trace)
            assert endpoint_report['endpoint_sha256'] == sha(args.preceding_endpoint)
            assert endpoint_report['all_recorded_samples_bitexact']
            assert endpoint_report['independent_physical_audit_already_passed']
            assert endpoint_report['completed_controls'] == args.global_start
            assert endpoint_report['compared_physics_steps'] == args.global_start * 10
            with np.load(args.preceding_endpoint, allow_pickle=False) as endpoint:
                assert str(endpoint['original_trace_sha256']) == sha(args.preceding_trace)
                assert int(endpoint['completed_controls']) == args.global_start
                assert integration_spec == int(endpoint['integration_state_spec']) == int(mujoco.mjtState.mjSTATE_INTEGRATION)
                assert (endpoint_report['integration_state_size'] == len(initial_integration) ==
                        len(endpoint['final_integration']) == mujoco.mj_stateSize(native, integration_spec) == 291)
                np.testing.assert_array_equal(initial_integration, endpoint['final_integration'])
                np.testing.assert_array_equal(t['qpos'][0], endpoint['qpos'])
                np.testing.assert_array_equal(t['qvel'][0], endpoint['qvel'])
                np.testing.assert_array_equal(initial_warnings, endpoint['warning_counts'])
                np.testing.assert_array_equal(initial_warning_lastinfo, endpoint['warning_lastinfo'])
                assert t['physics_time'][0] == float(endpoint['time'])
        initialization = 'separate hold continuous with preceding full integration state/pose/velocity/clock'
    if args.preceding_endpoint is not None and args.global_start != timeline['total_requested_controls']:
        raise ValueError('independent endpoint is only valid for a separate hold after the full lifecycle')
    selected = (counts == 10) & (controls >= phase['control_start']) & (controls < phase['control_stop'])
    source_frames, poses = frames[selected], t['qpos'][1:][selected]
    original_model_path = ROOT.parent / 'GR00T-WholeBodyControl/gear_sonic/data/robots/g1/g1_29dof.xml'
    source_model = mujoco.MjModel.from_xml_path(str(original_model_path))
    tasks, convention = neutral_wrist_hand_tasks(source_model, native)
    tasks = [next(task for task in tasks if task.name == name) for name in ('left_hand', 'right_hand', 'head_proxy')]
    data, measured, feet = mujoco.MjData(native), [], []
    for pose in poses:
        data.qpos[:] = pose
        mujoco.mj_kinematics(native, data)
        measured.append([data.xpos[native.body(task.target_body).id] +
            data.xmat[native.body(task.target_body).id].reshape(3, 3) @ task.target_point for task in tasks])
        feet.append([data.xpos[native.body(side + '_ankle_roll_link').id].copy() for side in ('left', 'right')])
    metrics, gates = None, {}
    if len(poses):
        root = poses[:, :3]
        ref = original['source_qpos29'][source_frames]
        ref_points = original['source_task_position_w'][source_frames]
        hands = np.percentile(np.linalg.norm((np.asarray(measured) - root[:, None]) -
            (ref_points - ref[:, None, :3]), axis=-1), 95, axis=0)
        native_root = motion['body_pos_w'][source_frames, 0]
        native_feet = motion['body_pos_w'][source_frames][:, [6, 12]]
        foot = np.percentile(np.linalg.norm((np.asarray(feet) - root[:, None]) -
            (native_feet - native_root[:, None]), axis=-1), 95, axis=0)
        metrics = dict(source_controls=len(poses), requested_source_controls=phase['requested_controls'],
            original_root_p95_m=float(np.percentile(np.linalg.norm(root - ref[:, :3], axis=-1), 95)),
            original_yaw_p95_deg=float(np.percentile(yaw_error(poses[:, 3:7], ref[:, 3:7]), 95)),
            original_relative_hand_head_p95_m=hands.tolist(), world_axis_relative_foot_p95_m=foot.tolist(),
            native_leg_rmse_rad=float(np.sqrt(np.mean((poses[:, 7:19] - motion['joint_pos'][source_frames, :12]) ** 2))))
        gates = dict(root=metrics['original_root_p95_m'] <= .20, yaw=metrics['original_yaw_p95_deg'] <= 15,
            hands=bool(np.all(hands[:2] <= .15)), head=bool(hands[2] <= .10),
            feet=bool(np.all(foot <= .12)), legs=metrics['native_leg_rmse_rad'] <= .15)
    quiet = None
    if boundaries[-1] >= 1500:
        q, dq = t['physics_qpos'][-1500:], t['physics_qvel'][-1500:]
        qframes = np.repeat(frames, counts)[-1500:]
        original_q = original['source_qpos29'][qframes]
        speed = np.max(np.abs(dq[:, 6:]), axis=-1)
        quiet = dict(samples=1500, interval=[float(t['physics_time'][-1501]), float(t['physics_time'][-1])],
            root_xy_p95_m=float(np.percentile(np.linalg.norm(q[:, :2] - original_q[:, :2], axis=-1), 95)),
            original_yaw_p95_deg=float(np.percentile(yaw_error(q[:, 3:7], original_q[:, 3:7]), 95)),
            root_linear_speed_p95_mps=float(np.percentile(np.linalg.norm(dq[:, :3], axis=-1), 95)),
            joint_speed_p95_radps=float(np.percentile(speed, 95)), joint_speed_max_radps=float(speed.max()),
            tilt_max_rad=float(np.arccos(np.clip(1 - 2 * np.sum(q[:, 4:6] ** 2, axis=-1), -1, 1)).max()))
        quiet['gates'] = {key: quiet[key] <= limit for key, limit in (
            ('root_xy_p95_m', .05), ('original_yaw_p95_deg', 5), ('root_linear_speed_p95_mps', .05),
            ('joint_speed_p95_radps', .5), ('joint_speed_max_radps', 2), ('tilt_max_rad', .15))}
    complete = count == args.requested_controls and bool(np.all(counts == 10))
    lifecycle = args.global_start == 0 and args.requested_controls == timeline['total_requested_controls']
    physical_pass = bool(physics['independent_segment_pass'])
    report = dict(kind='independent_recorded_state_intent_inspection', clip=args.clip,
        requested_controls=args.requested_controls, recorded_controls=count, global_start=args.global_start,
        intended_segment_completed=complete, independent_physical_pass=physical_pass,
        source_metrics=metrics, source_metric_gates=gates, quiet_last_three_seconds=quiet,
        full_lifecycle_source_intent_pass=bool(lifecycle and complete and physical_pass and metrics and
            len(poses) == phase['requested_controls'] and all(gates.values())),
        requested_segment_quiet_pass=bool(complete and physical_pass and quiet and all(quiet['gates'].values())),
        dynamics_executed=False, timing_or_live_teleoperation_qualified=False, hardware_authorized=False,
        convention=convention, initialization_checked=initialization,
        hashes={str(p): sha(p) for p in (args.trace, args.physical_audit, args.reference, receipt_path,
            original_path, original_model_path, timeline_path, args.bundle / 'manifest.json',
            args.bundle / 'contract.json', args.bundle / 'native_prepared.xml',
            args.bundle / 'prepared_model_arrays.npz', Path(__file__),
            ROOT / 'gear_sonic/utils/g1_true23_hand_frame_tasks.py')})
    if args.preceding_trace is not None:
        report['hashes'][str(args.preceding_trace)] = sha(args.preceding_trace)
    if args.preceding_endpoint is not None:
        report['hashes'][str(args.preceding_endpoint)] = sha(args.preceding_endpoint)
        report['hashes'][str(endpoint_report_path)] = sha(endpoint_report_path)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    (args.output / 'source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k: report[k] for k in ('source_metrics', 'source_metric_gates',
        'full_lifecycle_source_intent_pass', 'requested_segment_quiet_pass')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bundle', 'trace', 'reference', 'physical-audit', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--clip', required=True, choices=('pico', 'walk002', 'walk003', 'walk008'))
    parser.add_argument('--requested-controls', type=int, required=True)
    parser.add_argument('--global-start', type=int, default=0)
    parser.add_argument('--preceding-trace', type=Path)
    parser.add_argument('--preceding-endpoint', type=Path,
        help='Independent full integration endpoint when the preceding immutable trace omitted it')
    main(parser.parse_args())
