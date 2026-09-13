"""Read-only tracking diagnostics on fixed windows of an archived PICO run."""

import json
from pathlib import Path

import mujoco
import numpy as np

from .inspect_recorded_intent import ROOT, sha, yaw_error
from gear_sonic.utils.g1_true23_hand_frame_tasks import neutral_wrist_hand_tasks


def main():
    base = Path('E:/codex-artifacts/sonic23_teleop_resume_20260911')
    bundle = ROOT / 'artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1'
    trace_path = base / 'pico_full_control_lm_bounded_segment_comparison_v1/full_checkpoint.npz'
    comparison_path = trace_path.parent / 'report.json'
    bounded_path = base / 'control_lm_continuation_3800_intent_v1/report.json'
    reference_path = Path('E:/codex-artifacts/sonic23_teleop_six_hour_20260910/mjbatch_intent_floor_inputs_v1/pico/reference.npz')
    original_path = bundle / 'pico/original29.npz'
    timeline_path = bundle / 'pico/timeline.json'
    manifest_path = bundle / 'manifest.json'
    original_model_path = ROOT.parent / 'GR00T-WholeBodyControl/gear_sonic/data/robots/g1/g1_29dof.xml'
    manifest = json.loads(manifest_path.read_text())
    timeline = json.loads(timeline_path.read_text())
    comparison = json.loads(comparison_path.read_text())
    bounded = json.loads(bounded_path.read_text())
    assert sha(trace_path) == 'ef5635e2b240fb3886588520e725f8017de89cec32d4a9f83b56601ae9d5d5d0'
    assert comparison['all_bitexact'] and comparison['full_checkpoint_controls'] == 4000
    assert timeline['total_requested_controls'] == 6530
    source_phase = next(p for p in timeline['phases'] if p['name'] == 'source_motion')
    assert source_phase['control_start'] == 350 and source_phase['requested_controls'] == 5780
    assert sha(original_path) == manifest['cases']['pico']['original29.npz']
    assert sha(bundle / 'native_prepared.xml') == manifest['portable_xml_sha256']
    assert sha(bundle / 'prepared_model_arrays.npz') == manifest['prepared_arrays_sha256']
    for name, digest in manifest['meshes'].items():
        assert sha(bundle / 'meshes' / name) == digest
    receipt_path = reference_path.parent / 'portable_receipt.json'
    receipt = json.loads(receipt_path.read_text())
    assert receipt['reference_sha256'] == sha(reference_path)
    assert receipt['original29_reference_sha256'] == sha(original_path)
    assert receipt['timeline_sha256'] == sha(timeline_path)
    native = mujoco.MjModel.from_xml_path(str(bundle / 'native_prepared.xml'))
    with np.load(bundle / 'prepared_model_arrays.npz', allow_pickle=False) as arrays:
        for name in arrays.files:
            getattr(native, name)[:] = arrays[name]
        mujoco.mj_setConst(native, mujoco.MjData(native))
        for name in arrays.files:
            np.testing.assert_array_equal(getattr(native, name), arrays[name])
    source_model = mujoco.MjModel.from_xml_path(str(original_model_path))
    tasks, convention = neutral_wrist_hand_tasks(source_model, native)
    tasks = [next(t for t in tasks if t.name == name) for name in ('left_hand', 'right_hand', 'head_proxy')]
    with np.load(trace_path, allow_pickle=False) as trace:
        poses = trace['qpos'][1:].copy()
        frames = trace['source_frame'].copy()
        assert len(poses) == len(frames) == 4000
        np.testing.assert_array_equal(trace['physics_substeps'], np.full(4000, 10))
        np.testing.assert_array_equal(trace['qpos'], trace['physics_qpos'][::10])
        np.testing.assert_array_equal(frames, np.arange(4000) + 11)
    with np.load(reference_path, allow_pickle=False) as reference:
        motion = {name: reference[name].copy() for name in ('body_pos_w', 'joint_pos')}
    with np.load(original_path, allow_pickle=False) as original:
        target = {name: original[name].copy() for name in ('source_qpos29', 'source_task_position_w')}
    data = mujoco.MjData(native)
    windows = []
    for start, stop in ((3700, 3800), (3800, 3900), (3900, 4000)):
        q, f = poses[start:stop], frames[start:stop]
        hands, feet = [], []
        for pose in q:
            data.qpos[:] = pose
            mujoco.mj_kinematics(native, data)
            hands.append([data.xpos[native.body(t.target_body).id] +
                          data.xmat[native.body(t.target_body).id].reshape(3, 3) @ t.target_point for t in tasks])
            feet.append([data.xpos[native.body(side + '_ankle_roll_link').id].copy() for side in ('left', 'right')])
        root, original = q[:, :3], target['source_qpos29'][f]
        hand_error = np.linalg.norm((np.asarray(hands) - root[:, None]) -
            (target['source_task_position_w'][f] - original[:, None, :3]), axis=-1)
        foot_error = np.linalg.norm((np.asarray(feet) - root[:, None]) -
            (motion['body_pos_w'][f][:, [6, 12]] - motion['body_pos_w'][f, 0, None]), axis=-1)
        metrics = dict(original_root_p95_m=float(np.percentile(np.linalg.norm(root - original[:, :3], axis=-1), 95)),
            original_yaw_p95_deg=float(np.percentile(yaw_error(q[:, 3:7], original[:, 3:7]), 95)),
            original_relative_hand_head_p95_m=np.percentile(hand_error, 95, axis=0).tolist(),
            world_axis_relative_foot_p95_m=np.percentile(foot_error, 95, axis=0).tolist(),
            native_leg_rmse_rad=float(np.sqrt(np.mean((q[:, 7:19] - motion['joint_pos'][f, :12]) ** 2))))
        gates = dict(root=metrics['original_root_p95_m'] <= .20, yaw=metrics['original_yaw_p95_deg'] <= 15,
            hands=bool(np.all(np.asarray(metrics['original_relative_hand_head_p95_m'][:2]) <= .15)),
            head=metrics['original_relative_hand_head_p95_m'][2] <= .10,
            feet=bool(np.all(np.asarray(metrics['world_axis_relative_foot_p95_m']) <= .12)),
            legs=metrics['native_leg_rmse_rad'] <= .15)
        if start == 3800:
            for name, value in metrics.items():
                np.testing.assert_array_equal(value, bounded['source_metrics'][name])
            assert gates == bounded['source_metric_gates']
        windows.append(dict(global_controls=[start, stop], source_seconds=[(start - 350) * .02, (stop - 350) * .02],
            metrics=metrics, unchanged_threshold_diagnostic=gates))
    report = dict(kind='fixed_recorded_state_recovery_window_diagnostic', requested_full_controls=6530,
        recorded_full_controls=4000, requested_source_controls=5780, windows=windows,
        bounded_3800_3900_metrics_bitexact=True, full_source_pass=False,
        subset_is_not_full_tracking_qualification=True, dynamics_executed=False,
        timing_or_live_teleoperation_qualified=False, convention=convention,
        hashes={str(p): sha(p) for p in (trace_path, comparison_path, bounded_path, reference_path,
            receipt_path, original_path, timeline_path, manifest_path, original_model_path,
            bundle / 'native_prepared.xml', bundle / 'prepared_model_arrays.npz', Path(__file__),
            ROOT / 'gear_sonic/utils/g1_true23_hand_frame_tasks.py')})
    out = base / 'pico_post_recovery_tracking_windows_v1'
    out.mkdir(exist_ok=False)
    (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    (out / 'source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(windows, indent=2))


if __name__ == '__main__':
    main()
