"""Read-only original-intent metrics for a bounded native23 physical continuation."""
import argparse
import json
from pathlib import Path
import sys

BASE = Path(__file__).parent
SNAPSHOT = BASE.parent/'recovery_probe_v1/source_snapshot'
sys.path.insert(0, str(SNAPSHOT))
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from gear_sonic.utils.g1_true23_mjbatch_mpc import load_native_bundle, load_motion_override, sha256

ROOT = Path('/mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof')
TASK = Path('/mnt/e/codex-artifacts/sonic23_teleop_six_hour_20260910')
BUNDLE = ROOT/'artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1'


def yaw(q):
    x, y, z, w = np.moveaxis(q[..., [1, 2, 3, 0]], -1, 0)
    return np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))


def summarize(arrays, selected):
    if not np.any(selected):
        return None
    values = {}
    for key, value in arrays.items():
        if key.endswith('_error_m') or key.endswith('_error_deg') or key.endswith('_error_rad'):
            a = value[selected]
            values[key.removesuffix('_error_m').removesuffix('_error_deg').removesuffix('_error_rad')] = dict(
                p95=np.percentile(a, 95, axis=0).tolist(), maximum=np.max(a, axis=0).tolist(), mean=np.mean(a, axis=0).tolist())
    error = arrays['joint_error'][selected]
    controls = arrays['global_control'][selected]
    frames = arrays['source_frame'][selected]
    return dict(controls=len(controls), global_control_start=int(controls[0]), global_control_stop_exclusive=int(controls[-1]+1),
        source_frame_start=int(frames[0]), source_frame_stop_exclusive=int(frames[-1]+1),
        goal_sample_source_seconds_start=float(arrays['goal_sample_source_seconds'][selected][0]),
        goal_sample_source_seconds_end=float(arrays['goal_sample_source_seconds'][selected][-1]),
        physical_source_seconds_start=float(arrays['physical_source_seconds_end'][selected][0]-.02),
        physical_source_seconds_end=float(arrays['physical_source_seconds_end'][selected][-1]),
        root_and_task_metrics=values,
        native_joint_rmse_by_hardware_joint=np.sqrt(np.mean(error**2, axis=0)).tolist(),
        native_leg_rmse_rad=float(np.sqrt(np.mean(error[:, :12]**2))),
        native_arm_rmse_rad=float(np.sqrt(np.mean(error[:, 13:]**2))),
        native_waist_rmse_rad=float(np.sqrt(np.mean(error[:, 12]**2))))


def main(args):
    assert mujoco.__version__ == '3.2.3'
    directory = args.trace.parent
    result_dir = args.output
    result_dir.mkdir(exist_ok=False)
    report = json.loads((directory/'report.json').read_text())
    request = json.loads((directory/'request.json').read_text())
    assert report['trace_sha256'] == sha256(args.trace)
    assert request['actual_initial_control'] == args.initial_control
    with np.load(args.trace, allow_pickle=False) as a:
        trace = {key:a[key].copy() for key in a.files}
    native, contract, original, timeline, manifest = load_native_bundle(BUNDLE, args.clip)
    reference = TASK/'mjbatch_intent_floor_inputs_v1'/args.clip/'reference.npz'
    motion, _ = load_motion_override(reference, BUNDLE, args.clip, native, contract, original, timeline, manifest)
    assert sha256(reference) == request['motion_override']['reference_sha256']
    original29_path = BUNDLE/args.clip/'original29.npz'
    assert sha256(original29_path) == request['motion_override']['original29_sha256']
    with np.load(original29_path, allow_pickle=False) as a:
        wanted = a['source_task_position_w'].copy()
        wanted_quat = a['source_task_quaternion_wxyz'].copy()
        source_qpos = a['source_qpos29'].copy()
    phase = next(p for p in timeline['phases'] if p['name']=='source_motion')
    n = len(trace['target'])
    assert trace['qpos'].shape == (n+1, 30) and trace['qvel'].shape == (n+1, 29)
    assert len(trace['physics_torque']) == n*10, 'partial control must be separately reported, not treated as complete'
    controls = np.arange(args.initial_control, args.initial_control+n)
    frames = controls+11
    np.testing.assert_array_equal(trace['source_frame'], frames)
    np.testing.assert_array_equal(trace['qpos'], trace['physics_qpos'][::10])
    np.testing.assert_array_equal(trace['qvel'], trace['physics_qvel'][::10])
    assert np.all(controls>=phase['control_start']) and np.all(controls<phase['control_stop'])
    # Reproduce the neutral-wrist convention geometrically instead of assuming
    # numeric hand offsets. This changes no model or source task geometry.
    source_model_path = ROOT.parent/'GR00T-WholeBodyControl/gear_sonic/data/robots/g1/g1_29dof.xml'
    source_model = mujoco.MjModel.from_xml_path(str(source_model_path))
    source_data, data = mujoco.MjData(source_model), mujoco.MjData(native)
    for m, d in ((source_model, source_data), (native, data)):
        d.qpos[:] = 0
        d.qpos[2:4] = [.8, 1]
        mujoco.mj_kinematics(m, d)
    points = []
    ids = []
    convention = []
    for side, offset in [('left', (.18, -.025, 0)), ('right', (.18, .025, 0))]:
        syaw = source_model.body(side+'_wrist_yaw_link').id
        sroll = int(source_model.joint(side+'_wrist_roll_joint').bodyid[0])
        target = native.body(side+'_wrist_roll_rubber_hand').id
        np.testing.assert_allclose(source_data.xpos[sroll], data.xpos[target], atol=1e-9, rtol=0)
        np.testing.assert_allclose(source_data.xmat[sroll], data.xmat[target], atol=1e-9, rtol=0)
        world = source_data.xpos[syaw]+source_data.xmat[syaw].reshape(3,3)@np.asarray(offset)
        point = source_data.xmat[sroll].reshape(3,3).T@(world-source_data.xpos[sroll])
        points.append(point)
        ids.append(target)
        convention.append(dict(task=side+'_hand', body=native.body(target).name, point=point.tolist()))
    points.append(np.array([0.,0.,.35])); ids.append(native.body('torso_link').id)
    convention.append(dict(task='head_proxy', body='torso_link', point=points[-1].tolist()))
    actual, orientation, feet = [], [], []
    feet_ids = [native.body(side+'_ankle_roll_link').id for side in ('left','right')]
    assert [i-1 for i in feet_ids] == [6,12]
    poses = trace['qpos'][1:]
    for pose in poses:
        data.qpos[:] = pose
        mujoco.mj_kinematics(native, data)
        actual.append([data.xpos[i]+data.xmat[i].reshape(3,3)@point for i,point in zip(ids,points)])
        orientation.append(data.xquat[ids].copy())
        feet.append(data.xpos[feet_ids].copy())
    actual, orientation, feet = map(np.asarray, (actual, orientation, feet))
    root = poses[:,:3]
    desired_root = source_qpos[frames,:3]
    native_root = motion['body_pos_w'][frames,0]
    desired_feet = motion['body_pos_w'][frames][:,[6,12]]
    yaw_difference = yaw(poses[:,3:7])-yaw(source_qpos[frames,3:7])
    yaw_difference = np.arctan2(np.sin(yaw_difference),np.cos(yaw_difference))
    angle = (Rotation.from_quat(orientation.reshape(-1,4)[:,[1,2,3,0]]).inv()*Rotation.from_quat(wanted_quat[frames].reshape(-1,4)[:,[1,2,3,0]])).magnitude().reshape(-1,3)
    arrays = dict(global_control=controls, source_frame=frames,
        goal_sample_source_seconds=(controls-phase['control_start'])*.02,
        physical_source_seconds_end=(controls-phase['control_start']+1)*.02,
        physical_time=trace['physics_time'][10::10],
        original_root_world_error_m=np.linalg.norm(root-desired_root, axis=-1),
        original_root_yaw_error_deg=np.rad2deg(np.abs(yaw_difference)),
        original_hand_head_world_error_m=np.linalg.norm(actual-wanted[frames], axis=-1),
        original_hand_head_relative_error_m=np.linalg.norm((actual-root[:,None])-(wanted[frames]-desired_root[:,None]),axis=-1),
        original_hand_head_orientation_error_rad=angle,
        native_feet_world_error_m=np.linalg.norm(feet-desired_feet,axis=-1),
        native_feet_same_world_axis_relative_error_m=np.linalg.norm((feet-root[:,None])-(desired_feet-native_root[:,None]),axis=-1),
        native_root_world_error_m=np.linalg.norm(root-native_root,axis=-1),
        joint_error=poses[:,7:]-motion['joint_pos'][frames], actual_task_positions=actual, actual_feet=feet)
    old_failure_source_seconds=68.17
    windows = dict(all_observed=summarize(arrays,np.ones(n,dtype=bool)),
        completed_controls_after_old_failure_time=summarize(arrays, arrays['physical_source_seconds_end']>old_failure_source_seconds),
        final_200ms=summarize(arrays,np.arange(n)>=max(0,n-10)))
    np.savez_compressed(result_dir/'metrics.npz',**arrays)
    result = dict(kind='independent_bounded_continuation_original_intent_metrics',clip=args.clip,
        initial_control=args.initial_control, completed_controls=n, requested_controls=request['requested_controls'],
        source_requested_controls=phase['requested_controls'], no_full_source_or_lifecycle_claim=True,
        recovery_qualified=False, producer_failure=report['failure'], windows=windows,
        timing='postcontrol measured qpos[c+1] compared with original source frame c+11; source goal time (c-350)*.02 and completion time (c-350+1)*.02 both saved',
        metric_convention='same world axes for relative hand/head and feet; no independent heading alignment',
        hand_head_order=['left_hand','right_hand','head_proxy'], feet_order=['left','right'], hand_convention=convention,
        FK_only_no_physics_integration=True, independent_physics_assessment='parent referee owns full native replay',
        hashes={str(p):sha256(p) for p in (Path(__file__),args.trace,directory/'report.json',directory/'request.json',reference,original29_path,source_model_path,BUNDLE/'contract.json',BUNDLE/'prepared_model_arrays.npz',result_dir/'metrics.npz')})
    with (result_dir/'report.json').open('x') as f:
        json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--trace',type=Path,required=True)
    parser.add_argument('--initial-control',type=int,required=True)
    parser.add_argument('--clip',default='pico',choices=('pico','walk002','walk003','walk008'))
    parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args())
