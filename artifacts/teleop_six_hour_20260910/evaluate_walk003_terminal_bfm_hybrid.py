"""One explicit offline MPC-target-to-BFM terminal-standing hybrid trial.

Continuous native323 physics from reference frame10; original source/return
targets preserved. No resets, root assistance, planner or private rollout.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import mujoco
import numpy as np


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def read(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def stat(x):return dict(p50=float(np.percentile(x,50)),p95=float(np.percentile(x,95)),maximum=float(np.max(x)))


def direct_history(source,contract,count):
    """Independent algebra, not BFMHistory/state_and_terms calls."""
    q=source['qpos'][:count];dq=source['qvel'][:count]
    previous=np.zeros((count,23),np.float32)
    previous[1:]=((source['target'][:count-1]-np.asarray(contract['default_q']))*np.asarray(contract['kp'])/(.25*np.asarray(contract['training_effort']))).astype(np.float32)
    quat=q[:,3:7].copy();quat/=np.linalg.norm(quat,axis=1)[:,None]
    w,x,y,z=quat.T
    gravity=np.c_[-2*(x*z-w*y),-2*(y*z+w*x),-(1-2*(x*x+y*y))].astype(np.float32)
    terms=dict(actions=previous,base_ang_vel=(dq[:,3:6]*.25).astype(np.float32),
        dof_pos=(q[:,7:]-np.asarray(contract['default_q'])).astype(np.float32),
        dof_vel=dq[:,6:].astype(np.float32),projected_gravity=gravity)
    state=np.c_[terms['dof_pos'],terms['dof_vel'],gravity,terms['base_ang_vel']]
    pieces=[]
    for name,values in sorted(terms.items()):
        h=np.zeros((count,4,values.shape[1]),np.float32)
        for back in range(1,5):h[back:,back-1]=values[:-back]
        pieces.append(h.reshape(count,-1))
    return state,previous,np.concatenate(pieces,axis=1)


def new_trace(data):
    keys=('qpos','qvel','target','source_frame','global_control','controller_mode','joint_error','root_error',
          'state','history','previous_action','action','inference_ms','range_excess','velocity_ratio','effort_ratio',
          'physics_qpos','physics_qvel','physics_torque','physics_actuator_torque','physics_substeps','physics_time',
          'physics_warning_counts','physics_warning_lastinfo')
    out={k:[] for k in keys}
    for key in ('qpos','physics_qpos'):out[key].append(data.qpos.copy())
    for key in ('qvel','physics_qvel'):out[key].append(data.qvel.copy())
    out['physics_time'].append(float(data.time))
    out['physics_warning_counts'].append(np.asarray(data.warning.number,dtype=np.int64).copy())
    out['physics_warning_lastinfo'].append(np.asarray(data.warning.lastinfo,dtype=np.int64).copy())
    return out


QUIET_THRESHOLDS=dict(window_seconds=3,root_XY_error_p95_m=.05,original_heading_abs_p95_deg=5.,
    root_linear_speed_p95_mps=.05,max_joint_speed_p95_radps=.5,max_joint_speed_max_radps=2.,tilt_max_rad=.15)


def yaw(q):
    w,x,y,z=np.asarray(q).T
    return np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))


def standing_windows(arrays,motion,original,original29):
    result={}
    for seconds in (1,3):
        steps=min(seconds*500,len(arrays['physics_torque']))
        q,dq=arrays['physics_qpos'][-steps:],arrays['physics_qvel'][-steps:]
        frames=np.repeat(arrays['source_frame'],arrays['physics_substeps'])[-steps:]
        metric=motion['body_pos_w'][frames,0]
        goal=original['body_pos_w'][frames,0]
        wanted29=original29['source_qpos29'][frames]
        heading=yaw(q[:,3:7])-yaw(wanted29[:,3:7])
        heading=np.abs(np.rad2deg(np.arctan2(np.sin(heading),np.cos(heading))))
        tilt=np.arccos(np.clip(1-2*np.sum(q[:,4:6]**2,axis=1),-1,1))
        result[f'last_{seconds}s']=dict(physics_samples=steps,
            interval_seconds=[float(arrays['physics_time'][-steps-1]),float(arrays['physics_time'][-1])],
            declared_v4_root_error_m=stat(np.linalg.norm(q[:,:3]-metric,axis=1)),
            original_BFM_root_error_m=stat(np.linalg.norm(q[:,:3]-goal,axis=1)),
            horizontal_root_error_m=stat(np.linalg.norm(q[:,:2]-wanted29[:,:2],axis=1)),
            original_heading_absolute_error_deg=stat(heading),root_tilt_rad=stat(tilt),
            horizontal_displacement_from_window_start_m=float(np.linalg.norm(q[-1,:2]-q[0,:2])),
            maximum_absolute_joint_speed_radps=stat(np.max(np.abs(dq[:,6:]),axis=1)),
            joint_speed_rms_radps=float(np.sqrt(np.mean(dq[:,6:]**2))),
            root_linear_speed_mps=stat(np.linalg.norm(dq[:,:3],axis=1)),
            final_maximum_joint_speed_radps=float(np.abs(dq[-1,6:]).max()),
            final_declared_v4_root_error_m=float(np.linalg.norm(q[-1,:3]-metric[-1])),
            final_original_BFM_root_error_m=float(np.linalg.norm(q[-1,:3]-goal[-1])))
    return result


def quiet_diagnostic(windows,strict):
    tail=windows['last_3s']
    gates=dict(all_prior_strict_physical_and_warning_gates=bool(strict),
        full_three_second_window=tail['physics_samples']==1500,
        original_root_XY=tail['horizontal_root_error_m']['p95']<=QUIET_THRESHOLDS['root_XY_error_p95_m'],
        original_heading=tail['original_heading_absolute_error_deg']['p95']<=QUIET_THRESHOLDS['original_heading_abs_p95_deg'],
        root_linear_speed=tail['root_linear_speed_mps']['p95']<=QUIET_THRESHOLDS['root_linear_speed_p95_mps'],
        joint_speed_p95=tail['maximum_absolute_joint_speed_radps']['p95']<=QUIET_THRESHOLDS['max_joint_speed_p95_radps'],
        joint_speed_maximum=tail['maximum_absolute_joint_speed_radps']['maximum']<=QUIET_THRESHOLDS['max_joint_speed_max_radps'],
        maximum_tilt=tail['root_tilt_rad']['maximum']<=QUIET_THRESHOLDS['tilt_max_rad'])
    return dict(quiet_standing_diagnostic_pass=all(gates.values()),gates=gates,thresholds=QUIET_THRESHOLDS,
        raw_last_three_second_distributions=tail,existing_fifteen_source_gates_changed=False,hardware_certification=False)


def main(args):
    assert mujoco.__version__=='3.2.3'
    args.output.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(args.repo))
    from gear_sonic.utils.g1_true23_mjbatch_mpc import load_native_bundle,motion_states
    from gear_sonic.utils.g1_true23_mjbatch_bfm_seed import Native23BFMRolloutSeed
    from gear_sonic.utils.g1_true23_bfm_seed_observations import BFMHistory
    native,contract,original,timeline,manifest=load_native_bundle(args.bundle,'walk003')
    motion=read(args.reference)
    source=read(args.producer/'trace.npz')
    original29=read(args.bundle/'walk003/original29.npz')
    baseline=read(args.baseline/'trace.npz')
    baseline_report=json.loads((args.baseline/'report.json').read_text())
    assert sha(args.baseline/'trace.npz')==baseline_report['trace_sha256']
    assert baseline_report['engine_warning_counts']==[0]*8 and baseline_report['failure'] is None
    for name in ('physics_qpos','physics_qvel','physics_torque'):
        np.testing.assert_array_equal(baseline[name],source[name])
    producer_report=json.loads((args.producer/'report.json').read_text())
    request=json.loads((args.producer/'request.json').read_text())
    assert manifest==request['model_manifest']
    assert sha(args.producer/'trace.npz')==producer_report['trace_sha256']
    assert sha(args.producer/'request.json')==producer_report['request_sha256']
    assert sha(args.reference)==request['motion_override']['reference_sha256']
    assert sha(args.bundle/'walk003/native_original.npz')==request['motion_override']['base_native_reference_sha256']
    assert sha(args.bundle/'walk003/original29.npz')==request['motion_override']['original29_sha256']
    phase=next(p for p in timeline['phases'] if p['name']=='source_motion')
    standing=next(p for p in timeline['phases'] if p['name']=='returned_standing')
    switch=standing['control_start'];count=len(motion['joint_pos'])-11
    assert switch==1269 and count==1569 and phase['control_stop']==1169
    assert len(source['target'])==count and np.all(source['physics_substeps']==10)
    expected_state,expected_action,expected_history=direct_history(source,contract,switch+1)
    seed=Native23BFMRolloutSeed(native,contract,original,args.onnx,dependency_directory=args.dependencies,threads=1)
    kp,kd,effort,speed=[np.asarray(contract[k]) for k in ('kp','kd','native_effort','native_velocity')]
    limits=native.jnt_range[1:]
    initial=motion_states(motion)[10]
    np.testing.assert_array_equal(initial[:30],source['qpos'][0])
    np.testing.assert_array_equal(initial[30:],source['qvel'][0])
    data=mujoco.MjData(native);data.qpos[:],data.qvel[:]=initial[:30],initial[30:]
    data.qacc_warmstart[:]=0;mujoco.mj_forward(native,data)
    paths=[Path(__file__),args.producer/'trace.npz',args.producer/'report.json',args.producer/'request.json',
        args.baseline/'trace.npz',args.baseline/'report.json',
        args.reference,args.reference.parent/'portable_receipt.json']
    paths += [args.bundle/k for k in ('manifest.json','native_prepared.xml','prepared_model_arrays.npz','contract.json',
        'walk003/native_original.npz','walk003/original29.npz','walk003/timeline.json')]
    paths += [args.onnx/k for k in ('manifest.json','actor.onnx','backward.onnx')]
    paths += [args.repo/'gear_sonic/utils'/name for name in ('g1_true23_mjbatch_mpc.py','g1_true23_mjbatch_ilqr_core.py',
        'g1_true23_bfm_seed_observations.py','g1_true23_mjbatch_bfm_seed.py')]
    provenance=dict(kind='explicit_offline_recorded_MPC_targets_then_BFM_standing_hybrid',clip='walk003',
        hashes={str(p.resolve()):sha(p) for p in paths},mujoco=mujoco.__version__,python=sys.executable,
        BFM_identity=seed.identity(),motion_override=request['motion_override'],
        explicit_local_motion_override=str(args.reference.resolve()),
        switch_control_zero_based=switch,switch_simulated_seconds=switch*.02,
        preceding_source_and_return_targets_unchanged=True,
        BFM_goal_reference=str((args.bundle/'walk003/native_original.npz').resolve()),
        BFM_goal_reference_sha256=sha(args.bundle/'walk003/native_original.npz'),
        BFM_position_gain=1.,BFM_yaw_gain=2.,BFM_goal_horizon=8,
        previous_action_before_switch='actual applied MPC target normalized using training effort/kp/action scale, no action-history clipping',
        previous_action_after_switch='BFM raw actor times5 before native target clipping, published policy convention',
        history='four preceding measured observations newest-first within alphabetically sorted terms; each stores its own preceding action',
        physical_initialization='declared v4 reference frame10, independently equality checked against passing producer',
        physical_state_rewrites_after_initialization=0,root_assistance_forces=0,
        ground_truth_root_feedback=True,private_policy_rollouts=0,shared_sources_modified=False,
        extension='optional250 controls/5s continuously after original lifecycle, held last original BFM goal; stored separately',
        physical_abort_range_rad=.01,strict_physical_acceptance_range_rad=1e-6,
        quiet_standing_diagnostic_thresholds_declared_before_trial=QUIET_THRESHOLDS,
        quiet_standing_is_separate_from_existing_fifteen_recorded_source_gates=True,
        offline_planning=True,timing_qualified=False,hardware_authorized=False)
    write(args.output/'provenance.json',provenance)
    for p in paths:
        if p.suffix=='.py':(args.output/('source_'+p.name)).write_bytes(p.read_bytes())
    (args.output/'runner_snapshot.py').write_bytes(Path(__file__).read_bytes())
    history_verified=0;prefix_steps=0;switch_receipt=None
    controller_history=None;controller_previous=None
    started=time.perf_counter()

    def segment(start,stop):
        nonlocal history_verified,prefix_steps,switch_receipt,controller_history,controller_previous
        trace=new_trace(data);failure=None
        for control in range(start,stop):
            frame=min(control+11,len(motion['joint_pos'])-1)
            if control==switch:
                controller_history=BFMHistory()
                for key in controller_history.data:controller_history.data[key][:]=seed.history.data[key]
                controller_previous=seed.previous_action.copy()
            previous=seed.previous_action.copy() if control<switch else controller_previous.copy()
            state,terms=seed._terms(data.qpos,data.qvel,previous)
            active_history=seed.history if control<switch else controller_history
            history=np.concatenate([active_history.data[k].reshape(-1) for k in sorted(active_history.data)]).copy()
            if control<=switch:
                np.testing.assert_array_equal(state,expected_state[control])
                np.testing.assert_array_equal(previous,expected_action[control])
                np.testing.assert_array_equal(history,expected_history[control])
                history_verified+=1
            tick=time.perf_counter()
            if control<switch:
                target=source['target'][control].copy()
                seed.record_control(control,data.qpos,data.qvel,target)
                action=seed.previous_action.copy()
            else:
                if control==switch:
                    switch_receipt=dict(control=control,seconds=float(data.time),prefix_physics_steps=prefix_steps,
                        actual_state_equals_producer=True,all_preceding_histories_and_boundary_exact=True,
                        history_comparisons=history_verified,previous_action_max_abs=float(np.abs(previous).max()),
                        history_action_max_abs=float(np.abs(seed.history.data['actions']).max()),
                        preceding_effective_action_components_outside_five=seed.actual_action_components_outside_five)
                    np.savez_compressed(args.output/'verified_switch_inputs.npz',qpos=data.qpos.copy(),qvel=data.qvel.copy(),
                        state=state,previous_action=previous,history=history,independent_expected_history=expected_history[control],
                        independent_expected_state=expected_state[control],independent_expected_previous_action=expected_action[control])
                    write(args.output/'verified_switch_receipt.json',switch_receipt)
                np.testing.assert_array_equal(controller_history.before_update(terms),history)
                goal=seed._goal(frame,data.qpos)
                raw=seed.sessions['actor'].run(None,dict(state=state[None],last_action=previous[None],history=history[None],z=goal))[0][0]
                assert raw.shape==(23,) and np.isfinite(raw).all()
                action=raw*5
                target=np.clip(np.asarray(contract['default_q'])+action*.25*np.asarray(contract['training_effort'])/kp,limits[:,0],limits[:,1])
                controller_previous=action.copy()
            inference_ms=(time.perf_counter()-tick)*1000
            peak_range=peak_speed=peak_effort=0.
            for sub in range(10):
                assert not np.any(data.xfrc_applied) and not np.any(data.qfrc_applied)
                data.ctrl[:]=np.clip(kp*(target-data.qpos[7:])-kd*data.qvel[6:],-effort,effort)
                mujoco.mj_step(native,data)
                for key,value in (('physics_qpos',data.qpos),('physics_qvel',data.qvel),('physics_torque',data.ctrl),
                    ('physics_actuator_torque',data.qfrc_actuator[6:]),('physics_warning_counts',data.warning.number),
                    ('physics_warning_lastinfo',data.warning.lastinfo)):trace[key].append(np.asarray(value).copy())
                trace['physics_time'].append(float(data.time))
                peak_range=max(peak_range,float(np.maximum(limits[:,0]-data.qpos[7:],data.qpos[7:]-limits[:,1]).max()))
                peak_speed=max(peak_speed,float(np.max(np.abs(data.qvel[6:])/speed)))
                peak_effort=max(peak_effort,float(np.max(np.abs(data.qfrc_actuator[6:])/effort)))
                global_step=control*10+sub+1
                if control<switch:
                    for name,value in (('physics_qpos',data.qpos),('physics_qvel',data.qvel),('physics_torque',data.ctrl)):
                        index=global_step-1 if name=='physics_torque' else global_step
                        if not np.array_equal(value,source[name][index]):
                            failure=dict(kind='prefix_physics_not_bit_exact',control=control,substep=sub+1,field=name,
                                maximum_difference=float(np.abs(value-source[name][index]).max()))
                    prefix_steps+=1
                tilt=float(np.arccos(np.clip(1-2*np.sum(data.qpos[4:6]**2),-1,1)))
                if np.any(data.warning.number) or abs(data.time-global_step*.002)>1e-8:
                    failure=dict(kind='engine_warning_or_clock',control=control,substep=sub+1)
                elif not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all() or data.qpos[2]<.25 or tilt>1.2:
                    failure=dict(kind='fall_or_nonfinite',control=control,substep=sub+1,height=float(data.qpos[2]),tilt=tilt)
                elif peak_range>.01 or peak_speed>1 or peak_effort>1+1e-6:
                    failure=dict(kind='native_limit_abort',control=control,substep=sub+1,range_excess=peak_range,velocity_ratio=peak_speed)
                if failure:break
            mujoco.mj_kinematics(native,data)
            values=dict(qpos=data.qpos.copy(),qvel=data.qvel.copy(),target=target.copy(),source_frame=frame,global_control=control,
                controller_mode=int(control>=switch),joint_error=data.qpos[7:]-motion['joint_pos'][frame],
                root_error=data.qpos[:3]-motion['body_pos_w'][frame,0],state=state,history=history,
                previous_action=previous,action=action.copy(),inference_ms=inference_ms,
                range_excess=peak_range,velocity_ratio=peak_speed,effort_ratio=peak_effort,physics_substeps=sub+1)
            for key,value in values.items():trace[key].append(value)
            if (control+1)%250==0 or failure:print(json.dumps(dict(control=control+1,physics_seconds=float(data.time),BFM=control>=switch,failure=failure)),flush=True)
            if failure:break
        return {key:np.asarray(value) for key,value in trace.items()},failure

    def save(directory,arrays,failure,extension=False):
        np.savez_compressed(directory/'trace.npz',**arrays)
        n,steps=len(arrays['target']),len(arrays['physics_torque'])
        q=arrays['physics_qpos'];dq=arrays['physics_qvel']
        excess=float(np.maximum(0,np.maximum(limits[:,0]-q[:,7:],q[:,7:]-limits[:,1])).max())
        warnings=np.max(arrays['physics_warning_counts'],axis=0).astype(int).tolist()
        speedratio=float(np.max(np.abs(dq[:,6:])/speed));effortratio=float(np.max(np.abs(arrays['physics_torque'])/effort))
        tilt=np.arccos(np.clip(1-2*np.sum(q[:,4:6]**2,axis=1),-1,1))
        strict=failure is None and excess<=1e-6 and speedratio<=1 and effortratio<=1+1e-9 and warnings==[0]*8 and q[:,2].min()>=.25 and tilt.max()<=1.2
        result=dict(kind='post_lifecycle_BFM_hold_extension' if extension else provenance['kind'],clip='walk003',
            motion_override=request['motion_override'],reference_sha256=sha(args.reference),
            original29_sha256=request['motion_override']['original29_sha256'],
            completed_controls=n,complete_controls=int(np.sum(arrays['physics_substeps']==10)),
            requested_controls=250 if extension else count,physics_steps=steps,
            simulated_seconds=float(arrays['physics_time'][-1]-arrays['physics_time'][0]),
            absolute_physics_start_seconds=float(arrays['physics_time'][0]),absolute_physics_end_seconds=float(arrays['physics_time'][-1]),
            full_source_completed=bool(not extension and n>=phase['control_stop'] and np.all(arrays['physics_substeps'][:phase['control_stop']]==10)),
            full_lifecycle_completed=bool(not extension and n==count and np.all(arrays['physics_substeps']==10)),
            failure=failure,engine_warning_counts=warnings,engine_warning_sampling_seconds=.002,
            range_excess_max=excess,velocity_ratio_max=speedratio,effort_ratio_max=effortratio,
            strict_physical_limits_pass=bool(strict),switch_control=switch,switch_seconds=switch*.02,
            prefix_physics_steps_bit_exact=prefix_steps,prefix_measured_histories_verified=history_verified,
            BFM_controls=int(np.sum(arrays['controller_mode']==1)),BFM_original_goal_reference=provenance['BFM_goal_reference'],
            standing_windows=standing_windows(arrays,motion,original,original29),
            physics_state_rewrites_after_initialization=0,root_assistance_forces=0,
            post_lifecycle_extension=extension,extension_does_not_replace_or_pad_original_lifecycle=True,
            timing_qualified=False,online_controller_qualified=False,hardware_authorized=False,
            trace_sha256=sha(directory/'trace.npz'),provenance_sha256=sha(args.output/'provenance.json'))
        result['quiet_standing_diagnostic']=quiet_diagnostic(result['standing_windows'],strict)
        write(directory/'report.json',result)
        return result

    arrays,failure=segment(0,count)
    result=save(args.output,arrays,failure)
    extension_result=None
    if result['strict_physical_limits_pass'] and result['full_lifecycle_completed']:
        extension=args.output/'post_lifecycle_hold_5s';extension.mkdir()
        extra,extra_failure=segment(count,count+250)
        extension_result=save(extension,extra,extra_failure,extension=True)
        np.testing.assert_array_equal(extra['qpos'][0],arrays['qpos'][-1])
        np.testing.assert_array_equal(extra['qvel'][0],arrays['qvel'][-1])
    baseline_windows=standing_windows(baseline,motion,original,original29)
    baseline_strict=(baseline_report['range_excess_max']<=1e-6 and baseline_report['velocity_ratio_max']<=1
        and baseline_report['effort_ratio_max']<=1+1e-9 and baseline_report['engine_warning_counts']==[0]*8
        and baseline_report['failure'] is None and baseline_report['actual_clock_max_abs_error_seconds']<=1e-8)
    summary=dict(kind='bounded_terminal_BFM_hybrid_comparison',lifecycle=result,extension=extension_result,
        baseline_standing_windows=baseline_windows,baseline_quiet_standing_diagnostic=quiet_diagnostic(baseline_windows,baseline_strict),switch=switch_receipt,
        elapsed_seconds=time.perf_counter()-started,physics_reexecuted_once_continuously=True,
        scope='one original full-source nominal case; standing-only controller switch; no robust or live-control pass',
        new_lifecycle_trace_sha256=sha(args.output/'trace.npz'),hardware_authorized=False)
    assert all(sha(path)==digest for path,digest in provenance['hashes'].items()),'Input changed during hybrid trial'
    write(args.output/'comparison.json',summary)
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('repo','bundle','reference','producer','baseline','onnx','dependencies','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    main(parser.parse_args())
