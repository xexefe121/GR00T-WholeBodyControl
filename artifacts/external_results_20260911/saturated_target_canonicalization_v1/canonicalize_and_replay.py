"""Exact clipped-PD target canonicalization on saved native23 trajectories."""
import json
from pathlib import Path
import sys
import time

BASE=Path(__file__).parent
SNAPSHOT=BASE.parent/'recovery_probe_v1/source_snapshot'
sys.path.insert(0,str(SNAPSHOT))
import mujoco
import numpy as np
from gear_sonic.utils.g1_true23_mjbatch_mpc import load_native_bundle,load_motion_override,motion_states,sha256

ROOT=Path('/mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof')
TASK=Path('/mnt/e/codex-artifacts/sonic23_teleop_six_hour_20260910')
BUNDLE=ROOT/'artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1'
CASES=[('walk003',TASK/'mjbatch_full_v1/walk003_v4_native323_allmargin_full_v1',0),
       ('pico',BASE.parent/'hard_feasibility_restoration_continuation_3740_v1',3740)]


def archive(path):
    with np.load(path,allow_pickle=False) as a:return {k:a[k].copy() for k in a.files}


def metrics(targets,previous,contract,selected):
    delta=np.diff(np.vstack((previous,targets)),axis=0)[selected]
    values=targets[selected]
    action=(values-np.asarray(contract['default_q']))*np.asarray(contract['kp'])/(.25*np.asarray(contract['training_effort']))
    return dict(controls=len(values),target_delta_abs_p95_rad=float(np.percentile(np.abs(delta),95)),
        target_delta_abs_p99_rad=float(np.percentile(np.abs(delta),99)),target_delta_abs_max_rad=float(np.max(np.abs(delta))),
        target_delta_rms_rad=float(np.sqrt(np.mean(delta**2))),
        target_vector_step_l2_p95_rad=float(np.percentile(np.linalg.norm(delta,axis=1),95)),
        normalized_action_abs_max=float(np.max(np.abs(action))),normalized_action_abs_p95=float(np.percentile(np.abs(action),95)),
        normalized_action_rms=float(np.sqrt(np.mean(action**2))),normalized_action_outside5_components=int(np.sum(np.abs(action)>5)),
        normalized_action_outside5_fraction=float(np.mean(np.abs(action)>5)))


def run_case(clip,producer,control0):
    dest=BASE/(clip if control0==0 else 'restoration65')
    dest.mkdir(exist_ok=False)
    src=archive(producer/'trace.npz')
    report=json.loads((producer/'report.json').read_text())
    assert sha256(producer/'trace.npz')==report['trace_sha256']
    native,c,original,timeline,manifest=load_native_bundle(BUNDLE,clip)
    reference=TASK/'mjbatch_intent_floor_inputs_v1'/clip/'reference.npz'
    motion,_=load_motion_override(reference,BUNDLE,clip,native,c,original,timeline,manifest)
    kp,kd,effort=[np.asarray(c[k]) for k in ('kp','kd','native_effort')]
    lo,hi=np.asarray(c['joint_limits']).T
    targets=src['target'];n=len(targets)
    assert len(src['physics_torque'])==n*10
    np.testing.assert_array_equal(src['qpos'],src['physics_qpos'][::10])
    np.testing.assert_array_equal(src['qvel'],src['physics_qvel'][::10])
    # The standalone segment starts from its already received reference sample.
    initial_previous=motion['joint_pos'][control0+10].copy()
    previous=initial_previous.copy()
    canonical=[];intervals=[];fully_saturated=[];fallback=[]
    for t,target in enumerate(targets):
        q=src['physics_qpos'][t*10:t*10+10,7:]
        dq=src['physics_qvel'][t*10:t*10+10,6:]
        logged=src['physics_torque'][t*10:t*10+10]
        before=np.clip(kp*(target-q)-kd*dq,-effort,effort)
        np.testing.assert_array_equal(before,logged,err_msg=f'original torque mismatch control{t}')
        saturated=np.all(np.abs(logged)==effort,axis=0)
        low,high=lo.copy(),hi.copy()
        for j in np.flatnonzero(saturated):
            plus=logged[:,j]==effort[j]
            minus=logged[:,j]==-effort[j]
            if np.any(plus):low[j]=max(low[j],float(np.max(q[plus,j]+(effort[j]+kd[j]*dq[plus,j])/kp[j])))
            if np.any(minus):high[j]=min(high[j],float(np.min(q[minus,j]+(-effort[j]+kd[j]*dq[minus,j])/kp[j])))
        candidate=target.copy()
        valid=saturated&(low<=high)
        candidate[valid]=np.minimum(np.maximum(previous[valid],low[valid]),high[valid])
        after=np.clip(kp*(candidate-q)-kd*dq,-effort,effort)
        bad=np.any(after!=logged,axis=0)
        # Floating-point boundary failures never weaken the exact torque test.
        candidate[bad]=target[bad]
        np.testing.assert_array_equal(np.clip(kp*(candidate-q)-kd*dq,-effort,effort),logged)
        np.testing.assert_array_equal(candidate[~saturated],target[~saturated])
        canonical.append(candidate.copy());intervals.append(np.stack((low,high),axis=-1));fully_saturated.append(saturated);fallback.append(bad)
        previous=candidate.copy()
    canonical=np.asarray(canonical);saturated=np.asarray(fully_saturated);fallback=np.asarray(fallback)
    changed=canonical!=targets
    phase=next(p for p in timeline['phases'] if p['name']=='source_motion')
    source_selection=(np.arange(n)+control0>=phase['control_start'])&(np.arange(n)+control0<phase['control_stop'])
    summaries={}
    for name,selected in [('full_segment',np.ones(n,dtype=bool)),('source_only',source_selection)]:
        summaries[name]=dict(original=metrics(targets,initial_previous,c,selected),canonical=metrics(canonical,initial_previous,c,selected),
            changed_components=int(np.sum(changed[selected])),changed_components_over1e6rad=int(np.sum(np.abs(canonical[selected]-targets[selected])>1e-6)),
            changed_controls=int(np.sum(np.any(changed[selected],axis=1))),fully_saturated_components=int(np.sum(saturated[selected])),
            fallback_components=int(np.sum(fallback[selected])),component_count=int(targets[selected].size),
            target_adjustment_abs_max_rad=float(np.max(np.abs(canonical[selected]-targets[selected]))),
            changed_components_by_joint=np.sum(changed[selected],axis=0).tolist())
    out=dest/'targets.npz'
    np.savez_compressed(out,original_targets=targets,canonical_targets=canonical,initial_previous_reference_target=initial_previous,
        fully_saturated=saturated,roundoff_boundary_fallback=fallback,derived_intervals=np.asarray(intervals),
        source_selection=source_selection,global_control=np.arange(n)+control0)
    result=dict(clip=clip,global_initial_control=control0,controls=n,initial_previous_canonical_target='already_received_v4_jointreference_frame_'+str(control0+10),
        all_saved_substep_torques_bitexact=True,unsaturated_components_unchanged=True,
        rule='All10substeps saturated required. Intersect nativebounds with positive saturation lowerbounds and negative saturation upperbounds. Choose closest previouscanonicaltarget, revert component tooriginal if any recomputedtorque differs.',
        scope='offline command-sequence equivalence; normalized controllerhistory changes and requires separatevalidation foronlinepolicy',
        summaries=summaries,hashes={str(p):sha256(p) for p in (Path(__file__),producer/'trace.npz',producer/'report.json',reference,BUNDLE/'contract.json',out)})
    print(json.dumps(dict(clip=clip,control0=control0,summaries=summaries)),flush=True)
    if clip=='walk003' and np.any(changed):
        # One continuous native replay from declared frame10. No source state
        # copies or actual position clipping after this initialization.
        initial=motion_states(motion)[10]
        np.testing.assert_array_equal(initial[:30],src['qpos'][0]);np.testing.assert_array_equal(initial[30:],src['qvel'][0])
        data=mujoco.MjData(native);data.qpos[:]=initial[:30];data.qvel[:]=initial[30:];mujoco.mj_forward(native,data)
        warnings=[];lastinfo=[];clocks=[];physics_qpos=[data.qpos.copy()];physics_qvel=[data.qvel.copy()];physics_torque=[]
        for t,target in enumerate(canonical):
            for sub in range(10):
                data.ctrl[:]=np.clip(kp*(target-data.qpos[7:])-kd*data.qvel[6:],-effort,effort)
                mujoco.mj_step(native,data);step=t*10+sub+1
                np.testing.assert_array_equal(data.qpos,src['physics_qpos'][step],err_msg=f'qpos at{step}')
                np.testing.assert_array_equal(data.qvel,src['physics_qvel'][step],err_msg=f'qvel at{step}')
                np.testing.assert_array_equal(data.ctrl,src['physics_torque'][step-1],err_msg=f'torque at{step}')
                assert not np.any(data.warning.number) and abs(data.time-step*.002)<1e-8
                physics_qpos.append(data.qpos.copy());physics_qvel.append(data.qvel.copy());physics_torque.append(data.ctrl.copy())
                warnings.append(data.warning.number.copy());lastinfo.append(data.warning.lastinfo.copy());clocks.append(float(data.time))
            mujoco.mj_kinematics(native,data)
        replay_path=dest/'canonical_native323_replay.npz'
        np.savez_compressed(replay_path,physics_qpos=physics_qpos,physics_qvel=physics_qvel,physics_torque=physics_torque,
            warning_counts=warnings,warning_lastinfo=lastinfo,physics_time=clocks,canonical_targets=canonical)
        result['independent_native_replay']=dict(full_lifecycle_controls=n,source_controls=int(source_selection.sum()),physics_steps=n*10,
            all_qpos_qvel_torque_bitexact=True,warning_counts=np.max(warnings,axis=0).tolist(),warning_lastinfo=np.max(lastinfo,axis=0).tolist(),
            clock_max_error=float(np.max(np.abs(np.asarray(clocks)-np.arange(1,n*10+1)*.002))),
            trace_sha256=sha256(replay_path),physics_statewrites_after_initialization=0,
            model_gains_effort_limits_changed=False,originalphysicaltrackingunchanged=True,policy_history_equivalence_claimed=False)
        print(json.dumps(dict(clip=clip,replay=result['independent_native_replay'])),flush=True)
    with (dest/'report.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    return result


if __name__=='__main__':
    started=time.perf_counter()
    assert mujoco.__version__=='3.2.3' and np.__version__=='1.26.4'
    cases=[run_case(*case) for case in CASES]
    with (BASE/'report.json').open('x') as f:
        json.dump(dict(kind='exact_saved_native_PD_saturation_target_canonicalization',cases=cases,
            elapsed_seconds=time.perf_counter()-started,no_optimizer=True,no_training=True,no_hardware=True),f,indent=2,allow_nan=False);f.write('\n')
