"""Independent saved-array audit of all velocity probes, plus fixed BFM repeats.

No physics, optimizer, replanning, checkpoint selection or new training labels.
The nine repeated centers are fixed: first acquisition/source/return in each
dataset. Each checks its original center and all46 signed probes:432 BFM calls.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import traceback
from types import SimpleNamespace

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
NEW=Path('/mnt/e/codex-artifacts/sonic23_teleop_resume_20260911')
OLD=NEW.parent/'sonic23_teleop_six_hour_20260910'
KEYS=('actions','base_ang_vel','dof_pos','dof_vel','projected_gravity')


def sha(p):
    value=hashlib.sha256()
    with Path(p).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):
            value.update(block)
    return value.hexdigest()


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def atomic_json(path,value):
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def load(p):
    with np.load(p,allow_pickle=False) as a:
        return {k:a[k].copy() for k in a.files}


def exact(a,b,name):
    a,b=np.asarray(a),np.asarray(b)
    assert a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes(),name


def local(p):
    p=str(p).replace('\\','/')
    return Path('/mnt/'+p[0].lower()+p[2:]) if len(p)>2 and p[1]==':' else Path(p)


def main(args):
    assert np.__version__=='1.26.4'
    import mujoco
    assert mujoco.__version__=='3.2.3'
    def forbidden_step(*unused,**unused_named):
        raise RuntimeError('Physics stepping is prohibited in this audit')
    mujoco.mj_step=mujoco.mj_step1=mujoco.mj_step2=forbidden_step
    generation=args.generation
    producer=read(generation/'report.json')
    assert producer['complete'] and producer['centers']==3057 and producer['probe_rows']==140622
    assert producer['sign_order']==[-1.,1.] and producer['physics_steps']==producer['optimizer_updates']==0
    assert producer['inference_calls']==dict(actor=143679,backward=3057)
    for name,digest in producer['output_sha256'].items():
        assert sha(generation/name)==digest,name
    experiment=generation.parent
    receipt=experiment/'generation_frozen_inputs_v2.json'
    assert sha(receipt)=='a79321d317438c2a3817b0db74e8acb1b6ec92479e44736da1d587ad1d912f08'
    freeze=read(receipt)
    assert sha(receipt)==producer['input_frozen_receipt_sha256']
    for name,digest in freeze['input_sha256'].items():
        assert sha(local(name))==digest,name
    source=experiment/'source_snapshot_v2'
    assert local(freeze['source_directory'])==source
    for name,digest in freeze['source_sha256'].items():
        assert sha(source/name)==digest,name
    assert producer['source_generation_sha256']==freeze['source_sha256']['generate_velocity_chords.py']
    assert producer['all_frozen_sources_and_inputs_rehashed_at_completion'] is True
    sys.path.insert(0,str(source))
    from gear_sonic.utils.g1_true23_mjbatch_mpc import load_native_bundle,load_motion_override
    from gear_sonic.utils.g1_true23_mpc_student import GoalFeatures
    from gear_sonic.utils.g1_true23_bfm_seed_observations import state_and_terms
    from gear_sonic.utils.g1_true23_mjbatch_bfm_seed import Native23BFMRolloutSeed
    bundle=ROOT/'artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1'
    reference=OLD/'mjbatch_intent_floor_inputs_v1/walk003/reference.npz'
    native,c,original,timeline,manifest=load_native_bundle(bundle,'walk003')
    motion,_=load_motion_override(reference,bundle,'walk003',native,c,original,timeline,manifest)
    original29=load(bundle/'walk003/original29.npz')
    goals=GoalFeatures(motion,original29,c)
    default,kp,effort,caps,limits=[np.asarray(c[k]) for k in ('default_q','kp','training_effort','native_velocity','joint_limits')]
    center=load(generation/'centers.npz')
    widths={'features':1069,'base_target':23,'base_action':23,'state':52,'teacher_target':23,
            'teacher_feedback_raw':23,'teacher_feedback_clipped':23,'teacher_native_clipped':23,
            'teacher_branch_changed':23,'perturbed_joint_velocity':None}
    probes={k:np.load(generation/(k+'.npy'),mmap_mode='r',allow_pickle=False) for k in widths}
    for key,width in widths.items():
        assert probes[key].shape==(3057,23,2)+(() if width is None else (width,)),key
        dtype=np.bool_ if key in ('teacher_feedback_clipped','teacher_native_clipped','teacher_branch_changed') else (
            np.float32 if key in ('features','base_action','state') else np.float64)
        assert probes[key].dtype==np.dtype(dtype),key
    exact(center['dataset'],np.repeat(np.arange(3,dtype=np.int64),1019),'dataset order')
    exact(center['control'],np.tile(np.arange(250,1269,dtype=np.int64),3),'control order')
    exact(center['source_frame'],center['control']+11,'goal frame')
    exact(center['joint_limits'],limits,'limits')
    exact(center['native_velocity'],caps,'caps')
    exact(center['joint_span'],np.diff(limits,axis=1).ravel().astype(np.float32),'spans')
    assert np.all(np.abs(center['qvel'][:,6:])+.01*caps<caps)
    configurations=[
        (NEW/'fast_controller_nominal_pilot_v1/labels/labels.npz',
         OLD/'mjbatch_full_v1/walk003_v4_native323_allmargin_full_v1'),
        (NEW/'fresh_expert_labels_resume_v1/labels/labels.npz',
         NEW/'student_actual_oracle_control1_resume1001_v1/nominal'),
        (NEW/'bfm_entry250_labels_v1/labels/labels.npz',
         NEW/'bfm_entry250_actual_oracle_v1/nominal')]
    bindings=[]
    for dataset,(label_path,expert_path) in enumerate(configurations):
        labels=load(label_path)
        ids=np.flatnonzero((labels['control']>=250)&(labels['control']<1269))
        span=slice(dataset*1019,(dataset+1)*1019)
        for key in ('features','residual_rad','base_target','expert_target','base_action','previous_action','history','state'):
            exact(center[key][span],labels[key][ids],'original center '+key)
        exact(center['qpos'][span],labels['teacher_qpos'][:-1][ids],'original qpos')
        exact(center['qvel'][span],labels['teacher_qvel'][:-1][ids],'original qvel')
        records=read(expert_path/'plans.json')
        by_control={record['control']:record for record in records}
        assert len(by_control)==len(records)
        old_trace=load(expert_path/'trace.npz') if dataset==0 else None
        cache={}
        for i in range(dataset*1019,(dataset+1)*1019):
            control=int(center['control'][i]);start=int(center['plan_control'][i]);at=int(center['plan_local'][i])
            assert start+at==control and start in by_control
            record=by_control[start]
            committed=record['controls_committed'] if dataset==0 else record['executed_controls']
            assert 0<=at<committed
            accepted=-1 if dataset==0 else int(any(it['accepted'] for it in record['solver_feasibility']['iterations']))
            assert int(center['plan_accepted_update'][i])==accepted
            if dataset==0:
                values=(old_trace['planned_state'][control],old_trace['planned_target'][control],old_trace['feedback_gain'][control])
            else:
                if start not in cache:
                    plan_path=expert_path/'plans'/('plan_%05d.npz'%start)
                    cache[start]=load(plan_path)
                    bindings.append((str(plan_path),sha(plan_path)))
                plan=cache[start]
                values=(plan['nominal_states'][at],plan['targets'][at],plan['gains'][at])
            for key,value in zip(('planned_state','planned_target','gain'),values):
                exact(center[key][i],value,'original committed '+key)
        bindings.extend([(str(label_path),sha(label_path)),(str(expert_path/'plans.json'),sha(expert_path/'plans.json'))])
        if dataset==0:
            bindings.append((str(expert_path/'trace.npz'),sha(expert_path/'trace.npz')))
    core=NEW/'bfm_entry250_actual_oracle_v1/source_snapshot_v1/gear_sonic/utils/g1_true23_mjbatch_ilqr_core.py'
    tree=ast.parse(core.read_text())
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('quat_mul','quat_log')]
    planner=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Planner')
    functions.append(next(n for n in planner.body if isinstance(n,ast.FunctionDef) and n.name=='difference'))
    scope={'np':np}
    exec(compile(ast.fix_missing_locations(ast.Module(body=functions,type_ignores=[])),str(core),'exec'),scope)
    fake=SimpleNamespace(nq=30,nv=29,lin_q=np.r_[0:3,7:30],lin_v=np.r_[0:3,6:29],free=[(0,0)])
    def target(i,v):
        tangent=scope['difference'](fake,center['planned_state'][i,None],np.r_[center['qpos'][i],v][None])[0]
        raw=center['gain'][i]@tangent
        clipped=np.clip(raw,-.1,.1)
        pre=center['planned_target'][i]+clipped
        return np.clip(pre,limits[:,0],limits[:,1]),raw,clipped,pre
    def features(q,v,previous,base,frame):
        prior_target=np.clip(default+previous*.25*effort/kp,limits[:,0],limits[:,1])
        return np.r_[goals(q,v,prior_target,frame),base-default,previous].astype(np.float32)
    audit_request=dict(kind='independent_all_velocity_chord_arrays_and_fixed_BFM_repeat',
                       generation_report_sha256=sha(generation/'report.json'),all_probes=140622,
                       repeated_centers=[d*1019+p for d in range(3) for p in (0,100,919)],
                       repeated_actor_budget=423,repeated_backward_budget=9,
                       physics_steps=0,optimizer_updates=0,source_sha256=sha(Path(__file__)))
    audit_request.update(input_receipt_sha256=sha(receipt),output_sha256=producer['output_sha256'],
                         plan_and_label_hashes=dict(bindings),core_sha256=sha(core))
    (args.output/'request.json').write_text(json.dumps(audit_request,indent=2)+'\n')
    stats=dict(feedback_clipped_probes=0,feedback_clipped_components=0,native_clipped_probes=0,
               native_clipped_components=0,branch_changed_probes=0)
    maximum_delta=0.
    for i in range(3057):
        q,v,previous=center['qpos'][i],center['qvel'][i],center['previous_action'][i]
        args.active=dict(stage=np.asarray('PURE_ARRAY_CHECK'),center=np.int64(i),qpos=q,qvel=v,
                         previous_action=previous,history=center['history'][i])
        exact(center['history'][i],np.concatenate([center['history_'+k][i].reshape(-1) for k in KEYS]),'history flatten')
        state,_=state_and_terms(q[7:],v[6:],q[3:7],v[3:6],previous,default)
        exact(state,center['state'][i],'center state')
        exact(features(q,v,previous,center['base_target'][i],int(center['source_frame'][i])),center['features'][i],'center feature rebuild')
        t,r,clipped,pre=target(i,v)
        for key,value in [('expert_target',t),('teacher_feedback_raw',r),('teacher_feedback_correction',clipped),('teacher_preclip',pre)]:
            exact(center[key][i],value,'center '+key)
        center_fb=np.where(r<-.1,-1,np.where(r>.1,1,0))
        center_native=np.where(pre<limits[:,0],-1,np.where(pre>limits[:,1],1,0))
        for joint in range(23):
            for sign_index,sign in enumerate((-1.,1.)):
                changed=v.copy();changed[6+joint]+=sign*.01*caps[joint]
                at=(i,joint,sign_index)
                args.active.update(axis=np.int64(joint),sign=np.float64(sign),qvel=changed)
                exact(probes['perturbed_joint_velocity'][at],changed[6+joint],'perturbation value')
                assert np.all(np.abs(changed[6:])<caps)
                base=default+probes['base_action'][at]*.25*effort/kp
                exact(probes['base_target'][at],base,'raw action to base')
                state,_=state_and_terms(q[7:],changed[6:],q[3:7],changed[3:6],previous,default)
                exact(probes['state'][at],state,'probe state')
                expected_feature=center['features'][i].copy()
                expected_feature[23+joint]=changed[6+joint]
                expected_feature[1023:1046]=base-default
                exact(probes['features'][at],expected_feature,'all1069 probe features')
                t,r,clipped,pre=target(i,changed)
                exact(probes['teacher_target'][at],t,'teacher target')
                exact(probes['teacher_feedback_raw'][at],r,'teacher full matvec')
                fb=r!=clipped;nc=pre!=t
                changed_branch=(np.where(r<-.1,-1,np.where(r>.1,1,0))!=center_fb)|(
                    np.where(pre<limits[:,0],-1,np.where(pre>limits[:,1],1,0))!=center_native)
                exact(probes['teacher_feedback_clipped'][at],fb,'feedback clip flags')
                exact(probes['teacher_native_clipped'][at],nc,'native clip flags')
                exact(probes['teacher_branch_changed'][at],changed_branch,'branch flags')
                assert np.isfinite(base).all() and np.isfinite(expected_feature).all() and np.isfinite(t).all() and np.isfinite(r).all()
                assert np.all(t>=limits[:,0]) and np.all(t<=limits[:,1])
                maximum_delta=max(maximum_delta,float(np.max(np.abs(t-center['expert_target'][i]))))
                stats['feedback_clipped_probes']+=int(fb.any());stats['feedback_clipped_components']+=int(fb.sum())
                stats['native_clipped_probes']+=int(nc.any());stats['native_clipped_components']+=int(nc.sum())
                stats['branch_changed_probes']+=int(changed_branch.any())
        if (i+1)%250==0:
            atomic_json(args.output/'progress.json',dict(stage='PURE_ARRAY_CHECK',checked_centers=i+1,
                        checked_probes=(i+1)*46,inference_calls=args.calls))
            print(json.dumps(dict(checked_centers=i+1,checked_probes=(i+1)*46)),flush=True)
    assert maximum_delta<=.2+1e-12
    for key,value in stats.items():
        assert producer[key]==value,(key,producer[key],value)
    assert producer['zero_gain_centers']==int(np.all(center['gain']==0,axis=(1,2)).sum())
    seed=Native23BFMRolloutSeed(native,c,original,ROOT/'artifacts/teleop_six_hour_20260910/bfm_onnx_v2',
        dependency_directory='/mnt/e/codex_sonic_runtime/bfm_seed_20260910/onnx_deps',threads=1)
    calls=args.calls
    for i in audit_request['repeated_centers']:
        q,v,previous,history=[center[k][i] for k in ('qpos','qvel','previous_action','history')]
        args.active=dict(stage=np.asarray('BFM_REPEAT_GOAL'),center=np.int64(i),qpos=q,qvel=v,
                         previous_action=previous,history=history,source_frame=center['source_frame'][i])
        assert calls['backward']<9
        calls['backward']+=1
        atomic_json(args.output/'progress.json',dict(stage='BFM_REPEAT_GOAL',center=i,inference_calls=calls))
        z=seed._goal(int(center['source_frame'][i]),q)
        args.active['returned_goal_latent']=z
        exact(z,center['goal_latent'][i],'cached latent repeat')
        states=[(-1,-1,center['state'][i],center['base_action'][i])]
        states.extend((j,s,probes['state'][i,j,s],probes['base_action'][i,j,s]) for j in range(23) for s in range(2))
        for joint,sign_index,state,expected in states:
            actual_velocity=v.copy()
            if joint>=0:
                actual_velocity[6+joint]+=(-1.,1.)[sign_index]*.01*caps[joint]
            args.active.update(stage=np.asarray('BFM_REPEAT_ACTOR'),axis=np.int64(joint),sign_index=np.int64(sign_index),
                               qvel=actual_velocity,nominal_qvel=v,state=state,expected_base_action=expected)
            args.active.pop('returned_actor_output',None)
            args.active.pop('returned_base_action',None)
            assert calls['actor']<423
            calls['actor']+=1
            atomic_json(args.output/'progress.json',dict(stage='BFM_REPEAT_ACTOR',center=i,axis=joint,
                        sign_index=sign_index,inference_calls=calls))
            returned=seed.sessions['actor'].run(None,dict(state=state[None],last_action=previous[None],history=history[None],z=z))[0]
            args.active['returned_actor_output']=returned
            raw=returned[0]*5
            args.active['returned_base_action']=raw
            exact(raw,expected,'fixed independent actor repeat')
    assert calls==dict(actor=423,backward=9)
    assert seed.recorded_controls==0 and not np.any(seed.previous_action)
    assert all(not np.any(a) for a in seed.history.data.values())
    for name,digest in producer['output_sha256'].items():
        assert sha(generation/name)==digest,name
    assert sha(receipt)==audit_request['input_receipt_sha256']
    for name,digest in freeze['input_sha256'].items():
        assert sha(local(name))==digest,name
    for name,digest in freeze['source_sha256'].items():
        assert sha(source/name)==digest,name
    for name,digest in bindings:
        assert sha(name)==digest,name
    assert sha(core)==audit_request['core_sha256']
    assert sha(Path(__file__))==audit_request['source_sha256']
    assert sha(generation/'report.json')==audit_request['generation_report_sha256']
    report=dict(kind='independent_all_velocity_chord_array_and_fixed_actor_audit',passed=True,
                centers=3057,probes=140622,all_nominal_labels_and_committed_plan_bindings_byteexact=True,
                all_state_feature_target_and_branch_arrays_byteexact=True,static_native_speed_checks_passed=True,
                maximum_teacher_center_delta_rad=maximum_delta,clipping_counts=stats,
                fixed_BFM_repeats_byteexact=True,audit_inference_calls=calls,
                no_BFM_history_prior_mutation=True,physics_steps=0,optimizer_updates=0,
                probes_are_not_qualified_physical_trajectories=True,
                generation_report_sha256=sha(generation/'report.json'),request_sha256=sha(args.output/'request.json'),
                plan_and_label_hashes=dict(bindings),core_sha256=sha(core),source_sha256=sha(Path(__file__)))
    (args.output/'source.py').write_bytes(Path(__file__).read_bytes())
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    atomic_json(args.output/'progress.json',dict(stage='COMPLETE',checked_centers=3057,checked_probes=140622,
                                               inference_calls=calls,passed=True))
    print(json.dumps({k:v for k,v in report.items() if k!='plan_and_label_hashes'}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generation',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    args.active={}
    args.calls=dict(actor=0,backward=0)
    (args.output/'source.py').write_bytes(Path(__file__).read_bytes())
    try:
        main(args)
    except BaseException as error:
        np.savez_compressed(args.output/'failure_active.npz',**args.active)
        failure=dict(
            kind='independent_velocity_chord_audit_failure',passed=False,
            exception=type(error).__name__,message=str(error),traceback=traceback.format_exc(),
            inference_attempts=args.calls,active_sha256=sha(args.output/'failure_active.npz'),
            source_sha256=sha(args.output/'source.py'))
        atomic_json(args.output/'failure.json',failure)
        atomic_json(args.output/'report.json',failure)
        raise
