"""One fixed 3057-center/140622-probe BFM generation. No physics or optimizer."""
import argparse
import time
import numpy as np
from chord_common import (BASE,NEW,KEYS,ROWS,AXES,SIGNS,ACTOR_BUDGET,BACKWARD_BUDGET,
    read,write,write_atomic,sha,archive,exact,assert_clearance,assert_frozen,dataset_configs)
from saved_committed_map import difference_function,plans_for_dataset,committed_target

DEST = BASE/'generation'

class CountedSession:
    def __init__(self, session, name, counts, times):
        self.session,self.name,self.counts,self.times = session,name,counts,times
    def run(self, *args, **kwargs):
        if self.name not in ('actor','backward'):
            raise RuntimeError('Unexpected ONNX session call: '+self.name)
        limit = ACTOR_BUDGET if self.name=='actor' else BACKWARD_BUDGET
        if self.counts[self.name] >= limit:
            raise RuntimeError('ONNX call budget exhausted: '+self.name)
        self.counts[self.name] += 1
        started = time.perf_counter()
        try:
            return self.session.run(*args,**kwargs)
        finally:
            self.times[self.name] += time.perf_counter()-started

def bfm_actor(seed,qpos,qvel,previous,history,latent,active):
    state,_ = seed._terms(qpos,qvel,previous)
    active['returned_state']=state.copy()
    # Byte-identical actor input shapes and arithmetic to frozen infer_base.
    raw = seed.sessions['actor'].run(None,dict(state=state[None],last_action=previous[None],
        history=history[None],z=latent))[0][0]*5
    active['returned_base_action']=raw.copy()
    c=seed.contract
    target=c['default_q']+raw*.25*c['training_effort']/c['kp']
    active['returned_base_target']=target.copy()
    assert np.isfinite(raw).all() and np.isfinite(target).all()
    return raw,target,state

def save_failure(active,arrays,progress,exc):
    flush_errors=[]
    for name,value in arrays.items():
        try:value.flush()
        except BaseException as flush_error:flush_errors.append(dict(array=name,message=str(flush_error)))
    np.savez_compressed(DEST/'failure_active.npz',**{key:np.asarray(value) for key,value in active.items()})
    return dict(type=type(exc).__name__,message=str(exc),active_snapshot_sha256=sha(DEST/'failure_active.npz'),
        flush_errors=flush_errors,completed_centers=progress['centers_completed'],completed_probes=progress['probes_completed'],
        committed_prefix_order='C order (center3057,axis23,sign2[-1,+1]); only flat probe indices strictly below completed_probes are committed',
        active_probe_committed=bool('flat_probe_index' in active and int(active['flat_probe_index'])<progress['probes_completed']),
        first_uncommitted_probe_index=progress['probes_completed'])

def seed_unmutated(seed, history):
    assert seed.recorded_controls == 0
    exact(seed.previous_action,np.zeros(23,np.float32),'unmutated seed prior')
    for key in KEYS:
        exact(seed.history.data[key],history[key],'unmutated seed '+key)

def reconstruct_centers(seed,c,limits,difference):
    outputs=[]
    for dataset,config in enumerate(dataset_configs()):
        labels=archive(config['labels']);trace=archive(config['trace'])
        ids=np.flatnonzero((labels['control']>=250)&(labels['control']<1269))
        np.testing.assert_array_equal(labels['control'][ids],np.arange(250,1269))
        np.testing.assert_array_equal(labels['source_frame'][ids],np.arange(261,1280))
        np.testing.assert_array_equal(labels['teacher_qpos'][:-1][ids],trace['qpos'][250:1269])
        np.testing.assert_array_equal(labels['teacher_qvel'][:-1][ids],trace['qvel'][250:1269])
        exact(labels['joint_limits'],limits,'native limits')
        exact(labels['joint_span'],np.diff(limits,axis=1).ravel().astype(np.float32),'native spans')
        plans=plans_for_dataset(config['name'],trace)
        named={key:np.zeros_like(seed.history.data[key]) for key in KEYS}
        previous=np.zeros(23,np.float32)
        for control in range(1269):
            qpos,qvel=trace['qpos'][control],trace['qvel'][control]
            state,terms=seed._terms(qpos,qvel,previous)
            history=np.concatenate([named[key].reshape(-1) for key in KEYS]).copy()
            if control>=250:
                at=int(ids[control-250]);plan=plans[control]
                for key,actual in [('state',state),('history',history),('previous_action',previous),
                    ('expert_target',trace['target'][control])]:
                    exact(actual,labels[key][at],config['name']+' '+str(control)+' '+key)
                for key in KEYS:
                    if 'history_'+key in labels:
                        exact(named[key],labels['history_'+key][at],'named history '+key)
                target,raw,correction,preclip=committed_target(difference,plan['planned_state'],
                    plan['planned_target'],plan['gain'],qpos,qvel,limits)
                exact(target,labels['expert_target'][at],'full nominal committed target')
                row={key:labels[key][at].copy() for key in ['features','residual_rad','base_target','expert_target',
                    'base_action','previous_action','history','state']}
                row.update(dataset=np.int64(dataset),control=np.int64(control),source_frame=np.int64(control+11),
                    qpos=qpos.copy(),qvel=qvel.copy(),plan_control=np.int64(plan['plan_control']),
                    plan_local=np.int64(plan['local']),plan_accepted_update=np.int64(plan['accepted_update']),
                    planned_state=plan['planned_state'],planned_target=plan['planned_target'],gain=plan['gain'],
                    teacher_feedback_raw=raw,teacher_feedback_correction=correction,teacher_preclip=preclip,
                    **{'history_'+key:named[key].copy() for key in KEYS})
                outputs.append(row)
            # Pure provenance reconstruction; no BFMHistory method or actor call.
            for key in KEYS:
                named[key][1:]=named[key][:-1].copy()
                named[key][0]=terms[key]
            use_raw = (dataset==1 and control==0) or (dataset==2 and control<250)
            previous=(trace['action'][control].copy() if use_raw else
                ((trace['target'][control]-np.asarray(c['default_q']))*np.asarray(c['kp'])/
                 (.25*np.asarray(c['training_effort']))).astype(np.float32))
    assert len(outputs)==ROWS
    return {key:np.asarray([row[key] for row in outputs]) for key in outputs[0]}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--clearance',required=True)
    args=parser.parse_args();receipt,clearance=assert_clearance(args.clearance)
    DEST.mkdir(exist_ok=False)
    counts=dict(actor=0,backward=0);times=dict(actor=0.,backward=0.)
    started=time.perf_counter();progress=dict(stage='STARTING',centers_completed=0,probes_completed=0,
        actor_calls=0,backward_calls=0,optimizer_calls=0,physics_steps=0,
        committed_probe_prefix_order='C order (center,axis,sign[-1,+1]); flat indices < probes_completed only')
    arrays={};active={'stage':np.asarray('STARTING')}
    def persist(stage=None):
        if stage:progress['stage']=stage
        progress.update(actor_calls=counts['actor'],backward_calls=counts['backward'],elapsed_seconds=time.perf_counter()-started)
        write_atomic(DEST/'progress.json',progress)
    try:
        import mujoco
        def forbidden_step(*args,**kwargs):raise RuntimeError('Physics stepping is prohibited in chord generation')
        mujoco.mj_step=mujoco.mj_step1=mujoco.mj_step2=forbidden_step
        from student_linear_runtime import (LinearFeatures,BUNDLE,REFERENCE,ONNX,DEPS,Native23BFMRolloutSeed)
        from gear_sonic.utils.g1_true23_mjbatch_mpc import load_native_bundle,load_motion_override
        assert mujoco.__version__=='3.2.3'
        assert np.__version__=='1.26.4'
        native,c,original,timeline,manifest=load_native_bundle(BUNDLE,'walk003')
        motion,_=load_motion_override(REFERENCE,BUNDLE,'walk003',native,c,original,timeline,manifest)
        original29=archive(BUNDLE/'walk003/original29.npz')
        seed=Native23BFMRolloutSeed(native,c,original,ONNX,dependency_directory=DEPS,threads=1)
        initial_history={k:seed.history.data[k].copy() for k in KEYS}
        for name,session in list(seed.sessions.items()):
            seed.sessions[name]=CountedSession(session,name,counts,times)
        limits=np.asarray(c['joint_limits']);caps=np.asarray(c['native_velocity'])
        span=np.diff(limits,axis=1).ravel().astype(np.float32)
        builder=LinearFeatures(motion,original29,c);difference=difference_function()
        centers=reconstruct_centers(seed,c,limits,difference)
        assert np.all(np.abs(centers['qvel'][:,6:])+.01*caps<caps)
        seed_unmutated(seed,initial_history)
        cached=[]
        persist('VERIFYING_ALL_NOMINAL_BFM_CENTERS')
        for i in range(ROWS):
            q,v,previous,history=tuple(centers[key][i] for key in ('qpos','qvel','previous_action','history'))
            active=dict(stage=np.asarray('CENTER'),center=np.int64(i),dataset=centers['dataset'][i],control=centers['control'][i],
                qpos=q.copy(),qvel=v.copy(),previous_action=previous.copy(),history=history.copy(),
                **{'history_'+k:centers['history_'+k][i].copy() for k in KEYS},
                **{'expected_'+k:centers[k][i].copy() for k in ('base_action','base_target','state','features')})
            z=seed._goal(int(centers['source_frame'][i]),q)
            active['goal_latent']=z.copy()
            raw,base,state=bfm_actor(seed,q,v,previous,history,z,active)
            x=builder(q,v,int(centers['source_frame'][i]),base,previous)
            active['returned_features']=x.copy()
            for key,actual in [('base_action',raw),('base_target',base),('state',state),('features',x)]:
                exact(actual,centers[key][i],'BFM center '+str(i)+' '+key)
            exact(centers['history'][i],np.concatenate([centers['history_'+k][i].reshape(-1) for k in KEYS]),'center named flatten')
            cached.append(z.copy());progress['centers_completed']=i+1
            if (i+1)%100==0:persist()
        assert counts==dict(actor=ROWS,backward=ROWS)
        seed_unmutated(seed,initial_history)
        centers['goal_latent']=np.asarray(cached)
        centers['joint_span']=span;centers['joint_limits']=limits;centers['native_velocity']=caps
        np.savez_compressed(DEST/'centers.npz',**centers)
        write(DEST/'center_parity.json',dict(all_3057_center_targets_base_state_history_prior_features_byteexact=True,
            actor_calls=counts['actor'],backward_calls=counts['backward'],centers_sha256=sha(DEST/'centers.npz'),
            no_probes_attempted=True,seed_history_prior_count_unmutated=True))
        # Probe arrays are created only after every original center passed.
        specs={'features':(1069,np.float32),'base_target':(23,np.float64),'base_action':(23,np.float32),
            'state':(52,np.float32),'teacher_target':(23,np.float64),'teacher_feedback_raw':(23,np.float64),
            'teacher_feedback_clipped':(23,np.bool_),'teacher_native_clipped':(23,np.bool_),
            'teacher_branch_changed':(23,np.bool_),'perturbed_joint_velocity':(None,np.float64)}
        arrays={key:np.lib.format.open_memmap(DEST/(key+'.npy'),mode='w+',dtype=dtype,
            shape=(ROWS,AXES,2)+(() if width is None else (width,))) for key,(width,dtype) in specs.items()}
        persist('GENERATING_FIXED_AXIS_PROBES')
        for i in range(ROWS):
            q,v,previous,history=tuple(centers[key][i] for key in ('qpos','qvel','previous_action','history'))
            center_feedback=centers['teacher_feedback_raw'][i]
            center_preclip=centers['teacher_preclip'][i]
            center_feedback_branch=np.where(center_feedback<-.1,-1,np.where(center_feedback>.1,1,0))
            center_native_branch=np.where(center_preclip<limits[:,0],-1,np.where(center_preclip>limits[:,1],1,0))
            for j in range(AXES):
                changed_features=np.zeros(1069,bool);changed_features[23+j]=True;changed_features[1023:1046]=True
                for sign_index,sign in enumerate(SIGNS):
                    changed=v.copy();changed[6+j]+=sign*.01*caps[j]
                    active=dict(stage=np.asarray('PROBE'),center=np.int64(i),dataset=centers['dataset'][i],
                        control=centers['control'][i],axis=np.int64(j),sign_index=np.int64(sign_index),sign=np.float64(sign),
                        flat_probe_index=np.int64((i*AXES+j)*2+sign_index),qpos=q.copy(),qvel=changed.copy(),
                        nominal_qvel=v.copy(),previous_action=previous.copy(),history=history.copy(),goal_latent=centers['goal_latent'][i].copy(),
                        **{'history_'+k:centers['history_'+k][i].copy() for k in KEYS})
                    assert np.all(np.abs(changed[6:])<caps)
                    assert np.count_nonzero(changed!=v)==1
                    raw,base,state=bfm_actor(seed,q,changed,previous,history,centers['goal_latent'][i],active)
                    x=builder(q,changed,int(centers['source_frame'][i]),base,previous)
                    active['returned_features']=x.copy()
                    exact(x[~changed_features],centers['features'][i,~changed_features],'unchanged probe feature mask')
                    mask=np.ones(52,bool);mask[23+j]=False
                    exact(state[mask],centers['state'][i,mask],'unchanged BFM state mask')
                    target,feedback,correction,preclip=committed_target(difference,centers['planned_state'][i],
                        centers['planned_target'][i],centers['gain'][i],q,changed,limits)
                    active.update(returned_teacher_target=target.copy(),returned_teacher_feedback_raw=feedback.copy(),
                        returned_teacher_feedback_correction=correction.copy(),returned_teacher_preclip=preclip.copy())
                    assert np.isfinite(target).all() and np.isfinite(feedback).all() and np.isfinite(x).all()
                    assert np.all(target>=limits[:,0]) and np.all(target<=limits[:,1])
                    assert np.max(np.abs(target-centers['expert_target'][i]))<=.2+1e-12
                    feedback_branch=np.where(feedback<-.1,-1,np.where(feedback>.1,1,0))
                    native_branch=np.where(preclip<limits[:,0],-1,np.where(preclip>limits[:,1],1,0))
                    values=dict(features=x,base_target=base,base_action=raw,state=state,teacher_target=target,
                        teacher_feedback_raw=feedback,teacher_feedback_clipped=feedback!=correction,
                        teacher_native_clipped=preclip!=target,
                        teacher_branch_changed=(feedback_branch!=center_feedback_branch)|(native_branch!=center_native_branch),
                        perturbed_joint_velocity=changed[6+j])
                    for key,value in values.items():arrays[key][i,j,sign_index]=value
                    progress['probes_completed']+=1
            seed_unmutated(seed,initial_history)
            if (i+1)%25==0:
                for value in arrays.values():value.flush()
                persist()
                print('centers',ROWS,'probe_center',i+1,'actor',counts['actor'],'backward',counts['backward'],flush=True)
        for value in arrays.values():value.flush()
        assert counts==dict(actor=ACTOR_BUDGET,backward=BACKWARD_BUDGET)
        assert progress['probes_completed']==ROWS*AXES*2
        seed_unmutated(seed,initial_history)
        persist('VERIFYING_FINAL_FROZEN_INPUTS')
        final_receipt=assert_frozen()
        assert final_receipt==receipt
        result=dict(kind='one_exact_bounded_velocity_chord_generation',complete=True,centers=ROWS,axis_pairs=ROWS*AXES,
            probe_rows=ROWS*AXES*2,total_rows=ROWS*(1+AXES*2),sign_order=list(SIGNS),axis_native_speed_radius=.01,
            all_center_byte_parity=True,all_feature_masks_byte_parity=True,all_static_native_velocity_checks_passed=True,
            all_exact_committed_targets_with_both_original_clips=True,teacher_center_delta_bound_rad=.2,
            feedback_clipped_probes=int(np.any(arrays['teacher_feedback_clipped'],axis=-1).sum()),
            feedback_clipped_components=int(arrays['teacher_feedback_clipped'].sum()),
            native_clipped_probes=int(np.any(arrays['teacher_native_clipped'],axis=-1).sum()),
            native_clipped_components=int(arrays['teacher_native_clipped'].sum()),
            branch_changed_probes=int(np.any(arrays['teacher_branch_changed'],axis=-1).sum()),
            zero_gain_centers=int(np.all(centers['gain']==0,axis=(1,2)).sum()),
            inference_calls=counts,inference_seconds=times,seed_history_prior_count_unmutated=True,
            teacher_scope='exact clipped already-committed map; no replan or perturbed-trajectory feasibility claim',
            input_frozen_receipt_sha256=sha(BASE/'generation_frozen_inputs_v2.json'),clearance_sha256=sha(args.clearance),
            all_frozen_sources_and_inputs_rehashed_at_completion=True,
            output_sha256={path.name:sha(path) for path in sorted(DEST.iterdir()) if path.suffix in ('.npz','.npy')},
            source_generation_sha256=sha(__file__),optimizer_updates=0,physics_steps=0,hardware_authorized=False,
            elapsed_seconds=time.perf_counter()-started)
        write(DEST/'report.json',result);persist('COMPLETE')
        print('GENERATION COMPLETE',counts,flush=True)
    except BaseException as exc:
        progress['failure']=save_failure(active,arrays,progress,exc);persist('FAILED')
        raise

if __name__=='__main__':main()
